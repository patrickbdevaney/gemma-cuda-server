// test_linear_real.cu — W4A4 gate: GPU activation-quantize + REAL checkpoint weight -> nvfp4_gemm
// vs numpy reference (scripts/nvfp4_ref.py). Exercises loader + quantizer + GEMM end-to-end on real data.
#include "fp4_gemm.h"
#include "nvfp4_quant.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

static std::vector<uint8_t> rd(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb"); if (!f) { fprintf(stderr, "open %s\n", p.c_str()); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> v(n); if (fread(v.data(), 1, n, f) != (size_t)n) exit(1); fclose(f); return v;
}
#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "/tmp/lincase";
    int M, N, K; float w_gscale, gs_X;
    { FILE* f = fopen((dir + "/dims.txt").c_str(), "r");
      if (!f || fscanf(f, "%d %d %d %e %e", &M, &N, &K, &w_gscale, &gs_X) != 5) { fprintf(stderr,"dims\n"); exit(1);} fclose(f);}
    printf("M=%d N=%d K=%d  w_gscale=%.6e gs_X=%.6e\n", M, N, K, w_gscale, gs_X);

    auto X   = rd(dir + "/X.bin");          // fp32 [M,K]
    auto Wp  = rd(dir + "/W_packed.bin");   // [N,K/2]
    auto Wsf = rd(dir + "/W_scales.bin");   // [N,K/16] e4m3 logical
    auto Yr  = rd(dir + "/Y_ref.bin");      // fp32 [M,N] row-major
    const float* yref = (const float*)Yr.data();

    // swizzle real weight scales on host
    size_t Wsz = nvfp4_scale_buffer_bytes(N, K);
    std::vector<uint8_t> Wsw(Wsz);
    nvfp4_swizzle_scales(Wsf.data(), Wsw.data(), N, K);

    // device alloc
    float* dX; uint8_t *dXp, *dXs, *dWp, *dWs; float* dD; void* ws;
    size_t Xsz = nvfp4_scale_buffer_bytes(M, K);
    CU(cudaMalloc(&dX, X.size())); CU(cudaMalloc(&dXp, (size_t)M*K/2)); CU(cudaMalloc(&dXs, Xsz));
    CU(cudaMalloc(&dWp, Wp.size())); CU(cudaMalloc(&dWs, Wsz)); CU(cudaMalloc(&dD,(size_t)M*N*4));
    size_t wsb = 32u<<20; CU(cudaMalloc(&ws, wsb));
    CU(cudaMemcpy(dX, X.data(), X.size(), cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dWp, Wp.data(), Wp.size(), cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dWs, Wsw.data(), Wsz, cudaMemcpyHostToDevice));
    CU(cudaMemset(dXs, 0, Xsz));

    // GPU quantize activation
    nvfp4_quantize_activations(dX, dXp, dXs, M, K, gs_X, 0);
    CU(cudaDeviceSynchronize());

    // optional: compare GPU activation quant codes vs numpy
    { auto Xpr = rd(dir + "/Xq_packed_ref.bin");
      std::vector<uint8_t> Xpg((size_t)M*K/2); CU(cudaMemcpy(Xpg.data(), dXp, Xpg.size(), cudaMemcpyDeviceToHost));
      size_t diff = 0; for (size_t i=0;i<Xpg.size();++i) if (Xpg[i]!=Xpr[i]) diff++;
      printf("activation-quant code mismatch vs numpy: %zu/%zu (%.3f%%)\n", diff, Xpg.size(), 100.0*diff/Xpg.size()); }

    cublasLtHandle_t lt; cublasLtCreate(&lt);
    // real convention: dequant divides by global_scale -> pass reciprocals so alpha = 1/(gs_X*w_gscale)
    int rc = nvfp4_gemm(dD, dXp, dXs, 1.0f/gs_X, dWp, dWs, 1.0f/w_gscale, M, N, K, lt, ws, wsb, 0);
    CU(cudaDeviceSynchronize());
    if (rc) { fprintf(stderr, "gemm rc=%d\n", rc); return 1; }

    std::vector<float> D((size_t)M*N);
    CU(cudaMemcpy(D.data(), dD, D.size()*4, cudaMemcpyDeviceToHost)); // col-major [M,N] ld=M

    double max_abs=0, sum_abs=0, ref_amax=0, sum_ref=0;
    for (int i=0;i<M*N;++i){ ref_amax=fmax(ref_amax,fabs(yref[i])); sum_ref+=fabs(yref[i]); }
    for (int m=0;m<M;++m) for (int n=0;n<N;++n){
        double got=D[(size_t)m+(size_t)n*M], exp=yref[(size_t)m*N+n], ad=fabs(got-exp);
        sum_abs+=ad; if(ad>max_abs)max_abs=ad;
    }
    double rel_l1 = sum_abs / (sum_ref + 1e-9);
    printf("ref_amax=%.4f  max_abs=%.5f  mean_abs=%.6e  rel_L1=%.5e\n",
           ref_amax, max_abs, sum_abs/(M*N), rel_l1);
    printf("D[0,0]=%.5f ref=%.5f | D[3,7]=%.5f ref=%.5f\n", D[0], yref[0], D[3+7*M], yref[3*N+7]);
    bool pass = rel_l1 < 1e-3 && max_abs < 5e-3 * (ref_amax + 1.0);
    printf("%s\n", pass ? "GATE PASS ✅" : "GATE FAIL ❌");
    return pass ? 0 : 2;
}
