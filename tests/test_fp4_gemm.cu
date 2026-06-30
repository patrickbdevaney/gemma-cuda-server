// test_fp4_gemm.cu — Phase 3.1 gate: cublasLt NVFP4 GEMM vs numpy float64 reference.
// Loads the synthetic testcase from scripts/gen_fp4_gemm_testcase.py and compares.
#include "fp4_gemm.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

static std::vector<uint8_t> readfile(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", p.c_str()); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> v(n); if (fread(v.data(), 1, n, f) != (size_t)n) { exit(1); } fclose(f);
    return v;
}
#define CU(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){fprintf(stderr,"cuda err %s at %d: %s\n",#x,__LINE__,cudaGetErrorString(e_));exit(1);} } while(0)

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "/tmp/fp4case";
    int M, N, K; float Ag, Bg;
    { FILE* f = fopen((dir + "/dims.txt").c_str(), "r");
      if (!f || fscanf(f, "%d %d %d %e %e", &M, &N, &K, &Ag, &Bg) != 5) { fprintf(stderr, "bad dims\n"); exit(1); }
      fclose(f); }
    printf("M=%d N=%d K=%d  Ag=%.6e Bg=%.6e\n", M, N, K, Ag, Bg);

    auto Apk = readfile(dir + "/A_packed.bin");   // [M, K/2]
    auto Bpk = readfile(dir + "/B_packed.bin");   // [N, K/2]
    auto Asf = readfile(dir + "/A_scales.bin");   // [M, K/16] e4m3 logical
    auto Bsf = readfile(dir + "/B_scales.bin");   // [N, K/16] e4m3 logical
    auto Yref = readfile(dir + "/Y_ref.bin");     // [M, N] float32 row-major
    const float* yref = (const float*)Yref.data();

    // swizzle scales (host)
    size_t Asz = nvfp4_scale_buffer_bytes(M, K), Bsz = nvfp4_scale_buffer_bytes(N, K);
    std::vector<uint8_t> Asw(Asz), Bsw(Bsz);
    nvfp4_swizzle_scales(Asf.data(), Asw.data(), M, K);
    nvfp4_swizzle_scales(Bsf.data(), Bsw.data(), N, K);

    // device buffers
    uint8_t *dA, *dB, *dAs, *dBs; float* dD; void* ws;
    CU(cudaMalloc(&dA, Apk.size())); CU(cudaMalloc(&dB, Bpk.size()));
    CU(cudaMalloc(&dAs, Asz)); CU(cudaMalloc(&dBs, Bsz));
    CU(cudaMalloc(&dD, (size_t)M * N * sizeof(float)));
    size_t wsb = 32u << 20; CU(cudaMalloc(&ws, wsb));
    CU(cudaMemcpy(dA, Apk.data(), Apk.size(), cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dB, Bpk.data(), Bpk.size(), cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dAs, Asw.data(), Asz, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dBs, Bsw.data(), Bsz, cudaMemcpyHostToDevice));

    cublasLtHandle_t lt; cublasLtCreate(&lt);
    int rc = nvfp4_gemm(dD, dA, dAs, Ag, dB, dBs, Bg, M, N, K, lt, ws, wsb, 0);
    CU(cudaDeviceSynchronize());
    if (rc != 0) { fprintf(stderr, "nvfp4_gemm failed rc=%d\n", rc); return 1; }

    std::vector<float> D((size_t)M * N);
    CU(cudaMemcpy(D.data(), dD, D.size() * sizeof(float), cudaMemcpyDeviceToHost)); // col-major [M,N] ld=M

    // compare D[m + n*M] (col-major) vs yref[m*N + n] (row-major)
    double max_abs = 0, max_rel = 0, sum_abs = 0; int nbad = 0;
    double ref_amax = 0; for (int i = 0; i < M * N; ++i) ref_amax = fmax(ref_amax, fabs(yref[i]));
    for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
        double got = D[(size_t)m + (size_t)n * M], exp = yref[(size_t)m * N + n];
        double ad = fabs(got - exp); sum_abs += ad;
        double rel = ad / (fabs(exp) + 1e-6);
        if (ad > max_abs) max_abs = ad;
        if (rel > max_rel) max_rel = rel;
        if (ad > 1e-2 * (fabs(exp) + ref_amax * 1e-3)) nbad++;
    }
    printf("ref_amax=%.5f  max_abs=%.6f  mean_abs=%.6e  max_rel=%.4e  nbad=%d/%d\n",
           ref_amax, max_abs, sum_abs / (M * N), max_rel, nbad, M * N);
    // sample
    printf("sample D[0,0]=%.5f ref=%.5f | D[1,2]=%.5f ref=%.5f\n",
           D[0], yref[0], D[1 + 2 * M], yref[1 * N + 2]);

    bool pass = (max_abs < 1e-3 * (ref_amax + 1.0)) && nbad == 0;
    printf("%s\n", pass ? "GATE PASS ✅" : "GATE FAIL ❌");
    return pass ? 0 : 2;
}
