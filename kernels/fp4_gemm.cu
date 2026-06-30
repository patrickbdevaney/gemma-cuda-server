// fp4_gemm.cu — NVFP4 block-scaled GEMM via cublasLt (CUDA 13, Blackwell sm_110a).
// Recipe per reference/CUBLASLT_FP4_RECIPE.md (cuBLAS 13.3 "1D Block Scaling Factors Layout").
#include "fp4_gemm.h"
#include <cstdio>
#include <cstring>

// ---- scale-factor swizzle (logical [outer, K/16] E4M3 -> 128x4-tile layout) ----
// Tile = 128 outer x 4 block-cols = 512 E4M3 bytes. Tiles row-major over (outer/128, block/4).
// within-tile = (m%32)*16 + ((m%128)/32)*4 + (b%4).
static inline int round_up(int x, int m) { return (x + m - 1) / m * m; }

size_t nvfp4_scale_buffer_bytes(int outer, int K) {
    int KB = K / 16;                       // blocks along K
    int tile_rows = round_up(outer, 128) / 128;
    int tile_cols = round_up(KB, 4) / 4;
    return (size_t)tile_rows * tile_cols * 512;
}

void nvfp4_swizzle_scales(const uint8_t* logical, uint8_t* swizzled, int outer, int K) {
    int KB = K / 16;
    int tile_cols = round_up(KB, 4) / 4;
    size_t total = nvfp4_scale_buffer_bytes(outer, K);
    memset(swizzled, 0, total);
    for (int m = 0; m < outer; ++m) {
        for (int b = 0; b < KB; ++b) {
            size_t tile = (size_t)((m / 128) * tile_cols + (b / 4));
            int within = (m % 32) * 16 + ((m % 128) / 32) * 4 + (b % 4);
            swizzled[tile * 512 + within] = logical[(size_t)m * KB + b];
        }
    }
}

#define LT_CHECK(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cublasLt error %d at %s:%d\n", (int)s_, __FILE__, __LINE__); return (int)s_; } } while(0)

// ---- FP4-weight GEMV for single-token decode (W4A16: FP4 weight x fp32 activation) ----
// y[N] = sum_k (fp4(wp[n,k]) * e4m3(ws[n,k/16]) / w_gscale) * x[k].  Raw (unswizzled) weight+scales.
__device__ __forceinline__ float e2m1_dec(uint8_t c){ const float t[8]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f}; float v=t[c&7]; return (c&8)?-v:v; }
__device__ __forceinline__ float e4m3_dec(uint8_t b){ int s=(b>>7)&1,e=(b>>3)&0xF,man=b&7; float v=(e==0)?(man*0.125f*0.015625f):((1.f+man*0.125f)*exp2f((float)(e-7))); return s?-v:v; }
__global__ void fp4_gemv_kernel(float* y, const uint8_t* wp, const uint8_t* ws, float wg_inv,
                                const float* x, int N, int K){
    int n=blockIdx.x; if(n>=N) return;
    const uint8_t* wpn=wp+(size_t)n*(K/2); const uint8_t* wsn=ws+(size_t)n*(K/16);
    __shared__ float red[256]; float acc=0.f;
    for(int k=threadIdx.x;k<K;k+=blockDim.x){
        uint8_t byte=wpn[k>>1]; uint8_t code=(k&1)?(byte>>4):(byte&0xF);
        acc += e2m1_dec(code) * (e4m3_dec(wsn[k>>4])*wg_inv) * x[k];
    }
    red[threadIdx.x]=acc; __syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s)red[threadIdx.x]+=red[threadIdx.x+s]; __syncthreads(); }
    if(threadIdx.x==0) y[n]=red[0];
}
void fp4_gemv(float* y, const uint8_t* wp, const uint8_t* ws, float w_gscale, const float* x,
              int N, int K, cudaStream_t s){
    fp4_gemv_kernel<<<N,256,0,s>>>(y, wp, ws, 1.0f/w_gscale, x, N, K);
}

int nvfp4_gemm(float* dD,
               const uint8_t* dA_packed, const uint8_t* dA_scales_swz, float a_gscale,
               const uint8_t* dB_packed, const uint8_t* dB_scales_swz, float b_gscale,
               int M, int N, int K,
               cublasLtHandle_t lt, void* workspace, size_t ws_bytes, cudaStream_t stream) {
    // TN GEMM, all operands col-major:
    //   A described [K,M] col-major (= our [M,K] row-major packed), op_T -> [M,K]
    //   B described [K,N] col-major (= our [N,K] row-major packed), op_N -> [K,N]
    //   D = [M,N] col-major, ld=M
    cublasLtMatmulDesc_t op;
    LT_CHECK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
    LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
    LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

    void* aptr = (void*)dA_scales_swz; void* bptr = (void*)dB_scales_swz;
    int32_t vmode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &aptr, sizeof(aptr)));
    LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &bptr, sizeof(bptr)));
    LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &vmode, sizeof(vmode)));
    LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &vmode, sizeof(vmode)));

    cublasLtMatrixLayout_t lA, lB, lD;
    LT_CHECK(cublasLtMatrixLayoutCreate(&lA, CUDA_R_4F_E2M1, K, M, K)); // [K,M] col-major
    LT_CHECK(cublasLtMatrixLayoutCreate(&lB, CUDA_R_4F_E2M1, K, N, K)); // [K,N] col-major
    LT_CHECK(cublasLtMatrixLayoutCreate(&lD, CUDA_R_32F, M, N, M));     // [M,N] col-major

    // global per-tensor scales fold into alpha (block mode has no separate A/B global input)
    float alpha = a_gscale * b_gscale, beta = 0.0f;

    cublasLtMatmulPreference_t pref;
    LT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    LT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                  &ws_bytes, sizeof(ws_bytes)));
    cublasLtMatmulHeuristicResult_t heur[8];
    int nres = 0;
    LT_CHECK(cublasLtMatmulAlgoGetHeuristic(lt, op, lA, lB, lD, lD, pref, 8, heur, &nres));
    if (nres == 0) { fprintf(stderr, "nvfp4_gemm: no cublasLt algo found (M=%d N=%d K=%d)\n", M, N, K); return -100; }

    cublasStatus_t st = cublasLtMatmul(lt, op, &alpha,
                                       dA_packed, lA, dB_packed, lB,
                                       &beta, dD, lD, dD, lD,
                                       &heur[0].algo, workspace, ws_bytes, stream);

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(lA); cublasLtMatrixLayoutDestroy(lB); cublasLtMatrixLayoutDestroy(lD);
    cublasLtMatmulDescDestroy(op);
    LT_CHECK(st);
    return 0;
}
