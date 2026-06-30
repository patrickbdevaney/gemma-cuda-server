# cuBLASLt block-scaled NVFP4 (FP4 E2M1, block-16, FP8-E4M3 scales) GEMM recipe

Target: NVIDIA Blackwell (SM100/SM120; compute capability 10.0+), CUDA 13.0, cublasLt 13.x.

Ground truth used:
- Local header `/usr/local/cuda-13.0/include/cublasLt.h` (enum line numbers cited inline).
- Local header `/usr/local/cuda-13.0/include/library_types.h` (data-type enums).
- cuBLAS 13.3 Library PDF, section "16/32-Element 1D Block Scaling for FP8 and FP4 Data Types" ->
  "1D Block Scaling Factors Layout" (https://docs.nvidia.com/cuda/pdf/CUBLAS_Library.pdf). LAYOUT QUOTED VERBATIM below.
- NVIDIA CUDALibrarySamples `cuBLASLt/LtNvfp4Matmul` (sample_cublasLt_LtNvfp4Matmul.cu / .h / main.cpp).
- NVIDIA blog "Boosting Matrix Multiplication ... cuBLAS 12.9".

--------------------------------------------------------------------------------
## 1. cudaDataType enums (library_types.h)
- FP4 operands A, B, D tensors:    `CUDA_R_4F_E2M1`  = 33   (nv_fp4_e2m1, 2 values packed per byte)
- NVFP4 per-block scale factors:   `CUDA_R_8F_UE4M3` = 28   (== `CUDA_R_8F_E4M3`; unsigned alias, sign bit ignored)
- MXFP scale factors (block-32):   `CUDA_R_8F_UE8M0` = 30   (exponent-only; NOT used for NVFP4)
- Global / D-input scalar scale:   `CUDA_R_32F`             (FP32 tensorwide scale)
- C tensor in the NVIDIA sample:   `CUDA_R_16BF`            (bf16 accumulate-in target; C may also be fp32/fp16)

The C++ pointer types in the sample: A/B/D data are `__nv_fp4_e2m1` (stored via `StorageType<>::type`,
i.e. packed bytes); A/B/D-out block scales are `__nv_fp8_e4m3*`; the D input scalar scale is `float*`.

## 2. Descriptor: compute type, scale type, attributes
- `cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F)`  (computeType=CUBLAS_COMPUTE_32F, scaleType=CUDA_R_32F)
- Transpose: sample uses `CUBLASLT_MATMUL_DESC_TRANSA = CUBLAS_OP_T`, `TRANSB = CUBLAS_OP_N` (TN GEMM, the supported FP4 form).

Scale-MODE attributes (cublasLtMatmulDescAttributes_t; header lines 1545-1581). Value is `int32_t`
holding a `cublasLtMatmulMatrixScale_t`:
- `CUBLASLT_MATMUL_DESC_A_SCALE_MODE` (31)      <- `CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3`
- `CUBLASLT_MATMUL_DESC_B_SCALE_MODE` (32)      <- `CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3`
- `CUBLASLT_MATMUL_DESC_C_SCALE_MODE` (33)      <- `CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F` (if C scaled)
- `CUBLASLT_MATMUL_DESC_D_SCALE_MODE` (34)      <- `CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F`  (FP32 input scalein_D)
- `CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE` (37)  <- `CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3` (only if D is FP4 and you want cuBLAS to emit per-block output scales)

Scale-POINTER attributes (header lines 1423-1576), all device pointers:
- `CUBLASLT_MATMUL_DESC_A_SCALE_POINTER` (17)    -> A block scales (`__nv_fp8_e4m3*`)
- `CUBLASLT_MATMUL_DESC_B_SCALE_POINTER` (18)    -> B block scales (`__nv_fp8_e4m3*`)
- `CUBLASLT_MATMUL_DESC_C_SCALE_POINTER` (19)    -> C scale (optional)
- `CUBLASLT_MATMUL_DESC_D_SCALE_POINTER` (20)    -> D input FP32 scalar scale (`float*`, scalein_D, "compresses" values pre-quant)
- `CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER` (36)-> where cuBLAS WRITES output block scales (`__nv_fp8_e4m3*`)

cublasLtMatmulMatrixScale_t values (header lines 914-934):
  0 SCALAR_32F | 1 VEC16_UE4M3 (NVFP4, block 16, UE4M3 scales) | 2 VEC32_UE8M0 (MXFP, block 32, UE8M0) |
  3 OUTER_VEC_32F | 4 VEC128_32F | 5 BLK128x128_32F.
NVFP4 == mode 1; MXFP4/MXFP8 == mode 2. Mixing FP4 precision with non-VEC16 scale modes errors.

NVFP4 two-level scaling note: cuBLAS gives ONE block scale per operand (the UE4M3 VEC16). The per-tensor
GLOBAL fp32 dequant scales for A and B (A_gscale, B_gscale) are NOT separate cuBLAS inputs in block mode --
fold them into alpha:  alpha_eff = alpha * A_gscale * B_gscale. The global scale for the OUTPUT is the
FP32 D_SCALE_POINTER (scalein_D = Amax(e2m1)*Amax(e4m3)/Amax(D)).

## 3. *** SCALE-FACTOR LAYOUT (THE CRITICAL PART) -- verbatim from cuBLAS 13.3 docs ***
"Scaling factors are stored using a tiled layout." Each tile is **128 (outer) x 4 (inner) scale factors**.
A single tile covers a **128 x 64** region of the data tensor for VEC16_UE4M3 (128 rows x 4 blocks x 16
elems), and 128 x 128 for VEC32_UE8M0.

Coordinate convention: `inner` = K for A and B (M for C/D); `outer` = M for A (N for B,C,D).

Within ONE 128x4 tile -- verbatim pseudocode:
```
// Indices -> offset
offset = (outer % 32) * 16 + (outer / 32) * 4 + inner
// Offset -> Indices
outer = ((offset % 16) / 4) * 32 + (offset / 16)
inner = (offset % 4)
```
Across tiles, tiles are arranged ROW-MAJOR. For a scale tensor with `sf_inner_dim` scale factors per row
(i.e. blocks-per-row padded up to a multiple of 4), the base offset of the tile whose top-left scale-factor
coordinate is `(sf_outer, sf_inner)` is -- verbatim:
```
// note sf_inner is a multiple of 4 due to the tiling layout
offset = (sf_inner + sf_outer * sf_inner_dim) * 128
```
=> each tile occupies 128*4 = 512 contiguous scale bytes.

FULL element mapping for a scale at matrix coords (outer=row m, inner=block index b = k/16):
```
sf_inner_dim   = roundUp(numBlocksPerRow, 4)         // numBlocksPerRow = K/16 for A,B
tile_base      = ( (b & ~3) + (m & ~127) * sf_inner_dim ) * 128
within_tile    = (m % 32) * 16 + ((m % 128) / 32) * 4 + (b % 4)
phys_offset    = tile_base + within_tile             // in units of 1 scale byte (UE4M3)
```

PADDING / ALLOCATION rules (verbatim): "when tensor dimensions are not multiples of the tile size ... it
is necessary to still allocate full tile for storage and fill out of bounds values with zeroes." So:
- outer (M for A; N for B) is padded UP to a multiple of **128**.
- inner block count (K/16) is padded UP to a multiple of **4** (= sf_inner_dim).
- Total A-scale bytes = roundUp(M,128) * roundUp(K/16, 4)  (1 byte each, UE4M3).
- "Starting addresses of scaling factors must be **16B aligned**."
- "the layout ... does not allow transposition" -- even if A/B data are transposed, the scale layout is
  unchanged. Output kernels may write extra zeros out of bounds; do not rely on OOB persistence.

## 4. Operand memory layout, packing, alignment
- Layouts created with `cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_4F_E2M1, rows, cols, ld)`. cuBLAS FP4 is
  the standard column-major cublasLt convention; the supported form is TN: A passed as op(A)=A^T
  (`TRANSA=CUBLAS_OP_T`), B as `TRANSB=CUBLAS_OP_N`. In the sample, Adesc dims are
  (transa==OP_N ? m:k, transa==OP_N ? k:m, lda); D/C are (m, n, ld) with D type CUDA_R_4F_E2M1, C CUDA_R_16BF.
- FP4 nibble packing: 2 values per byte. Lower-index (even) element occupies the **low nibble** (bits 0-3),
  the next (odd) element the high nibble (bits 4-7). The packed contiguous dim is K (the 16-wide block dim),
  so each byte holds two adjacent-in-K FP4 values sharing block/scale structure.
- Dimension constraints: the contiguous (K) dim must be even (whole bytes); leading dims of FP4 packed data
  effectively multiples of 2 elements. For the block-scaled path keep K a multiple of 16 (block size) and,
  for clean scale tiling, M/N multiples of 128 and K a multiple of 64 (=4 blocks); cuBLAS pads internally but
  you must allocate the padded scale tiles as in section 3. ld (in elements) for FP4 must give 16B-aligned rows.

## 5. Minimal complete C++ snippet (names every real enum/attribute)
```cpp
#include <cublasLt.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

// D[M,N] = alpha * op(A)[M,K] * op(B)[K,N]   (TN: op(A)=A^T, op(B)=B)
// A_packed,B_packed: __nv_fp4_e2m1 bytes (2 vals/byte). A_scales,B_scales: __nv_fp8_e4m3 in the 128x4
// tiled layout of section 3. A_gscale,B_gscale: host float global dequant scales (folded into alpha).
// D_out_scale: __nv_fp8_e4m3* output block scales (cuBLAS writes); d_in_scale: float* device (scalein_D).
void nvfp4_gemm(__nv_fp4_e2m1* D, float* d_in_scale, __nv_fp8_e4m3* D_out_scale,
                const __nv_fp4_e2m1* A_packed, const __nv_fp8_e4m3* A_scales, float A_gscale,
                const __nv_fp4_e2m1* B_packed, const __nv_fp8_e4m3* B_scales, float B_gscale,
                int M, int N, int K, void* workspace, size_t wsBytes,
                cublasLtHandle_t lt, cudaStream_t stream)
{
  cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
  float alpha = A_gscale * B_gscale, beta = 0.0f;   // fold per-tensor global scales into alpha
  auto modeAB = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;   // NVFP4 block-16 UE4M3
  auto modeDscalar = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
  auto modeDout = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;

  cublasLtMatmulDesc_t op;
  cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
  cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
  cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &modeAB, sizeof(modeAB));
  cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &modeAB, sizeof(modeAB));
  cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_D_SCALE_MODE, &modeDscalar, sizeof(modeDscalar));
  cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE, &modeDout, sizeof(modeDout));
  cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &A_scales, sizeof(A_scales));
  cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &B_scales, sizeof(B_scales));
  cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &d_in_scale, sizeof(d_in_scale));
  cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER, &D_out_scale, sizeof(D_out_scale));

  // TN: Adesc is (K, M, lda=K); Bdesc is (K, N, ldb=K); Ddesc (M, N, ldd=M)
  cublasLtMatrixLayout_t Ad, Bd, Cd, Dd;
  cublasLtMatrixLayoutCreate(&Ad, CUDA_R_4F_E2M1, K, M, K);
  cublasLtMatrixLayoutCreate(&Bd, CUDA_R_4F_E2M1, K, N, K);
  cublasLtMatrixLayoutCreate(&Cd, CUDA_R_16BF,   M, N, M);  // C unused (beta=0) but layout required
  cublasLtMatrixLayoutCreate(&Dd, CUDA_R_4F_E2M1, M, N, M);

  cublasLtMatmulPreference_t pref;
  cublasLtMatmulPreferenceCreate(&pref);
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsBytes, sizeof(wsBytes));

  cublasLtMatmulHeuristicResult_t heur{}; int got = 0;
  cublasLtMatmulAlgoGetHeuristic(lt, op, Ad, Bd, Cd, Dd, pref, 1, &heur, &got);
  if (got == 0) { /* CUBLAS_STATUS_NOT_SUPPORTED */ return; }

  cublasLtMatmul(lt, op, &alpha, A_packed, Ad, B_packed, Bd, &beta,
                 /*C*/nullptr, Cd, /*D*/D, Dd, &heur.algo, workspace, wsBytes, stream);

  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(Dd); cublasLtMatrixLayoutDestroy(Cd);
  cublasLtMatrixLayoutDestroy(Bd); cublasLtMatrixLayoutDestroy(Ad);
  cublasLtMatmulDescDestroy(op);
}
```
If D is a wide type (fp16/bf16/fp32) instead of FP4, drop D_OUT_SCALE_MODE/POINTER and the D_SCALE scalar,
set Ddesc type accordingly; then per-block output scaling does not apply.

Sources: cuBLAS 13.3 Library PDF (block-scaling layout, quoted §3); local cublasLt.h / library_types.h enums;
NVIDIA CUDALibrarySamples cuBLASLt/LtNvfp4Matmul; NVIDIA cuBLAS 12.9 blog.
