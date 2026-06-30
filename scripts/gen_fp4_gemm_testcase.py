#!/usr/bin/env python3
"""Generate a synthetic NVFP4 GEMM test case + float64 reference (no torch/ml_dtypes).

Models a Linear layer Y = X @ W^T computed as a cublasLt TN FP4 GEMM:
  A = activations [M, K] (NVFP4),  B = weights [N, K] (NVFP4),  D = Y [M, N].
Both operands: FP4 E2M1 codes (2/byte) + per-16 block scales in FP8-E4M3 + per-tensor fp32 global scale.

To keep the reference EXACT (isolating GEMM/accumulation correctness from quantization rounding):
  - FP4 codes are random 0..15 mapped through the exact E2M1 value table.
  - Block scales are drawn ONLY from exactly-E4M3-representable values: (8+m)/8 * 2^E,
    m in 0..7, E in [-2..2]  -> exact float and exact E4M3 byte (sign 0, exp E+7, mant m).
  - Global scales are arbitrary fp32 (folded into alpha on the cublasLt side).

Outputs raw little-endian binaries into out dir for the C++ test to consume.
Scales are emitted in LOGICAL [outer, K//16] order; the C++ side swizzles to cuBLAS tile layout.
"""
import numpy as np, os, struct, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fp4case"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(1234)

M = int(sys.argv[2]) if len(sys.argv) > 2 else 128
N = int(sys.argv[3]) if len(sys.argv) > 3 else 256
K = int(sys.argv[4]) if len(sys.argv) > 4 else 128   # M,N mult 128; K mult 64 (and 16) per recipe tiling
assert K % 16 == 0   # block size; M/N arbitrary (scale buffer pads internally)
KB = K // 16                      # blocks along K

# Exact E2M1 value table (index 0..15): sign-magnitude, mag {0,.5,1,1.5,2,3,4,6}
E2M1 = np.array([0,0.5,1,1.5,2,3,4,6, -0.0,-0.5,-1,-1.5,-2,-3,-4,-6], dtype=np.float64)

def rand_codes(rows, cols):
    return rng.integers(0, 16, size=(rows, cols), dtype=np.uint8)

def pack_fp4(codes):  # [rows, K] uint8 codes -> [rows, K//2] bytes, even idx in LOW nibble
    lo = codes[:, 0::2].astype(np.uint8)
    hi = codes[:, 1::2].astype(np.uint8)
    return (lo | (hi << 4)).astype(np.uint8)

def rand_e4m3_scales(rows, cols):
    """Return (float_vals[rows,cols], e4m3_bytes[rows,cols]) — all exactly representable."""
    m = rng.integers(0, 8, size=(rows, cols)).astype(np.int32)     # mantissa 0..7
    E = rng.integers(-2, 3, size=(rows, cols)).astype(np.int32)    # exponent -2..2
    vals = ((8 + m).astype(np.float64) / 8.0) * (2.0 ** E)         # (1+m/8)*2^E, exact
    exp_field = (E + 7).astype(np.uint8)                           # bias 7
    byts = ((exp_field << 3) | m.astype(np.uint8)).astype(np.uint8)  # sign 0
    return vals, byts

def dequant(codes, scales_f, gscale):  # codes[R,K], scales_f[R,KB] -> float64 [R,K]
    vals = E2M1[codes]                                  # [R,K]
    sc = np.repeat(scales_f, 16, axis=1)                # [R,K]
    return vals * sc * gscale

A_codes = rand_codes(M, K); B_codes = rand_codes(N, K)
A_sf, A_sb = rand_e4m3_scales(M, KB); B_sf, B_sb = rand_e4m3_scales(N, KB)
A_g = np.float32(0.0123); B_g = np.float32(0.789)

A_deq = dequant(A_codes, A_sf, float(A_g))
B_deq = dequant(B_codes, B_sf, float(B_g))
Y_ref = (A_deq @ B_deq.T).astype(np.float64)            # [M,N], TN: D[m,n]=sum_k A[m,k]B[n,k]

def dump(name, arr):
    arr.tofile(os.path.join(OUT, name))

dump("A_packed.bin", pack_fp4(A_codes)); dump("B_packed.bin", pack_fp4(B_codes))
dump("A_scales.bin", A_sb); dump("B_scales.bin", B_sb)         # logical [outer, KB] uint8 e4m3
dump("Y_ref.bin", Y_ref.astype(np.float32))
with open(os.path.join(OUT, "dims.txt"), "w") as f:
    f.write(f"{M} {N} {K}\n{float(A_g):.8e} {float(B_g):.8e}\n")
print(f"wrote testcase to {OUT}: M={M} N={N} K={K} KB={KB}")
print(f"Y_ref range [{Y_ref.min():.4f}, {Y_ref.max():.4f}] mean|Y|={np.abs(Y_ref).mean():.4f}")
