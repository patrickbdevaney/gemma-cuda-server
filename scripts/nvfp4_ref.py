#!/usr/bin/env python3
"""nvfp4_ref.py — numpy NVFP4 reference for the W4A4 real-Linear gate (no torch/ml_dtypes).

Reads a REAL weight (default layer0 q_proj) from the Gemma-4 NVFP4 checkpoint, dequantizes it
per the compressed-tensors convention  W = fp4_code * e4m3_blockscale / weight_global_scale,
quantizes a random activation X[M,K] to NVFP4 the same way (global_scale = 2688/amax(X)),
and emits the reference  Y = dequant(X) @ dequant(W)^T  plus raw bytes for the CUDA test.

Convention (compressed_tensors/quantization/utils/helpers.py:329):
  global_scale = (FP8_E4M3_MAX=448 * FP4_E2M1_MAX=6) / amax = 2688/amax
  stored e4m3 block scale = global_scale * (block_amax/6);  dequant val = code * e4m3 / global_scale
"""
import numpy as np, os, sys, json, struct

HOME = os.path.expanduser("~")
CKPT = sys.argv[1] if len(sys.argv) > 1 else HOME + "/models/gemma-4-26B-A4B-it-NVFP4/model.safetensors"
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "model.language_model.layers.0.self_attn.q_proj"
OUT = sys.argv[3] if len(sys.argv) > 3 else "/tmp/lincase"
os.makedirs(OUT, exist_ok=True)
M = 128
rng = np.random.default_rng(7)

# ---- minimal safetensors reader ----
def st_open(path):
    f = open(path, "rb"); hlen = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(hlen)); data0 = 8 + hlen
    return f, hdr, data0
def st_get(f, hdr, data0, name):
    meta = hdr[name]; b, e = meta["data_offsets"]; f.seek(data0 + b); raw = f.read(e - b)
    return meta["dtype"], meta["shape"], raw

# ---- E2M1 / E4M3 tables ----
E2M1 = np.array([0,0.5,1,1.5,2,3,4,6, 0,-0.5,-1,-1.5,-2,-3,-4,-6], dtype=np.float64)  # idx by 4-bit code
def e4m3_byte_to_val(b):
    s = -1.0 if (b >> 7) else 1.0; exp = (b >> 3) & 0xF; man = b & 0x7
    if exp == 0: v = (man / 8.0) * 2.0**(-6)
    elif exp == 0xF and man == 0x7: v = np.nan
    else: v = (1 + man / 8.0) * 2.0**(exp - 7)
    return s * v
E4M3 = np.array([e4m3_byte_to_val(b) for b in range(256)], dtype=np.float64)
# encoder grid (finite, positive incl 0), sorted unique -> nearest
_pos_vals = np.unique(E4M3[np.isfinite(E4M3) & (E4M3 >= 0)])
def quant_e4m3_pos(x):  # x>=0 array -> (e4m3_bytes uint8, decoded float vals)
    idx = np.searchsorted(_pos_vals, x)
    idx = np.clip(idx, 0, len(_pos_vals) - 1)
    lo = np.clip(idx - 1, 0, len(_pos_vals) - 1)
    pick_lo = np.abs(x - _pos_vals[lo]) <= np.abs(x - _pos_vals[idx])
    vals = np.where(pick_lo, _pos_vals[lo], _pos_vals[idx])
    # map value back to byte
    byte_of = {v: b for b, v in enumerate(E4M3) if np.isfinite(v) and v >= 0}
    flat = np.array([byte_of[v] for v in vals.ravel()], dtype=np.uint8).reshape(x.shape)
    return flat, vals
_fp4_mag = np.array([0,0.5,1,1.5,2,3,4,6], dtype=np.float64)
def quant_fp4(x):  # signed -> 4-bit codes (uint8) + decoded value
    a = np.abs(x); idx = np.searchsorted(_fp4_mag, a); idx = np.clip(idx, 0, 7)
    lo = np.clip(idx - 1, 0, 7)
    pick_lo = np.abs(a - _fp4_mag[lo]) <= np.abs(a - _fp4_mag[idx])
    mi = np.where(pick_lo, lo, idx)
    code = mi.astype(np.uint8) | ((x < 0).astype(np.uint8) << 3)
    val = np.sign(x) * _fp4_mag[mi]
    return code.astype(np.uint8), val

def unpack_fp4(packed, K):  # [N, K/2] uint8 -> codes [N, K]
    lo = packed & 0xF; hi = (packed >> 4) & 0xF
    out = np.empty((packed.shape[0], K), dtype=np.uint8)
    out[:, 0::2] = lo; out[:, 1::2] = hi
    return out
def pack_fp4(codes):  # [N,K] -> [N,K/2]
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)

def nvfp4_quant_rows(X, gscale):
    """X[R,K] fp32 -> (packed[R,K/2], e4m3_bytes[R,K/16], dequant[R,K]) per convention."""
    R, K = X.shape; KB = K // 16
    blk = X.reshape(R, KB, 16)
    bamax = np.abs(blk).max(axis=2)                          # [R,KB]
    e4m3_real = gscale * (bamax / 6.0)                       # stored scale (pre-e4m3-round)
    e4m3_b, e4m3_v = quant_e4m3_pos(e4m3_real)               # [R,KB]
    local = np.where(e4m3_v > 0, e4m3_v / gscale, 1.0)       # effective real scale per block
    codes, _ = quant_fp4(blk / local[:, :, None])            # [R,KB,16]
    codes = codes.reshape(R, K)
    deq = (E2M1[codes].reshape(R, KB, 16) * local[:, :, None]).reshape(R, K)
    return pack_fp4(codes), e4m3_b.astype(np.uint8), deq

# ---- load real weight ----
f, hdr, data0 = st_open(CKPT)
_, wp_shape, wp_raw = st_get(f, hdr, data0, PREFIX + ".weight_packed")
_, ws_shape, ws_raw = st_get(f, hdr, data0, PREFIX + ".weight_scale")
_, _, wg_raw = st_get(f, hdr, data0, PREFIX + ".weight_global_scale")
N, Kh = wp_shape; K = Kh * 2
Wp = np.frombuffer(wp_raw, dtype=np.uint8).reshape(N, Kh)
Wcodes = unpack_fp4(Wp, K)
Wsb = np.frombuffer(ws_raw, dtype=np.uint8).reshape(N, K // 16)
Wsv = E4M3[Wsb]                                              # e4m3 decoded
w_gscale = float(np.frombuffer(wg_raw, dtype=np.float32)[0])
W_deq = (E2M1[Wcodes].reshape(N, K // 16, 16) * (Wsv / w_gscale)[:, :, None]).reshape(N, K)
print(f"weight {PREFIX}: N={N} K={K} w_gscale={w_gscale:.6g}  W_deq range[{W_deq.min():.4f},{W_deq.max():.4f}]")

# ---- random activation, quantize ----
X = (rng.standard_normal((M, K)) * 0.5).astype(np.float32)
gs_X = float(2688.0 / max(np.abs(X).max(), 1e-9))
Xp, Xsb, X_deq = nvfp4_quant_rows(X.astype(np.float64), gs_X)
Y_ref = (X_deq @ W_deq.T).astype(np.float32)                 # [M,N]

# ---- dump ----
X.tofile(OUT + "/X.bin")                                     # fp32 [M,K] (GPU quantizes this)
Wp.tofile(OUT + "/W_packed.bin"); Wsb.tofile(OUT + "/W_scales.bin")
Y_ref.tofile(OUT + "/Y_ref.bin")
# also dump numpy's own X quantization for an optional quantizer-only check
Xp.tofile(OUT + "/Xq_packed_ref.bin"); Xsb.tofile(OUT + "/Xq_scales_ref.bin")
with open(OUT + "/dims.txt", "w") as fo:
    fo.write(f"{M} {N} {K}\n{w_gscale:.8e} {gs_X:.8e}\n")
print(f"wrote {OUT}: M={M} N={N} K={K} gs_X={gs_X:.4g}  Y_ref range[{Y_ref.min():.3f},{Y_ref.max():.3f}] mean|Y|={np.abs(Y_ref).mean():.4f}")
