#!/usr/bin/env python3
"""Reference for rmsnorm + rotate-half RoPE (numpy). bf16 weights via truncation to match GPU decode."""
import numpy as np, os, sys
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ewcase"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(3)
ROWS, DIM = 5, 2816
EPS = 1e-6
HEADS, HD, ROT = 16, 256, 256
THETA = 10000.0

def to_bf16_trunc(x):  # fp32 -> bf16 bits (truncate low 16), return (uint16, decoded fp32)
    u = x.astype(np.float32).view(np.uint32)
    b = (u >> 16).astype(np.uint16)
    dec = (b.astype(np.uint32) << 16).view(np.float32)
    return b, dec

# rmsnorm
X = (rng.standard_normal((ROWS, DIM)) * 1.3).astype(np.float32)
wb, wdec = to_bf16_trunc((rng.standard_normal(DIM) * 0.2 + 1.0).astype(np.float32))
ms = (X.astype(np.float64) ** 2).mean(1, keepdims=True) + EPS
RN = (X / np.sqrt(ms) * wdec.astype(np.float64)).astype(np.float32)
X.tofile(OUT + "/rn_x.bin"); wb.tofile(OUT + "/rn_w.bin"); RN.tofile(OUT + "/rn_ref.bin")

# rope
Xr = (rng.standard_normal((ROWS, HEADS, HD)) * 1.0).astype(np.float32)
half = ROT // 2
inv = THETA ** (-(np.arange(half) * 2.0) / ROT)        # [half]
pos = np.arange(ROWS)[:, None]
ang = pos * inv[None, :]                                 # [ROWS, half]
cos = np.cos(ang).astype(np.float32); sin = np.sin(ang).astype(np.float32)
Xo = Xr.copy().astype(np.float64)
x0 = Xo[..., :half].copy(); x1 = Xo[..., half:ROT].copy()
Xo[..., :half]    = x0 * cos[:, None, :] - x1 * sin[:, None, :]
Xo[..., half:ROT] = x1 * cos[:, None, :] + x0 * sin[:, None, :]
Xr.tofile(OUT + "/rope_x.bin"); cos.tofile(OUT + "/rope_cos.bin"); sin.tofile(OUT + "/rope_sin.bin")
Xo.astype(np.float32).tofile(OUT + "/rope_ref.bin")
with open(OUT + "/dims.txt", "w") as f:
    f.write(f"{ROWS} {DIM}\n{HEADS} {HD} {ROT}\n")
print(f"wrote {OUT}: rmsnorm[{ROWS},{DIM}] rope[{ROWS},{HEADS},{HD}] rot={ROT}")
