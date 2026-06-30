#!/usr/bin/env python3
"""Reference for single-sequence causal SDPA with GQA + optional sliding window (numpy float64).
Mirrors HF eager_attention_forward: scores = (Q.K)*scaling, causal+window mask -> -inf,
softmax over keys (fp64), out = softmax @ V. scaling=1.0 (Gemma4, no 1/sqrt(d)).

usage: gen_attention_ref.py <out_dir> <seq> <n_heads> <n_kv> <head_dim> <window> [seed]
"""
import numpy as np, os, sys

OUT    = sys.argv[1]
SEQ    = int(sys.argv[2])
HEADS  = int(sys.argv[3])
NKV    = int(sys.argv[4])
HD     = int(sys.argv[5])
WINDOW = int(sys.argv[6])
SEED   = int(sys.argv[7]) if len(sys.argv) > 7 else 7
SCALING = 1.0

os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(SEED)
group = HEADS // NKV

Q = (rng.standard_normal((SEQ, HEADS, HD)) * 0.5).astype(np.float32)
K = (rng.standard_normal((SEQ, NKV,   HD)) * 0.5).astype(np.float32)
V = (rng.standard_normal((SEQ, NKV,   HD)) * 0.5).astype(np.float32)

Qd, Kd, Vd = Q.astype(np.float64), K.astype(np.float64), V.astype(np.float64)
out = np.zeros((SEQ, HEADS, HD), dtype=np.float64)

i = np.arange(SEQ)[:, None]
j = np.arange(SEQ)[None, :]
mask = (j <= i)                                  # causal
if WINDOW > 0:
    mask &= ((i - j) < WINDOW)                   # sliding window

for h in range(HEADS):
    kv = h // group
    scores = (Qd[:, h, :] @ Kd[:, kv, :].T) * SCALING     # [SEQ, SEQ]
    scores = np.where(mask, scores, -np.inf)
    m = scores.max(axis=1, keepdims=True)
    e = np.exp(scores - m)
    w = e / e.sum(axis=1, keepdims=True)                  # softmax over keys
    out[:, h, :] = w @ Vd[:, kv, :]

Q.tofile(OUT + "/q.bin")
K.tofile(OUT + "/k.bin")
V.tofile(OUT + "/v.bin")
out.astype(np.float32).tofile(OUT + "/out_ref.bin")
with open(OUT + "/dims.txt", "w") as f:
    f.write(f"{SEQ} {HEADS} {NKV} {HD} {WINDOW}\n")
print(f"wrote {OUT}: Q[{SEQ},{HEADS},{HD}] K/V[{SEQ},{NKV},{HD}] window={WINDOW} amax={np.abs(out).max():.3e}")
