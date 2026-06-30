# OPTIMIZATION LOG — append-only record of every candidate tried

Champion metric: base decode tok/s (primes, sustained). Correctness = gate_self.sh (must PASS).
Format: `[cycle] candidate | correctness | base tok/s | champion? | note`

## Pre-loop history (from session, condensed — already in mainline; do NOT re-try as new)
- HW FP4 decode (`__nv_cvt_fp4x2_to_halfraw2`) + fp16 acts + half2: 15.8→18.5 (GEMV), →21.4 (MoE). CHAMPION path.
- MoE down warp-per-(t,d) 8-experts-fused 1-reduce: 21.4→25.9. CHAMPION.
- MoE gateup warp-per-output: 25.9→26.6. CHAMPION.
- Parallel router top8 (128-thread reduction-argmax): →28.2. CHAMPION.
- w4a16 verify GEMM half2 + draft batched lm_head + device argmax: DFlash 16→35.7 (predictable). CHAMPION (DFlash).
- DFlash causal-within-block mask fix (sliding layers): code accept 3.3→4.1. Correct (draft bug fix).
- TRIED+LOST: register-block fp4_gemv RB=4 (14.9, regressed — register pressure). warp-per-vocab lmhead (regressed).
  MoE down RB=2 (+0.4% marginal). constant-LUT (neutral). __ldcs streaming (neutral, kept).
- TRIED+REFUTED: warp-per-output MoE w/ SCALAR dequant (regressed 3x — needs HW decode). W4A4 taps (worse accept).
  off-by-one tap layers [0,5,10,16,21,26] (worse). Tensor-Core FP4 mma for M=1 decode (wrong tool, research-refuted).

## Loop cycles

### [0] Baseline established
- correctness: PASS | base 29.89 tok/s | CHAMPION (initial) | DFlash 30.58 predictable. Loop infra added.
