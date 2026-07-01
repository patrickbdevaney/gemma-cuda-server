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

### [1] Device-argmax base-decode fast path
- change: base decode was `k_lmhead -> cudaMemcpy 1MB logits D2H -> host argmax over 262144` every step;
  added `ftok` path = `k_argmax<<<1>>>` on device + copy 1 int. Skips 1MB D2H + serial host argmax.
- correctness: PASS | base 29.89 -> **30.73 tok/s (+2.8%)** | **CHAMPION** | clean isolated win.

### [2] CUDA graph capture of the base decode step
- change: base/position -> `__device__ g_base` (read by k_rope_tables/k_store_kv/sdpa_cache); token via DS->dids;
  decode_step self-advances on-device (k_advance: dids=argmax, g_base++). Warmup 1 eager step (lazy init) then
  cudaStreamBeginCapture(perThread) decode_step -> instantiate -> replay (1 graph launch replaces ~1000 kernel
  launches/step). Build needs `--default-stream per-thread` (scripts/build.sh). NOGRAPH=1 = eager fallback.
- correctness: PASS | base 30.73 -> **34.18 tok/s (+11.2%)** | **CHAMPION** | launch-overhead was ~24% of step;
  CUDA_LAUNCH_BLOCKING probe (30.4->17.8) confirmed launch-sensitivity. DFlash unaffected (eager).

### [3] MoE gateup ILP (2 accumulators + 2-wide vi unroll)  — LOST
- ncu: gateup 9.74% mem / 32% compute = LATENCY-bound (247us x 30 = 7.4ms/step). Tried 2 fp32 accumulators +
  2-wide vi unroll for ILP. correctness PASS but base 34.18 -> 33.55 (REGRESSED, likely register pressure).
  Discarded, reverted. Champion stays 34.18. NEXT: check gateup occupancy/registers; or different ILP shape.

### [4] DFlash verify forward as CUDA graph
- change: verify_step (fixed M=k+1) captured as graph (reads DS->dids block + g_base, writes taps_blk + darg),
  draft stays eager (variable context shape). correctness: PASS + base/DFlash PARITY.
- DFlash (predictable) 30.6 -> 31.34 tok/s (+2.4%). Modest because DRAFT now dominates the step (eager).
  Base champion 34.18 unchanged (base > DFlash on predictable). Committed (no regression, improves DFlash mode).
  NEXT: draft is the DFlash bottleneck -> profile + graph/optimize the draft (query forward fixed BLK=16;
  context forward variable C -> make incremental). Path to DFlash>base>110.

### [5] Draft k_linear_bf16 -> half2 (fp16 acts + bf16->fp16 weight, __hfma2)
- draft was 54% of DFlash (k_linear_bf16 43% incl 65ms lm_head, compute-bound: 15 scalar dots/weight-elem).
  half2: fp16 activations (k_df32to16) + bf16->fp16 weight -> __hfma2 (2 MAC/instr), reuse W across M<=16.
  Draft is approximate (verify confirms w/ target) so fp16 draft doesn't break parity.
- correctness: PASS + base/DFlash PARITY, acceptance unchanged (11.14). DFlash 31.34 -> **37.42 tok/s (+19.4%)**,
  now BEATS base 34.18. **CHAMPION (fastest decode mode = DFlash-predictable 37.42)**. Path to 110 continues.

### [6] Draft bigM (context) -> half2  — NEUTRAL, reverted
- context fc/kv_proj use k_linear_bf16_bigM (warp-per-(m,n)) -> weight read M=C times (redundant), so it's
  MEMORY-bound on redundant reads, not compute-bound. half2 gave 37.42->37.26 (neutral). Reverted.
- ROOT FIX (next): context forward reprocesses ALL C tokens each propose (O(C), grows). Make INCREMENTAL:
  persistent draft context-KV cache, only project the newly-committed na+1 positions each step (M<=16 ->
  efficient warp-per-n). Big win especially as sequence grows. Champion stays DFlash 37.42.

### [7] Incremental draft context K/V  — CHAMPION
- context forward was O(C): reprocessed ALL committed positions each propose. Now persists a per-layer draft
  context-KV cache (DraftModel.ctx_done) and projects only the newly-committed positions [ctx_done..C-1]
  (<=16 -> efficient half2 warp-per-n) each step. Also fixed latent bug: half2 xf16 buffer was 16*8192 but
  fc has K=FCIN=16896 (overflow when fc hit the M<=16 path) -> sized 16*FCIN.
- correctness: PASS + PARITY, acceptance unchanged (11.14). DFlash 37.42 -> **42.05 tok/s (+12.4%)**. CHAMPION.
  Scales better as sequence grows (O(committed) not O(C) per step). base 34.18 unchanged.

### [8] Verify lmhead -> half2 warp-per-vocab  — CHAMPION
- verify k_lmhead_batched (block-per-vocab, scalar, shared-staging) was 75ms for M=15 — 3x slower than the
  draft's k_linear_bf16 (half2 warp-per-n, 25ms) for the SAME projection. Added k_lmhead_batched_h2 (half2:
  fp16 hidden + bf16->fp16 embed, reuse embed row across M). correctness PASS + PARITY, accept unchanged.
- DFlash 42.05 -> **51.22 tok/s (+21.8%)**. CHAMPION. Now matches vLLM base (52). base 34.18 unchanged.

### [9] lmhead uint4 (aligned embed) — LOST
- ncu: verify lmhead 94% L1/TEX (memory-INSTRUCTION bound), 63% L2, 52% compute. Tried uint4 (8 bf16/load,
  half the load instructions) w/ aligned embed copy. 4-deep hfma2 chain: 49.24 (regressed). 2-deep tree: 50.08.
  Both < uint2 champion 51.22 -> the load-instruction reduction is offset by ILP/register cost; uint2 optimal.
  Reverted. Champion stays DFlash 51.22.
