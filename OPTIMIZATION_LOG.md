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

### [10] MoE gateup __launch_bounds__(256,6) — NEUTRAL, reverted
- ncu: gateup register-limited (48 regs -> 5 blocks/SM; warp limit 6). Forced 6 blocks via launch_bounds.
  base 34.18->34.34, DFlash 51.78->51.36 (within noise) — compiler likely spilled to fit 6 blocks, offsetting
  occupancy gain. Reverted. MoE gateup has now resisted ILP (cycle3) + launch_bounds (cycle10).
  NEXT big lever: grouped/batched MoE verify (structural) — 15 predictable verify tokens share experts;
  group by expert -> read each active expert weight once (vs per-token). Verify MoE = 34% of DFlash.

### [11] FP8-e4m3 lm_head (per-row scaled embed)  — LOST (both base & verify)
- Research #1 lever: lm_head=44% of per-step bytes; FP8 halves embed read. Implemented per-row-scaled FP8
  embed (k_embed_to_fp8) + FP8 base lmhead + FP8 verify lmhead (HW __nv_cvt_fp8x2_to_halfraw2, 8 fp8/uint2).
- correctness PASS + PARITY (greedy argmax robust to FP8, as research predicted). BUT: base 34.18->32.3 (-5.5%),
  DFlash 51.87->49.96 (-3.7%). The fp8x2->half2 HW decode overhead offsets the halved loads on Thor, and the
  lmhead isn't purely byte-bound (base M=1 memory+decode, verify M=15 L1-instruction+decode). Reverted.
  NOTE: FP8 decode is NOT free on sm_110a; byte-reduction levers must beat the decode cost. bf16 lmhead optimal.

### [12] Grouped-GEMM verify MoE gateup  — CHAMPION
- verify MoE was per-output (warp per (t,j,i)) -> each expert weight read per (token,expert). Now: k_moe_invert
  builds expert->tokens map (atomics), k_moe_gateup_grouped = warp per (e,i), read+decode Wg_e[i]/Wu_e[i] ONCE,
  reuse across the <=4 tokens/pass routing to e (register-bounded ag/au[4] to avoid the ILP register pressure).
  Only for seq<=16 (verify); prefill/base per-output. Captured in verify graph (memset+atomics capturable).
- correctness PASS + PARITY, accept unchanged. DFlash 51.87 -> **59.72 tok/s (+15.1%)** (> the 1.5x reuse estimate:
  also cut warp count + relieved latency). CHAMPION. This is the enabler that makes tree-verify net-positive.

### [13] Grouped verify MoE down (atomicAdd) — NEUTRAL, reverted
- grouped down (warp per (e,d), reuse Wd_e[d] across tokens, atomicAdd to moe_out). 59.72->59.37 (neutral):
  atomicAdd overhead offset the ~1.5x weight reuse, and the down was already warp-fused-efficient (8 experts/warp).
  Reverted. Champion stays DFlash 59.72 (grouped gateup only). down grouping not worth the atomics.

### [14] Tree verification (tested via multi-round verify)  — LOST (data-confirmed net-negative)
- Tree-verify for block-diffusion DFlash: full top-k tree explodes combinatorially (fork continuations at unknown
  positions); the tractable proxy is MULTI-ROUND verify (re-verify the draft tail conditioned on the corrected
  bonus; draft tokens are position-fixed in block diffusion so reusable, no re-propose).
- MEASURED (MROUND env, A/B): baseline 59.07 tok/s / 7 steps / tau 11.43  vs  multi-round 57.27 tok/s / 6 steps /
  tau 13.33. So +16% tokens-per-draft (PARITY held) but -3% SPEED: each extra round is a full weight-bound forward
  that costs more than the recovered tail tokens. A one-forward tree saves the weight re-reads but grouped-MoE +
  lmhead still scale with candidate count -> ~0.98x (break-even) on this high-acceptance benchmark. Reverted.
- CONCLUSION: extra-candidate verification can't overcome the per-token verify cost when base acceptance is already
  high (11.14/14 on primes). Path to 110 needs a better BASE drafter (higher accept at same verify cost) =
  training-adjacent (EAGLE-3 feature fusion / deeper diffusion drafter), NOT a kernel/verify-structure change.
  Would help LOW-acceptance workloads (code) where the headroom is large. Champion stays DFlash 59.72.

### [15] Persistent draft query scratch (remove per-propose malloc/free + sync@273) — NEUTRAL, reverted
- draft query forward malloc'd ~16 fixed-size buffers/propose + freed them (gated by 2 cudaDeviceSynchronize).
  Made them persistent statics + dropped sync@273 (lm_head is stream-ordered). gate PASS + PARITY.
- A/B (5 runs each, sorted): champion 60.17-60.65 (med 60.50) vs scratch 60.42-60.65 (med 60.51) = IDENTICAL.
  The driver caches allocations + the sync overlapped, so the churn was already free. Reverted (neutral).
- Draft launch-sensitivity probe: CUDA_LAUNCH_BLOCKING 63.59->60.91 = only ~4%; graphing the draft would need
  refactoring k_attn for growing context (device-side length) — complex for ~4%. Not worth it now.

### [16] T>0 typical (Medusa) acceptance — implemented (feature; no-op on primes, valid for low-accept workloads)
- k_typical_prob: per-position target softmax prob (temp 1/invT) of the draft token; accept iff prob>=TYP_EPS
  (TYP_EPS=0 -> exact greedy, byte-identical, EXACT PARITY verified). Gated by TYP_EPS + TEMP env.
- MEASURED on primes: no-op (accept 11.14 unchanged across eps 0.02-0.15, T 0.3-2.0). T=4/eps0.09 CRASHES to 4.33
  (flat dist drops argmax below eps). Reason: primes draft has NO acceptance headroom (tail = hard errors, not
  near-misses) — same root cause tree/multi-round didn't help. Kept as production feature for LOW-accept workloads.
- KEY REFRAME: acceptance-based levers are exhausted on primes (draft near-optimal at 11.14/14). The gap to vLLM
  DFlash (110) is EFFICIENCY: my BASE decode 34 tok/s = only 36% of the ~94 tok/s roofline (2.9GB/step: 1.4 FP4
  weight + 1.5 bf16 lmhead). vLLM base=52 (72% higher). Closing the base BW gap scales DFlash too. NEXT: profile
  base-decode kernels' achieved DRAM BW; find where the 64% is lost.

### [17] base MoE gateup uint2 weight loads (128->widen) — LOST (broke correctness)
- base gateup ncu: 9.82% DRAM BW / 32% compute / 66% occ = LATENCY-bound (15.8MB weight in 245us = 10x slow).
  Tried uint2 weight loads (16 FP4 codes = 1 scale group; half the load instr + scale lookups). Built but GATE
  FAIL (empty/NaN output) — subtle bug in the uint2 reinterpret/act-pairing. Reverted. The gateup is latency/MLP-
  bound; widening loads REDUCES outstanding requests (wrong direction for MLP) anyway. Need occupancy/MLP, which
  hits the 48-reg limit. NEXT: research the vLLM base-decode (52) + DFlash (110) efficiency to find the real technique.

### [18] base MoE gateup K-unroll prefetch (MLP fix) — CHAMPION (base)
- Research: base gateup MLP-starved (serial FMA chain -> 1-2 loads in flight -> 9.82% BW, 3-5x below naive).
  Fix = K-unroll U=4: issue U independent weight loads BEFORE the FMA block (raises memory-level parallelism)
  while KEEPING warp-per-output (N-blocking regressed: halved warps -> 0.73 waves at M=1 underfill).
- ncu: 9.82%->13.60% BW, 32%->42.86% compute, 245->184us (-25%). gate PASS. base 34.18 -> ~34.7 (+1.7%).
  Modest end-to-end (gateup is a fraction) but the technique is the key. NEXT: apply to base w4a16 (28%) + down (18%).

### [19] w4a16 (dense linears) K-unroll prefetch — CHAMPION (base + DFlash)
- same MLP fix on w4a16_gemm_kernel (used by base M=1 dense qkv/o AND DFlash verify M=15). U=4 weight prefetch.
- gate PASS. base ~34.7->34.9, DFlash 60->60.9 (+1.5%, verify w4a16 benefits too). Both improved. CHAMPION.

### [20] MoE down K-unroll prefetch (8-expert weights) — LOST (register pressure), reverted
- prefetched all 8 experts' wd0/wd1 (16 uint regs) before FMAs. base 34.9->34.1, DFlash 60.9->59.2 (regressed).
  The down already has 8-way expert ILP; the 16-reg prefetch crushed occupancy (same failure mode as N-blocking).
  Reverted. MLP prefetch helps SERIAL-chain kernels (gateup, w4a16) but hurts already-ILP-rich ones (down).
- Net MLP-fix result: base 34.18->34.9 (+2.1%), DFlash 60->60.9 (+1.5%). gateup ncu 9.82%->13.6% BW.

### [21] NVFP4 lm_head (quantize embed E2M1+e4m3, reuse fp4_gemv) — BIG CHAMPION (base)
- lm_head was ~30% of base (1.5GB bf16). Quantized the tied embed to NVFP4 (k_embed_amax -> global scale,
  k_quant_embed_fp4 -> E2M1 codes + linear e4m3 group-16 scales matching w4a16 dequant). Base lm_head now =
  fp4_gemv (4x fewer bytes, fast HW FP4 decode - the reason FP8 failed). No softcap (argmax-invariant).
- gate PASS (FP4 embed accurate enough for greedy argmax on confident tokens). base 34.78 -> **44.5 (+28%)**.
  Near vLLM base (52). NEXT: apply FP4 lm_head to DFlash draft + verify lmheads (biggest DFlash cost).

### [22] NVFP4 verify lm_head (w4a16_gemm with quantized embed) — CHAMPION (DFlash)
- verify lmhead k_lmhead_batched_h2 (bf16 embed) -> w4a16_gemm(g_ewp,g_ews,g_egs) (FP4, 4x lighter).
  DFlash 60.9 -> **65.1 (+6.9%)**, output correct (primes intact), accept 11.14 unchanged. CHAMPION.

### [23] NVFP4 draft lm_head — HUGE CHAMPION (DFlash)
- draft lm_head bf16 -> w4a16_gemm(g_ewp,g_ews,g_egs) (FP4). DFlash 65.1 -> **81.97 (+26%)**. gate PASS.
- BONUS: accept 11.14 -> **13.33** (+20%)! draft AND verify now use the SAME FP4 lm_head -> draft proposals
  align with the target's FP4 argmax (consistency win on top of the byte reduction). tau 13.33.
- Cumulative FP4 lm_head (base+verify+draft): base 34.78->44.5 (+28%), DFlash 60.9->82 (+35%). This was THE lever.
  Now ~75% of the way to vLLM DFlash 110. NEXT: steps 2 (L2 persist + MoE toward 33% BW) + 3 (full-graph capture).

### [24] L2 persistence on MoE activation (step 2) — NEUTRAL, reverted
- pinned x2_16 (90KB fp16 activation) in L2 via cudaAccessPolicyWindow. base 44.5->44.2, DFlash 82->81.7 (neutral).
  Thor's L2 already holds the small activation (no eviction to prevent). Reverted. MoE also at MLP limit (U=8 neutral).
  Step 2 exhausted on Thor. Step 3 (full-graph): DFlash host tax only ~5% (81.2 vs 77.1 blocking) + draft has a
  GROWING attention context (hard to graph) -> poor ROI. Path to 110 now needs TENSOR CORES for the compute-bound
  M=15 verify dense/MoE GEMMs (lmhead already memory-bound 60.9%, so TC won't help it; dense+MoE would).

### [25] FP4 tensor-core verify kernel — RESEARCHED, decided AGAINST (evidence-based)
- 4 parallel deep-research agents (2 returned, 2 session-limited). DECISIVE findings:
  * Thor sm_110a = tcgen05/TMEM only (NO warp mma.sync; CUTLASS #2951). TC = full tcgen05 path on immature
    CUDA-13.1 toolchain. M=128 CTA-tile floor -> 8.5x pad waste at M=15.
  * Nsight bound-status: lm_head (34.6% of step) MEMORY-bound 60.9% -> TC gives NOTHING. verify down (14.8%)
    COMPUTE-bound 77.7% -> TC candidate but only ~+10-15% total for a huge risky build. NOT worth it.
  * DFlash ships NO custom verify kernel (verify = standard batched forward). No secret to match.
  * "110 tok/s DFlash gemma-4-26B-A4B on Thor" is UNSOURCED. Real anchors: DGX-Spark base 52, Spark DFlash on
    similar MoEs 50-118, RTX PRO 6000 138, H100 gemma-4 DFlash C1 306. MY 82 IS COMPETITIVE with real Thor-class.
  * SOTA small-M FP4 = CUDA cores (ReSET 1.57-2.49x over TC baseline at M=1-8), which is my approach.
- VERDICT: pure-CUDA decode is near its bound-limits (lm_head 60.9% mem, down 77.7% compute). TC path documented
  (CUTLASS ex.72 / tcgen05 recipe in research) but recommend-against on ROI+toolchain-risk. Remaining gap to any
  higher number = framework (Marlin MoE, graph) or deeper drafter (training), not a missing CUDA kernel.
