# Performance research: what actually limits NVFP4 decode on Thor (before the pivot)

Two independent web-research passes converged on the same cited verdict.

## VERDICT: batch-1 NVFP4 decode is MEMORY-BANDWIDTH-BOUND. Do NOT pivot to Tensor-Core FP4 mma.

- **Thor / GB10 memory bandwidth = 273 GB/s** (128GB LPDDR5x, 256-bit, ~8.533 Gbps/pin). This sets the ceiling.
- **Single-token decode = GEMV (M=1)**, arithmetic intensity ~1-2 FLOP/byte → memory-bound. Tensor Cores
  accelerate compute-bound GEMM with row reuse (large M); at M=1 there's no reuse → FP4 mma units sit idle.
- **cuBLASLt/cutlass FP4 at M=1 is pathological**: pads M→128 (TRT-LLM #4412: FP4 *slower than FP8* in decode
  for this reason). cutlass NVFP4 is GEMM-only, 128-row M-tile minimum, NO GEMV path. Reserve FP4 mma for PREFILL.
- **Production engines use CUDA-core dequant-GEMV / W4A16-Marlin / trtllm-gen low-latency kernels for decode**,
  not FP4 mma. A hand CUDA-core NVFP4 GEMV beat vLLM's cutlass NVFP4 GEMM at M=1 by 1.57-2.49x (Blackwell hackathon).
- TRT-LLM does NOT officially support Thor yet; its fast trtllm-gen FP4 cubins have no sm_110 → won't dispatch.
  Portable primitives are CUDA-core: weightOnlyBatchedGemv, cudaCoreGemmNVFP4.cu.

## Published A4B NVFP4 decode on GB10 (concurrency=1) — the realistic ceiling
| model | active | quant | tok/s |
|---|---|---|---|
| Nemotron-3-Nano 30B | 3.5B | W4A16 | 74.75 (~93% of byte-ceiling) |
| **Gemma-4-26B-A4B (OUR MODEL)** | 4B | NVFP4 | **52** (vLLM base) |
| Qwen3.6-35B-A3B | 3B | NVFP4 W4A4 | 66.9 (CUDA graphs) → 97 (tuned) |

=> Our base target is ~52 tok/s (vLLM parity). We're at 15.7 (3.3x slower = kernel inefficiency, NOT hardware).
=> 110 tok/s (the vLLM+DFlash figure) needs ~52 base + DFlash speculation on top + byte reduction.

## Why our kernels are at ~10% bandwidth + ranked fixes (no Tensor Cores)
1. **CUDA GRAPHS = biggest lever** (+186% measured on GB10: 23.4→66.9 tok/s). Decode is so short that
   kernel-launch/host-sync latency dominates on low-bandwidth devices. (arXiv 2605.30571 "Memory-Bound but Not
   Bandwidth-Limited" — lower-bandwidth devices are launch-bound, not DRAM-bound.)
2. **Kill fp32 activations.** GEMV reads the activation vector per output row; fp32 = 4 bytes/elem vs 0.5 for
   4-bit weights → activations can dominate. Use fp16 (exllama, __hfma2) or int8+dp4a (llama.cpp mmvq Q8_1).
3. **128-bit coalesced weight loads** (ld.global.v4 / int4), consecutive lanes read consecutive packed words.
   We currently use uint32 (32-bit). Marlin lop3 trick decodes 2 fp16 per op.
4. **Warp-per-output + __shfl warp-shuffle reduction, NO shared-mem reduction** (we do this for GEMV; MoE doesn't).
5. **Fill all SMs**: split-K across grid.z with atomicAdd when rows < SM count (exllama). Thor has 20 SMs.
6. **Fused MoE: ONE kernel for all 8 active experts** (not 8 launches), mask-not-pad (DeepGEMM masked grouped
   GEMM), adaptive small-block grid to fill SMs (SGLang: 207µs vs vLLM 370µs/layer batch-1).
7. **Byte reduction beyond experts**: quantize attention/lm_head residual (BF16) + compress KV — these non-FP4
   bytes pull the real A4B ceiling below the naive 137. PTX cache hints: L1::no_allocate on streamed weights,
   evict_last on reused activation.

## Reference implementations to copy
- llama.cpp `ggml-cuda/mmvq.cu` + `vecdotq.cuh` (Q8_1 activation + dp4a) — the canonical batch-1 4-bit matvec.
- exllamav2 `q_gemm_kernel.cuh` (int4 128-bit loads, fp16 acts, split-K grid.z atomics, 4 cols/thread).
- TRT-LLM `weightOnlyBatchedGemv/` + `cudaCoreGemmNVFP4.cu` (CUDA-core, Bs1/2/4 tiles).
- vLLM `fused_moe` + `moe_align_block_size` (on-device sort), `fused_marlin_moe`; DeepGEMM masked grouped GEMM.
- Veitner NVFP4-GEMV CuTeDSL posts; Blackwell NVFP4 hackathon writeup (2000µs→22µs, profiler: memory-bound).
- ⚠️ NVFP4 correctness: e4m3 group-16 block-scale exponent rebias (bias 7) + per-tensor fp32 global scale
  separately — vLLM #34694 shipped a rebias bug that underflowed weights to zero. (We already match this.)

## CORRECTED PLAN (replaces the Tensor-Core pivot)
1. CUDA graphs over the decode step (kill launch overhead) — try first, likely biggest single win.
2. Bandwidth GEMV: fp16/int8 activations, 128-bit loads, fill SMs (split-K).
3. Fused single-launch MoE GEMV.
4. Byte reduction (residual quant + KV) if chasing >90 tok/s.
5. DFlash on top of a fast base to approach/beat 110.
Realistic: base ~52 (vLLM parity) is the near-term target; 110 needs base ~52 + DFlash.
