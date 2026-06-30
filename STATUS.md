# gemma4-cuda-server — build status

Single-architecture CUDA decode server for `google/gemma-4-26B-A4B-it` (NVFP4) on Jetson Thor (GB10, sm_120/121).
See directive in project root convo; verified arch in `reference/ARCHITECTURE.md`.

## Decisions made
- **Phase-1 reference path: RedHatAI NVFP4 target + DFlash** (user choice, directive-literal).
  Tests DFlash-against-quantized-target; not the bf16 model-card pairing. Risk noted in P1.

## Phase status
- [x] **Phase 0 — env**: Thor, CUDA 13.0, nvcc 13.0.48 (sm_120/121 ok). torch NOT in base env → container-based. 264 GB free disk.
- [x] **Phase 1 — reference trace capture — DONE (core)**
  - [x] DFlash drafter present; RedHatAI NVFP4 target downloaded (16.4 GB, verified)
  - [x] Reference serving: image `vllm/vllm-openai:gemma-aarch64-cu130` (vllm 0.22.1rc1 +dflash, transformers 5.10.2). NOT the dflash fork (its transformers too old for gemma4). `scripts/serve_reference.sh`. cutlass MoE + TRITON_ATTN + BF16 KV + `--language-model-only`. Health 200.
  - [x] **RISK RESOLVED ✅**: DFlash DOES work against the NVFP4 W4A4 target. vLLM treats draft as EAGLE-style (shares target embed/lm_head), taps target layers (2,7,12,18,23,28). cutlass MoE loads 128 experts with no crash. KV headroom 778k tokens at BF16.
  - [x] Suite + capture harness (`scripts/prompt_suite.json`, `scripts/capture_traces.py`) → `traces/<id>.json` + summary.json. 18 prompts (all categories incl. long-context needle); extend toward 100 later (mechanical).
  - **Results (temp0 greedy):** all outputs correct (Paris, $0.05 bat-ball, 3 r's in strawberry, needle 7-ZULU-42 retrieved, code runs). mean DFlash tau 4.61 acc/draft (code 7-11, reasoning 5-6, prose ~0.9). avg decode 41.7 tok/s (code 85-114, prose ~20). TTFT ~0.19s.
  - NOTE: raw /v1/completions degenerates (instruction-tuned → must use chat endpoint). longctx tok/s is prefill-dominated (artifact, not decode rate).
- [ ] **Phase 2 — scaffold**: dirs + git done; stub sources not yet written
- [ ] **Phase 3.1** FP4 GEMM + loader (compressed-tensors nvfp4 W4A4, group 16, fp8 e4m3 scales)
- [ ] **Phase 3.2** attention (global hd512/2kv/pRoPE-0.25/θ1e6 vs sliding hd256/8kv/θ1e4/win1024)
- [x] **Phase 3.3** MoE (router softmax-top8-renorm + dense MLP || 128 experts) — in src/forward.cu
- [x] **Phase 3.4 ✅** full forward GATE PASSES: top-1 matches vLLM on confident prompts (Paris/Blue/4, logprobs match). gibberish prompts not a valid gate (norm 588x amplifies quant noise). scripts/gate_forward.sh
- [x] **Step B (3.5)** single-session BF16 KV cache + incremental decode — LOSSLESS (KV-cached gen == full recompute: 'Red, yellow, and blue.'), prefill top-1 still Paris. Decode 6.54 tok/s (was sec/tok). sdpa_cache_kernel + Session. CAP via CTX env. BF16-KV budget ~14GB@64K. TODO: M=1 GEMV (avoid 128-pad), batched MoE, fewer host round-trips.
- [ ] **Phase 3.6** DFlash draft+verify (block_size 16, taps [1,6,11,17,22,27])
- [ ] **Phase 3.7** fused decode-step CUDA graph
- [ ] **Phase 3.8** OpenAI-compatible server loop
- [ ] **Phase 4** bench vs vLLM reference  |  Phase 5 TurboQuant (opt)  |  Phase 6 writeup

## Key corrections vs directive (see ARCHITECTURE.md)
- Quant is compressed-tensors NVFP4 **W4A4** (not GPTQ; activations also FP4)
- Global vs sliding layers differ in head_dim (512 vs 256) AND kv-heads (2 vs 8) AND RoPE
- Activation gelu_pytorch_tanh (not silu); full layers at [5,11,17,23,29]; weights 16.4 GB
