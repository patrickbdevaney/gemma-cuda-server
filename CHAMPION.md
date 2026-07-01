# CHAMPION — best-known configuration (single lineage)

**Champion metric:** base decode tok/s (primes, sustained 60-tok, max of 2 runs). Base is the foundation
that DFlash builds on; secondary metrics tracked too.

| field | value |
|---|---|
| **BASE decode** | **44.5 tok/s** (FP4 lm_head) |
|  DFlash (predictable, DK=14) 37.42 tok/s OLD:| 30.58 tok/s, accept 11.14/14 |
| correctness | gate_self.sh PASS (primes + Paris/Blue/4/Au) |
| commit | cycle23 NVFP4-lm_head |
| date | 2026-06-30 |
| context depth | prompt ~25 tok, decode 60 |
| memory | 16.4 GB weights |

Hardware ceiling note: A4B active ≈ 1.4 GB FP4 weight/token; at Thor 273 GB/s → ~190 tok/s memory-bound
ceiling for base (realistic ~70-90 well-tuned). DFlash on predictable can exceed base. Target: well-exceed 110.

Reproduce: `bash scripts/gate_self.sh && bash scripts/bench.sh`

---
## BANKED — pure-CUDA baseline (2026-07-01)
Stable, gated baseline. base 44 tok/s, DFlash 82 (easy) / 58 (hard workload), accept 13.33/14.
Full loop: base 29.9->44 (+47%), DFlash 30.6->82 (+168%). THE lever = NVFP4 lm_head.
Reference: vLLM DFlash = 100-105 tok/s on same model/hardware (mature flashinfer/cutlass TC + full graph
+ bf16 lm_head). Our draft is BETTER (tau 10-13 vs vLLM 7.84); our gap is per-step kernel efficiency.
TC-verify/full-graph synthesis toward 110-140 is being developed in ~/gemma-cuda-hybrid (CUTLASS FP4
proven to compile+run on Thor sm_110a/CUDA13.0). This repo stays the stable pure-CUDA reference.
