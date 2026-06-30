# Gemma-4-26B-A4B-it — VERIFIED architecture (read from real config.json, not assumed)

Source: `RedHatAI/gemma-4-26B-A4B-it-NVFP4/config.json` (sha 24eef43, 2026-06-18).
Cross-checked against `z-lab/gemma-4-26B-A4B-it-DFlash/config.json`.
Everything here is read from the actual checkpoint. Where the original directive
disagreed, the directive is wrong — flagged with ⚠️.

## Top-level
- arch `Gemma4ForConditionalGeneration`, `model_type: gemma4` (multimodal wrapper:
  text_config + vision_config + audio_config(null)). We serve TEXT ONLY.
- `tie_word_embeddings: true` → lm_head tied to embed_tokens (and lm_head is in the
  quant ignore-list → stays bf16). Embedding table is shared, unquantized.

## Text model (`text_config`, model_type `gemma4_text`)
- `num_hidden_layers: 30` ✅
- `hidden_size: 2816` (matches DFlash draft hidden_size → draft taps target hidden states)
- `hidden_activation: gelu_pytorch_tanh`  ⚠️ NOT silu (DFlash draft uses silu; target does not)
- `final_logit_softcapping: 30.0` — logits are softcapped; MUST replicate for logit match
- `vocab_size: 262144`, `attention_bias: false`, `rms_norm_eps: 1e-6`

### Attention — HYBRID, and the two layer types differ in MORE than window
30-entry `layer_types`: pattern = 5×sliding then 1×full, repeated.
**full_attention (global) layers = indices [5, 11, 17, 23, 29]** (5 of them).
The other 25 are sliding_attention.

|                     | sliding (25 layers) | full/global (5 layers) |
|---------------------|---------------------|------------------------|
| head_dim            | **256**             | **512** (`global_head_dim`) ⚠️ different! |
| num_attention_heads | 16                  | 16 |
| num_key_value_heads | **8**               | **2** (`num_global_key_value_heads`) ⚠️ different! |
| sliding_window      | **1024**            | n/a (full) |
| RoPE type           | `default`           | `proportional` (p-RoPE) |
| RoPE theta          | **10000**           | **1000000** |
| partial_rotary      | 1.0 (full)          | **0.25** (only 25% of dims rotated) |

- ⚠️ This is the **p-RoPE** the directive asked about: global layers use proportional
  RoPE with `partial_rotary_factor: 0.25`; sliding use default full RoPE. Different theta too.
- `attention_k_eq_v: true` — MUST confirm semantics from modeling_gemma4.py (K and V
  projection/weights equal?). Affects KV cache layout. UNRESOLVED until modeling code read.
- KV-cache sizing differs per layer type: global layers store 2 KV heads × 512 head_dim;
  sliding store 8 KV heads × 256 head_dim but only a 1024-token window. Budget accordingly.

### MoE
- `enable_moe_block: true`, `num_experts: 128`, `top_k_experts: 8`
- `moe_intermediate_size: 704` (per-expert FFN width)
- `intermediate_size: 2112` (the dense/shared path width) — shared-expert handling TBD from modeling code
- routers (`layers.N.router.proj`, all 30) are in the quant ignore-list → **bf16**

## Quantization (NVFP4 / compressed-tensors — ⚠️ NOT GPTQ as directive claimed)
- `quant_method: compressed-tensors`, `format: nvfp4-pack-quantized`, llm-compressor v0.14.1
- **W4A4**: BOTH weights AND input_activations are 4-bit:
  - weights: num_bits 4, group_size 16, strategy `tensor_group`, symmetric, scale_dtype `float8_e4m3fn`, static
  - input_activations: num_bits 4, group_size 16, `dynamic: local`, static_minmax → quantized at runtime
- targets `Linear`; **ignore-list** (stay bf16): all 30 `router.proj`, ALL `vision_tower.*`,
  `embed_vision.embedding_projection`, `lm_head`.
- `kv_cache_scheme: null` → checkpoint does NOT bake KV quant; FP8 KV is a serve-time choice (vLLM `--kv-cache-dtype fp8`). ✅ consistent with directive.
- Single shard `model.safetensors` = 16.4 GB (⚠️ directive said ~13–15 GB).

## DFlash draft (`z-lab/gemma-4-26B-A4B-it-DFlash`)
- arch `DFlashDraftModel`, `model_type: qwen3`, 5 layers, `block_size: 16`
- `dflash_config.target_layer_ids: [1,6,11,17,22,27]`, `num_target_layers: 30`, `mask_token_id: 4`
- hidden_size 2816 (matches target), head_dim 128, 32 q-heads / 8 kv-heads, sliding_window **2048**
  (⚠️ draft window 2048 ≠ target sliding window 1024 — independent KV), `final_logit_softcapping: 30.0`
- layer_types: 4×sliding + 1×full. Activation silu. tie_word_embeddings false.

## OPEN QUESTIONS — resolve from modeling_gemma4.py before writing attention/MoE kernels
1. `attention_k_eq_v: true` exact semantics (shared K/V weights? shared cache?).
2. Global-attention shape: 16 q-heads × head_dim 512 with 2 KV heads — confirm q/k/v proj dims.
3. Shared/dense expert vs the 128 routed experts — is `intermediate_size 2112` a always-on shared FFN?
4. p-RoPE `proportional` exact formula + which 25% of dims are rotated.
5. Query/key normalization (Gemma uses qk-norm in some variants?) — confirm.
