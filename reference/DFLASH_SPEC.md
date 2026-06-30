# DFlash Speculative Decoding — Implementable Spec (single-session, greedy)

Target: `gemma-4-26B-A4B-it` (NVFP4, 30 layers, hidden 2816, final_logit_softcap 30).
Drafter: `z-lab/gemma-4-26B-A4B-it-DFlash` (`DFlashDraftModel`, qwen3-style, 5 layers).
`num_speculative_tokens = 15`, `block_size = 16` (block_size = 1 bonus + 15 drafts).

Line citations are into `reference/dflash_src/vllm_dflash_full.txt`
(== `/tmp/dflash_src.txt`, identical line numbers). Three concatenated files:
`dflash.py` (1–308), `llm_base_proposer.py` (309–2026), `qwen3_dflash.py` (2027–2643).

> **Core idea.** DFlash is *parallel* (single-forward) block-diffusion drafting.
> The drafter is a tiny 5-layer transformer that does **cross-attention**: its
> Keys/Values are precomputed from the **target model's fused hidden states**
> (context), and its Queries are `[bonus_token, MASK×15]`. One draft forward
> denoises all 15 MASK positions at once → 15 draft tokens. Verification is the
> **standard greedy longest-prefix rule** done by the target (NOT in these files).

---

## 0. Shapes / constants (confirmed from checkpoint + config)

```
hidden               = 2816
n_draft_layers       = 5
n_heads (q)          = 32      head_dim = 128   q_size  = 4096
n_kv_heads           = 8                          kv_size = 1024   (GQA, 4:1)
attn scaling         = head_dim**-0.5 = 1/sqrt(128)      [2115]   (NOTE: target uses 1.0)
mlp                  = SiLU gate, gate/up [5632,2816], down [2816,5632]   (Qwen2MLP) [2209]
rms_norm_eps         = 1e-6
rope_theta           = 1e6, neox style, head_dim 128            [2192]
sliding_window       = 2048 (4 sliding layers + 1 full) — draft's OWN window, independent of target
fc.weight            = [2816, 16896]   16896 = 6 * 2816  (fuses 6 target taps)  [2295-2303]
hidden_norm.weight   = [2816]   (RMS-norm on fused target hidden, context branch only) [2304]
norm.weight          = [2816]   (final RMS-norm after layer 5)                          [2308]
per-layer            : q_proj[4096,2816] k_proj[1024,2816] v_proj[1024,2816] o_proj[2816,4096]
                       q_norm[128] k_norm[128]  (NO v_norm, NO biases)
mask_token_id        = 4
target_layer_ids     = [1, 6, 11, 17, 22, 27]   (into target's 30 decoder layers)
```

**Draft has NO `embed_tokens` and NO `lm_head` in the checkpoint** → both are
**shared from the target** (`_maybe_share_embeddings` [1602-1664], `_maybe_share_lm_head`
[1671-1720]). See Q-embed note in §3.

---

## Q1. Target hidden-state taps (which / how combined / where)

- **Which:** the residual-stream hidden states output by target decoder layers
  `[1, 6, 11, 17, 22, 27]` (EAGLE3 "aux hidden state" convention — the layer's
  output residual, i.e. value of the residual stream *after* that decoder layer).
  6 vectors of dim 2816 each. Driven by `target_layer_ids` in `dflash_config`
  ([2287-2288]); `use_aux_hidden_state=True` ([304], [2260-2263]).
- **At which positions:** for every token position the target runs a forward over
  in the current step (prefill = all prompt positions; verify step = the block of
  verified positions). One 6-tap vector per position.
- **How combined:** concatenate the 6 taps **in `target_layer_ids` order** →
  `[T, 16896]`, then a single linear `fc` (no bias) → `[T, 2816]`. This is
  `combine_hidden_states` → `self.model.fc(hidden_states)` ([2596-2608], called at
  [775-778] at the very start of `propose`).
- A second RMS-norm `hidden_norm` is applied to this fused vector, but **only in
  the context-KV branch** (inside `precompute_and_store_context_kv` [2404-2409]),
  not here. So full context pipeline = `concat6 → fc → hidden_norm → per-layer KV`.

> Implementer: you must add taps to your Gemma forward that copy the residual
> stream after layers 1,6,11,17,22,27 into a `[T,6,2816]` buffer (concat order =
> that list). Everything else (fc, hidden_norm) lives in the drafter.

---

## Q2. Draft input construction / where MASK comes from

This **is** block diffusion. For one propose call, per request, the draft's
**query** sequence is exactly `block_size = 16` tokens:

```
query_ids = [ next_token_id ,  MASK , MASK , ... , MASK ]   # 1 bonus + 15 masks
            (=4)              (15 copies of mask_token_id=4)
```

Built by `copy_and_expand_dflash_inputs_kernel` ([135-162], kernel body itself is
in `vllm/v1/spec_decode/utils.py`, NOT in these files — semantics inferred):
`next_token_id` written to query slot 0, `parallel_drafting_token_id` (=4) to the
other 15 ([155-158]). `num_query_per_req = 1 + num_speculative_tokens = 16` ([110]).

The **context** (everything before the block) does **not** enter as tokens — it is
injected purely as precomputed K/V (§ below). Query tokens enter as **embeddings**
via the shared target `embed_tokens` (`embed_input_ids` [2313-2314, 2552-2558]).

**Two-stream layout for one propose:**
- Context stream: `num_context` positions, fused target hidden → K/V only,
  pre-inserted into the draft KV cache at `context_slot_mapping`
  (`precompute_and_store_context_kv` [271-275, 2371-2463]).
- Query stream: 16 positions/req, embeddings → flow through the 5 layers; their
  Q attends to (all context K/V) ∪ (the 16 query K/V). `token_indices_to_sample`
  selects the **15 MASK positions** to read logits from ([120-124]; size
  `batch*num_speculative_tokens`).

**Attention is NON-CAUSAL** within the query block by default:
`dflash_causal = dflash_config.get("causal", False)` ([72]); `new_cad.causal =
self.dflash_causal` ([195]); enforced at [292-300]. So all 16 query positions
attend bidirectionally to each other and to all context — this is what lets one
forward denoise the whole block.

---

## Q3. Propose (draft) algorithm — SINGLE forward, parallel

DFlash sets `parallel_drafting = True`, so `propose` returns after **one** draft
forward ([834-842]): `if num_speculative_tokens == 1 or self.parallel_drafting:`
→ sample → return. There is **no iterative denoising loop** for DFlash (the
multi-step loop at [897-971] is for EAGLE/MTP and is not taken).

Per propose step (single session, greedy/T0):

```
# --- inputs from the target's most recent forward ---
taps6        : [C, 6, 2816]   # residual after target layers 1,6,11,17,22,27, for the C context positions
ctx_pos      : [C]            # absolute sequence positions of those context tokens
next_token   : int           # token the target just sampled (the bonus); goes at position P0
P0           = ctx_pos[-1] + 1

# --- (A) build & insert context K/V into draft KV cache (precompute_and_store_context_kv) ---
fused   = fc(concat_last_dim(taps6))                 # [C,2816]   ([775-778],[2605])
fused_n = rmsnorm(fused, hidden_norm.weight, eps)    # [C,2816]   ([2404-2409])
for l in 0..4:                                        # one fused GEMM over all layers in vLLM
    K_l = k_proj[l] @ fused_n ; V_l = v_proj[l] @ fused_n          # [C,1024] each ([2410-2420])
    K_l = per_head_rmsnorm(K_l, k_norm[l].weight, eps)            # per 128-dim head ([2422-2430])
    K_l = rope(K_l, ctx_pos, theta=1e6, neox)                     # V NOT roped ([2432-2447])
    write (K_l, V_l) into draft_kv_cache[layer l] at slots for ctx_pos   # ([2452-2463])
# NOTE: every layer's context K/V is projected from the SAME fused_n (shared input),
#       NOT from a per-layer residual stream. Key DFlash trick.

# --- (B) draft forward over the 16 query tokens ---
qids = [next_token, 4,4,...,4]            # len 16
qpos = [P0, P0+1, ..., P0+15]
h = embed_tokens(qids)                    # shared target table, dim 2816 (see Q-embed note)
resid = None
for l in 0..4:
    h, resid = decoder_layer_l(qpos, h, resid)        # see §Q7 for exact ops
                                                      # K/V of query tokens also written to cache,
                                                      # Q attends to context-slots ∪ query-slots (non-causal)
h = rmsnorm_combine(h, resid, norm.weight, eps)       # final norm ([2483])

# --- (C) sample 15 draft tokens from the 15 MASK positions ---
logits = lm_head(h[mask_positions])       # shared target lm_head ; mask_positions = query slots 1..15
draft_ids[0..14] = argmax(logits, -1)     # greedy ([716],[724-725])
return draft_ids                          # [15]
```

`_greedy_sample` = `compute_logits(...).argmax(-1)` ([712-716]). At T0 everything is
argmax; probabilistic path ([2026]) is unused (`# we always use argmax` [1982-1985]).

**Output of one propose():** a flat block of **15 candidate token ids** (no tree).
Combined with the already-known `next_token` (bonus), the verifier sees 16
positions `P0..P0+15`.

---

## Q4. Verify algorithm — standard greedy spec-decode (NOT in these files)

DFlash uses vLLM's generic rejection sampler; the accept/reject code lives in the
gpu_model_runner + `vllm/v1/sample/rejection_sampler.py`, **not** in the three
provided files. `propose()` only returns draft ids; verification is external.

At **temperature 0 (greedy)** the rule is the standard one:

```
# target runs ONE forward over the 16 verify positions P0..P0+15:
#   inputs = [next_token, draft_0, draft_1, ..., draft_14]    (the proposed block)
#   produces target_argmax[i] = argmax target logits at position P0+i, i=0..15
# (target_argmax[i] is the model's prediction for the token AFTER position P0+i)

accepted = [next_token]                 # bonus is always accepted (target already sampled it)
for i in 0..14:                         # check each of the 15 drafts
    if draft_ids[i] == target_argmax[i]:    # draft token == what target would greedily emit
        accepted.append(draft_ids[i])
    else:
        break                           # first mismatch ends acceptance
# append one bonus token = target_argmax at the first unaccepted position:
accepted.append(target_argmax[len(accepted)-1])
```

So per step you commit `n_accepted (0..15) + 1 bonus` new tokens, i.e. **1..16**
tokens, with exactly **one** target forward per block. (Reported acceptance length
≈ 4.3–8.6 tokens/step for block 16.) `num_rejected_tokens = block - 1 - n_accepted`
is plumbed back to the next propose ([134, 170-171, 1095-1096]). No tree, no
typical-acceptance for DFlash at T0.

---

## Q5. Decode loop / chaining

```
prefill: target forward over prompt → taps6 for all prompt positions, sample next_token (pos = len).
         propose(context = all prompt positions) → 15 drafts.
loop:
  1. target verify-forward over the 16 block positions [bonus, draft_0..draft_14]
     → 16 sets of taps6 (one per block position) + 16 target_argmax.
  2. greedy accept (Q4) → commit n_accepted+1 tokens; new next_token = last committed.
  3. the committed positions' taps6 become the NEW context for the next propose:
       - fuse + hidden_norm + per-layer KV (Q3-A), append to draft KV cache at their slots.
       - (rejected block positions' draft KV are discarded / overwritten via num_rejected.)
  4. propose again (Q3-B/C) at P0 = (last committed position)+1 → 15 new drafts.
```

- **One target forward per accepted block** (the verify pass). The drafter adds one
  small forward per block (negligible).
- Draft & target KV caches both grow by `n_accepted+1` per step.
- `next_token_ids` carry-over: bonus from verify becomes query slot 0 of next block
  ([137],[1048],[1207-1238]).

---

## Q6. Draft KV cache

Yes — the draft maintains its **own paged KV cache** across propose steps
(separate from the target; block table, `block_size` from the attn backend
[1932-1935]). Two kinds of entries occupy it:

1. **Context K/V** for confirmed positions — written by
   `precompute_and_store_context_kv` from the target's fused hidden (Q3-A). These
   are the durable entries; they persist and are attended to by all later blocks.
2. **Query K/V** for the 16 block positions — written during the draft forward.
   These are transient: on the next step the accepted positions' *context* K/V
   (from real target hidden) overwrite them, and rejected ones are dropped.

Window: the draft's own `sliding_window = 2048` (4 sliding layers + 1 full
attention layer), **independent** of the target's window. Positions are absolute
sequence positions (RoPE uses them directly).

> Single-session simplification: keep one contiguous draft K/V buffer of shape
> `[5 layers, max_seq, 8 kv_heads, 128]`. Each step: append context K/V for the
> newly committed positions; run the block attending to `[0 .. P0-1]` (context)
> plus the 16 in-block positions (non-causal); never persist in-block query K/V.

---

## Q7. Exact draft forward (qwen3 specifics; diffs vs Gemma target)

Per `DFlashQwen3DecoderLayer.forward` ([2221-2240]) and `DFlashQwen3Attention`
([2152-2177]). All RMSNorms are **qwen3/vLLM style: plain `weight` (NOT `1+weight`),
fp32 compute, `x*rsqrt(mean(x²)+eps)*weight`, eps=1e-6** ([2149-2150, 2216-2219]).

```
layer l (input h, residual r):
  if r is None: r = h ; h = input_layernorm(h)
  else:         h, r = input_layernorm(h, r)            # fused: normalizes (h+r), updates r=h+r
  # attention (query stream):
  q,k,v = split(qkv_proj @ h, [4096,1024,1024])
  q = per_head_rmsnorm(q, q_norm.weight)   # reshape [.,32,128], norm each 128 head ([2166-2168])
  k = per_head_rmsnorm(k, k_norm.weight)   # reshape [.,8,128]                       ([2169-2171])
  q,k = rope(q,k, positions, theta=1e6, neox)                                        ([2173])
  a = attention(q,k,v, scale=1/sqrt(128), kv = context_slots ∪ block_slots, causal=False)
  h = o_proj @ a                                                                     ([2176])
  h, r = post_attention_layernorm(h, r)    # fused norm(h+r), r=h+r                  ([2238])
  h = down_proj( silu(gate_proj @ h) * (up_proj @ h) )   # Qwen2MLP                  ([2239])
  return h, r
# after 5 layers:
h, _ = norm(h, r)     # final RMSNorm on (h+r)                                       ([2483])
logits = lm_head(h)   # shared target lm_head
```

**Differences from the Gemma-4 target (do NOT copy target conventions into draft):**
| Aspect | Draft (qwen3) | Gemma-4 target |
|---|---|---|
| RMSNorm weight | plain `weight` | plain `weight` (both plain; eps placement differs slightly) |
| attn scaling | **1/sqrt(128)** [2115] | **1.0** (no 1/sqrt(d)) |
| v_norm | **none** | has v_norm (no-scale) |
| RoPE theta / head_dim | 1e6 / 128 | (target's own) / 256 |
| MLP | dense SiLU gate, 5632 | MoE (router + experts) |
| logit softcap | 30 in config but **irrelevant at T0** (monotonic, argmax unchanged) | 30, applied |

**Q-embed scale (AMBIGUITY — resolve before bring-up):** the Gemma target scales
embeddings by `sqrt(2816) ≈ 53.0` (cast to bf16 first). The drafter's
`embed_input_ids` ([2313-2314, 2472]) calls the **shared** target `embed_tokens`
with **NO scale** in the drafter code path. In vLLM/Gemma the `* embed_scale`
happens in the *target's* model.forward, not inside the embedding module, so the
shared table returns the **raw, unscaled** row to the drafter. **To match this
reference, feed the drafter UNSCALED embeddings** (i.e. do not apply sqrt(2816)).
This is the single most likely place to get a silent mismatch — verify against a
known-good draft logit if possible.

**lm_head:** shared target head, `logits = lm_head @ h` (tied/raw table, not
pre-scaled). softcap-30 may be applied by the target's LogitsProcessor but is
order-preserving so does not change the greedy argmax draft tokens.

---

## Ambiguities / not in the provided source (flagged)
1. **`copy_and_expand_dflash_inputs_kernel`** ([135-162]) body is in
   `spec_decode/utils.py` (not provided). The exact query **positions** and
   slot-mapping are inferred: bonus at `P0 = last_ctx_pos+1`, masks at `P0+1..P0+15`
   (contiguous). High confidence from `num_query_per_req` and RoPE needs, but
   confirm the position of the bonus token vs masks if logits don't match.
2. **Accept/reject code** (Q4) is the generic vLLM rejection sampler, not in these
   files — described as the standard greedy longest-prefix + 1 bonus rule.
3. **Q-embed scale** (Q7) — reference path applies none; flagged above.
4. **hidden_norm vs input_layernorm asymmetry:** context K/V use `hidden_norm`
   on the fused target hidden; query tokens use the normal per-layer
   `input_layernorm` on embeddings. Confirmed in source ([2404-2409] vs [2228-2231]).
