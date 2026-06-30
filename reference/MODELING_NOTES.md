# Gemma-4-26B-A4B Text Model — Exact Numerical Spec (ground truth for CUDA kernels)

Source of truth: HuggingFace Transformers `models/gemma4/modeling_gemma4.py` (extracted from
`vllm/vllm-openai:gemma-aarch64-cu130`), cross-checked against vLLM `model_executor/models/gemma4.py`.
All `file:line` citations below are into `reference/transformers_src/modeling_gemma4.py` unless prefixed
`[vllm]` (→ `reference/transformers_src/vllm_gemma4.py`) or `[cfg]` (→ `configuration_and_filelist.txt`)
or `[rope]` (→ `reference/transformers_src/rope_proportional.txt`).

> NOTE: the Transformers *default* `Gemma4TextConfig` values are for the small `e2b` model
> (hidden 2304, 8 heads). The **gemma-4-26B-A4B** values used in this spec are the ones supplied in the
> task (hidden 2816, 16 heads, etc.). Formulas are config-driven, so they hold for the 26B model.

## 0. Config (gemma-4-26B-A4B)
```
hidden_size                 = 2816
num_hidden_layers           = 30
num_attention_heads         = 16
head_dim (sliding)          = 256
global_head_dim (full)      = 512
num_key_value_heads (slid)  = 8
num_global_key_value_heads  = 2      (full-attn, used only because attention_k_eq_v=True)
attention_k_eq_v            = True
num_kv_shared_layers        = 0      (=> NO kv-sharing; see §3e)
vocab_size                  = 262144
tie_word_embeddings         = True
rms_norm_eps                = 1e-6
hidden_activation           = gelu_pytorch_tanh
sliding_window              = 1024
final_logit_softcapping     = 30.0
# MoE
enable_moe_block            = True
num_experts                 = 128
top_k_experts               = 8
moe_intermediate_size       = 704     (per-expert FFN)
intermediate_size           = 2112    (the DENSE MLP that runs in PARALLEL — see §5)
use_double_wide_mlp         = False
# PLE (Per-Layer Embeddings) — SEE §8, easy to miss
hidden_size_per_layer_input = 256
vocab_size_per_layer_input  = 262144
# layer types: full_attention at indices [5,11,17,23,29], sliding elsewhere ([cfg]:207-218)
# RoPE: sliding -> default theta 1e4 ; full -> proportional theta 1e6, partial_rotary_factor 0.25
```

---

## 1. Embedding scaling  ✅ YES, multiply by sqrt(hidden_size)
`Gemma4TextScaledWordEmbedding` (1459-1470). `embed_scale = hidden_size**0.5` set at 1602-1604.
```
inputs_embeds = embed_tokens(input_ids) * embed_scale         # 1470
embed_scale   = sqrt(2816) = 53.0660...
```
DTYPE GOTCHA (1470, comment 1601): the scale is cast to the weight dtype BEFORE the multiply:
`super().forward(ids) * self.embed_scale.to(self.weight.dtype)`. In bf16, sqrt(2816) rounds
(e.g. 53.0660 → nearest bf16 53.0). Embedding lookup is the raw table row, then scaled.
There is a SECOND scaled embedding (PLE) with scale `sqrt(256)=16.0` (1618-1623) — see §8.

---

## 2. RMSNorm  ⚠️ NOT the (1+weight) Gemma2/3 variant
`Gemma4RMSNorm` (193-211). Computed in fp32, multiplied by `weight` DIRECTLY (no `1+`):
```
def norm(x):                                   # x in fp32 (caller passes x.float(), line 208)
    ms = mean(x**2, dim=-1, keepdim=True) + eps         # 203, eps=1e-6 ADDED INSIDE mean (not sqrt)
    xn = x * pow(ms, -0.5)                               # 205  (pow, not rsqrt — JAX parity)
    if with_scale: xn = xn * weight.float()             # 210  ← plain weight, weight initialised to ONES (200)
    return xn.type_as(input)                             # 211  cast back to input dtype at the very end
```
- eps is added to the mean-square BEFORE the inverse-sqrt (Gemma convention), not outside.
- weight stored centered at 1.0 (init `ones`, 200). DO NOT add 1.0. (vLLM uses plain `RMSNorm`, [vllm]:428.)
- `with_scale=False` variant (no weight, pure normalize): used by `v_norm` and `router.norm`.

### Every norm in ONE text decoder layer, in dataflow order (1370-1456)
1. `input_layernorm`            RMSNorm(2816), scaled              (1378)
2. `self_attn.q_norm`           RMSNorm(head_dim), scaled          (1210) — per-head, on Q
3. `self_attn.k_norm`           RMSNorm(head_dim), scaled          (1214) — per-head, on K
4. `self_attn.v_norm`           RMSNorm(head_dim), **NO scale**    (1215) — per-head, on V
5. `post_attention_layernorm`   RMSNorm(2816), scaled              (1379)
6. `pre_feedforward_layernorm`  RMSNorm(2816), scaled              (1380) — feeds DENSE mlp
7. `router.norm`                RMSNorm(2816), **NO scale**        (1342) — inside router
8. `pre_feedforward_layernorm_2`RMSNorm(2816), scaled              (1397) — feeds MoE experts
9. `post_feedforward_layernorm_1`RMSNorm(2816), scaled             (1395) — on dense-mlp output
10.`post_feedforward_layernorm_2`RMSNorm(2816), scaled             (1396) — on moe output
11.`post_feedforward_layernorm` RMSNorm(2816), scaled              (1381) — on (h1+h2) before residual
12.`post_per_layer_input_norm`  RMSNorm(2816), scaled              (1389) — PLE branch (§8)
Final model norm: `model.norm` RMSNorm(2816) scaled, after all layers (1608,1730).
(Norms 7–11 #7,#8,#10 exist only because `enable_moe_block=True`; with MoE off only 1,2,3,4,5,6,11,12.)

---

## 3. Attention  (`Gemma4TextAttention` 1178-1291)

### 3a. attention_k_eq_v = True  — what it does (1190, 1220-1224, 1259, 1265)
`use_alternative_attention = attention_k_eq_v and not is_sliding`  → TRUE only on FULL-attn layers.
When true:
- `v_proj` is **None** (1220-1224): there is NO value projection weight matrix.
- In forward, `value_states = key_states` where `key_states` is the **raw `k_proj(x)` output**
  (1259), captured BEFORE k_norm/RoPE are applied to K.
- K and V then diverge:  K = RoPE(k_norm(k_proj x)) (1261-1263);  V = v_norm(k_proj x) (1265, NO RoPE).
So K and V **share the same projection weights** (`k_proj`) but get different normalization
(k_norm w/ scale vs v_norm w/o scale) and only K is rotated. [vllm] confirms: K weights are loaded into
both K and V slots of fused qkv; V gets v_norm, no RoPE ([vllm]:511-531).
Sliding layers: `attention_k_eq_v` does NOT apply → normal independent `v_proj` exists.

### 3b. Projection dims
SLIDING layer (head_dim=256, kv=8):
```
q_proj: 2816 -> 16*256 = 4096        (1207)
k_proj: 2816 ->  8*256 = 2048        (1217)
v_proj: 2816 ->  8*256 = 2048        (1220) present
o_proj: 4096 -> 2816                 (1226)
num_key_value_groups = 16/8 = 2      (1194)  ; bias = attention_bias = False
```
FULL-attn layer (global_head_dim=512, kv=num_global_key_value_heads=2, k_eq_v):
```
q_proj: 2816 -> 16*512 = 8192        (1207)
k_proj: 2816 ->  2*512 = 1024        (1217)
v_proj: None  (V reuses k_proj out)  (1220-1224)
o_proj: 8192 -> 2816                 (1226)
num_key_value_groups = 16/2 = 8      (1194)
```
head_dim selection: `global_head_dim if (not is_sliding and global_head_dim) else head_dim` (1189).

### 3c. Per-head QK-norm: YES
RMSNorm applied to EACH head vector (length head_dim) of Q and K (q_norm 1245, k_norm 1261),
and to V (v_norm, no-scale, 1265). Order: project → reshape to (...,n_heads,head_dim) → norm → RoPE(Q,K only).

### 3d. Query / logit scaling  ⚠️ scaling = 1.0  (NO 1/sqrt(d), NO query_pre_attn_scalar)
`self.scaling = 1.0` (1195), passed as `scaling=` to the attention interface (1284).
`eager_attention_forward`: `attn_weights = (Q @ Kᵀ) * scaling` with scaling=1.0 (838) — i.e. **raw dot
product, no temperature**. This holds for BOTH 256- and 512-dim heads. Magnitude control comes entirely
from q_norm/k_norm (learned RMS scale). [vllm]:402-405 explicitly: "Gemma4 uses scaling=1.0 … Q/K norms
handle scaling implicitly." There is NO `query_pre_attn_scalar`.

### 3e. Attention logit softcapping: REMOVED
`Gemma4TextAttention.forward` does NOT pass `softcap` to the attention interface (1277-1287);
`eager_attention_forward`'s `softcap` defaults to None → tanh-cap branch (840-843) is skipped.
Gemma2's `attn_logit_softcapping (~50)` is GONE for gemma4 text. [vllm] passes
`attn_logit_softcapping` which is absent from gemma4 config → None ([vllm]:592).

### 3f. Sliding window
`sliding_window = config.sliding_window` (=1024) on sliding layers, None on full (1187).
Mask built by `create_sliding_window_causal_mask` (1701). Causal AND within window. Inclusive geometry
(from `sliding_window_mask_function` 1909-1922): keep key kv for query q iff `0 <= (q-kv) < window`,
i.e. the current token and the **1023 preceding** tokens are visible (window of 1024 inclusive of self).
(`num_kv_shared_layers=0` ⇒ `is_kv_shared_layer` is False for all layers; the shared-KV path 1252-1256
is DEAD for this model. Implement plain per-layer KV.)

---

## 4. RoPE / proportional p-RoPE
Convention: **rotate_half** (NOT interleaved). `rotate_half(x)=cat(-x[d/2:], x[:d/2])` (780-784);
`apply = x*cos + rotate_half(x)*sin` (787-806). Rotary pairs dim `i` with dim `i + head_dim/2`.
cos/sin built as `emb = cat(freqs, freqs)`; `cos=emb.cos()*attn_scaling`, attn_scaling=1.0 (1171-1175).
RoPE is applied to the **full head_dim** tensor; partial rotation is achieved by zeroing high inv_freqs.

### 4a. SLIDING layers — rope_type "default", theta=1e4, dim=head_dim=256 (1123-1157)
```
dim = 256
inv_freq[j] = 1 / (1e4 ** ( (2j) / 256 )),  j=0..127      # 128 freqs, FULL rotation
```
All 256 dims rotate; pairs (i, i+128) for i in [0,128).

### 4b. FULL-attn layers — rope_type "proportional", theta=1e6, partial_rotary_factor=0.25
`_compute_proportional_rope_parameters` [rope] uses `head_dim_key="global_head_dim"` (1115-1116) ⇒ head_dim=512.
```
head_dim    = 512
rope_angles = int(0.25 * 512 // 2) = int(64) = 64
inv_freq_rotated[j] = 1 / (1e6 ** ( (2j) / 512 )),  j=0..63   # NOTE divisor is head_dim(512), 64 entries
nope_angles = 512//2 - 64 = 192
inv_freq    = concat( inv_freq_rotated[64] , zeros[192] )     # length 256
```
Then `emb = cat(inv_freq, inv_freq)` (len 512); zero freqs ⇒ cos=1, sin=0 ⇒ those dims pass through.
Net effect with rotate_half pairing (i, i+256):
- pairs i in [0,64)  : ROTATED by angle `inv_freq_rotated[i] * position`   (128 dims = 25% of 512)
- dims i in [64,256) and [320,512) : UNCHANGED (NoPE pass-through).
So the rotated coordinates are indices {0..63} ⊗ {256..319}; everything else is identity.
"proportional" vs "default": proportional always emits a head_dim-length encoding but zero-pads the
inv_freq beyond `rope_angles` (partial rotation); default rotates the entire dim. Also note the exponent
denominator in proportional is the FULL head_dim (512), while default divides by its own `dim`.

cos/sin are computed in fp32 then cast to activation dtype (1169-1175).

---

## 5. MoE block  (`enable_moe_block=True` ⇒ every layer is DENSE-MLP ∥ MoE)  (1429-1444)

### 5a. There is a DENSE MLP **and** routed experts, run in PARALLEL and summed.
- DENSE MLP (`Gemma4TextMLP`, 1069-1085): gate/up/down, `intermediate_size=2112`, gated GELU-tanh:
  `down( gelu_tanh(gate(x)) * up(x) )`. This is the always-on "shared expert" equivalent.
- ROUTED experts (`Gemma4TextExperts`, 1294-1331): 128 experts, `moe_intermediate_size=704` each.
There is NO separate "shared_expert" module beyond the dense MLP. `intermediate_size` (2112) = dense MLP;
`moe_intermediate_size` (704) = per-routed-expert.

### 5b. Router  (`Gemma4TextRouter`, 1334-1367) — SOFTMAX over all experts, then renormalized top-8
```
x  = router.norm(residual)                 # RMSNorm, NO scale (1348)
x  = x * router.scale * (hidden_size**-0.5)# learned per-dim 'scale' (init 1) * 1/sqrt(2816)  (1349)
logits = proj(x)                           # Linear 2816 -> 128, no bias (1351)
probs  = softmax(logits, dim=-1)           # softmax over ALL 128 experts (1352)   ← NOT sigmoid
w, idx = topk(probs, k=8)                  # (1355)
w = w / w.sum(-1, keepdim=True)            # RENORMALIZE top-8 to sum 1 (1362)
w = w * per_expert_scale[idx]              # learned per-expert scale (init 1) (1365)
```
Router input is the **raw pre-norm residual** (attention-block output), 1433-1434
(`hidden_states_flat = residual.reshape(...)`). [vllm] default matches softmax→topk→renorm→per_expert_scale
([vllm]:185-203). No global routed_scaling_factor beyond `per_expert_scale`.

### 5c. Expert FFN (1325-1328): gated GELU-tanh, weights packed as gate_up.
```
gate, up = chunk( gate_up_proj[e] @ x , 2 )      # gate_up_proj: [128, 2*704, 2816]
y = gelu_tanh(gate) * up                          # act_fn = gelu_pytorch_tanh (1305)
y = down_proj[e] @ y                              # down_proj: [128, 2816, 704]
y = y * top_k_weights                             # scale by router weight (1328)
```
Output = sum over the token's 8 selected experts (index_add, 1329).

### 5d. Combine (1430-1441)
```
h1 = post_feedforward_layernorm_1( dense_mlp( pre_feedforward_layernorm(residual) ) )   # 1426-1430
h2 = post_feedforward_layernorm_2( experts( pre_feedforward_layernorm_2(residual) ) )   # 1435-1438
hidden_states = h1 + h2                                                                 # 1441
```
NOTE the two FF branches read the SAME `residual` through DIFFERENT pre-norms
(`pre_feedforward_layernorm` for dense, `pre_feedforward_layernorm_2` for MoE).

### 5e. enable_moe_block / use_double_wide_mlp
- `enable_moe_block` (global): if True, every decoder layer instantiates router+experts+the 3 extra
  norms and runs the §5d parallel block; if False, only the dense MLP path (1429 guard).
- `use_double_wide_mlp` (1074-1077): doubles the DENSE MLP intermediate_size, but ONLY on kv-shared
  layers (`is_kv_shared_layer`). With `num_kv_shared_layers=0` this is INACTIVE for the 26B model
  (intermediate_size stays 2112).

---

## 6. Final logits  (`Gemma4ForCausalLM.forward`, 1889-1893)
lm_head is `nn.Linear(2816, 262144, bias=False)` (1830), **tied** to `embed_tokens.weight` (1820).
Softcapping applied AFTER lm_head, to the logits:
```
logits = lm_head(hidden_states)                      # 1889 (after model.norm)
if final_logit_softcapping is not None:              # = 30.0
    logits = logits / 30.0
    logits = tanh(logits)
    logits = logits * 30.0                            # 1891-1893  => 30*tanh(logits/30)
```
(The tied lm_head uses the RAW embedding table — it is NOT pre-scaled by the §1 embed_scale.)

---

## 7. Decoder-layer dataflow (exact)  (`Gemma4TextDecoderLayer.forward`, 1399-1456; [vllm]:701-762)

Same skeleton for sliding and full layers; only attention internals (head_dim, kv heads, k_eq_v, RoPE
table) differ. `gelu_tanh` = gelu_pytorch_tanh.

```
# --- Attention block (norm wraps sublayer, residual added AFTER post-norm) ---
residual = x
h = input_layernorm(x)
h = self_attn(h)                       # see §3 (sliding: hd256/kv8 ; full: hd512/kv2/k_eq_v)
h = post_attention_layernorm(h)
x = residual + h                       # 1422-1423

# --- Feed-forward block (DENSE ∥ MoE) ---
residual = x
dense = mlp( pre_feedforward_layernorm(x) )           # dense MLP, int=2112
if enable_moe_block:                                  # TRUE for 26B-A4B
    h1 = post_feedforward_layernorm_1(dense)
    moe = experts( pre_feedforward_layernorm_2(x),    # x == residual (pre-norm)
                   *router(x) )                        # router reads raw x
    h2 = post_feedforward_layernorm_2(moe)
    h  = h1 + h2
else:
    h  = dense
h = post_feedforward_layernorm(h)
x = residual + h                       # 1443-1444

# --- PLE injection (see §8); present because hidden_size_per_layer_input=256 ---
residual = x
g = gelu_tanh( per_layer_input_gate(x) )              # Linear 2816->256
g = g * per_layer_input                               # per-token PLE vector for THIS layer (256-d)
p = per_layer_projection(g)                           # Linear 256->2816
p = post_per_layer_input_norm(p)
x = residual + p                       # 1446-1453

# --- per-layer scalar ---
x = x * layer_scalar                   # buffer, shape (1,), init 1.0, loaded from ckpt (1382,1455)
return x
```
Full-attn vs sliding: identical control flow; full layers additionally use the 512-d heads / 2 kv heads /
k_eq_v V-sharing / proportional-RoPE table, sliding use 256-d / 8 kv / default-RoPE / windowed mask.

---

## 8. ⚠️ SURPRISES the task config did not mention — MUST implement

### 8a. Per-Layer Embeddings (PLE)  (1613-1631, 1675-1815, decoder 1446-1453)
A SECOND embedding table feeds a 256-d signal into EVERY layer. If `hidden_size_per_layer_input` (=256)
is nonzero this is ACTIVE. Pipeline:
```
# token-identity component (get_per_layer_inputs, 1738-1780):
ple_tok = embed_tokens_per_layer(input_ids)            # table [262144, 30*256], scale sqrt(256)=16.0
ple_tok = ple_tok.reshape(B, S, 30, 256)               # one 256-vec per layer
# context component (project_per_layer_inputs, 1782-1815):
proj = per_layer_model_projection(inputs_embeds) * (hidden_size**-0.5)   # Linear 2816->30*256, *1/sqrt(2816)
proj = proj.reshape(B, S, 30, 256)
proj = per_layer_projection_norm(proj)                 # RMSNorm(256), scaled
per_layer_inputs = (proj + ple_tok) * (2**-0.5)        # 1815  (1/sqrt(2) blend)
# layer i consumes per_layer_inputs[:, :, i, :] in the decoder PLE branch (§7).
```
`per_layer_input_scale = 2**-0.5` (1624); `per_layer_model_projection_scale = hidden_size**-0.5` (1630).
ACTION: confirm against the real checkpoint's `config.json` whether `hidden_size_per_layer_input` is 256
(PLE on) or 0 (off). The code path defaults to 256 (PLE ON). If on, the kernel MUST implement it.

### 8b. `layer_scalar` (1382,1455): every layer multiplies its output by a learned scalar buffer
(shape (1,), init 1.0). Applied AFTER the PLE branch. [vllm]:758-760 applies to all text layers.

### 8c. RMSNorm is plain `weight` (not `1+weight`) — §2. Different from Gemma2/Gemma3. Do not add 1.

### 8d. Attention has scaling=1.0 (no 1/sqrt(d)) and NO attn-logit softcap — §3d/§3e.

---

## Cross-check status
vLLM `gemma4.py` independently confirms: scaling=1.0 / no query_pre_attn_scalar ([vllm]:402-405);
k_eq_v V=K with v_norm + no RoPE on V ([vllm]:511-531); MoE softmax→topk→renorm→per_expert_scale
([vllm]:185-203); parallel dense+MoE combine and PLE+layer_scalar dataflow ([vllm]:726-760);
proportional partial_rotary per layer-type. No contradictions found.
NOT FOUND / needs check against checkpoint config.json: exact runtime value of
`hidden_size_per_layer_input` (PLE on/off) and `num_kv_shared_layers` (assumed 0).
```

---
## RESOLVED via loader gate + direct modeling read (2026-06-30, authoritative)

**weight_packed second dim is PACKED = in_features/2** (2 FP4/byte). So e.g. layer-5 full-attn
o_proj `[2816,4096]` ⇒ in_features 8192 = 16×512. No qk/v head_dim asymmetry — full-attn uses
head_dim 512 throughout.

**Per-layer-type attention geometry (verified vs checkpoint shapes):**
- SLIDING (25 layers, idx not in {5,11,17,23,29}): head_dim 256, 16 q-heads / 8 kv-heads.
  q_proj[4096,2816] k/v_proj[2048,2816] o_proj[2816,4096]. RoPE default θ1e4, full 256-dim rot. window 1024 causal.
- FULL (5 layers {5,11,17,23,29}): head_dim 512 (global_head_dim), 16 q / 2 kv heads. use_alternative_attention (k_eq_v).
  q_proj[8192,2816] k_proj[1024,2816] NO v_proj o_proj[2816,8192]. RoPE proportional θ1e6 partial_rotary 0.25.

**k_eq_v dataflow (full-attn, modeling §1258-1266):** kp = k_proj(x); V = v_norm(kp) [no RoPE, no weight];
K = RoPE(k_norm(kp)); Q = RoPE(q_norm(q_proj(x))). GQA repeat kv to 16 heads.

**Norms:** Gemma4RMSNorm = x*(mean(x²)+1e-6)^-0.5 *[weight if with_scale], fp32. q_norm/k_norm have weight
[head_dim]; v_norm with_scale=False (NO weight tensor, still normalizes). Decoder norms (input/post_attn/
pre_ff/post_ff) all have weight [2816].

**Attention scores:** scaling=1.0 (NO 1/sqrt(d)), softcap=None (NO attn-logit softcap). softmax in fp32.
Temperature is set entirely by learned q_norm/k_norm weights. THIS IS UNUSUAL — verify in 3.4 logit gate.

**Open for 3.2:** exact 'proportional' RoPE attention_scaling factor on cos/sin (see rope_proportional.txt).

---
## EXACT decoder-layer dataflow (modeling §1399-1456, all 30 layers enable_moe_block=true, PLE off)
```
residual = h
h = input_layernorm(h); h = self_attn(h); h = post_attention_layernorm(h); h = residual + h
residual = h                                  # FF-block input (pre any FF norm)
h_mlp   = mlp( pre_feedforward_layernorm(h) ) # DENSE MLP, intermediate 2112, gated gelu_tanh
hs1     = post_feedforward_layernorm_1(h_mlp)
# MoE path operates on `residual` (the PRE-FF-norm input!), with its OWN norms:
probs,topw,topi = router(residual)            # see router below
hs2     = experts( pre_feedforward_layernorm_2(residual), topi, topw )
hs2     = post_feedforward_layernorm_2(hs2)
h       = hs1 + hs2
h       = post_feedforward_layernorm(h); h = residual + h
h      *= layer_scalar                        # bf16 [1] per-layer scalar
```
Router (Gemma4TextRouter §1347): hn = router.norm(h)[no-weight RMSNorm]; hn = hn * router.scale[2816] * (2816**-0.5);
scores = hn @ router.proj.weight^T [128]; probs = softmax(scores); topw,topi = topk(probs,8);
topw /= topw.sum(); topw *= per_expert_scale[topi].
Experts (§1325): per token's expert e: gate=Wg@x, up=Wu@x; h=gelu_tanh(gate)*up; out=Wd@h; out*=topw; sum per token.
Embedding: token_emb * sqrt(2816)=53.066 (bf16). Final: norm(h) -> lm_head(tied embed, bf16) -> 30*tanh(logits/30).
Activation quant for every NVFP4 Linear uses that Linear's STORED input_global_scale (static), gscale=input_global_scale.
