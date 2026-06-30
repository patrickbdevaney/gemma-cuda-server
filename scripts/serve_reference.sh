#!/bin/bash
# serve_reference.sh — Phase 1 ground-truth reference server for gemma4-cuda-server.
#
# Serves RedHatAI/gemma-4-26B-A4B-it-NVFP4 (target) + z-lab DFlash draft in the
# proven Thor DFlash vLLM container, to capture logits / accept-reject / timing
# traces that every custom kernel is later diffed against.
#
# Flags distilled from ~/dflash-setup/scripts/_common.sh (battle-tested on Thor SM110)
# and reference/ARCHITECTURE.md. Deviations from the project directive, with reasons:
#
#   ⚠️ KV cache = auto (BF16), NOT fp8.
#      DFlash REJECTS a quantized KV cache in this vLLM fork (issue #41559). The
#      directive's "fp8 KV + DFlash" pairing is impossible in the reference stack.
#      The custom server (Phase 3+) is free to do fp8 KV since it isn't bound by
#      vLLM's limitation — but its DFlash ground-truth trace is captured at BF16 KV.
#      (A separate fp8-KV-WITHOUT-DFlash run can ground-truth the attention kernel's
#      fp8 path independently.)
#
#   • MoE backend = cutlass. marlin FP4-MoE faults at large expert counts on Thor
#     (122B/256-expert crash); 128 experts is unproven on marlin, so use the
#     known-safe cutlass path. VLLM_USE_FLASHINFER_MOE_FP4=0 mandatory (no Thor kernel).
#   • attention-backend = TRITON_ATTN. flashinfer paged-decode has a kv_cache_sf API
#     mismatch in this fork; also supports the non-causal DFlash drafter.
#   • --language-model-only: checkpoint arch is Gemma4ForConditionalGeneration (MM
#     wrapper); we serve text only.
#   • No reasoning/tool parser on this first bring-up: trace capture uses raw greedy
#     /v1/completions (parsers only reshape chat output, and --reasoning-parser gemma4
#     may not be registered in this dflash container — adding it risks a startup crash).
#     Add `--reasoning-parser gemma4 --tool-call-parser gemma4 --enable-auto-tool-choice`
#     once confirmed registered, for the agentic-format prompts.
#   • No --enable-prefix-caching (DFlash + hybrid-attn bug #40624).
#   • Stop with `docker kill`, NEVER `docker stop` (Thor page-cache leak).
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-$HOME/models/gemma-4-26B-A4B-it-NVFP4}"
DRAFT_DIR="${DRAFT_DIR:-$HOME/models/gemma-4-26B-A4B-DFlash}"
MODEL_NAME="${MODEL_NAME:-gemma4-26b-nvfp4-ref}"
# Image: gemma-aarch64 ships vllm 0.22.1rc1 (HAS dflash) + transformers 5.10.2 (full gemma4)
# + gemma4_unified.py. The dflash fork (vllm-dflash-thor:ddtree) has an older transformers
# that does NOT register the gemma4 config (load fails: "Transformers does not recognize gemma4").
# NOTE: this image's ENTRYPOINT is ["vllm","serve"] → pass `/model <flags>`, NOT `vllm serve /model`.
IMAGE="${IMAGE:-vllm/vllm-openai:gemma-aarch64-cu130}"
PORT="${PORT:-8000}"
GPU_UTIL="${GPU_UTIL:-0.60}"          # 16.4GB weights + ~1GB draft + tiny KV → lots of headroom
MAX_LEN="${MAX_LEN:-65536}"           # directive's 64K context budget
MAX_SEQS="${MAX_SEQS:-1}"             # single-session
MAX_BATCHED="${MAX_BATCHED:-8192}"
NUM_SPEC="${NUM_SPEC:-15}"            # draft block_size=16 → k=16; cudagraph sizes multiples of 16
DRAFT_MAX_LEN="${DRAFT_MAX_LEN:-2048}" # draft sliding_window is 2048; cap its KV
MOE_BACKEND="${MOE_BACKEND:-cutlass}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-TRITON_ATTN}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"   # set 1 if CUDA-graph capture crashes after load
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-900}"

[ -f "$MODEL_DIR/config.json" ] || { echo "ERROR: target config missing: $MODEL_DIR"; exit 1; }
[ -f "$DRAFT_DIR/config.json" ] || { echo "ERROR: draft config missing: $DRAFT_DIR"; exit 1; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "ERROR: image not found: $IMAGE"; exit 1; }

EAGER_ARG=""; [ "$ENFORCE_EAGER" = "1" ] && EAGER_ARG="--enforce-eager"
SPEC_CONFIG="{\"method\":\"dflash\",\"num_speculative_tokens\":${NUM_SPEC},\"model\":\"/drafter\",\"max_model_len\":${DRAFT_MAX_LEN}}"

echo "=================================================================="
echo " Phase-1 reference: $MODEL_NAME"
echo " target=$MODEL_DIR  draft=$DRAFT_DIR (DFlash k=$((NUM_SPEC+1)))"
echo " image=$IMAGE port=$PORT gpu_util=$GPU_UTIL max_len=$MAX_LEN"
echo " moe=$MOE_BACKEND attn=$ATTENTION_BACKEND kv=auto(BF16, DFlash#41559)"
echo "=================================================================="

# kill any prior server on this port (docker kill, never docker stop)
for c in $(docker ps -q --filter "publish=${PORT}" 2>/dev/null); do docker kill "$c" >/dev/null 2>&1 || true; done
sudo nvpmodel -m 1  >/dev/null 2>&1 || true
sudo jetson_clocks  >/dev/null 2>&1 || true
mkdir -p "$HOME/thor-vllm-cache/vllm-dflash" "$HOME/thor-vllm-cache/flashinfer"

CONTAINER="gemma4-ref-$(date +%s)"
set -x
docker run -d --name "$CONTAINER" \
  --runtime nvidia --gpus all --ipc=host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=16g \
  -e HF_HUB_DISABLE_XET=1 \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -v "${MODEL_DIR}:/model:ro" \
  -v "${DRAFT_DIR}:/drafter:ro" \
  -v "$HOME/thor-vllm-cache/vllm-dflash:/root/.cache/vllm" \
  -v "$HOME/thor-vllm-cache/flashinfer:/root/.cache/flashinfer" \
  "$IMAGE" \
  /model \
    --served-model-name "$MODEL_NAME" \
    --speculative-config "$SPEC_CONFIG" \
    --quantization compressed-tensors \
    --kv-cache-dtype auto \
    --attention-backend "$ATTENTION_BACKEND" \
    --moe-backend "$MOE_BACKEND" \
    $EAGER_ARG \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAX_LEN" \
    --max-num-seqs "$MAX_SEQS" \
    --max-num-batched-tokens "$MAX_BATCHED" \
    --enable-chunked-prefill \
    --trust-remote-code \
    --language-model-only \
    --port "$PORT"
set +x
echo "$CONTAINER" > "$HOME/thor-vllm-cache/current-gemma4-ref-container"

echo "waiting for http://localhost:${PORT}/health (timeout ${HEALTH_TIMEOUT}s)..."
deadline=$((SECONDS + HEALTH_TIMEOUT))
while true; do
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:${PORT}/health" 2>/dev/null || echo 000)
  [ "$code" = "200" ] && { echo; echo ">>> READY: $MODEL_NAME on :$PORT (container $CONTAINER)"; exit 0; }
  if ! docker ps -q --filter "name=$CONTAINER" | grep -q .; then
    echo; echo "!!! container exited before healthy. Last 60 log lines:"; docker logs "$CONTAINER" 2>&1 | tail -60; exit 1
  fi
  [ $SECONDS -ge $deadline ] && { echo; echo "!!! TIMEOUT. Last 40 log lines:"; docker logs "$CONTAINER" 2>&1 | tail -40; exit 1; }
  sleep 4
done
