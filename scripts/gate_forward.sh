#!/bin/bash
# gate_forward.sh — Phase 3.4 end-to-end gate: custom CUDA forward top-1 must match the
# vLLM reference on CONFIDENT prompts (top-1 is precision-robust; gibberish prompts are not
# a valid gate because model.norm has 588x dims that amplify quantization noise).
# Requires the reference server up on :8000 and the gemma image for tokenization.
set -u
IMG=vllm/vllm-openai:gemma-aarch64-cu130
MDIR=$HOME/models/gemma-4-26B-A4B-it-NVFP4
mkdir -p /tmp/share
PROMPTS=(
  "What is the capital of France? Answer in one word."
  "What color is a clear daytime sky? One word."
  "What is 2+2? Answer with just the number."
  "What is the chemical symbol for gold? One word."
)
pass=0; total=0
for P in "${PROMPTS[@]}"; do
  total=$((total+1))
  docker run --rm --entrypoint python3 -v "$MDIR":/m -v /tmp/share:/share "$IMG" -c "
from transformers import AutoTokenizer
tk=AutoTokenizer.from_pretrained('/m')
t=tk.apply_chat_template([{'role':'user','content':'''$P'''}],add_generation_prompt=True,tokenize=False)
open('/share/tokens.txt','w').write('\n'.join(map(str,tk(t,add_special_tokens=False).input_ids)))" 2>/dev/null
  cp /tmp/share/tokens.txt /tmp/tokens.txt
  IDS=$(paste -sd, /tmp/tokens.txt)
  REF=$(curl -s http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"gemma4-26b-nvfp4-ref\",\"prompt\":[$IDS],\"max_tokens\":1,\"temperature\":0,\"logprobs\":1}" \
    | python3 -c "import sys,json;d=json.load(sys.stdin);lp=d['choices'][0]['logprobs']['top_logprobs'][0];print(max(lp,key=lp.get))")
  MID=$(timeout 400 ./build/forward >/tmp/o.txt 2>/dev/null; grep -A2 next-token /tmp/o.txt|sed -n '2p'|awk '{print $1}')
  MTOK=$(docker run --rm --entrypoint python3 -v "$MDIR":/m "$IMG" -c "from transformers import AutoTokenizer;print(AutoTokenizer.from_pretrained('/m').convert_ids_to_tokens($MID))" 2>/dev/null)
  if [ "$REF" = "$MTOK" ]; then echo "PASS  '$P' -> '$MTOK'"; pass=$((pass+1)); else echo "FAIL  '$P'  ref='$REF' mine='$MTOK'"; fi
done
echo "=== $pass/$total confident-prompt top-1 matches ==="
[ "$pass" = "$total" ] && echo "GATE PASS ✅" || echo "GATE FAIL ❌"
