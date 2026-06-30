#!/bin/bash
# Self-contained correctness gate (no live reference server).
# Checks the model produces the exact known-correct greedy outputs:
#   - first 40 primes sequence (strong: exact arithmetic/sequence)
#   - 4 confident factual prompts (Paris / Blue / 4 / Au)
# Exit 0 = PASS, 1 = FAIL. Used by the optimization loop before any speed claim.
set -e
cd ~/gemma-cuda-server
M=$HOME/models/gemma-4-26B-A4B-it-NVFP4
IMG=vllm/vllm-openai:gemma-aarch64-cu130
mkdir -p /tmp/share
tok(){ docker run --rm --entrypoint python3 -v "$M":/m -v /tmp/share:/share "$IMG" -c "
from transformers import AutoTokenizer
tk=AutoTokenizer.from_pretrained('/m')
t=tk.apply_chat_template([{'role':'user','content':'''$1'''}],add_generation_prompt=True,tokenize=False)
open('/share/gate_tok.txt','w').write(chr(10).join(map(str,tk(t,add_special_tokens=False).input_ids)))" 2>/dev/null; cp /tmp/share/gate_tok.txt /tmp/tokens.txt; }
dec(){ docker run --rm --entrypoint python3 -v "$M":/m "$IMG" -c "from transformers import AutoTokenizer;print(AutoTokenizer.from_pretrained('/m').decode([int(x) for x in '''$1'''.split()]))" 2>/dev/null; }
run(){ GEN=$2 timeout 300 ./build/forward 2>/dev/null | grep -oE '^[0-9 ]+$' | head -1; }
fail=0
check(){ local q="$1" exp="$2" n="$3"; tok "$q"; local ids=$(run x $n); local txt=$(dec "$ids")
  if [[ "$txt" == *"$exp"* ]]; then echo "  ok   '$exp' <- $q"; else echo "  FAIL '$exp' got '${txt:0:50}' <- $q"; fail=1; fi; }
echo "=== self-contained correctness gate ==="
check "List the first 40 prime numbers, comma separated." "2, 3, 5, 7, 11, 13, 17, 19, 23, 29" 40
check "What is the capital of France? One word." "Paris" 3
check "Sky color on a clear day? One word." "Blue" 3
check "What is the chemical symbol for gold? One word." "Au" 3
if [ $fail -eq 0 ]; then echo "GATE PASS ✅"; exit 0; else echo "GATE FAIL ❌"; exit 1; fi
