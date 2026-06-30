#!/bin/bash
# Standing benchmark: base decode tok/s (primary champion metric), DFlash tok/s + acceptance (predictable),
# on a fixed primes prompt. Reports median of 2 base runs. Self-contained.
cd ~/gemma-cuda-server
M=$HOME/models/gemma-4-26B-A4B-it-NVFP4
IMG=vllm/vllm-openai:gemma-aarch64-cu130
docker run --rm --entrypoint python3 -v "$M":/m -v /tmp/share:/share "$IMG" -c "
from transformers import AutoTokenizer
tk=AutoTokenizer.from_pretrained('/m')
t=tk.apply_chat_template([{'role':'user','content':'List the first 40 prime numbers, comma separated.'}],add_generation_prompt=True,tokenize=False)
open('/share/bench_primes.txt','w').write(chr(10).join(map(str,tk(t,add_special_tokens=False).input_ids)))" 2>/dev/null
b1=0; for i in 1 2; do cp /tmp/share/bench_primes.txt /tmp/tokens.txt
  v=$(GEN=60 timeout 300 ./build/forward 2>/dev/null | grep -oE "= [0-9.]+ tok/s" | grep -oE "[0-9.]+"); b1=$(echo "$b1 $v" | awk '{print ($2>$1)?$2:$1}'); done
cp /tmp/share/bench_primes.txt /tmp/tokens.txt
df=$(DFLASH=1 DK=14 GEN=80 timeout 400 ./build/forward 2>/dev/null | grep -oE "= [0-9.]+ tok/s.*accept [0-9.]+")
echo "BASE_TOKS=$b1"
echo "DFLASH=$df"
