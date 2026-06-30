import os
def main():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tk=AutoTokenizer.from_pretrained("/models/gemma-4-26B-A4B-it-NVFP4")
    prompt=tk.apply_chat_template([{"role":"user","content":"Write a Python function quicksort(arr) that sorts a list using the quicksort algorithm. Include a docstring and handle the empty list case."}],add_generation_prompt=True,tokenize=False)
    print("=== building engine ===",flush=True)
    llm=LLM(model="/models/gemma-4-26B-A4B-it-NVFP4",
        speculative_config={"method":"dflash","model":"/models/gemma-4-26B-A4B-DFlash","num_speculative_tokens":15},
        max_model_len=2048,gpu_memory_utilization=0.55,enforce_eager=True,trust_remote_code=True,disable_log_stats=False)
    o=llm.generate([prompt],SamplingParams(temperature=0,max_tokens=120))
    print("OUTPUT:",repr(o[0].outputs[0].text[:100]),flush=True)
    try:
        mets=llm.llm_engine.get_metrics(); acc=draft=0
        for m in mets:
            n=m.name
            if 'spec' in n or 'accept' in n or ('draft' in n):
                v=getattr(m,'value',None); print("METRIC",n,v,flush=True)
                if 'accepted' in n and v: acc=v
                if 'draft' in n and 'num' in n and v: draft=v
        if draft: print(f"ACCEPTANCE: {acc}/{draft} = {acc/draft:.3f}  mean_accept={15*acc/draft:.2f}/block",flush=True)
    except Exception as e: print("metric err",repr(e)[:200],flush=True)
    print("=== DONE ===",flush=True)
if __name__=="__main__":
    main()
