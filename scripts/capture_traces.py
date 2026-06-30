#!/usr/bin/env python3
"""capture_traces.py — Phase-1 ground-truth trace capture for gemma4-cuda-server.

Hits the reference vLLM server (NVFP4 target + DFlash, BF16 KV) over its
OpenAI-compatible chat API at temperature=0 (greedy) and stores, per prompt:
  - the generated text + per-token top-k logprobs (the logit ground truth)
  - DFlash accept/reject signal: /metrics counter deltas around the request
    (drafts, draft tokens, accepted tokens, per-position acceptance)
  - timing: total wall-clock, TTFT (streamed), completion tokens, decode tok/s

One JSON per prompt under traces/<id>.json, plus a summary.json.

These traces are the reference every custom kernel (Phase 3+) is diffed against.
Greedy + temp 0 makes them deterministic and comparable.

Usage:
  python3 scripts/capture_traces.py --base http://localhost:8000 \
      --model gemma4-26b-nvfp4-ref --suite scripts/prompt_suite.json --out traces
"""
import argparse, json, os, time, urllib.request, urllib.error, re

def http_post(url, payload, stream=False, timeout=600):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)

def get_metrics(base):
    """Return dict of spec_decode counters (floats) keyed by a short name."""
    try:
        raw = urllib.request.urlopen(base + "/metrics", timeout=30).read().decode()
    except Exception:
        return {}
    m = {}
    for line in raw.splitlines():
        if line.startswith("#") or "spec_decode" not in line:
            continue
        # vllm:spec_decode_num_accepted_tokens_total{...} 26.0
        mobj = re.match(r"(vllm:spec_decode_[a-z_]+)(\{[^}]*\})?\s+([0-9.eE+-]+)", line)
        if not mobj:
            continue
        name, labels, val = mobj.group(1), mobj.group(2) or "", float(mobj.group(3))
        if "_created" in name:  # skip unix-timestamp gauges
            continue
        pos = ""
        pm = re.search(r'position="(\d+)"', labels)
        if pm:
            pos = "_pos" + pm.group(1)
        m[name.replace("vllm:spec_decode_", "") + pos] = m.get(name.replace("vllm:spec_decode_", "") + pos, 0.0) + val
    return m

def metrics_delta(before, after):
    keys = set(before) | set(after)
    d = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in keys}
    # keep only changed counters
    d = {k: v for k, v in d.items() if abs(v) > 1e-9}
    drafts = d.get("num_drafts_total", 0.0)
    draft_toks = d.get("num_draft_tokens_total", 0.0)
    accepted = d.get("num_accepted_tokens_total", 0.0)
    summary = {
        "drafts": drafts,
        "draft_tokens": draft_toks,
        "accepted_tokens": accepted,
        "mean_accepted_per_draft": (accepted / drafts) if drafts else None,
        "draft_acceptance_rate": (accepted / draft_toks) if draft_toks else None,
        "per_position": {k: v for k, v in sorted(d.items()) if "_pos" in k},
    }
    return summary

def build_messages(p):
    if p.get("_build") == "needle":
        filler = ("This is filler context line for retrieval testing. " * 4 + "\n") * p["filler_repeats"]
        content = (filler[: len(filler) // 2] + "\n" + p["needle"] + "\n" + filler[len(filler) // 2:]
                   + "\n\nQuestion: " + p["question"])
        return [{"role": "user", "content": content}]
    return p["messages"]

def stream_ttft(base, model, messages, max_tokens):
    """One streamed greedy call → (ttft_s, total_s, n_chunks)."""
    payload = {"model": model, "messages": messages, "temperature": 0,
               "max_tokens": max_tokens, "stream": True}
    t0 = time.perf_counter(); ttft = None; n = 0
    try:
        resp = http_post(base + "/v1/chat/completions", payload, stream=True)
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                obj = json.loads(body)
            except Exception:
                continue
            delta = obj.get("choices", [{}])[0].get("delta", {})
            if delta.get("content"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                n += 1
    except Exception:
        pass
    return ttft, time.perf_counter() - t0, n

def capture_one(base, model, p, outdir):
    messages = build_messages(p)
    max_tokens = p.get("max_tokens", 128)

    before = get_metrics(base)
    payload = {"model": model, "messages": messages, "temperature": 0,
               "max_tokens": max_tokens, "logprobs": True, "top_logprobs": 5}
    t0 = time.perf_counter()
    resp = http_post(base + "/v1/chat/completions", payload)
    body = json.loads(resp.read().decode())
    wall = time.perf_counter() - t0
    after = get_metrics(base)

    ch = body["choices"][0]
    content = ch["message"]["content"]
    lp = (ch.get("logprobs") or {}).get("content", []) or []
    tokens = [{
        "token": t["token"],
        "logprob": t["logprob"],
        "top": [{"token": x["token"], "logprob": x["logprob"]} for x in t.get("top_logprobs", [])],
    } for t in lp]
    usage = body.get("usage", {})
    comp = usage.get("completion_tokens") or len(tokens)

    # separate streamed timing pass for TTFT (greedy → identical text)
    ttft, stream_wall, _ = stream_ttft(base, model, messages, max_tokens)

    rec = {
        "id": p["id"], "category": p.get("category"),
        "messages": messages if p.get("_build") != "needle" else "[needle-built, omitted]",
        "max_tokens": max_tokens,
        "content": content,
        "n_completion_tokens": comp,
        "tokens": tokens,
        "usage": usage,
        "timing": {
            "wall_s": round(wall, 4),
            "ttft_s": round(ttft, 4) if ttft else None,
            "stream_wall_s": round(stream_wall, 4),
            "decode_tok_s": round(comp / wall, 2) if wall > 0 else None,
        },
        "dflash": metrics_delta(before, after),
    }
    with open(os.path.join(outdir, p["id"] + ".json"), "w") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    return rec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--model", default="gemma4-26b-nvfp4-ref")
    ap.add_argument("--suite", default="scripts/prompt_suite.json")
    ap.add_argument("--out", default="traces")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    suite = json.load(open(args.suite))["prompts"]
    summary = []
    for p in suite:
        try:
            rec = capture_one(args.base, args.model, p, args.out)
            d = rec["dflash"]; tm = rec["timing"]
            print(f"[{rec['id']:14s}] {rec['category']:14s} toks={rec['n_completion_tokens']:4d} "
                  f"tok/s={tm['decode_tok_s']} ttft={tm['ttft_s']} "
                  f"tau={d.get('mean_accepted_per_draft')} :: {rec['content'][:50]!r}")
            summary.append({"id": rec["id"], "category": rec["category"],
                            "n_tokens": rec["n_completion_tokens"], **tm,
                            "mean_accepted_per_draft": d.get("mean_accepted_per_draft"),
                            "draft_acceptance_rate": d.get("draft_acceptance_rate")})
        except urllib.error.HTTPError as e:
            print(f"[{p['id']}] HTTP {e.code}: {e.read().decode()[:200]}")
        except Exception as e:
            print(f"[{p['id']}] ERROR: {e}")
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if summary:
        ok = [s for s in summary if s["decode_tok_s"]]
        avg_tps = sum(s["decode_tok_s"] for s in ok) / len(ok)
        taus = [s["mean_accepted_per_draft"] for s in summary if s["mean_accepted_per_draft"]]
        print(f"\n=== {len(summary)} prompts | avg decode {avg_tps:.1f} tok/s | "
              f"mean tau {sum(taus)/len(taus):.2f}" if taus else f"\n=== {len(summary)} prompts | avg decode {avg_tps:.1f} tok/s")

if __name__ == "__main__":
    main()
