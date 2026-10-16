#!/usr/bin/env python3
"""Smoke-test a local completion API and record single-request streaming latency."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time
from urllib.request import Request, urlopen


def request_json(base_url, path, payload=None):
    request = Request(base_url + path, data=None if payload is None else json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=180) as response:
        return json.load(response)


def stream_trial(base_url, model, prompt, tokens):
    payload = {"model": model, "prompt": prompt, "temperature": 0, "max_tokens": tokens,
               "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True},
               "logprobs": 0}
    request = Request(base_url + "/v1/completions", data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    arrivals = []
    chunks = []
    usage = None
    done = False
    with urlopen(request, timeout=180) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                done = True
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(event["error"])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                count = len((choice.get("logprobs") or {}).get("tokens", []))
                if count:
                    arrivals.append((time.perf_counter() - started, count))
                chunks.append(choice.get("text", ""))
    elapsed = time.perf_counter() - started
    counted = sum(count for _, count in arrivals)
    if not done or not usage or counted != usage["completion_tokens"] or counted != tokens:
        raise RuntimeError(f"Incomplete stream: done={done}, counted={counted}, usage={usage}")
    first, first_count = arrivals[0]
    last = arrivals[-1][0]
    if last <= first:
        raise RuntimeError("Need multiple timed chunks to measure decode throughput")
    return {"prompt": prompt, "usage": usage, "ttft_seconds": first,
            "total_seconds": elapsed, "decode_window_seconds": last - first,
            "decode_tokens_per_second": (counted - first_count) / (last - first),
            "stream_token_chunks": len(arrivals), "first_chunk_tokens": first_count,
            "text": "".join(chunks)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="specferry-tlm")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path(".cache/runs/tlm-opt/benchmark.json"))
    args = parser.parse_args()
    if args.tokens < 2:
        parser.error("--tokens must be at least 2")
    base_url = args.base_url.rstrip("/")
    with urlopen(base_url + "/health", timeout=10) as response:
        assert response.status == 200
    models = request_json(base_url, "/v1/models")
    if args.model not in {entry["id"] for entry in models["data"]}:
        raise RuntimeError(f"Requested model is absent: {models}")
    warmup = stream_trial(base_url, args.model, "A short explanation of how computers work:", 64)
    samples = []
    for prompt in ("The solar system consists of", "人工智能是一门"):
        samples.append(request_json(base_url, "/v1/completions",
                                   {"model": args.model, "prompt": prompt,
                                    "temperature": 0, "max_tokens": 96}))
    prompts = ["In a quiet village near the mountains, there lived a",
               "The main difference between a computer processor and memory is",
               "To make a simple vegetable soup, first prepare"]
    trials = []
    for prompt in prompts:
        trial = stream_trial(base_url, args.model, prompt, args.tokens)
        trials.append(trial)
        print(f"TTFT={trial['ttft_seconds']:.3f}s, decode={trial['decode_tokens_per_second']:.2f} tokens/s",
              flush=True)
    result = {"recorded_at": datetime.now(timezone.utc).isoformat(), "base_url": base_url,
              "model": args.model, "concurrency": 1, "temperature": 0,
              "benchmark_ignore_eos": True, "models": models, "warmup": warmup,
              "samples": samples, "trials": trials,
              "measurement": "Client-observed streaming decode: tokens after the first chunk divided "
                             "by the interval from first to last token chunk; includes HTTP scheduling. "
                             "TTFT includes API overhead and prefill, not isolated GPU prefill time.",
              "median_decode_tokens_per_second": statistics.median(
                  trial["decode_tokens_per_second"] for trial in trials),
              "median_ttft_seconds": statistics.median(trial["ttft_seconds"] for trial in trials)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
