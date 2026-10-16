#!/usr/bin/env python3
"""Measure cached-chunk API cost, then simulate serial draft/verify throughput.

This is NOT speculative decoding: the completion endpoint projects logits for
only the last input position. Acceptance checks and KV rollback are absent.
Measured chunk time is an optimistic proxy, not a full verification benchmark.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import statistics
import time
from urllib.request import Request, urlopen
import uuid


def post(base, path, payload):
    request = Request(base + path, data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urlopen(request, timeout=120) as response:
        result = json.load(response)
    return result, time.perf_counter() - started


def counters(base):
    with urlopen(base + "/metrics", timeout=10) as response:
        lines = response.read().decode().splitlines()
    names = ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total")
    return {name: sum(float(line.rsplit(" ", 1)[1]) for line in lines
                      if line.startswith(name + "{")) for name in names}


def expected_output(k, acceptance):
    # One correction/bonus token plus the accepted draft prefix. A constant
    # conditional acceptance probability gives P(first i accepted) = a**i.
    return sum(acceptance ** i for i in range(k + 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path(".cache/runs/verify-proxy/results.json"))
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("--repeats must be at least 3")
    base = args.base_url.rstrip("/")
    salt = "specferry-verify-proxy-" + uuid.uuid4().hex
    tokenized, _ = post(base, "/tokenize", {"model": "specferry-tlm", "prompt":
        "A research team studies efficient language model inference. " * 100,
        "add_special_tokens": True})
    source = tokenized["tokens"]
    if len(source) < 545:
        raise RuntimeError("Tokenized fixture is too short")
    before = counters(base)
    expected_queries = expected_hits = 0
    serial = 1000
    rows = []
    for context in (128, 512):
        prefix = source[:context]

        def run(k):
            nonlocal serial, expected_queries, expected_hits
            serial += 1
            # Every request has a unique first uncached token. Replaying the
            # same suffix would cache the candidate block and invalidate timing.
            suffix = [serial] + source[context:context + k]
            payload = {"model": "specferry-tlm", "prompt": prefix + suffix,
                       "cache_salt": salt + str(context), "temperature": 0,
                       "max_tokens": 1, "ignore_eos": True}
            result, elapsed = post(base, "/v1/completions", payload)
            usage = result["usage"]
            if usage["prompt_tokens"] != context + k + 1 or usage["completion_tokens"] != 1:
                raise RuntimeError(f"Unexpected token accounting: {usage}")
            expected_queries += context + k + 1
            return elapsed

        run(0)  # Populate this experiment's isolated, block-aligned prefix.
        ks = (0, 1, 2, 4, 8, 16, 32)
        for k in ks:
            for _ in range(3):
                run(k)
                expected_hits += context
        samples = {k: [] for k in ks}
        order = list(ks) * args.repeats
        random.Random(20261015 + context).shuffle(order)
        for k in order:
            samples[k].append(run(k))
            expected_hits += context
        for k in ks:
            row = {"context_tokens": context, "draft_tokens": k,
                   "uncached_input_tokens": k + 1, "seconds": samples[k],
                   "median_seconds": statistics.median(samples[k]),
                   "min_seconds": min(samples[k]), "max_seconds": max(samples[k])}
            rows.append(row)
            print(f"context={context}, K={k}, new positions={k+1}, "
                  f"median={1000*row['median_seconds']:.2f} ms", flush=True)
    # Engine metrics are exported asynchronously; wait for this run's requests.
    deadline = time.monotonic() + 20
    while True:
        after = counters(base)
        observed_queries = after["vllm:prefix_cache_queries_total"] - before["vllm:prefix_cache_queries_total"]
        observed_hits = after["vllm:prefix_cache_hits_total"] - before["vllm:prefix_cache_hits_total"]
        if observed_queries >= expected_queries or time.monotonic() >= deadline:
            break
        time.sleep(1)
    if observed_queries != expected_queries or observed_hits != expected_hits:
        raise RuntimeError(f"Cache evidence mismatch; do not use timings: "
                           f"queries={observed_queries}/{expected_queries}, "
                           f"hits={observed_hits}/{expected_hits}")
    baseline = json.loads(Path('.cache/runs/tlm-opt/benchmark.json').read_text())[
        'median_decode_tokens_per_second']
    simulations = []
    for row in rows:
        k = row['draft_tokens']
        if not k:
            continue
        for rate in (2, 10, 20, 50, 100):
            for acceptance in (0.5, 0.8, 0.95, 1.0):
                length = expected_output(k, acceptance)
                cycle = k / rate + row['median_seconds']
                threshold_denominator = length / baseline - row['median_seconds']
                simulations.append({"context_tokens": row['context_tokens'], "draft_tokens": k,
                    "draft_tokens_per_second": rate, "conditional_acceptance": acceptance,
                    "expected_output_tokens": length, "draft_seconds": k / rate,
                    "verify_proxy_seconds": row['median_seconds'], "network_seconds": 0,
                    "estimated_tokens_per_second": length / cycle,
                    "speedup_vs_vllm_baseline": length / cycle / baseline,
                    "break_even_draft_tokens_per_second": k / threshold_denominator
                        if threshold_denominator > 0 else None})
    result = {"recorded_at": datetime.now(timezone.utc).isoformat(),
              "method": "Native vLLM cached-prefix + K+1 fresh positions + one completion token; "
                        "local HTTP wall time, warmed and shuffled, single request at a time.",
              "limitations": ["Only final-position logits; missing other verification LM-head outputs.",
                  "No draft distribution, acceptance/rejection, rollback, DLM catch-up or network RTT.",
                  "Fresh candidate suffixes are synthetic; acceptance rates are assumptions, not measured.",
                  "Cache reuse checked in aggregate against exact query/hit counts.",
                  "Baseline 15.4 tokens/s was measured on short contexts; context lengths differ."],
              "baseline_tokens_per_second": baseline, "rows": rows,
              "cache_evidence": {"expected_queries": expected_queries, "observed_queries": observed_queries,
                                 "expected_hits": expected_hits, "observed_hits": observed_hits},
              "simulation_formula": "sum(a**i for i in range(K+1)) / (K/draft_rate + verify_proxy_seconds + RTT)",
              "simulations": simulations}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(f"Saved validated cache-proxy measurements to {args.output}", flush=True)


if __name__ == '__main__':
    main()
