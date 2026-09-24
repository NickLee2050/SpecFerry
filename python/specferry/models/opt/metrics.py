"""Host wall-time metrics; initialization and warmups are excluded."""

import math
from statistics import mean


def summarize(runs):
    if not runs:
        raise ValueError("no measured generations")
    intervals = []
    for run in runs:
        first, total = run["first_token_seconds"], run["total_seconds"]
        times = run["token_seconds"]
        if (
            len(times) != max(0, len(run["tokens"]) - 1)
            or any(
                not math.isfinite(t) or t < 0
                for t in [first, total, run["prefill_seconds"], *times]
            )
            or any(t <= 0 for t in times)
            or not run["prefill_seconds"] <= first <= total
            or first + sum(times) > total + 1e-6
        ):
            raise ValueError("inconsistent generation timings")
        intervals.extend(times)
    return {
        "ttft_seconds": mean(run["first_token_seconds"] for run in runs),
        "prefill_seconds": mean(run["prefill_seconds"] for run in runs),
        "decode_tokens_per_second": len(intervals) / sum(intervals) if intervals else None,
        "decode_intervals": len(intervals),
        "request_seconds": mean(run["total_seconds"] for run in runs),
    }
