"""Check SDK categorical samples independently of its random-number algorithm."""

import json

import numpy as np


def evaluate(directory, dtype, classes, count, evidence):
    """Use six-sigma marginal frequency bounds; do not expect CPU RNG identity."""
    checks = {}
    arrays = {}
    names = ("uniform.0", "uniform.1", "uniform.2", "counter", "weighted", "last", "fresh")
    for name in names:
        path = directory / f"{name}.bin"
        valid = path.is_file() and path.stat().st_size == count * 4
        values = np.fromfile(path, dtype="<i4") if valid else np.array([], dtype="<i4")
        valid = valid and bool(np.all((values >= 0) & (values < classes)))
        arrays[name] = values
        checks[f"{name}.range"] = {"passed": valid}

    def equal(left, right):
        return bool(np.array_equal(arrays[left], arrays[right]))

    checks["repeat"] = {"passed": equal("uniform.0", "uniform.2")}
    checks["fresh"] = {"passed": equal("uniform.0", "fresh")}
    checks["changed_seed"] = {"passed": not equal("uniform.0", "uniform.1")}
    checks["changed_counter"] = {"passed": not equal("uniform.0", "counter")}
    checks["last_token"] = {
        "passed": arrays["last"].size == count and bool(np.all(arrays["last"] == classes - 1))
    }

    def frequencies(name, values, probabilities):
        observed = np.bincount(values, minlength=8)[:8] / count
        limit = 6 * np.sqrt(probabilities * (1 - probabilities) / count) + 2 / count
        checks[name] = {
            "passed": bool(np.all(np.abs(observed - probabilities) <= limit)),
            "observed": observed.tolist(),
            "expected": probabilities.tolist(),
            "absolute_limits": limit.tolist(),
        }

    if checks["uniform.0.range"]["passed"]:
        # Eight equal-size buckets include the tail of the full OPT vocabulary.
        frequencies("uniform_frequency", arrays["uniform.0"] * 8 // classes, np.full(8, 0.125))
    if checks["weighted.range"]["passed"]:
        logits = np.log(np.arange(1, 9, dtype=np.float32))
        if dtype != "F32":
            logits = logits.astype(np.float16).astype(np.float32)
        probabilities = np.exp(logits.astype(np.float64))
        probabilities /= probabilities.sum()
        frequencies("weighted_frequency", arrays["weighted"], probabilities)
        checks["suppressed_tokens"] = {"passed": bool(np.all(arrays["weighted"] < 8))}

    path = directory / "execution.json"
    execution = json.loads(path.read_text()) if path.is_file() else {}
    checks["lifecycle"] = {
        "passed": (
            evidence.get("returncode") == 0
            and evidence.get("process_group_exited") is True
            and evidence.get("device_recovery_required") is False
            and execution.get("status") == "executed"
            and execution.get("released") is True
        )
    }
    failed = [name for name, check in checks.items() if not check["passed"]]
    return {
        "status": "failed" if failed else "numerical_pass",
        "dtype": dtype,
        "classes": classes,
        "samples_per_case": count,
        "checks": checks,
        "failures": failed,
        "evidence": evidence,
        "hardware_execution_proven": False,
    }
