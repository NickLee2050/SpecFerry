"""Shared command options and result handling for native compute checks."""

import argparse
from pathlib import Path

from .device import RECOVERY_ROOT, clean_execution, device_lock, run_device, write_json


def positive_seconds(value):
    seconds = int(value)
    if seconds <= 0:
        raise argparse.ArgumentTypeError("timeout must be positive")
    return seconds


def add_run_arguments(parser):
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=positive_seconds, help="process deadline in seconds")
    parser.add_argument("--prepare-only", action="store_true", help="do not open NP101")
    parser.add_argument(
        "--sdk-timing", action="store_true", help="record public SDK call durations"
    )
    parser.add_argument(
        "--trace-driver", action="store_true", help="record ioctl calls with strace"
    )


def run_check(binary, arguments, output, options, *, evaluate=None, timeout=None):
    with device_lock(RECOVERY_ROOT):
        evidence = run_device(
            binary,
            arguments,
            output / "device",
            Path("/usr/lib/ljmicro"),
            timeout
            if timeout is not None
            else (120 if options.timeout is None else options.timeout),
            sdk_timing=options.sdk_timing,
            trace_driver=options.trace_driver,
        )
    report = {
        "status": "numerical_pass" if clean_execution(evidence) else "failed",
        "evidence": evidence,
    }
    if evaluate is not None and clean_execution(evidence):
        report = evaluate(evidence)
    write_json(output / "result.json", report)
    print(f"{report['status']}; result: {output / 'result.json'}")
    return int(report["status"] != "numerical_pass")
