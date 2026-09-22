#!/usr/bin/env python3
"""Check categorical sampling, seed replay, frequency and the final vocabulary token."""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from specferry.validation.device import device_lock, fingerprint, run_device, write_json
from specferry.validation.sampling import evaluate

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("F16", "F32", "F16_TO_F32"), default="F16_TO_F32")
    parser.add_argument("--classes", type=int, choices=(8, 50272), default=50272)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not 1024 <= args.samples <= 8192:
        parser.error("use a fresh output directory and 1024..8192 samples for frequency checks")
    output = args.output.resolve()
    output.mkdir(parents=True)
    try:
        with device_lock(ROOT / ".cache/runs"):
            binary = output / "np101_sampling_check"
            shutil.copy2(ROOT / "build/tests/np101_sampling_check", binary)
            write_json(output / "binary.json", {"sha256": fingerprint(binary)})
            evidence = run_device(
                binary,
                [str(output / "device"), args.dtype, str(args.classes), str(args.samples)],
                output / "device",
                Path("/usr/lib/ljmicro"),
                120,
                shader_header=Path("/usr/inc/CL/cl_viv_vx_ext.h"),
            )
        report = evaluate(output / "device", args.dtype, args.classes, args.samples, evidence)
        write_json(output / "validation.json", report)
        print(f"{report['status']}: {report['failures']}; {output / 'validation.json'}")
        print("Numerical SDK validation; exclusive NPU execution remains unproven.")
        return 0 if report["status"] == "numerical_pass" else 1
    except (OSError, ValueError, RuntimeError) as error:
        write_json(output / "failure.json", {"status": "failed", "error": str(error)})
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
