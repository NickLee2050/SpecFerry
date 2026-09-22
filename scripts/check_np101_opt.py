#!/usr/bin/env python3
"""Check an OPT decoder slice against independent official CPU trajectories."""

import argparse
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from specferry.models.opt.validation import evaluate, prepare
from specferry.validation.device import device_lock, fingerprint, run_device, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / ".cache/np101/opt-350m")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[0])
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--capacity", type=int, default=512)
    parser.add_argument("--prompt", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="allow numerical pass without full hardware evidence",
    )
    args = parser.parse_args()
    if args.output.exists() or args.timeout < 1:
        parser.error("use a new output directory and positive timeout")
    torch.set_num_threads(4)
    output = args.output.resolve()
    output.mkdir(parents=True)
    try:
        print("Preparing independent OPT reference ...", flush=True)
        metadata = prepare(
            args.model.resolve(),
            args.trace.resolve(),
            output / "fixture",
            args.layers,
            args.steps,
            args.capacity,
            args.prompt,
        )
        if args.prepare_only:
            print(f"Prepared OPT slice: {output}")
            return 0
        with device_lock(ROOT / ".cache/runs"):
            binary = output / "np101_opt_decoder_check"
            shutil.copy2(ROOT / "build/tests/np101_opt_decoder_check", binary)
            write_json(
                output / "sources.json",
                {
                    "binary_sha256": fingerprint(binary),
                    "sources": {
                        str(path.relative_to(ROOT)): fingerprint(path)
                        for directory in ("native", "python", "scripts", "tests")
                        for path in sorted((ROOT / directory).rglob("*"))
                        if path.is_file() and path.suffix in (".py", ".cpp", ".hpp")
                    },
                },
            )
            print("Running OPT decoder slice ...", flush=True)
            evidence = run_device(
                binary,
                [
                    str(args.model.resolve()),
                    str(output / "fixture"),
                    str(output / "device"),
                    str(args.steps),
                ],
                output / "device",
                Path("/usr/lib/ljmicro"),
                args.timeout,
                shader_header=Path("/usr/inc/CL/cl_viv_vx_ext.h"),
            )
        report = evaluate(output / "fixture", output / "device", metadata, evidence)
        write_json(output / "opt.json", report)
        if evidence["returncode"] != 0 or evidence["device_recovery_required"]:
            print(
                f"SDK process failed (exit {evidence['returncode']}); "
                f"phase={report['execution'].get('phase', 'unknown')}, "
                f"completed steps={report['execution'].get('steps', 0)}. "
                f"See {output / 'device'}",
                file=sys.stderr,
            )
            return 1
        failures = [name for name, check in report["checks"].items() if not check["passed"]]
        print(f"{report['status']}: {len(failures)} failed comparisons; {output / 'opt.json'}")
        if report["status"] != "numerical_pass":
            print("First failures:", failures[:8])
            return 1
        return 0 if args.diagnostic else 2
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        write_json(output / "failure.json", {"status": "failed", "error": str(error)})
        print(f"OPT validation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
