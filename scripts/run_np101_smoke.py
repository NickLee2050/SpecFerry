#!/usr/bin/env python3
"""Run the NP101 SDK smoke test with a timeout, library fingerprints, SDK target output and driver trace."""

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("build/np101_smoke"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sdk-lib", type=Path, default=Path("/usr/lib/ljmicro"))
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 1000 or args.timeout < 1:
        parser.error("invalid repeats or timeout")
    binary, output = args.binary.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "smoke.json").unlink(missing_ok=True)
    (output / "smoke.strace").unlink(missing_ok=True)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(args.sdk_lib.resolve())
    env["VIV_VX_ENABLE_PRINT_TARGET"] = "1"
    command = [str(binary), "--output", str(output / "smoke.json"), "--repeats", str(args.repeats)]
    tracer = shutil.which("strace")
    if tracer:
        command = [
            tracer,
            "-f",
            "-tt",
            "-T",
            "-yy",
            "-e",
            "trace=openat,close,ioctl,mmap",
            "-o",
            str(output / "smoke.strace"),
            *command,
        ]
    evidence = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "binary_elf": subprocess.run(
            ["readelf", "-h", str(binary)], capture_output=True, text=True
        ).stdout,
        "linked_libraries": subprocess.run(
            ["ldd", str(binary)], env=env, capture_output=True, text=True
        ).stdout,
        "sdk_library_dir": str(args.sdk_lib.resolve()),
        "print_target": True,
        "timeout": False,
    }
    with (output / "smoke.log").open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=output,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            evidence["timeout"] = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
    evidence["returncode"] = code
    trace = (output / "smoke.strace").read_text() if tracer else ""
    successes = re.findall(r"ioctl\([^\n]*</dev/galcore[^\n]*= 0", trace)
    evidence["successful_galcore_ioctls"] = len(successes)
    evidence["driver_activity_observed"] = bool(successes)
    evidence["node_execution_verified"] = False
    evidence["hardware_acceptance"] = "requires_review_of_target_log_and_command_completion"
    evidence["note"] = (
        "Generic device ioctl success proves driver interaction, not individual node execution or absence of CPU/software paths."
    )
    (output / "smoke-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(
        json.dumps(
            {
                "returncode": code,
                "driver_ioctls": len(successes),
                "hardware_acceptance": evidence["hardware_acceptance"],
                "output": str(output),
            }
        )
    )
    return 1 if code else 0


if __name__ == "__main__":
    raise SystemExit(main())
