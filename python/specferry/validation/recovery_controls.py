"""Kernel health checks and an explicit vendor diagnostic; never clear the recovery gate."""

import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

from specferry.data.checkpoint import sha256

from .device import (
    RECOVERY_ROOT,
    clean_execution,
    device_lock,
    host_boot_id,
    run_device,
    write_json,
)

ROOT = RECOVERY_ROOT.parents[1]
SDK = Path("/usr/lib/ljmicro")
FAULT = re.compile(
    r"NP.*hang|EDMA.*timeout|\bBUG:|\bOops:|unable to handle|general protection fault|"
    r"invalid opcode|soft lockup|Hardware Error",
    re.I,
)


def vendor_control():
    """Reject changed/missing controls before opening the device; never silently rebuild."""
    pinned = json.loads((ROOT / "tests/fixtures/np101-recovery-controls.json").read_text())
    binary = ROOT / pinned["vendor_binary"]
    paths = {binary: pinned["vendor_sha256"]}
    paths.update((SDK / name, digest) for name, digest in pinned["sdk_sha256"].items())
    for path, digest in paths.items():
        if sha256(path) != digest:
            raise ValueError(f"Recovery control requires the frozen successful file: {path}")
    return binary


def snapshot(output, sudo_read_only):
    prefix = ["sudo", "-n"] if sudo_read_only else []
    log = subprocess.run(
        prefix + ["journalctl", "-k", "-b", "-o", "short-iso-precise", "--no-pager"],
        capture_output=True,
        text=True,
        check=True,
    )
    # journalctl can exit zero while explicitly reporting inaccessible system journals.
    if "permission" in log.stderr.lower() or "not seeing messages" in log.stderr.lower():
        raise RuntimeError(f"Cannot verify kernel health: {log.stderr.strip()}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".log").write_text(log.stdout)
    owner = subprocess.run(prefix + ["fuser", "/dev/galcore"], capture_output=True, text=True)
    if owner.returncode not in (0, 1) or (owner.returncode == 1 and owner.stderr.strip()):
        raise RuntimeError(f"Cannot verify device ownership: {owner.stderr.strip()}")
    record = {
        "checked_at": datetime.now().astimezone().isoformat(),
        "boot_id": host_boot_id(),
        "faults": [line for line in log.stdout.splitlines() if FAULT.search(line)],
        "owners": owner.stdout.strip(),
    }
    write_json(output.with_suffix(".json"), record)
    return record


def healthy(record):
    return not record["faults"] and not record["owners"]


def numerical_check(device):
    match = re.search(
        r"total=64 mismatches=0 max_abs_err=([\d.eE+-]+)", (device / "sdk.log").read_text()
    )
    return bool(match and float(match[1]) <= 0.1)


def recovery_record(output, name, evidence, postflight):
    marker = RECOVERY_ROOT / "np101-recovery-required.json"
    record = json.loads(marker.read_text()) if marker.exists() else {}
    record.update(
        reason=f"Recovery control {name}: failed execution or kernel/ownership gate",
        boot_id=host_boot_id(),
        pid_namespace=os.readlink("/proc/self/ns/pid"),
        members=evidence.get("remaining_processes", []),
        evidence=str(output),
        kernel_gate=postflight,
        established_cause="Execution or health gate failed; root cause is not established",
        sdk_call="Uninstrumented frozen control; exact outstanding SDK call is unknown",
        next_run_rule="Stop board work; analyze evidence, confirm both devices cold-restarted",
    )
    write_json(output / "incident.json", record)
    write_json(marker, record)


def run_vendor_control(output, options):
    binary = vendor_control()
    if options.prepare_only:
        print("vendor-conv", "deadline", options.timeout or 60, "command:", binary)
        return 0
    output.mkdir(parents=True, exist_ok=False)
    marker = RECOVERY_ROOT / "np101-recovery-required.json"
    # Retain the lock through delayed checks, so no repository test overlaps this gate.
    with device_lock(RECOVERY_ROOT, recovery_control=True):
        preflight = snapshot(output / "preflight", options.sudo_read_only)
        if not healthy(preflight):
            raise RuntimeError(
                "Recovery control preflight failed; inspect saved kernel log and ownership"
            )
        if marker.exists():
            previous = json.loads(marker.read_text())
            if not options.cold_restarted or previous.get("boot_id") == host_boot_id():
                raise RuntimeError(
                    "Confirm host and NP101 cold restart in a new boot before Recovery control"
                )
            shutil.copy2(marker, output / "previous-recovery-marker.json")
        if marker.exists():
            write_json(
                marker,
                dict(
                    reason="Vendor diagnostic does not establish device recovery",
                    boot_id=host_boot_id(),
                    pid_namespace=os.readlink("/proc/self/ns/pid"),
                    members=[],
                    evidence=str(output),
                ),
            )
        name = "vendor-conv"
        directory = output / name
        evidence = run_device(
            binary,
            [],
            directory / "device",
            SDK,
            options.timeout or 60,
            abort_check=lambda: bool(
                snapshot(directory / "monitor", options.sudo_read_only)["faults"]
            ),
            recovery_control=True,
        )
        try:
            immediate = snapshot(directory / "immediate", options.sudo_read_only)
            print(f"{name}: waiting 45s for delayed kernel faults", flush=True)
            time.sleep(45)
            postflight = snapshot(directory / "postflight", options.sudo_read_only)
        except (OSError, RuntimeError, subprocess.SubprocessError, KeyboardInterrupt):
            recovery_record(directory, name, evidence, {"health_check_incomplete": True})
            raise
        execution_pass = clean_execution(evidence)
        gate_pass = execution_pass and healthy(immediate) and healthy(postflight)
        report = {"control": name, "execution": evidence, "health_pass": gate_pass}
        try:
            report["numerical_pass"] = execution_pass and numerical_check(directory / "device")
        except (OSError, ValueError, KeyError):
            report["numerical_pass"] = False
        passed = gate_pass and report["numerical_pass"]
        if not passed:
            recovery_record(directory, name, evidence, postflight)
        write_json(
            output / "result.json",
            {"status": "passed" if passed else "failed", "controls": [report]},
        )
        print(f"{name}: {'PASS' if passed else 'FAIL'}", flush=True)
    return 0 if passed else 1
