"""Kernel health checks and an explicit vendor diagnostic; never clear the recovery gate."""

import json
import os
import re
import shutil
import subprocess
import time
from collections import Counter
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


def kernel_faults(log):
    """Retain the journal timestamp when a kernel message contains embedded newlines."""
    prefix, faults = "", []
    for line in log.splitlines():
        header = re.match(r"^\d{4}-\d{2}-\d{2}T\S+\s+\S+\s+kernel:", line)
        if header:
            prefix = header[0]
        if FAULT.search(line):
            faults.append(line if header or not prefix else prefix + " " + line.strip())
    return faults


def apply_fault_baseline(record, baseline):
    """Exclude only recorded, timestamped occurrences from this same host boot."""
    known = Counter()
    if baseline.get("boot_id") == record.get("boot_id"):
        known.update(
            fault for fault in baseline.get("faults", []) if re.match(r"^\d{4}-\d{2}-\d{2}T", fault)
        )
    current, historical = [], []
    for fault in record["faults"]:
        if known[fault]:
            historical.append(fault)
            known[fault] -= 1
        else:
            current.append(fault)
    return dict(record, faults=current, historical_faults=historical)


def power_cycle_baseline(previous, preflight):
    """A confirmed board reset may acknowledge known old faults, never unseen ones."""
    baseline = dict(boot_id=preflight["boot_id"], faults=[])
    if previous.get("boot_id") == preflight["boot_id"]:
        health = previous.get("final_health") or previous.get("kernel_gate") or {}
        baseline["faults"] = health.get("faults", []) + health.get("historical_faults", [])
    filtered = apply_fault_baseline(preflight, baseline)
    if not healthy(filtered):
        raise RuntimeError("Board power-cycle preflight has unrecorded faults or a device owner")
    return baseline, filtered


def snapshot(output, sudo_read_only, *, fault_baseline=None):
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
    # Sudo authorization covers journalctl only. Ownership checks remain unprivileged;
    # an unreadable owner is an error, never a clean-device result.
    owner = subprocess.run(["fuser", "/dev/galcore"], capture_output=True, text=True)
    if owner.returncode not in (0, 1) or (owner.returncode == 1 and owner.stderr.strip()):
        raise RuntimeError(f"Cannot verify device ownership: {owner.stderr.strip()}")
    record = {
        "checked_at": datetime.now().astimezone().isoformat(),
        "boot_id": host_boot_id(),
        "faults": kernel_faults(log.stdout),
        "owners": owner.stdout.strip(),
    }
    if fault_baseline is None:
        epoch = RECOVERY_ROOT / "np101-recovery-epoch.json"
        fault_baseline = json.loads(epoch.read_text()) if epoch.exists() else {}
    record = apply_fault_baseline(record, fault_baseline)
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
        next_run_rule="Stop board work; analyze evidence, confirm board power cycle and agreed checks",
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
        baseline = None
        preflight = snapshot(
            output / "preflight",
            options.sudo_read_only,
            fault_baseline={} if options.board_power_cycled else None,
        )
        if marker.exists():
            previous = json.loads(marker.read_text())
            if not options.board_power_cycled:
                raise RuntimeError("Confirm board power cycle before the agreed diagnostic")
            baseline, preflight = power_cycle_baseline(previous, preflight)
            write_json(output / "preflight.json", preflight)
            shutil.copy2(marker, output / "previous-recovery-marker.json")
        if not healthy(preflight):
            raise RuntimeError(
                "Recovery control preflight failed; inspect saved kernel log and ownership"
            )
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
                snapshot(directory / "monitor", options.sudo_read_only, fault_baseline=baseline)[
                    "faults"
                ]
            ),
            recovery_control=True,
        )
        try:
            immediate = snapshot(
                directory / "immediate", options.sudo_read_only, fault_baseline=baseline
            )
            print(f"{name}: waiting 45s for delayed kernel faults", flush=True)
            time.sleep(45)
            postflight = snapshot(
                directory / "postflight", options.sudo_read_only, fault_baseline=baseline
            )
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
