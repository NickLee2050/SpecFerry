"""Kernel health checks and occurrence-scoped recovery baselines."""

import json
import re
import subprocess
from collections import Counter
from datetime import datetime

from .device import (
    RECOVERY_ROOT,
    host_boot_id,
    write_json,
)

# The matched SoC driver's stuckDump mode replaces the terse NP hang message with this banner.
FAULT = re.compile(
    r"NP.*hang|EDMA.*timeout|\bBUG:|\bOops:|unable to handle|general protection fault|"
    r"invalid opcode|soft lockup|Hardware Error|GPU DRV CONFIG|Stop driver to keep scene",
    re.I,
)


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
