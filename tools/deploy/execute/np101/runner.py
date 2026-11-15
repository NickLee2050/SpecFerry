"""Shared command options and result handling for native compute checks."""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

from tools.deploy.execute.np101.device import (
    RECOVERY_ROOT,
    clean_execution,
    device_lock,
    run_device,
    write_json,
)
from tools.deploy.execute.np101.recovery_controls import healthy, power_cycle_baseline, snapshot

ROOT = RECOVERY_ROOT.parents[1]


def positive_seconds(value):
    seconds = int(value)
    if seconds <= 0:
        raise argparse.ArgumentTypeError("timeout must be positive")
    return seconds


def add_run_arguments(parser):
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=positive_seconds, help="process deadline in seconds")
    parser.add_argument(
        "--sudo-read-only", action="store_true", help="use sudo for read-only kernel journal checks"
    )
    parser.add_argument("--prepare-only", action="store_true", help="do not open NP101")
    parser.add_argument(
        "--sdk-timing", action="store_true", help="record public SDK call durations"
    )
    parser.add_argument(
        "--trace-driver", action="store_true", help="record ioctl calls with strace"
    )


def _read_json(path):
    return json.loads(path.read_text())


def run_guarded(
    binary,
    arguments,
    output,
    sdk,
    timeout,
    *,
    evaluate_result,
    sudo_read_only=False,
    sdk_timing=False,
    sdk_summary=False,
    trace_driver=False,
    numerical_diagnostic=False,
    board_power_cycled=False,
):
    """Hold the EP lock through validation and delayed checks; fail closed on interruption.

    Recovery exceptions are reserved for callers that already validated the prior
    numerical failure or received explicit board power-cycle confirmation.
    """
    if not 0 < timeout <= 300:
        raise ValueError("NP101 device runs require a timeout in (0, 300] seconds")
    output.mkdir(parents=True, exist_ok=True)
    device = output / "device"
    marker = RECOVERY_ROOT / "np101-recovery-required.json"
    with device_lock(RECOVERY_ROOT, recovery_control=numerical_diagnostic or board_power_cycled):
        baseline = None
        preflight = snapshot(
            output / "preflight", sudo_read_only, fault_baseline={} if board_power_cycled else None
        )
        if board_power_cycled:
            previous = _read_json(marker) if marker.exists() else {}
            baseline, preflight = power_cycle_baseline(previous, preflight)
            baseline.update(confirmed_at=preflight.get("checked_at"), evidence=str(output))
            write_json(output / "preflight.json", preflight)
            write_json(output / "board-power-cycle.json", baseline)
            # Preserve the old incident until this agreed workload passes every gate.
            write_json(output / "previous-recovery-marker.json", previous)
        if not healthy(preflight):
            raise RuntimeError("EP preflight failed; board not started")
        write_json(
            marker,
            dict(
                boot_id=preflight["boot_id"],
                members=[],
                evidence=str(output),
                pid_namespace=os.readlink("/proc/self/ns/pid"),
                reason="EP execution and delayed checks are incomplete",
            ),
        )
        report = dict(passed=False)
        try:
            evidence = run_device(
                binary,
                arguments,
                device,
                sdk,
                timeout,
                recovery_control=True,
                sdk_timing=sdk_timing,
                sdk_summary=sdk_summary,
                trace_driver=trace_driver,
                abort_check=lambda: bool(
                    snapshot(output / "monitor", sudo_read_only, fault_baseline=baseline)["faults"]
                ),
            )
            report["execution"] = evidence
            immediate = snapshot(output / "immediate", sudo_read_only, fault_baseline=baseline)
            report["immediate"] = immediate
            print("Process exited; waiting 45 seconds for delayed kernel checks", flush=True)
            time.sleep(45)
            postflight = snapshot(output / "postflight", sudo_read_only, fault_baseline=baseline)
            report.update(immediate=immediate, postflight=postflight)
            if clean_execution(evidence):
                report.update(evaluate_result(evidence))
            report["passed"] = report["passed"] and healthy(immediate) and healthy(postflight)
        except (
            OSError,
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
            RuntimeError,
            KeyboardInterrupt,
        ) as error:
            report["error"] = str(error)
        write_json(output / "health.json", report)
        if report["passed"]:
            if baseline is not None:
                # Publish only after numerical, release and delayed checks all pass.
                write_json(RECOVERY_ROOT / "np101-recovery-epoch.json", baseline)
            shutil.copy2(marker, output / "completed-run-marker.json")
            marker.unlink()
        else:
            phase = next(
                (
                    path
                    for path in [device / "phase.json", device / "execution.json"]
                    if path.exists()
                ),
                None,
            )
            incident = dict(
                boot_id=preflight["boot_id"],
                evidence=str(output),
                reason="EP execution failed or did not complete health checks",
                execution=report.get("execution"),
                kernel_gate=report.get("postflight", report.get("immediate")),
                phase=_read_json(phase) if phase else None,
                pid_namespace=os.readlink("/proc/self/ns/pid"),
                members=report.get("execution", {}).get("remaining_processes", []),
                established_cause="Not established; inspect phase/API and kernel timestamps",
                next_run_rule="Stop board work and preserve marker; review the incident ledger",
            )
            write_json(output / "incident.json", incident)
            write_json(marker, incident)
    return report


def run_check(binary, arguments, output, options, *, evaluate=None, timeout=None):
    def accept(evidence):
        report = (
            evaluate(evidence)
            if evaluate is not None
            else {"status": "numerical_pass", "evidence": evidence}
        )
        return dict(report, passed=report["status"] == "numerical_pass")

    report = run_guarded(
        binary,
        arguments,
        output,
        Path("/usr/lib/ljmicro"),
        timeout if timeout is not None else (120 if options.timeout is None else options.timeout),
        evaluate_result=accept,
        sudo_read_only=options.sudo_read_only,
        sdk_timing=options.sdk_timing,
        trace_driver=options.trace_driver,
    )
    if not report["passed"]:
        report["status"] = "failed"
    write_json(output / "result.json", report)
    print(f"{report['status']}; result: {output / 'result.json'}")
    return int(not report["passed"])
