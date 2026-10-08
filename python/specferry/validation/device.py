"""Serialize bounded SDK processes and stop retries after abnormal termination."""

import csv
import fcntl
import json
import os
import resource
import shutil
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

from specferry.data.checkpoint import sha256

RECOVERY_ROOT = Path(__file__).resolve().parents[3] / ".cache/runs"
VERIFY_TRACE_OPENVX_SHA256 = "2f7223e504b15886c1562889668a80af48b19d605461beac175d336d396ff108"

VERIFY_TRACE_GAL_SHA256 = "e60c55cd8caebebbda238119ba8c9c4d88996048f62769421dd0b538325de5c7"


class DeviceHealthFailure(RuntimeError):
    """A read-only health monitor failed or detected a fault during a child run."""


def verify_trace_environment(library: Path, sdk_lib: Path, output: Path, environment: dict):
    """Private graph offsets are diagnostic-only and valid for one verified SDK build."""
    library = library.resolve(strict=True)
    if sha256(sdk_lib / "libOpenVX.so") != VERIFY_TRACE_OPENVX_SHA256:
        raise ValueError("Verify tracing requires the pinned v105 libOpenVX binary")
    if sha256(sdk_lib / "libGAL.so") != VERIFY_TRACE_GAL_SHA256:
        raise ValueError("Execution tracing requires the pinned v105 libGAL binary")
    if environment.get("LD_PRELOAD"):
        raise ValueError("Verify tracing cannot be combined with another LD_PRELOAD")
    trace_path = output / "verify-internals.tsv"
    environment.update(LD_PRELOAD=str(library), SPECFERRY_VERIFY_TRACE=str(trace_path))
    return {
        "library": str(library),
        "library_sha256": sha256(library),
        "output": str(trace_path),
        "node_snapshots_output": str(trace_path) + "-nodes.tsv",
        "execution_output": str(trace_path) + "-execution.tsv",
        "private_layout_openvx_sha256": VERIFY_TRACE_OPENVX_SHA256,
        "execution_gal_sha256": VERIFY_TRACE_GAL_SHA256,
    }


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def clean_execution(evidence: dict, expected_returncode: int = 0) -> bool:
    return (
        evidence.get("returncode") == expected_returncode
        and evidence.get("timeout") is False
        and evidence.get("process_group_exited") is True
        and evidence.get("remaining_processes") == []
        and evidence.get("device_recovery_required") is False
    )


def process_group_members(group: int) -> list[dict]:
    members = []
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        try:
            fields = (path / "stat").read_text().rpartition(")")[2].split()
            if int(fields[2]) == group and fields[0] != "Z":
                members.append({"pid": int(path.name), "start_ticks": fields[19]})
        except (OSError, ValueError, IndexError):
            continue
    return members


def terminate_group(group: int, request: signal.Signals) -> None:
    try:
        os.killpg(group, request)
    except ProcessLookupError:
        # The child may finish between the timeout and the signal delivery.
        pass


def host_boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def require_recovered_device(root: Path) -> None:
    marker = root / "np101-recovery-required.json"
    if not marker.exists():
        return
    try:
        record = json.loads(marker.read_text())
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"NP101 recovery marker is incomplete; inspect and archive it after "
            f"confirming device recovery: {marker}"
        ) from error
    if record.get("boot_id") and record["boot_id"] != host_boot_id():
        # A reboot ends the previous driver's lifetime, even if PIDs are reused.
        marker.unlink()
        return
    if record.get("pid_namespace") != os.readlink("/proc/self/ns/pid"):
        raise RuntimeError(
            f"NP101 recovery must be checked from its original host PID namespace: {marker}"
        )
    for member in record["members"]:
        path = Path(f"/proc/{member['pid']}/stat")
        try:
            fields = path.read_text().rpartition(")")[2].split()
        except FileNotFoundError:
            continue
        if fields[19] == member["start_ticks"] and fields[0] != "Z":
            raise RuntimeError(
                f"NP101 diagnostic PID {member['pid']} has not exited; "
                f"recover the device before another run. See {marker}"
            )
    raise RuntimeError(
        "NP101 device recovery is required; process exit alone does not establish driver health. "
        f"After maintainer-confirmed recovery, archive this marker before retrying: {marker}"
    )


@contextmanager
def device_lock(root: Path):
    """Serialize this repository's diagnostics; never block another user's process."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".np101-diagnostic.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another SpecFerry device diagnostic is running") from error
        require_recovered_device(root)
        yield


def run_device(
    binary,
    arguments,
    output,
    sdk_lib,
    timeout,
    *,
    sdk_timing=False,
    sdk_summary=False,
    trace_driver=False,
    verify_trace=None,
    require_compiled_operations=False,
    abort_check=None,
):
    """Call under device_lock. A clean exit is not proof of exclusive NPU execution."""
    require_recovered_device(RECOVERY_ROOT)
    if require_compiled_operations and verify_trace is None:
        raise ValueError("compiled operation guard requires the pinned Verify trace")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if sdk_timing and sdk_summary:
        raise ValueError("SDK call tracing and summary timing are mutually exclusive")
    binary, output = binary.resolve(strict=True), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    header = Path("/usr/inc/CL/cl_viv_vx_ext.h")
    if header.is_file():
        shutil.copy2(header, output / header.name)
    env = os.environ | {
        "LD_LIBRARY_PATH": str(sdk_lib.resolve(strict=True)),
        "VIV_VX_ENABLE_PRINT_TARGET": "0",
        "VIV_VX_PROFILE": "0",
        "VIV_MEMORY_PROFILE": "1" if binary.name == "np101_memory_accounting_check" else "0",
        "SPECFERRY_SDK_TIMING": "calls" if sdk_timing else ("summary" if sdk_summary else "off"),
    }
    if env.get("LD_PRELOAD"):
        raise ValueError(
            "SDK diagnostics reject inherited LD_PRELOAD; use --trace-verify explicitly"
        )
    env.pop("SPECFERRY_WEIGHT_READBACK_REPORT", None)
    env.pop("SPECFERRY_VERIFY_TRACE", None)
    env.pop("SPECFERRY_VERIFY_REQUIRE_OPERATIONS", None)
    if require_compiled_operations:
        env["SPECFERRY_VERIFY_REQUIRE_OPERATIONS"] = "1"
    verify_trace_record = (
        verify_trace_environment(verify_trace, sdk_lib, output, env) if verify_trace else None
    )
    command = [str(binary), *map(str, arguments)]
    if trace_driver:
        command = [
            "strace",
            "-f",
            "-tt",
            "-T",
            "-yy",
            "-e",
            "trace=ioctl",
            "-o",
            str(output / "driver.strace"),
            *command,
        ]
    evidence = {
        "command": command,
        "boot_id": host_boot_id(),
        "binary_sha256": sha256(binary),
        "sdk_library_directory": str(sdk_lib.resolve()),
        "sdk_library_sha256": {
            name: sha256(sdk_lib / name)
            for name in (
                "libovxlib.so",
                "libOpenVX.so",
                "libGAL.so",
                "libArchModelSw.so",
                "libNNArchPerf.so",
            )
            if (sdk_lib / name).is_file()
        },
        "timeout": False,
        "driver_traced": trace_driver,
        "sdk_timing": sdk_timing,
        "sdk_summary": sdk_summary,
        "verify_trace": verify_trace_record,
        "require_compiled_operations": require_compiled_operations,
        "hardware_execution_proven": False,
    }
    # Preserve the exact executable/SDK identity even if the test crashes.
    write_json(output / "execution-evidence.json", evidence)
    print(f"Running {binary.name}; timeout {timeout}s; log: {output / 'sdk.log'}", flush=True)
    started = time.monotonic()
    child_usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    with (output / "sdk.log").open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=output,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    code = process.wait(timeout=min(5, remaining) if abort_check else remaining)
                    break
                except subprocess.TimeoutExpired:
                    if not abort_check:
                        raise
                    try:
                        failed = abort_check()
                    except Exception as error:
                        raise DeviceHealthFailure(f"Health monitor failed: {error}") from error
                    if failed:
                        raise DeviceHealthFailure("Health monitor detected a kernel fault")
        except (subprocess.TimeoutExpired, KeyboardInterrupt, DeviceHealthFailure) as error:
            evidence["timeout"] = isinstance(error, subprocess.TimeoutExpired)
            if isinstance(error, DeviceHealthFailure):
                evidence["health_monitor_failure"] = str(error)
            evidence["interrupted_by_user"] = isinstance(error, KeyboardInterrupt)
            code = None
            for request in (signal.SIGTERM, signal.SIGKILL):
                terminate_group(process.pid, request)
                try:
                    code = process.wait(timeout=5)
                    break
                except subprocess.TimeoutExpired:
                    pass
    members = process_group_members(process.pid)
    child_usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    # The interposer refuses unsafe tracing with _Exit(125). This can happen
    # after Context creation and skips SDK destructors despite the positive code.
    trace_aborted = verify_trace_record is not None and code == 125
    recovery = (
        bool(members)
        or code is None
        or code < 0
        or trace_aborted
        or evidence["timeout"]
        or evidence.get("interrupted_by_user", False)
        or bool(evidence.get("health_monitor_failure"))
    )
    evidence.update(
        returncode=code,
        process_group_exited=not members,
        remaining_processes=members,
        device_recovery_required=recovery,
        wall_seconds=time.monotonic() - started,
        host_cpu_user_seconds=child_usage_after.ru_utime - child_usage_before.ru_utime,
        host_cpu_system_seconds=child_usage_after.ru_stime - child_usage_before.ru_stime,
        host_peak_rss_kib=child_usage_after.ru_maxrss
        if child_usage_after.ru_maxrss > child_usage_before.ru_maxrss
        else None,
    )
    if verify_trace_record is not None:
        trace_path = Path(verify_trace_record["output"])
        try:
            with trace_path.open() as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
            ended = [row for row in rows if row.get("event") == "end"]
            verify_trace_record.update(
                trace_captured=any(row.get("api") == "vxVerifyGraph" for row in ended),
                completed_verify_calls=sum(row.get("api") == "vxVerifyGraph" for row in ended),
                events=len(rows),
            )
        except (OSError, ValueError, csv.Error) as error:
            verify_trace_record.update(trace_captured=False, capture_error=str(error))
    if recovery:
        RECOVERY_ROOT.mkdir(parents=True, exist_ok=True)
        write_json(
            RECOVERY_ROOT / "np101-recovery-required.json",
            {
                "reason": "Verify tracing aborted without SDK cleanup"
                if trace_aborted
                else "SDK process terminated abnormally or retained children",
                "boot_id": evidence["boot_id"],
                "process_group": process.pid,
                "pid_namespace": os.readlink("/proc/self/ns/pid"),
                "members": members,
                "evidence": str(output),
            },
        )
    write_json(output / "execution-evidence.json", evidence)
    print(f"Exit {code}; {evidence['wall_seconds']:.2f}s; log: {output / 'sdk.log'}", flush=True)
    return evidence
