"""Serialize bounded SDK processes and stop retries after abnormal termination."""

import fcntl
import json
import os
import shutil
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

RECOVERY_ROOT = Path(__file__).resolve().parents[3] / ".cache/runs"


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
    record = json.loads(marker.read_text())
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
    binary, arguments, output, sdk_lib, timeout, *, sdk_timing=False, trace_driver=False
):
    """Call under device_lock. A clean exit is not proof of exclusive NPU execution."""
    require_recovered_device(RECOVERY_ROOT)
    if timeout <= 0:
        raise ValueError("timeout must be positive")
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
        "SPECFERRY_SDK_TIMING": "calls" if sdk_timing else "off",
    }
    env.pop("SPECFERRY_WEIGHT_READBACK_REPORT", None)
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
        "timeout": False,
        "driver_traced": trace_driver,
        "sdk_timing": sdk_timing,
        "hardware_execution_proven": False,
    }
    print(f"Running {binary.name}; timeout {timeout}s; log: {output / 'sdk.log'}", flush=True)
    started = time.monotonic()
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
            code = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            evidence["timeout"] = isinstance(error, subprocess.TimeoutExpired)
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
    recovery = (
        bool(members)
        or code is None
        or code < 0
        or evidence["timeout"]
        or evidence.get("interrupted_by_user", False)
    )
    evidence.update(
        returncode=code,
        process_group_exited=not members,
        remaining_processes=members,
        device_recovery_required=recovery,
        wall_seconds=time.monotonic() - started,
    )
    if recovery:
        RECOVERY_ROOT.mkdir(parents=True, exist_ok=True)
        write_json(
            RECOVERY_ROOT / "np101-recovery-required.json",
            {
                "reason": "SDK process terminated abnormally or retained children",
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
