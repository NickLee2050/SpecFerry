"""Run one bounded SDK process and retain evidence without inferring NPU execution."""

import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

RECOVERY_ROOT = Path(__file__).resolve().parents[3] / ".cache/runs"


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    binary: Path,
    arguments: list[str],
    output: Path,
    sdk_lib: Path,
    timeout: int,
    artifact_prefix: str | None = None,
) -> dict:
    require_recovered_device(RECOVERY_ROOT)
    binary = binary.resolve(strict=True)
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(sdk_lib.resolve(strict=True))
    env["VIV_VX_ENABLE_PRINT_TARGET"] = "1"
    log_name = f"{artifact_prefix}.log" if artifact_prefix else "sdk.log"
    trace_name = f"{artifact_prefix}.strace" if artifact_prefix else "driver.strace"
    evidence_name = (
        f"{artifact_prefix}-evidence.json" if artifact_prefix else "execution-evidence.json"
    )
    command = [str(binary), *arguments]
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
            str(output / trace_name),
            *command,
        ]
    libraries = subprocess.run(
        ["ldd", str(binary)], env=env, capture_output=True, text=True, check=True
    ).stdout
    evidence = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "binary_sha256": fingerprint(binary),
        "linked_libraries": libraries,
        "sdk_library_dir": str(sdk_lib.resolve()),
        "timeout": False,
        "boot_id": host_boot_id(),
        "sdk_sha256": {
            path.name: fingerprint(path)
            for path in (sdk_lib / "libovxlib.so", sdk_lib / "libOpenVX.so", sdk_lib / "libGAL.so")
        },
    }
    with (output / log_name).open("w") as log:
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
        except subprocess.TimeoutExpired:
            evidence["timeout"] = True
            terminate_group(process.pid, signal.SIGTERM)
            try:
                code = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                terminate_group(process.pid, signal.SIGKILL)
                try:
                    code = process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    code = None
    # A tracer can exit before its SDK child. Even when every child exits, a
    # driver fault may appear later, so forced termination also requires recovery.
    members = process_group_members(process.pid)
    if members:
        time.sleep(0.2)
        members = process_group_members(process.pid)
    evidence["process_group_exited"] = not members
    evidence["remaining_processes"] = members
    interrupted = evidence["timeout"] or code is None or code < 0
    evidence["device_recovery_required"] = interrupted or bool(members)
    if evidence["device_recovery_required"]:
        marker = RECOVERY_ROOT / "np101-recovery-required.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        write_json(
            marker,
            {
                "reason": (
                    "SDK run timed out or was terminated by a signal"
                    if interrupted
                    else "SDK process group did not exit after the bounded run"
                ),
                "boot_id": evidence["boot_id"],
                "process_group": process.pid,
                "pid_namespace": os.readlink("/proc/self/ns/pid"),
                "members": members,
                "evidence": str(output),
            },
        )
    evidence["returncode"] = code
    successes = 0
    if tracer:
        with (output / trace_name).open(errors="replace") as trace:
            successes = sum(bool(re.search(r"ioctl\(.*</dev/galcore.*= 0", line)) for line in trace)
    evidence.update(
        {
            "successful_galcore_ioctls": successes,
            "driver_activity_observed": successes > 0,
            "hardware_execution_proven": False,
            "hardware_evidence_note": "Generic driver IO, target labels, and numeric agreement do not "
            "identify every node's actual execution device.",
        }
    )
    write_json(output / evidence_name, evidence)
    return evidence
