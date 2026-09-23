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


def snapshot_binary(source: Path, destination: Path) -> Path:
    """Keep the executable used by this run independent of later rebuilds."""
    shutil.copy2(source.resolve(strict=True), destination)
    return destination


def record_sources(root: Path, binary: Path, output: Path) -> None:
    """Record the current source fingerprints beside a run's executable snapshot.

    These hashes describe the working tree, not proof of what was compiled into
    the binary. The executable's own hash identifies the artifact that ran.
    """
    sources = {
        str(path.relative_to(root)): fingerprint(path)
        for directory in ("native", "python", "scripts", "tests")
        for path in sorted((root / directory).rglob("*"))
        if path.is_file() and path.suffix in (".cpp", ".hpp", ".py")
    }
    write_json(output, {"binary_sha256": fingerprint(binary), "sources": sources})


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


def wait_with_progress(process, output: Path, timeout: float, interval: float, label: str):
    """Poll an existing child; observations run outside its measured C++ calls."""
    if interval <= 0:
        return process.wait(timeout=timeout)
    started = time.monotonic()
    while True:
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise subprocess.TimeoutExpired(label, timeout)
        try:
            return process.wait(timeout=min(interval, remaining))
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - started
            phase, phase_seconds = "SDK call or initialization", None
            try:
                state = json.loads((output / "execution.json").read_text())
                phase = state.get("phase", phase)
                if state.get("phase_started_at"):
                    phase_seconds = max(0, time.time() - state["phase_started_at"])
            except (OSError, ValueError, TypeError):
                pass  # Startup and older native programs may have no progress file.
            detail = f"; phase {phase_seconds:.0f}s" if phase_seconds is not None else ""
            print(f"  {label}: {phase}; process {elapsed:.0f}s{detail}", flush=True)


def driver_wait_summary(path: Path) -> dict:
    """Separate submitting-thread latency from overlapping worker-thread waits."""

    def empty():
        return {
            "completed_calls": 0,
            "total_seconds": 0.0,
            "maximum_seconds": 0.0,
            "calls_at_least_one_second": 0,
            "unfinished_calls": 0,
        }

    threads = {}
    submission_thread = None
    pending = set()
    with path.open(errors="replace") as stream:
        for line in stream:
            fields = line.split()
            if not fields:
                continue
            thread = fields[0]
            # strace -f starts with the launched native process's loader calls.
            # Later SDK/shader workers must not redefine the submitting thread.
            if submission_thread is None and thread.isdecimal():
                submission_thread = thread
            resumed = "<... ioctl resumed>" in line and thread in pending
            if not resumed and ("/dev/galcore" not in line or "ioctl(" not in line):
                continue
            if "<unfinished ...>" in line:
                pending.add(thread)
                continue
            pending.discard(thread)
            match = re.search(r"<([0-9.]+)>$", line.rstrip())
            if not match:
                continue
            seconds = float(match[1])
            stats = threads.setdefault(thread, empty())
            stats["completed_calls"] += 1
            stats["calls_at_least_one_second"] += seconds >= 1
            stats["total_seconds"] += seconds
            stats["maximum_seconds"] = max(stats["maximum_seconds"], seconds)
    for thread in pending:
        threads.setdefault(thread, empty())["unfinished_calls"] += 1
    return threads.pop(submission_thread, empty()) | {
        "submission_thread": submission_thread,
        "other_threads": threads,
        "scope": "completed galcore ioctls on the native entry thread; other threads are separate and may overlap",
    }


def run_device(
    binary: Path,
    arguments: list[str],
    output: Path,
    sdk_lib: Path,
    timeout: int,
    artifact_prefix: str | None = None,
    shader_header: Path | None = None,
    trace_driver: bool = True,
    print_targets: bool = True,
    sdk_timing: str = "off",
    progress_interval: float = 0,
) -> dict:
    if sdk_timing not in ("off", "summary", "calls") or progress_interval < 0:
        raise ValueError("invalid SDK timing mode or progress interval")
    require_recovered_device(RECOVERY_ROOT)
    binary = binary.resolve(strict=True)
    output.mkdir(parents=True, exist_ok=True)
    shader_headers = []
    if shader_header is not None:
        header = shader_header.resolve(strict=True)
        if header.name != "cl_viv_vx_ext.h":
            raise ValueError("expected the installed SDK cl_viv_vx_ext.h header")
        # This SDK's shader compiler searches its working directory. Stage the
        # unmodified vendor header next to the run, never into the SDK installation.
        destination = output / header.name
        shutil.copy2(header, destination)
        shader_headers.append({"source": str(header), "sha256": fingerprint(destination)})
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(sdk_lib.resolve(strict=True))
    env["VIV_VX_ENABLE_PRINT_TARGET"] = "1" if print_targets else "0"
    env["SPECFERRY_SDK_TIMING"] = sdk_timing
    log_name = f"{artifact_prefix}.log" if artifact_prefix else "sdk.log"
    trace_name = f"{artifact_prefix}.strace" if artifact_prefix else "driver.strace"
    evidence_name = (
        f"{artifact_prefix}-evidence.json" if artifact_prefix else "execution-evidence.json"
    )
    command = [str(binary), *arguments]
    tracer = shutil.which("strace") if trace_driver else None
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
        "driver_traced": tracer is not None,
        "runtime_options": {
            name: env.get(name)
            for name in (
                "VIV_VX_ENABLE_PRINT_TARGET",
                "VIV_VX_PROFILE",
                "VIV_MEMORY_PROFILE",
                "VIV_VX_ENABLE_SHADER",
                "SPECFERRY_SDK_TIMING",
            )
        },
        "binary_sha256": fingerprint(binary),
        "linked_libraries": libraries,
        "sdk_library_dir": str(sdk_lib.resolve()),
        "shader_headers": shader_headers,
        "timeout": False,
        "boot_id": host_boot_id(),
        "sdk_sha256": {
            path.name: fingerprint(path)
            for path in (
                sdk_lib / name
                for name in (
                    "libovxlib.so",
                    "libOpenVX.so",
                    "libGAL.so",
                    "libArchModelSw.so",
                    "libNNArchPerf.so",
                )
            )
        },
    }
    started = time.monotonic()
    with (output / log_name).open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=output,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        write_json(output / evidence_name, evidence | {"status": "running", "pid": process.pid})
        try:
            code = wait_with_progress(process, output, timeout, progress_interval, output.name)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            evidence["timeout"] = isinstance(error, subprocess.TimeoutExpired)
            evidence["interrupted_by_user"] = isinstance(error, KeyboardInterrupt)
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
    interrupted = (
        evidence["timeout"]
        or evidence.get("interrupted_by_user", False)
        or code is None
        or code < 0
    )
    evidence["device_recovery_required"] = interrupted or bool(members)
    if evidence["device_recovery_required"]:
        marker = RECOVERY_ROOT / "np101-recovery-required.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        write_json(
            marker,
            {
                "reason": (
                    "SDK run timed out, was interrupted, or was terminated by a signal"
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
    evidence["status"] = "finished"
    evidence["wall_seconds"] = time.monotonic() - started
    if progress_interval:
        print(
            f"  {output.name}: exited {code}; process {evidence['wall_seconds']:.2f}s; "
            f"log: {output / log_name}",
            flush=True,
        )
    successes = 0
    if tracer:
        with (output / trace_name).open(errors="replace") as trace:
            successes = sum(bool(re.search(r"ioctl\(.*</dev/galcore.*= 0", line)) for line in trace)
        evidence["driver_waits"] = driver_wait_summary(output / trace_name)
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
