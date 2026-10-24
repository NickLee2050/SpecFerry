"""Standalone Python 3.9 supervisor deployed to NP101; no SDK imports or dependencies."""

import fcntl
import hashlib
import json
import os
import re
import signal
import struct
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

STATE = Path("/root/specferry-soc")
SDK = Path("/np101_driver/build/sdk/drivers")
DELAY_SECONDS = 45
RESERVE_KIB = 192 * 1024


def read_json(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bundle_file(root, name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("invalid bundle path: " + name)
    target = root / path
    if target.is_symlink() or not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("bundle path escapes its root: " + name)
    return target


def verify_bundle(root, check_sdk=True):
    manifest = read_json(root / "bundle.json")
    if (manifest.get("schema_version"), manifest.get("target")) != (1, "np101-soc-riscv64"):
        raise ValueError("unsupported SoC bundle")
    for name, expected in manifest["files"].items():
        path = bundle_file(root, name)
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError("bundle identity differs: " + name)
        if expected["executable"] and not os.access(path, os.X_OK):
            raise ValueError("missing executable permission: " + name)
    with (root / "bin/specferry_opt_graph_generate").open("rb") as binary:
        header = binary.read(20)
    if header[:6] != b"\x7fELF\x02\x01" or struct.unpack_from("<H", header, 18)[0] != 243:
        raise ValueError("SoC executable must be ELF64 little-endian RISC-V")
    if check_sdk:
        for name, digest in manifest["sdk"].items():
            if Path(name).name != name or sha256(SDK / name) != digest:
                raise ValueError("board SDK identity differs: " + name)
    return manifest


def owners():
    found = []
    for process in Path("/proc").glob("[0-9]*"):
        try:
            for fd in (process / "fd").iterdir():
                try:
                    if os.readlink(fd) == "/dev/galcore":
                        found.append(int(process.name))
                        break
                except FileNotFoundError:
                    pass
        except FileNotFoundError:
            pass
    return found


def snapshot():
    memory = Path("/proc/meminfo").read_text()
    pool = Path("/sys/kernel/debug/gc/meminfo").read_text()
    return {
        "monotonic_seconds": time.monotonic(),
        "mem_available_kib": int(re.search(r"^MemAvailable:\s+(\d+)", memory, re.M)[1]),
        "pool": pool,
        "pool_used": re.findall(r"^\s+Used\s*:\s*(\d+)", pool, re.M),
        "gfp": Path("/sys/kernel/debug/allocators/gfp/usage").read_text(),
    }


def kernel(output, name):
    log = subprocess.check_output(["dmesg"], text=True, timeout=5)
    (output / name).write_text(log)
    return log


def new_faults(log, pattern, acknowledged=()):
    """Subtract exact timestamped occurrences, never an entire fault category."""
    remaining = Counter(acknowledged)
    faults = []
    for line in log.splitlines():
        if pattern.search(line):
            if remaining[line]:
                remaining[line] -= 1
            else:
                faults.append(line)
    return faults


def terminate(child, actions):
    # Signal the whole group even if its leader has already exited.
    for request in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(child.pid, request)
            if request == signal.SIGTERM:
                os.killpg(child.pid, signal.SIGCONT)
            actions.append({"signal": request.name, "monotonic_seconds": time.monotonic()})
        except ProcessLookupError:
            break
        try:
            child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass


def group_members(group):
    members = []
    for process in Path("/proc").glob("[0-9]*"):
        try:
            fields = (process / "stat").read_text().rpartition(")")[2].split()
            if int(fields[2]) == group and fields[0] != "Z":
                members.append(int(process.name))
        except FileNotFoundError:
            pass
    return members


def supervise(command, cwd, environment, output, timeout, monitor):
    if not 0 < timeout <= 300:
        raise ValueError("child deadline must be in (0, 300] seconds")
    started = time.monotonic()
    result = {
        "child_start_monotonic_seconds": started,
        "child_start_wall_seconds": time.time(),
        "actions": [],
        "stop_reason": None,
    }
    child = None
    watchdog = None

    def expire():
        if child.poll() is None:
            result["stop_reason"] = "child deadline expired"
            expired_at = time.monotonic()
            terminate(child, result["actions"])
            # Storage trouble must not delay the signal itself.
            save(output / "watchdog.json", {"expired_at": expired_at, "child_pid": child.pid})

    try:
        with (output / "native.log").open("w") as log:
            child = subprocess.Popen(
                command,
                cwd=cwd,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            result["child_pid"] = child.pid
            save(output / "child.json", result)
            # A driver debugfs read can itself block. The deadline must not depend
            # on that monitor returning; this thread performs no device reads.
            watchdog = threading.Timer(max(0, timeout - (time.monotonic() - started)), expire)
            watchdog.start()
            while child.poll() is None:
                reason = monitor()
                if reason or time.monotonic() - started >= timeout:
                    result["stop_reason"] = reason or "child deadline expired"
                    break
                time.sleep(0.05)
    finally:
        if watchdog is not None:
            watchdog.cancel()
            watchdog.join()
        if child is not None:
            terminate(child, result["actions"])
            result["exit"] = child.poll()
            result["remaining_processes"] = group_members(child.pid)
        result["child_seconds"] = time.monotonic() - started
        save(output / "child.json", result)
    return result


def native_complete(output):
    record = read_json(output / "native/run.json")
    execution = read_json(output / "native/execution.json")
    return (
        record.get("status") == "executed"
        and record.get("released") is True
        and execution.get("released") is True
        and all(
            record.get(part + "_live_bytes") == 0
            and type(record.get(part + "_peak_bytes")) is int
            and 0 <= record[part + "_peak_bytes"] <= 2**30
            for part in ("const", "nonconst")
        )
    )


def run(root, output, timeout, diagnostic=None):
    if not 0 < timeout <= 300:
        raise ValueError("child deadline must be in (0, 300] seconds")
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "clean": False,
        "board_boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "bundle_sha256": sha256(root / "bundle.json"),
        "child_deadline_seconds": timeout,
        "faults": [],
        "child_started": False,
    }
    marker = STATE / "np101-recovery-required.json"
    stream = None
    descriptor = None
    before = None
    samples = []
    fault = None
    acknowledged = []
    try:
        if marker.exists():
            raise RuntimeError("board recovery marker remains; follow the incident ledger")
        # Installed libraries are checked before native loading. This does not load the driver.
        manifest = verify_bundle(root)
        fault = re.compile(manifest["fault_pattern"], re.I)
        result["fault_pattern"] = manifest["fault_pattern"]
        if diagnostic is not None:
            if (
                diagnostic["board_boot_id"] != result["board_boot_id"]
                or diagnostic["bundle_sha256"] != result["bundle_sha256"]
                or not diagnostic["authorization"]
                or diagnostic["timeout"] != timeout
            ):
                raise RuntimeError("diagnostic authorization does not match this run")
            acknowledged = diagnostic["kernel_faults"]
            if not all(re.match(r"^\[\s*\d+\.\d+\]", line) for line in acknowledged):
                raise ValueError("diagnostic baseline requires timestamped kernel occurrences")
            result["diagnostic"] = diagnostic
        if new_faults(kernel(output, "kernel-before.log"), fault, acknowledged) or owners():
            raise RuntimeError("kernel fault or existing galcore owner")
        parameters = {
            name: int((Path("/sys/module/galcore/parameters") / name).read_text(), 0)
            for name in manifest["driver_parameters"]
        }
        result["driver_parameters"] = parameters
        result["iomem"] = Path("/proc/iomem").read_text()
        if parameters != manifest["driver_parameters"]:
            raise RuntimeError("loaded driver pool/dump parameters differ")
        before = snapshot()
        result["before"] = before
        if (
            before["mem_available_kib"] < RESERVE_KIB
            or int(re.search(r"POOL SYSTEM:.*?Total\s*:\s*(\d+)", before["pool"], re.S)[1])
            != manifest["driver_parameters"]["contiguousSize"]
        ):
            raise RuntimeError("insufficient Linux memory or unexpected SYSTEM pool")
        descriptor = os.open("/dev/kmsg", os.O_RDONLY | os.O_NONBLOCK)
        os.lseek(descriptor, 0, os.SEEK_END)
        stream = (output / "kernel-stream.log").open("w")
        # Recheck after attaching the stream, closing the dmesg/stream handover gap.
        if new_faults(kernel(output, "kernel-before-child.log"), fault, acknowledged):
            raise RuntimeError("new kernel fault before native execution")
        pending = dict(result, output=str(output), reason="SoC execution/acceptance incomplete")
        save(marker, pending)
        save(output / "pending-marker.json", pending)
        next_sample = 0

        def monitor():
            nonlocal next_sample
            for _ in range(64):
                try:
                    raw = os.read(descriptor, 262144)
                except BlockingIOError:
                    break
                # EPIPE means lost kernel records: propagate and retain the gate.
                if not raw:
                    break
                text = raw.decode(errors="replace")
                stream.write(text)
                stream.flush()
                result["faults"].extend(line for line in text.splitlines() if fault.search(line))
            if result["faults"]:
                return "kernel fault"
            if time.monotonic() >= next_sample:
                state = snapshot()
                samples.append(state)
                next_sample = time.monotonic() + 0.5
                if state["mem_available_kib"] < RESERVE_KIB:
                    return "Linux memory reserve breached"
            return None

        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("SPECFERRY_", "VIV_", "VP_", "LD_"))
        }
        environment.update(
            LD_LIBRARY_PATH=str(SDK),
            VIV_MGPU_AFFINITY="1:0",
            VIVANTE_SDK_DIR=str(SDK.parent),
            LJMICRO_SDK_DIR=str(root / "runtime"),
            SPECFERRY_SDK_TIMING="calls" if diagnostic else "off",
            VIV_VX_PROFILE="0",
            VP_ENABLE_PRINT="0",
        )
        result["child_started"] = True
        result.update(
            supervise(
                [
                    str(root / "bin/specferry_opt_graph_generate"),
                    str(root / "weights"),
                    str(root / "request"),
                    str(output / "native"),
                ],
                root / "runtime",
                environment,
                output,
                timeout,
                monitor,
            )
        )
        monitor()
    except Exception as error:
        result["error"] = str(error)
        if (output / "child.json").is_file():
            result.update(read_json(output / "child.json"))
    finally:
        if result["child_started"]:
            delayed = time.monotonic()
            while time.monotonic() - delayed < DELAY_SECONDS:
                try:
                    monitor()
                except Exception as error:
                    result["monitor_error"] = str(error)
                time.sleep(0.1)
            result["delayed_seconds"] = time.monotonic() - delayed
        try:
            final = kernel(output, "kernel-delayed.log")
            result["kernel_faults"] = new_faults(final, fault, acknowledged) if fault else []
            result["after"] = snapshot()
            result["owners_delayed"] = owners()
            result["pool_released"] = bool(
                before and before["pool_used"] == result["after"]["pool_used"]
            )
            result["gfp_released"] = bool(before and before["gfp"] == result["after"]["gfp"])
            result["clean"] = bool(
                result.get("exit") == 0
                and not result.get("error")
                and not result.get("stop_reason")
                and not result.get("monitor_error")
                and result.get("remaining_processes") == []
                and not result["faults"]
                and not result["kernel_faults"]
                and not result["owners_delayed"]
                and result["pool_released"]
                and result["gfp_released"]
                and result.get("delayed_seconds", 0) >= 45
                and native_complete(output)
            )
        except Exception as error:
            result["final_check_error"] = str(error)
        if stream:
            stream.close()
        if descriptor is not None:
            os.close(descriptor)
        save(output / "memory-samples.json", samples)
        if marker.exists() and read_json(marker).get("output") == str(output):
            save(
                marker,
                dict(
                    result,
                    output=str(output),
                    reason="awaiting host numerical acceptance"
                    if result["clean"]
                    else "SoC run failed",
                ),
            )
        save(output / "result.json", result)
    return result


def accept(output, expected_digest):
    # Called only after host numerical checks; never resolves older recovery incidents.
    result = read_json(output / "result.json")
    marker = STATE / "np101-recovery-required.json"
    prior = marker if marker.exists() else output / "accepted-marker.json"
    if (
        sha256(output / "result.json") != expected_digest
        or not result["clean"]
        or result["board_boot_id"] != Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        or read_json(prior).get("output") != str(output)
        or owners()
    ):
        raise RuntimeError("acceptance does not match the pending run")
    # Reject new records even if they appeared after the delayed check.
    if new_faults(
        kernel(output, "kernel-accepted.log"),
        re.compile(result["fault_pattern"], re.I),
        result.get("diagnostic", {}).get("kernel_faults", []),
    ):
        raise RuntimeError("kernel fault after delayed check; retain recovery marker")
    if marker.exists():
        marker.rename(output / "accepted-marker.json")


def main():
    def interrupted(signum, frame):
        raise RuntimeError("board supervisor interrupted by signal " + str(signum))

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    command, root = sys.argv[1], Path(sys.argv[2]).resolve()
    if command == "verify":
        manifest = verify_bundle(root)
        print(
            json.dumps(
                {
                    "verified": True,
                    "bundle_sha256": sha256(root / "bundle.json"),
                    "files": len(manifest["files"]),
                }
            )
        )
        return
    STATE.mkdir(exist_ok=True)
    with (STATE / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if command == "run":
            run(
                root,
                Path(sys.argv[3]).resolve(),
                float(sys.argv[4]),
                json.loads(sys.argv[5]) if len(sys.argv) > 5 else None,
            )
        elif command == "accept":
            accept(root, sys.argv[3])
        else:
            raise ValueError("unknown board supervisor command")


if __name__ == "__main__":
    main()
