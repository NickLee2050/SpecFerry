"""Standalone deployment/health policy adapter for the independent C++ process supervisor."""

import fcntl
import hashlib
import json
import os
import re
import signal
import struct
import subprocess
import sys
import time
from collections import Counter
from concurrent import futures
from pathlib import Path

if __package__:
    from tools.deploy.execute import supervision
else:
    import supervision

SUPERVISOR_BINARY = Path(__file__).resolve().parent / "bin/specferry_supervise"

STATE = Path("/root/specferry-soc")
SDK = Path("/np101_driver/build/sdk/drivers")
DELAY_SECONDS = 10
RESERVE_KIB = 192 * 1024
HANG_FAILURE_LIMIT = 3
# Old bundles carry their original fault expression. Keep this minimum independent
# of that manifest so a stale expression cannot hide a storage failure.
STORAGE_FAULT_PATTERN = (
    r"mmc\d+:.*Timeout waiting for hardware(?: (?:cmd|data))? interrupt|"
    r"EXT4-fs error|Buffer I/O error|(?:blk_update_request|end_request):.*I/O error"
)


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


def memory_reserve_kib(manifest):
    """Allow an explicit smaller reserve for the constrained 512MiB Linux layout."""
    reserve = manifest.get("linux_memory_reserve_mib", RESERVE_KIB // 1024)
    if type(reserve) is not int or not 32 <= reserve <= 512:
        raise ValueError("Linux memory reserve must be an integer in [32, 512] MiB")
    return reserve * 1024


def native_command(root, manifest, output):
    """Keep model argument ordering explicit; never execute a manifest-provided command."""
    family = manifest.get("model_family", "opt")
    if family == "opt":
        return [
            str(root / "bin/specferry_opt_graph_generate"),
            str(root / "weights"),
            str(root / "request"),
            str(output / "native"),
        ]
    if family == "qwen3_5":
        return [
            str(root / "bin/specferry_qwen_generate"),
            str(root / "weights"),
            str(root / "model"),
            str(root / "request"),
            str(output / "native"),
        ]
    raise ValueError("unsupported SoC model family")


def verify_bundle(root, check_sdk=True, *, verify_payload=True):
    manifest = read_json(root / "bundle.json")
    if (manifest.get("schema_version"), manifest.get("target")) != (1, "np101-soc-riscv64"):
        raise ValueError("unsupported SoC bundle")
    memory_reserve_kib(manifest)
    if "verified_prefill_cache" in manifest or any(
        name in {"model/verified-prefill-cache.txt", "model/cache-producer.bin"}
        or name.startswith("model/verify-cache/")
        for name in manifest["files"]
    ):
        raise ValueError("FFN Verify cache package is retired")
    checks = []
    for name, expected in manifest["files"].items():
        path = bundle_file(root, name)
        if path.stat().st_size != expected["bytes"]:
            raise ValueError("bundle identity differs: " + name)
        if expected["executable"] and not os.access(path, os.X_OK):
            raise ValueError("missing executable permission: " + name)
        if verify_payload:
            checks.append((path, expected["bytes"], expected["sha256"], "bundle", name))
    if check_sdk:
        for name, digest in manifest["sdk"].items():
            if Path(name).name != name:
                raise ValueError("board SDK identity differs: " + name)
            path = SDK / name
            checks.append((path, path.stat().st_size, digest, "board SDK", name))

    def verify_file(item):
        path, _, digest, kind, name = item
        if sha256(path) != digest:
            raise ValueError(kind + " identity differs: " + name)

    # Independent hashes only; SDK graph initialization remains sequential.
    # Start large files first so the weight pack overlaps the smaller artifacts.
    checks.sort(key=lambda item: item[1], reverse=True)
    with futures.ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as workers:
        for _ in workers.map(verify_file, checks):
            pass
    executables = [Path(native_command(root, manifest, root)[0])]
    implementation = manifest.get("supervisor_implementation")
    if implementation is not None:
        if implementation != "cpp-process-v1":
            raise ValueError("unsupported SoC supervisor implementation")
        executables.append(root / "bin/specferry_supervise")
    for executable in executables:
        name = executable.relative_to(root).as_posix()
        if manifest["files"].get(name, {}).get("executable") is not True:
            raise ValueError("SoC executable must be included in the verified file inventory")
        with executable.open("rb") as binary:
            header = binary.read(20)
        if header[:6] != b"\x7fELF\x02\x01" or struct.unpack_from("<H", header, 18)[0] != 243:
            raise ValueError("SoC executable must be ELF64 little-endian RISC-V")

    return manifest


def deployment_files(root, manifest):
    result = {}
    for name in manifest["files"]:
        state = bundle_file(root, name).stat()
        result[name] = [
            state.st_dev,
            state.st_ino,
            state.st_size,
            state.st_mtime_ns,
            state.st_ctime_ns,
        ]
    return result


def verify_deployment(root):
    """Hash a deployment once, rejecting files changed during that verification."""
    digest = sha256(root / "bundle.json")
    before = deployment_files(root, read_json(root / "bundle.json"))
    manifest = verify_bundle(root)
    if sha256(root / "bundle.json") != digest or deployment_files(root, manifest) != before:
        raise RuntimeError("deployment changed during checksum verification")
    save(root / ".deployment-verified.json", {"bundle_sha256": digest, "files": before})
    return manifest


def require_verified_deployment(root, manifest):
    receipt = root / ".deployment-verified.json"
    if not receipt.is_file():
        raise RuntimeError("deployment verification required before execution")
    record = read_json(receipt)
    current = deployment_files(root, manifest)
    digest = sha256(root / "bundle.json")
    if record.get("bundle_sha256") != digest or record.get("files") != current:
        raise RuntimeError("deployment changed; repeat deployment verification")


def fault_pattern(manifest):
    return manifest["fault_pattern"] + "|" + STORAGE_FAULT_PATTERN


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


def inference_timeout_limit(manifest):
    # Qwen's first SoC acceptance has an explicitly authorized ten-minute budget.
    return 600 if manifest.get("model_family", "opt") == "qwen3_5" else 300


def supervise(command, cwd, environment, output, timeout, monitor, **options):
    return supervision.supervise(
        command,
        cwd,
        environment,
        output,
        timeout,
        monitor,
        supervisor_binary=SUPERVISOR_BINARY,
        save=save,
        **options,
    )


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


def recorded_hang(line, *, raw=False):
    """Match the pinned driver's exact event, including its kernel timestamp."""
    stamp = r"\d+,\d+,(\d+),[^;]*;" if raw else r"\[\s*(\d+)\.(\d{6})\]\s*"
    match = (
        re.fullmatch(stamp + r"(\[galcore\]: NP\[\d+\] hang, automatic recovery\.)", line)
        if isinstance(line, str)
        else None
    )
    if match is None:
        raise RuntimeError("bounded SoC retry only accepts recorded NP hang events")
    micros = int(match[1]) if raw else int(match[1]) * 1_000_000 + int(match[2])
    return micros, match[2 if raw else 3]


def hang_retry_state(previous, diagnostic, digest, boot_id):
    policy = diagnostic.get("soc_hang_retry")
    if policy is None:
        return None
    if (
        not isinstance(policy, dict)
        or set(policy) != {"first_failed_marker_sha256"}
        or not re.fullmatch(r"[0-9a-f]{64}", str(policy["first_failed_marker_sha256"]))
        or not diagnostic.get("authorization")
    ):
        raise RuntimeError("bounded SoC retry requires an explicit immutable fault-chain anchor")
    earlier = previous.get("diagnostic", {}).get("kernel_faults", [])
    faults, kernel_faults = previous.get("faults"), previous.get("kernel_faults")
    if not all(isinstance(value, list) for value in (earlier, faults, kernel_faults)):
        raise RuntimeError("bounded SoC retry requires complete fault records")
    baseline = earlier + kernel_faults
    if not baseline or diagnostic.get("kernel_faults") != baseline:
        raise RuntimeError("bounded SoC retry requires the exact complete historical baseline")
    for line in baseline:
        recorded_hang(line)
    # /dev/kmsg and dmesg name the same occurrence using different timestamp formats.
    if Counter(recorded_hang(line, raw=True) for line in faults) - Counter(
        recorded_hang(line) for line in kernel_faults
    ):
        raise RuntimeError("raw NP hang events are absent from the timestamped baseline")
    anchor = policy["first_failed_marker_sha256"]
    inherited = previous.get("soc_hang_retry")
    if inherited is None:
        if anchor != digest or not kernel_faults:
            raise RuntimeError("the initial retry must anchor the recorded failed run")
        failures = 1
    else:
        if not isinstance(inherited, dict) or inherited.get("first_failed_marker_sha256") != anchor:
            raise RuntimeError("changing the experiment name cannot reset its NP hang counter")
        failures = inherited.get("consecutive_hang_runs")
        if type(failures) is not int or failures < 0:
            raise RuntimeError("invalid persisted NP hang counter")
    accepted = STATE / "soc-hang-retry-accepted.json"
    if accepted.exists():
        record = read_json(accepted)
        if (
            record.get("board_boot_id") == boot_id
            and record.get("first_failed_marker_sha256") == anchor
            and record.get("restored_marker_sha256") == digest
        ):
            result_path = Path(record["result_path"])
            result = read_json(result_path)
            if (
                sha256(result_path) != record.get("result_sha256")
                or result.get("clean") is not True
                or result.get("board_boot_id") != boot_id
                or result.get("diagnostic", {}).get("board_marker_sha256") != digest
                or result.get("soc_hang_retry", {}).get("first_failed_marker_sha256") != anchor
            ):
                raise RuntimeError("accepted NP hang retry evidence differs")
            failures = 0
    if failures >= HANG_FAILURE_LIMIT:
        raise RuntimeError("three consecutive NP hang runs require a changed approach")
    return {"first_failed_marker_sha256": anchor, "consecutive_hang_runs": failures}


def retain_diagnostic_marker(marker, output, diagnostic, boot_id):
    """Preserve released failures, with explicit bounded SoC NP hang retries."""
    expected = diagnostic.get("board_marker_sha256") if diagnostic else None
    if not marker.exists():
        if expected is not None:
            raise RuntimeError("authorized board recovery marker is missing")
        return None
    if not expected:
        raise RuntimeError("board recovery marker remains; follow the incident ledger")
    original = marker.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    if digest != expected:
        raise RuntimeError("diagnostic board marker hash does not match")
    previous = json.loads(original)
    delayed = previous.get("delayed_seconds")
    retry = hang_retry_state(previous, diagnostic, digest, boot_id)
    if (
        previous.get("board_boot_id") != boot_id
        or (retry is None and any(previous.get(name) != [] for name in ("faults", "kernel_faults")))
        or any(previous.get(name) != [] for name in ("remaining_processes", "owners_delayed"))
        or previous.get("delayed_faults", []) != []
        or previous.get("pool_released") is not True
        or previous.get("gfp_released") is not True
        or type(delayed) not in (int, float)
        or not DELAY_SECONDS <= delayed < float("inf")
        or previous.get("monitor_error")
        or previous.get("final_check_error")
    ):
        raise RuntimeError("board marker does not establish fault-free release and delayed checks")
    retained = output / "retained-prior-board-marker.json"
    retained.write_bytes(original)
    result = {"path": str(retained), "sha256": digest}
    if retry is not None:
        result["soc_hang_retry"] = retry
    return result


def run(
    root, output, timeout, diagnostic=None, *, preparation_timeout=240, expected_bundle_sha256=None
):
    if expected_bundle_sha256 is None or sha256(root / "bundle.json") != expected_bundle_sha256:
        raise RuntimeError("selected bundle identity differs before board execution")
    execution_limit = inference_timeout_limit(read_json(root / "bundle.json"))
    if not 0 < timeout <= execution_limit:
        raise ValueError(f"inference deadline must be in (0, {execution_limit}] seconds")
    if not 0 < preparation_timeout <= 240:
        raise ValueError("preparation deadline must be in (0, 240] seconds")
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "clean": False,
        "board_boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "bundle_sha256": sha256(root / "bundle.json"),
        "preparation_limit_seconds": preparation_timeout,
        "inference_limit_seconds": timeout,
        "child_deadline_seconds": preparation_timeout + timeout,
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
        if marker.exists() and diagnostic is None:
            raise RuntimeError("board recovery marker remains; follow the incident ledger")
        # Installed libraries are checked before native loading. This does not load the driver.
        manifest = verify_bundle(root, verify_payload=False)
        reserve_kib = memory_reserve_kib(manifest)
        result["linux_memory_reserve_mib"] = reserve_kib // 1024
        result["fault_pattern"] = fault_pattern(manifest)
        fault = re.compile(result["fault_pattern"], re.I)
        if diagnostic is not None:
            if (
                diagnostic["board_boot_id"] != result["board_boot_id"]
                or diagnostic["bundle_sha256"] != result["bundle_sha256"]
                or not diagnostic["authorization"]
                or diagnostic["timeout"] != timeout
                or diagnostic.get("preparation_timeout") != preparation_timeout
            ):
                raise RuntimeError("diagnostic authorization does not match this run")
            acknowledged = diagnostic["kernel_faults"]
            if not all(re.match(r"^\[\s*\d+\.\d+\]", line) for line in acknowledged):
                raise ValueError("diagnostic baseline requires timestamped kernel occurrences")
            result["diagnostic"] = diagnostic
        retained = retain_diagnostic_marker(marker, output, diagnostic, result["board_boot_id"])
        if retained is not None:
            if "soc_hang_retry" in retained:
                result["soc_hang_retry"] = retained.pop("soc_hang_retry")
            result["retained_prior_board_marker"] = retained
        if new_faults(kernel(output, "kernel-before.log"), fault, acknowledged) or owners():
            raise RuntimeError("kernel fault or existing galcore owner")
        require_verified_deployment(root, manifest)
        result["payload_verification"] = "deployment-checksums-and-current-file-metadata"
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
            before["mem_available_kib"] < reserve_kib
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
                if state["mem_available_kib"] < reserve_kib:
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
            # A recovery exception authorizes execution; it does not request
            # per-call tracing. Keep diagnostic IO out of workload measurements.
            SPECFERRY_SDK_TIMING="off",
            VIV_VX_PROFILE="0",
            VP_ENABLE_PRINT="0",
        )
        result["child_started"] = True
        result.update(
            supervise(
                native_command(root, manifest, output),
                root / "runtime",
                environment,
                output,
                timeout,
                monitor,
                preparation_timeout=preparation_timeout,
                execution_limit=execution_limit,
            )
        )
        monitor()
    except Exception as error:
        result["error"] = str(error)
        if (output / "child.json").is_file():
            result.update(read_json(output / "child.json"))
    finally:
        if result["child_started"]:
            faults_before_delay = len(result["faults"])
            delayed = time.monotonic()
            while time.monotonic() - delayed < DELAY_SECONDS:
                try:
                    monitor()
                except Exception as error:
                    result["monitor_error"] = str(error)
                time.sleep(0.1)
            result["delayed_seconds"] = time.monotonic() - delayed
            result["delayed_faults"] = result["faults"][faults_before_delay:]
        try:
            final = kernel(output, "kernel-delayed.log")
            result["kernel_faults"] = new_faults(final, fault, acknowledged) if fault else []
            if (
                result["child_started"]
                and result.get("soc_hang_retry")
                and (result["faults"] or result["kernel_faults"])
            ):
                # One failed run counts once, regardless of duplicate or multiple log events.
                result["soc_hang_retry"]["consecutive_hang_runs"] += 1
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
                and result.get("delayed_seconds", 0) >= DELAY_SECONDS
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
    retained_bytes = None
    retained_digest = result.get("diagnostic", {}).get("board_marker_sha256")
    if retained_digest:
        retained = output / "retained-prior-board-marker.json"
        retained_bytes = retained.read_bytes()
        if hashlib.sha256(retained_bytes).hexdigest() != retained_digest:
            raise RuntimeError("retained board marker differs; retain recovery marker")
        # Retrying collection after an accepted diagnostic must leave the restored
        # older gate untouched, while still rejecting any unrelated newer marker.
        if marker.exists() and sha256(marker) == retained_digest:
            prior = output / "accepted-marker.json"
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
        re.compile(fault_pattern(result), re.I),
        result.get("diagnostic", {}).get("kernel_faults", []),
    ):
        raise RuntimeError("kernel fault after delayed check; retain recovery marker")
    if marker.exists() and prior == marker:
        marker.rename(output / "accepted-marker.json")
    if retained_bytes is not None:
        temporary = marker.with_suffix(marker.suffix + ".tmp")
        temporary.write_bytes(retained_bytes)
        temporary.replace(marker)
        retry = result.get("soc_hang_retry")
        if retry is not None:
            # A numerically accepted run breaks the streak without clearing its old gate.
            save(
                STATE / "soc-hang-retry-accepted.json",
                dict(
                    board_boot_id=result["board_boot_id"],
                    first_failed_marker_sha256=retry["first_failed_marker_sha256"],
                    restored_marker_sha256=retained_digest,
                    result_path=str(output / "result.json"),
                    result_sha256=expected_digest,
                ),
            )


def main():
    def interrupted(signum, frame):
        raise RuntimeError("board supervisor interrupted by signal " + str(signum))

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    command, root = sys.argv[1], Path(sys.argv[2]).resolve()
    if command == "verify":
        manifest = verify_deployment(root)
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
                preparation_timeout=float(sys.argv[6]) if len(sys.argv) > 6 else 240,
                expected_bundle_sha256=sys.argv[7] if len(sys.argv) > 7 else None,
            )
        elif command == "accept":
            accept(root, sys.argv[3])
        else:
            raise ValueError("unknown board supervisor command")


if __name__ == "__main__":
    main()
