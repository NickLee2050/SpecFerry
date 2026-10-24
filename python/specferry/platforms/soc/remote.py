"""SSH transport and host acceptance for the independent board supervisor."""

import json
import os
import re
import shlex
import subprocess
import tarfile
import tempfile
import time
import uuid
from pathlib import Path, PurePosixPath

from specferry.models.opt.graph_validation import evaluate, evaluate_benchmark
from specferry.validation.device import RECOVERY_ROOT, device_lock, host_boot_id
from specferry.validation.generation_metrics import summarize
from specferry.validation.recovery_controls import FAULT

from .board_runner import read_json, save, sha256, verify_bundle


def ssh(host, command, **kwargs):
    if not re.fullmatch(r"[A-Za-z0-9_.@-]+", host) or host.startswith("-"):
        raise ValueError("invalid SSH host alias")
    return subprocess.run(
        [
            "ssh",
            "-4",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            "-o",
            "ProxyCommand=none",
            host,
            command,
        ],
        check=True,
        timeout=kwargs.pop("timeout", 15),
        **kwargs,
    )


def remote_path(value):
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or not path.is_relative_to("/root/specferry-soc")
        or path == PurePosixPath("/root/specferry-soc")
    ):
        raise ValueError("remote paths must be below /root/specferry-soc")
    return str(path)


def python_command(code):
    return shlex.join(["python3", "-c", code])


def deploy(bundle, host, destination):
    bundle = bundle.resolve()
    destination = remote_path(destination)
    manifest = verify_bundle(bundle, check_sdk=False)
    # Reserve staging space before writing; never merge into an existing deployment.
    code = (
        "from pathlib import Path; import shutil; "
        f"p=Path({destination!r}); p.parent.mkdir(parents=True, exist_ok=True); "
        f"assert shutil.disk_usage(p.parent).free > {manifest['payload_bytes'] + 64 * 1024**2}; "
        "p.mkdir()"
    )
    ssh(host, python_command(code))
    with subprocess.Popen(
        ["tar", "--mtime=@0", "-C", str(bundle), "-cf", "-", "."], stdout=subprocess.PIPE
    ) as archive:
        try:
            ssh(
                host,
                shlex.join(["tar", "-xf", "-", "-C", destination]),
                stdin=archive.stdout,
                timeout=300,
            )
        finally:
            archive.stdout.close()
        if archive.wait():
            raise RuntimeError("bundle archive failed; remote staging is incomplete")
    script = str(PurePosixPath(destination) / "board_runner.py")
    expected = sha256(bundle / "board_runner.py")
    bootstrap = (
        "import hashlib,runpy,sys; "
        f"p={script!r}; assert hashlib.sha256(open(p,'rb').read()).hexdigest()=={expected!r}; "
        f"sys.argv=[p,'verify',{destination!r}]; runpy.run_path(p,run_name='__main__')"
    )
    verified = json.loads(
        ssh(host, python_command(bootstrap), capture_output=True, text=True, timeout=300).stdout
    )
    if verified.get("bundle_sha256") != sha256(bundle / "bundle.json"):
        raise RuntimeError("remote manifest differs after transfer")
    record = dict(verified, host=host, remote=destination, payload_bytes=manifest["payload_bytes"])
    save(bundle.with_name(bundle.name + "-deployment.json"), record)
    return record


def host_health(output, name):
    observed = subprocess.run(
        ["sudo", "-n", "journalctl", "-k", "-b", "--no-pager", "-o", "short-monotonic"],
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    (output / name).write_text(observed.stdout)
    if "permission" in observed.stderr.lower() or "not seeing messages" in observed.stderr.lower():
        raise RuntimeError("host kernel journal is inaccessible")
    if (
        FAULT.search(observed.stdout)
        or Path("/dev/galcore").exists()
        or Path("/sys/module/galcore").exists()
    ):
        raise RuntimeError("host kernel fault or EP isolation is not established")


def numerical_acceptance(bundle, board):
    options = read_json(bundle / "bundle.json")["request_options"]
    metadata = read_json(bundle / "request/reference.json")
    common = dict(
        variant="compiled-fullgraph",
        decoder_group_size=1,
        prefill_head_policy="every-token",
        kv_transfer="mapped-slot",
    )
    if options.get("benchmark", "0") == "1":
        checks, generations = evaluate_benchmark(
            board / "native",
            metadata,
            1,
            warmups=int(options.get("warmups", 0)),
            repeats=int(options.get("repeats", 1)),
            **common,
        )
    else:
        readback = options.get("readback", "none")
        checks, generations = evaluate(
            bundle / "request",
            board / "native",
            metadata,
            1,
            cache_readback=readback,
            cache_layer_range=[int(options["cache_first_layer"]), int(options["cache_last_layer"])]
            if readback == "kv"
            else None,
            readback_transport=options.get("readback_transport", "sdk"),
            **common,
        )
    profile = read_json(board / "native/platform.json")
    checks["platform"] = (
        profile.get("target") == "soc" and profile.get("sdk_identity") == "np101-soc-riscv64"
    )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "performance": summarize(generations) if all(checks.values()) else None,
    }


def collect(bundle, output):
    """Resume evidence retrieval after an SSH interruption; never launches another child."""
    session = read_json(output / "session.json")
    if session["bundle_sha256"] != sha256(bundle / "bundle.json"):
        raise ValueError("collection bundle differs from the launched run")
    verify_bundle(bundle, check_sdk=False)
    marker = RECOVERY_ROOT / "np101-recovery-required.json"
    with device_lock(RECOVERY_ROOT, recovery_control=True):
        if read_json(marker).get("evidence") != str(output.resolve()):
            raise RuntimeError("collection cannot clear an unrelated recovery marker")
        if not (output / "board").exists():
            with tempfile.TemporaryFile() as archive:
                ssh(
                    session["host"],
                    shlex.join(["tar", "-C", session["remote_output"], "-cf", "-", "."]),
                    stdout=archive,
                    timeout=90,
                )
                archive.seek(0)
                with tempfile.TemporaryDirectory(dir=output, prefix=".collect-") as staging:
                    with tarfile.open(fileobj=archive) as contents:
                        contents.extractall(staging, filter="data")
                    # An unfinished supervisor must remain collectible later.
                    read_json(Path(staging) / "result.json")
                    Path(staging).rename(output / "board")
        board = read_json(output / "board/result.json")
        passed = (
            board.get("clean") is True and board.get("bundle_sha256") == session["bundle_sha256"]
        )
        numerical = numerical_acceptance(bundle, output / "board") if passed else {"passed": False}
        host_health(output, "host-kernel-after.log")
        summary = {
            "passed": False,
            "checks_passed": passed and numerical["passed"],
            "board": board,
            "numerical": numerical,
            "host_boot_id": host_boot_id(),
            "scope": "formal OPT fullgraph C64/M1 mapped-slot",
        }
        save(output / "summary.json", summary)
        if summary["checks_passed"]:
            script = str(PurePosixPath(session["remote_bundle"]) / "board_runner.py")
            ssh(
                session["host"],
                shlex.join(
                    [
                        "python3",
                        script,
                        "accept",
                        session["remote_output"],
                        sha256(output / "board/result.json"),
                    ]
                ),
            )
            if session.get("diagnostic"):
                retained = output / "retained-prior-host-marker.json"
                if sha256(retained) != session["diagnostic"]["host_marker_sha256"]:
                    raise RuntimeError("retained recovery evidence changed")
            if session.get("diagnostic"):
                (output / "accepted-host-marker.json").write_bytes(marker.read_bytes())
                replacement = marker.with_suffix(".restore.tmp")
                replacement.write_bytes(retained.read_bytes())
                replacement.replace(marker)
            else:
                marker.rename(output / "accepted-host-marker.json")
            summary["passed"] = True
            summary["recovery_cleared"] = not bool(session.get("diagnostic"))
            save(output / "summary.json", summary)
        else:
            record = read_json(marker)
            record.update(
                reason="SoC execution or acceptance failed",
                board_boot_id=board.get("board_boot_id"),
                kernel_faults=board.get("kernel_faults", []),
                final_health=board,
            )
            save(marker, record)
            save(
                output / "incident.json",
                {
                    "observed": record,
                    "established_cause": None,
                    "next_action": "Stop board work and review SDK/kernel evidence; follow recovery ledger",
                },
            )
        return summary


def run(bundle, host, destination, output, timeout, diagnostic=None):
    if not 0 < timeout <= 300:
        raise ValueError("child deadline must be in (0, 300] seconds")
    bundle, output = bundle.resolve(), output.resolve()
    destination = remote_path(destination)
    verify_bundle(bundle, check_sdk=False)
    with device_lock(RECOVERY_ROOT, recovery_control=diagnostic is not None):
        marker = RECOVERY_ROOT / "np101-recovery-required.json"
        if diagnostic is not None and (
            sha256(marker) != diagnostic["host_marker_sha256"]
            or sha256(bundle / "bundle.json") != diagnostic["bundle_sha256"]
            or not diagnostic["authorization"]
            or diagnostic["timeout"] != timeout
        ):
            raise RuntimeError("diagnostic authorization does not match the retained gate/bundle")
        output.mkdir(parents=True, exist_ok=False)
        host_health(output, "host-kernel-before.log")
        remote_output = "/root/specferry-soc/runs/" + uuid.uuid4().hex
        session = dict(
            host=host,
            remote_bundle=destination,
            remote_output=remote_output,
            bundle_sha256=sha256(bundle / "bundle.json"),
            diagnostic=diagnostic,
        )
        save(output / "session.json", session)
        if diagnostic is not None:
            (output / "retained-prior-host-marker.json").write_bytes(marker.read_bytes())
        save(
            marker,
            {
                "reason": "SoC remote run/acceptance incomplete",
                "boot_id": host_boot_id(),
                "pid_namespace": os.readlink("/proc/self/ns/pid"),
                "members": [],
                "evidence": str(output),
                "remote_output": remote_output,
                "prior_marker": str(output / "retained-prior-host-marker.json")
                if diagnostic
                else None,
            },
        )
        script = str(PurePosixPath(destination) / "board_runner.py")
        # The detached process owns its watchdog. Losing this SSH does not kill the monitor.
        command = ["python3", script, "run", destination, remote_output, str(timeout)]
        if diagnostic is not None:
            command.append(json.dumps(diagnostic))
        launch = (
            "import subprocess,hashlib,pathlib; "
            f"assert hashlib.sha256(pathlib.Path({script!r}).read_bytes()).hexdigest()=={sha256(bundle / 'board_runner.py')!r}; "
            f"p=pathlib.Path({remote_output!r}); p.parent.mkdir(parents=True,exist_ok=True); "
            "log=open(str(p)+'.supervisor.log','w'); "
            f"c=subprocess.Popen({command!r},stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True); print(c.pid)"
        )
        try:
            try:
                pid = ssh(
                    host, python_command(launch), capture_output=True, text=True
                ).stdout.strip()
                save(output / "supervisor.json", {"remote_pid": int(pid)})
            except (subprocess.SubprocessError, OSError) as error:
                # Submission may have reached the board. Discover this run; never resubmit.
                save(output / "launch-transport-error.json", {"error": str(error)})
            # File/SDK hashing happens before the native child. Its SDK deadline stays <=300s.
            deadline = time.monotonic() + 300 + timeout + 60
            while time.monotonic() < deadline:
                code = f"from pathlib import Path; print(int(Path({remote_output + '/result.json'!r}).is_file()))"
                try:
                    if (
                        ssh(
                            host, python_command(code), capture_output=True, text=True
                        ).stdout.strip()
                        == "1"
                    ):
                        break
                except (subprocess.SubprocessError, OSError) as error:
                    save(
                        output / "transport-error.json",
                        {"error": str(error), "remote_output": remote_output},
                    )
                    time.sleep(5)
                    continue
                time.sleep(5)
            else:
                raise RuntimeError(
                    "supervisor result not available; preserve marker and collect evidence"
                )
        except BaseException as error:
            save(output / "interrupted.json", {"error": str(error), "remote_output": remote_output})
            raise
    for attempt in range(3):
        try:
            return collect(bundle, output)
        except (subprocess.SubprocessError, OSError) as error:
            save(output / "collection-transport-error.json", {"error": str(error)})
            if attempt == 2:
                raise
            time.sleep(5)
