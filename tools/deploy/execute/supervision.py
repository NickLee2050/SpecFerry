"""Device-independent adapter for the native process supervisor (also deployed standalone)."""

import json
import os
import subprocess
import time


def supervise(
    command,
    cwd,
    environment,
    output,
    timeout,
    monitor,
    *,
    supervisor_binary,
    save,
    preparation_timeout=None,
    execution_limit=300,
):
    if execution_limit not in (300, 600) or (
        execution_limit == 600 and preparation_timeout is None
    ):
        raise ValueError("extended execution requires a phased model workload")
    if not 0 < timeout <= execution_limit:
        raise ValueError(f"execution deadline must be in (0, {execution_limit}] seconds")
    if preparation_timeout is not None and not 0 < preparation_timeout <= 240:
        raise ValueError("preparation deadline must be in (0, 240] seconds")
    # The native process owns fork/exec, process groups, the private ready pipe and deadlines.
    # Python owns deployment/recovery policy; blocking this callback cannot delay termination.
    started = time.monotonic()
    result = dict(
        child_start_monotonic_seconds=started,
        child_start_wall_seconds=time.time(),
        child_started=False,
        remaining_processes=[],
        actions=[],
        stop_reason=None,
    )
    supervisor = None
    initial = b""
    try:
        with (output / "supervisor-native.log").open("wb") as log:
            supervisor = subprocess.Popen(
                [
                    str(supervisor_binary),
                    str(output),
                    str(preparation_timeout if preparation_timeout is not None else -1),
                    str(timeout),
                    str(execution_limit),
                    str(cwd),
                    "--",
                    *command,
                ],
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
            )
            os.set_blocking(supervisor.stdout.fileno(), False)
            os.set_blocking(supervisor.stdin.fileno(), False)
            published = False
            while supervisor.poll() is None:
                if not published:
                    try:
                        initial += os.read(supervisor.stdout.fileno(), 4096)
                    except BlockingIOError:
                        pass
                    if len(initial) > 4096:
                        raise RuntimeError("oversized native supervisor startup record")
                    if b"\n" in initial:
                        result = json.loads(initial)
                        save(output / "child-start.json", result)
                        published = True
                reason = monitor()
                if reason:
                    try:
                        os.write(
                            supervisor.stdin.fileno(),
                            (str(reason)[:2000].replace("\n", " ") + "\n").encode(),
                        )
                    except (BrokenPipeError, BlockingIOError):
                        pass
                    break
                time.sleep(0.05)
    finally:
        if supervisor is not None:
            # EOF is a fail-closed stop request, including exceptions and interrupted policy IO.
            try:
                supervisor.stdin.close()
            except BrokenPipeError:
                pass
            try:
                supervisor.wait(timeout=12)
            except subprocess.TimeoutExpired:
                # SIGTERM asks the native guard to terminate the owned process group. Never
                # SIGKILL the guard here: an uninterruptible child must remain visible/gated.
                supervisor.terminate()
                raise RuntimeError("native supervisor did not finish bounded release")
            finally:
                supervisor.stdout.close()
            if not (output / "child.json").is_file():
                result["error"] = "native supervisor exited without child evidence"
                save(output / "child.json", result)
            final = json.loads((output / "child.json").read_text())
            if supervisor.returncode or "exit" not in final:
                raise RuntimeError("native supervisor failed; retain incomplete execution evidence")
        else:
            save(output / "child.json", result)
    return json.loads((output / "child.json").read_text())
