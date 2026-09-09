#!/usr/bin/env python3
"""Collect host, toolchain, and NP101 device facts without exposing credentials."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=30)
        return {
            "command": args,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": args, "error": str(error)}


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError as error:
        return {"unavailable": str(error)}


def collect(include, libraries):
    root = Path(__file__).resolve().parents[1]
    pci = []
    for device in sorted(Path("/sys/bus/pci/devices").glob("*")):
        if read(device / "vendor") == "0x0709":
            pci.append(
                {
                    "address": device.name,
                    "vendor": read(device / "vendor"),
                    "device": read(device / "device"),
                    "revision": read(device / "revision"),
                    "driver": str((device / "driver").resolve())
                    if (device / "driver").exists()
                    else None,
                    "resource": read(device / "resource"),
                }
            )
    device = Path("/dev/galcore")
    node = {"path": str(device), "exists": device.exists()}
    if device.exists():
        s = device.stat()
        node.update(
            character_device=stat.S_ISCHR(s.st_mode),
            major=os.major(s.st_rdev),
            minor=os.minor(s.st_rdev),
            readable=os.access(device, os.R_OK),
            writable=os.access(device, os.W_OK),
        )
    library_info = []
    for name in (
        "libovxlib.so",
        "libOpenVX.so",
        "libGAL.so",
        "libArchModelSw.so",
        "libNNArchPerf.so",
    ):
        path = libraries / name
        entry = {"path": str(path), "realpath": str(path.resolve()), "exists": path.is_file()}
        if path.is_file():
            entry.update(
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                elf=command(["readelf", "-h", str(path)]),
                dependencies=command(["readelf", "-d", str(path)]),
            )
        library_info.append(entry)
    free = shutil.disk_usage(root).free
    return {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "python_executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "cpu": command(["lscpu"]),
        "memory": read("/proc/meminfo"),
        "disk_free_bytes": free,
        "disk_reserve_8gib_pass": free >= 8 * 1024**3,
        "sdk_include_dir": str(include),
        "sdk_library_dir": str(libraries),
        "sdk_version_header": read(include / "acuity-ovxlib-dev/vsi_nn_version.h"),
        "libraries": library_info,
        "tools": {
            name: command(args)
            for name, args in {
                "c_compiler": ["gcc", "--version"],
                "cxx_compiler": ["g++", "--version"],
                "compiler_target": ["g++", "-dumpmachine"],
                "cmake": ["cmake", "--version"],
                "strace": ["strace", "--version"],
            }.items()
        },
        "pci_devices": pci,
        "device_node": node,
        "galcore_module_loaded": Path("/sys/module/galcore").exists(),
        "driver_parameters": {
            name: read(Path("/sys/module/galcore/parameters") / name)
            for name in ("contiguousSize", "externalSize", "physSize", "enableNN", "type")
        },
        "runtime_options": {
            name: os.environ.get(name)
            for name in ("VIV_VX_ENABLE_PRINT_TARGET", "VIV_VX_PROFILE", "VIV_VX_ENABLE_SHADER")
        },
        "available_device_memory_bytes": None,
        "hardware_execution_proven": False,
        "note": "PCI binding identifies hardware; runtime command/completion evidence is still required.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sdk-include", type=Path, default=Path("/usr/inc"))
    p.add_argument("--sdk-lib", type=Path, default=Path("/usr/lib/ljmicro"))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = collect(args.sdk_include.resolve(), args.sdk_lib.resolve())
    (args.output / "environment.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "hardware-evidence.md").write_text(
        "# NP101 hardware and environment evidence\n\n"
        f"- Host: `{result['hostname']}` / `{result['machine']}`.\n"
        f"- PCI device(s): `{json.dumps(result['pci_devices'])}`.\n"
        f"- galcore loaded: `{result['galcore_module_loaded']}`; device node: `{json.dumps(result['device_node'])}`.\n"
        "- Run `scripts/check_np101_conv_relu_pool.py` with a freshly built binary to collect SDK output and device IO trace.\n"
        "- Hardware presence, HAL pool size, and a target label are not proof that every node ran on the NPU.\n"
        "- This environment check does not measure available model allocation capacity; do not substitute the advertised 4 GB.\n"
        "- No driver loading, reboot, root-only counter access, or onboard CPU deployment is performed.\n"
    )
    print(
        json.dumps(
            {
                "environment": str(args.output / "environment.json"),
                "disk_reserve_pass": result["disk_reserve_8gib_pass"],
                "pci_devices": len(result["pci_devices"]),
            }
        )
    )
    return 0 if result["disk_reserve_8gib_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
