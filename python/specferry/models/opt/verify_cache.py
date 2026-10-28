"""Package already validated SoC FFN caches; never compile or execute a device graph."""

import json
import shutil
import struct
import tempfile
from pathlib import Path

from specferry.data.checkpoint import sha256

from .export import verify_export

RECIPE = "batch-pointwise-f16-fp32-norm-v1"
WEIGHTS = (
    "fc1.weight",
    "fc1.bias",
    "fc2.weight",
    "fc2.bias",
    "final_layer_norm.weight",
    "final_layer_norm.bias",
)


def package_verify_cache(source, deployment, package, consumer, *, layers=12):
    """Bind an accepted cache prefix to one executable, SDK and complete weight identities."""
    if type(layers) is not int or layers not in (12, 19):
        raise ValueError("cache package requires twelve or nineteen layers")
    source, deployment, package, consumer = map(
        lambda path: Path(path).resolve(strict=True), (source, deployment, package, consumer)
    )
    verify_export(deployment)
    metadata = json.loads(source.read_text())
    model = json.loads((package / "manifest.json").read_text())
    acceptance = Path(metadata["acceptance"])
    evidence = json.loads(acceptance.read_text())
    if (
        metadata["schema_version"] != 1
        or metadata["recipe"] != RECIPE
        or metadata["capacity"] != 128
        or model["capacity"] not in (128, 256)
        or model["sdk"]["profile"] != "np101-soc-riscv64"
        or sha256(acceptance) != metadata["acceptance_sha256"]
        or evidence.get("passed") is not True
        or evidence.get("checks_passed") is not True
        or metadata["sdk"] != model["sdk"]["sha256"]
        or any(
            sha256(deployment / name) != digest
            for name, digest in metadata["deployment_sha256"].items()
        )
        or set(metadata["deployment_sha256"]) != {"weights.index", "deployment-manifest.json"}
    ):
        raise ValueError("cache source acceptance, deployment or SDK differs")
    passed = {row["layer"] for row in evidence["numerical"]["checks"] if row["passed"]}
    if not set(range(layers)) <= passed:
        raise ValueError("cache source lacks complete numerical acceptance")
    producer = Path(metadata["producer"])
    for binary in (producer, consumer):
        with binary.open("rb") as stream:
            header = stream.read(20)
        if (
            len(header) != 20
            or header[:6] != b"\x7fELF\x02\x01"
            or struct.unpack_from("<H", header, 18)[0] != 243
        ):
            raise ValueError("cache producer and consumer must be RISC-V64 ELF files")
    if sha256(producer) != metadata["producer_sha256"]:
        raise ValueError("cache producer identity differs")
    records = {
        record["name"]: record
        for record in json.loads((deployment / "deployment-manifest.json").read_text())["tensors"]
    }
    # The fixed recipe has only M16 hidden/parameter ports, no KV or position input.
    # Its accepted C128 producer is reusable at C256; pin the target geometry
    # separately and require complete target-model acceptance before claiming support.
    lines = [
        f"specferry-opt-verify-cache 1 {RECIPE} {layers} 1024 4096 16 {model['capacity']}",
        f'{sha256(consumer)} "cache-producer.bin" {metadata["producer_sha256"]}',
        f"np101-soc-riscv64 {len(metadata['sdk'])}",
        *(f"{name} {digest}" for name, digest in sorted(metadata["sdk"].items())),
    ]
    files = []
    for layer, entry in enumerate(metadata["graphs"][:layers]):
        binary = Path(entry["path"])
        expected = {
            f"decoder.layers.{layer}.{suffix}": records[f"decoder.layers.{layer}.{suffix}"][
                "sha256"
            ]
            for suffix in WEIGHTS
        }
        if (
            entry["layer"] != layer
            or entry["weight_sha256"] != expected
            or binary.is_symlink()
            or binary.suffix != ".nb"
            or sha256(binary) != entry["sha256"]
            or not 0 < binary.stat().st_size <= 64 * 1024**2
        ):
            raise ValueError(f"cache layer identity differs: {layer}")
        files.append(binary)
        lines.extend(
            [f"{layer} {json.dumps(binary.name)} {binary.stat().st_size} {entry['sha256']}"]
            + [expected[f"decoder.layers.{layer}.{suffix}"] for suffix in WEIGHTS]
        )
    if len(files) != layers:
        raise ValueError("cache package has an incomplete prefix")
    names = ("verify-cache", "cache-producer.bin", "verified-prefill-cache.txt")
    if any((package / name).exists() for name in names):
        raise FileExistsError("refusing to replace an existing cache package")
    with tempfile.TemporaryDirectory(prefix=".verify-cache-", dir=package) as temporary:
        pending = Path(temporary)
        for layer, binary in enumerate(files):
            directory = pending / "verify-cache" / f"layer-{layer}"
            directory.mkdir(parents=True)
            shutil.copyfile(binary, directory / binary.name)
            if sha256(directory / binary.name) != metadata["graphs"][layer]["sha256"]:
                raise ValueError("cache copy integrity check failed")
        shutil.copyfile(producer, pending / "cache-producer.bin")
        (pending / "verified-prefill-cache.txt").write_text("\n".join(lines) + "\n")
        # The manifest is the activation point; a partial deployment stays on the baseline.
        for name in names:
            (pending / name).rename(package / name)
    return dict(
        layers=layers,
        cache_bytes=sum(path.stat().st_size for path in files),
        consumer_sha256=sha256(consumer),
        producer_sha256=metadata["producer_sha256"],
        source_capacity=metadata["capacity"],
        runtime_capacity=model["capacity"],
    )
