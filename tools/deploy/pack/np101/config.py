"""Preparation options; artifact identity and device policy remain in bundle validation."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

DECODER_LAYOUTS = ("separate", "fused-attention-norm")
PREPARATION_SCHEMA_VERSION = 1
PREPARATION_PATHS = ("binary", "deployment", "package", "request", "sdk", "prefill_package")
# JSON uses the existing request_options names, while the CLI keeps its public spelling.
QWEN_OPTION_ARGUMENTS = {
    "projections": "qwen_projections",
    "kv_write_mode": "qwen_kv_write_mode",
    "ffn_batch_tokens": "qwen_ffn_batch",
    "delta_batch_tokens": "qwen_delta_batch",
    "decoder_layout": "qwen_decoder_layout",
    "state_commit": "qwen_state_commit",
}


@dataclass(frozen=True)
class QwenExecutionOptions:
    # Keep the existing manifest field order as well as its values.
    state_commit: str = "combined"
    projections: str = "shared-fcl"
    kv_write_mode: str = "copy-graph"
    ffn_batch_tokens: int = 1
    decoder_layout: str = "separate"
    delta_batch_tokens: int = 1

    def validate(self):
        if type(self.delta_batch_tokens) is not int or self.delta_batch_tokens not in (1, 8):
            raise ValueError("unsupported Qwen DeltaNet batch width")
        if self.delta_batch_tokens == 8 and (
            self.ffn_batch_tokens != 8
            or self.state_commit != "ffn"
            or self.decoder_layout != "fused-attention-norm"
            or self.projections != "shared-fcl"
            or self.kv_write_mode != "mapped"
        ):
            raise ValueError("M8 DeltaNet requires the fused M8 FFN-commit configuration")
        if self.projections not in ("shared-fcl", "matmul"):
            raise ValueError("unsupported Qwen projection backend")
        if self.state_commit not in ("combined", "ffn"):
            raise ValueError("unsupported Qwen state commit policy")
        if self.state_commit == "ffn" and self.ffn_batch_tokens == 1:
            raise ValueError("FFN state commit requires batched FFN")
        if self.kv_write_mode not in ("copy-graph", "mapped"):
            raise ValueError("unsupported Qwen KV write mode")
        if self.kv_write_mode == "mapped" and self.projections != "shared-fcl":
            raise ValueError("mapped Qwen KV requires shared FCL projections")
        if type(self.ffn_batch_tokens) is not int or self.ffn_batch_tokens not in (1, 8, 16):
            raise ValueError("unsupported Qwen FFN batch width")
        if self.decoder_layout not in DECODER_LAYOUTS:
            raise ValueError("unsupported Qwen decoder layout")
        if self.decoder_layout != "separate" and self.ffn_batch_tokens == 1:
            raise ValueError("optimized Qwen decoder layout requires batched FFN")
        if self.ffn_batch_tokens > 1 and (
            self.projections != "shared-fcl" or self.kv_write_mode != "mapped"
        ):
            raise ValueError("batched Qwen FFN requires mapped shared FCL projections")

    @classmethod
    def from_mapping(cls, values):
        if not isinstance(values, dict):
            raise ValueError("request_options must be a JSON object")
        unknown = values.keys() - QWEN_OPTION_ARGUMENTS.keys()
        if unknown:
            raise ValueError("unknown Qwen preparation option: " + ", ".join(sorted(unknown)))
        for key, value in values.items():
            expected = int if key in ("ffn_batch_tokens", "delta_batch_tokens") else str
            if type(value) is not expected:
                raise ValueError("invalid type for Qwen preparation option: " + key)
        result = cls(**values)
        result.validate()
        return result

    def as_mapping(self):
        return asdict(self)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate configuration key: " + key)
        result[key] = value
    return result


def read_preparation_config(path):
    """Read only preparation inputs, never SDK hashes, memory limits or retry policy."""
    path = Path(path).resolve()
    with path.open() as source:
        config = json.load(source, object_pairs_hook=_unique_object)
    if not isinstance(config, dict):
        raise ValueError("preparation configuration must be a JSON object")
    allowed = {"schema_version", "model_family", "request_options", *PREPARATION_PATHS}
    unknown = config.keys() - allowed
    if unknown:
        raise ValueError("unknown preparation field: " + ", ".join(sorted(unknown)))
    if (
        type(config.get("schema_version")) is not int
        or config["schema_version"] != PREPARATION_SCHEMA_VERSION
    ):
        raise ValueError("unsupported preparation schema_version")
    if config.get("model_family") not in ("opt", "qwen3_5"):
        raise ValueError("preparation model_family must be opt or qwen3_5")
    for name in PREPARATION_PATHS:
        if name in config:
            value = config[name]
            if not isinstance(value, str) or not value.strip():
                raise ValueError("preparation path must be a nonempty string: " + name)
            # File-relative paths make a saved plan independent of the calling shell's cwd.
            config[name] = (path.parent / value).resolve()
    options = config.get("request_options", {})
    if not isinstance(options, dict):
        raise ValueError("request_options must be a JSON object")
    if config["model_family"] == "opt":
        if options:
            raise ValueError("OPT execution options come from the validated request")
    else:
        # Validate the file itself, even if CLI overrides would hide a bad value.
        QwenExecutionOptions.from_mapping(options)

    return config


def resolve_preparation_arguments(args):
    """CLI overrides file values; absent values retain the existing model defaults."""
    config = read_preparation_config(args.config) if args.config else {}
    configured_model = {"opt": "opt", "qwen3_5": "qwen"}.get(config.get("model_family"))
    if args.model and configured_model and args.model != configured_model:
        raise ValueError("--model differs from configuration model_family")
    args.model = args.model or configured_model or "opt"
    for name in PREPARATION_PATHS:
        if getattr(args, name) is None and name in config:
            setattr(args, name, config[name])
    for name in ("package", "request", "sdk"):
        if getattr(args, name) is None:
            raise ValueError("provide --" + name + " or its preparation configuration field")
    options = dict(config.get("request_options", {}))
    for key, argument in QWEN_OPTION_ARGUMENTS.items():
        value = getattr(args, argument, None)
        if value is not None:
            options[key] = value
    if args.model == "qwen":
        args.execution_options = QwenExecutionOptions.from_mapping(options)
        if (args.execution_options.delta_batch_tokens == 8) != (args.prefill_package is not None):
            raise ValueError("M8 DeltaNet requires --prefill-package; M1 must omit it")
    elif options:
        raise ValueError("Qwen execution options require --model qwen")
    return args
