"""Component contracts shared by Qwen references, native fixtures and accounting."""

import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Components:
    hidden: int
    intermediate: int
    query_heads: int
    kv_heads: int
    head_dim: int
    capacity: int
    rotary_dim: int
    rope_theta: float
    epsilon: float
    delta_heads: int
    key_dim: int
    value_dim: int
    convolution_width: int
    layer_types: tuple[str, ...]

    def __post_init__(self):
        dimensions = (
            self.hidden,
            self.intermediate,
            self.query_heads,
            self.kv_heads,
            self.head_dim,
            self.capacity,
            self.rotary_dim,
            self.delta_heads,
            self.key_dim,
            self.value_dim,
            self.convolution_width,
        )
        if any(type(value) is not int or not 0 < value <= 2**32 - 1 for value in dimensions):
            raise ValueError("component dimensions must be positive uint32 values")
        if (
            self.query_heads % self.kv_heads
            or self.rotary_dim % 2
            or self.rotary_dim > self.head_dim
            or self.capacity > 512
            or self.convolution_width < 2
            or not math.isfinite(self.epsilon)
            or self.epsilon <= 0
            or not math.isfinite(self.rope_theta)
            or self.rope_theta <= 0
            or not self.layer_types
            or set(self.layer_types) - {"delta", "attention"}
        ):
            raise ValueError("unsupported component configuration")
        if self.channels > 2**32 - 1 or 2 * self.query_heads * self.head_dim > 2**32 - 1:
            raise ValueError("component dimensions overflow")

    @classmethod
    def from_text_config(cls, config, capacity=512):
        if config["linear_num_key_heads"] != config["linear_num_value_heads"]:
            raise ValueError("only equal DeltaNet key/value head counts are validated")
        rope = config["rope_parameters"]
        if (
            rope.get("rope_type", "default") != "default"
            or config.get("hidden_act", "silu") != "silu"
        ):
            raise ValueError("only default RoPE and SiLU Qwen components are implemented")
        if config.get("attention_bias", False) or not config.get("attn_output_gate", True):
            raise ValueError("only unbiased gated Qwen attention is implemented")
        return cls(
            config["hidden_size"],
            config["intermediate_size"],
            config["num_attention_heads"],
            config["num_key_value_heads"],
            config["head_dim"],
            capacity,
            int(config["head_dim"] * rope["partial_rotary_factor"]),
            rope["rope_theta"],
            config["rms_norm_eps"],
            config["linear_num_value_heads"],
            config["linear_key_head_dim"],
            config["linear_value_head_dim"],
            config["linear_conv_kernel_dim"],
            tuple(
                {"linear_attention": "delta", "full_attention": "attention"}[kind]
                for kind in config["layer_types"]
            ),
        )

    @property
    def channels(self):
        return self.delta_heads * (2 * self.key_dim + self.value_dim)

    @property
    def hidden_bytes(self):
        return self.hidden * 2

    @property
    def kv_slot_bytes(self):
        return 2 * self.kv_heads * self.head_dim * 2

    @property
    def recurrent_shape(self):
        return (1, self.delta_heads, self.key_dim, self.value_dim)

    @property
    def convolution_shape(self):
        return (1, self.channels, self.convolution_width)

    def state_bytes(self, layers):
        delta = sum(self.layer_types[layer] == "delta" for layer in layers)
        attention = len(layers) - delta
        return (
            delta
            * (2 * math.prod(self.recurrent_shape) * 4 + 2 * math.prod(self.convolution_shape) * 2)
            + attention * self.kv_slot_bytes * self.capacity
        )

    def transfers(self, layers):
        attention = sum(self.layer_types[layer] == "attention" for layer in layers)
        return {
            "uploads": 1 + 2 * attention,
            "upload_bytes": self.hidden_bytes + 8 * attention,
            "attention_layers": attention,
            "hidden_bytes": self.hidden_bytes,
            "kv_slot_bytes": self.kv_slot_bytes,
        }

    def write(self, directory: Path):
        values = asdict(self)
        parameters = [values[name] for name in self.__dataclass_fields__ if name != "layer_types"]
        (directory / "components.txt").write_text(
            "specferry-qwen-components 1\n"
            + " ".join(map(str, parameters))
            + "\n"
            + " ".join(self.layer_types)
            + "\n"
        )
        return values


# Explicit compatibility profile for historical Qwen fixtures/reports only.
# New runs derive Components from the verified deployment's text_config.
BASELINE = Components(
    1024,
    3584,
    8,
    2,
    256,
    512,
    64,
    10000000.0,
    1e-6,
    16,
    128,
    128,
    4,
    tuple("attention" if layer % 4 == 3 else "delta" for layer in range(24)),
)
