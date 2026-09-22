"""Explicit contracts for the retained OPT-350M checkpoint and component slices."""

import math
from dataclasses import dataclass

MODEL_ID = "facebook/opt-350m"
REVISION = "08ab08cc4b72ff5593870b5d527cf4230323703c"
CHECKPOINT_SHA256 = "a5223ae6f3c26c6d90003f96a6bcd9a4aaaef0d36fca6469112efeeb985f2842"
EPSILON = 1e-5  # nn.LayerNorm default in the pinned official OPT implementation.
TOLERANCES = {
    "cpu": {"atol": 0.02, "rtol": 0.01},
    "device": {"atol": 0.08, "rtol": 0.02},
}


@dataclass(frozen=True)
class Components:
    hidden: int
    intermediate: int
    heads: int
    layers: int
    capacity: int = 512
    epsilon: float = EPSILON

    def __post_init__(self):
        if any(
            type(x) is not int or not 0 < x < 2**32
            for x in (self.hidden, self.intermediate, self.heads, self.layers, self.capacity)
        ):
            raise ValueError("OPT dimensions must be positive uint32 values")
        if (
            self.hidden % self.heads
            or self.capacity > 512
            or not math.isfinite(self.epsilon)
            or self.epsilon <= 0
        ):
            raise ValueError("unsupported OPT component configuration")
        # The current FCL path reads each matrix in one bounded block.
        if self.hidden * max(self.hidden, self.intermediate) * 2 > 8 * 1024**2:
            raise ValueError("OPT projection exceeds the validated 8 MiB matrix size")

    @property
    def head_dim(self):
        return self.hidden // self.heads

    def select(self, layers):
        selected = tuple(layers)
        if (
            not selected
            or len(set(selected)) != len(selected)
            or any(type(i) is not int or not 0 <= i < self.layers for i in selected)
        ):
            raise ValueError("invalid OPT layer selection")
        return selected

    def state_bytes(self, selected):
        return len(self.select(selected)) * self.capacity * self.hidden * 4

    def write(self, path):
        path.write_text(
            "specferry-opt-components 1\n"
            f"{self.hidden} {self.intermediate} {self.heads} {self.layers} {self.capacity} {self.epsilon}\n"
        )


def validate_config(config):
    expected = {
        "model_type": "opt",
        "hidden_size": 1024,
        "ffn_dim": 4096,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "word_embed_proj_dim": 512,
        "vocab_size": 50272,
        "max_position_embeddings": 2048,
        "activation_function": "relu",
        "do_layer_norm_before": False,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("configuration differs from the verified OPT-350M checkpoint")
    if any(
        config.get(key, True) is not True
        for key in ("enable_bias", "layer_norm_elementwise_affine", "tie_word_embeddings")
    ):
        raise ValueError("OPT-350M requires affine normalization, biases and tied embeddings")
    return Components(1024, 4096, 16, 24)


def layer_shapes(config):
    hidden, intermediate = config.hidden, config.intermediate
    result = {}
    for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
        result[f"self_attn.{projection}.weight"] = [hidden, hidden]
        result[f"self_attn.{projection}.bias"] = [hidden]
    for name in ("self_attn_layer_norm", "final_layer_norm"):
        result[f"{name}.weight"] = [hidden]
        result[f"{name}.bias"] = [hidden]
    result.update(
        {
            "fc1.weight": [intermediate, hidden],
            "fc1.bias": [intermediate],
            "fc2.weight": [hidden, intermediate],
            "fc2.bias": [hidden],
        }
    )
    return result


def checkpoint_shapes(config):
    components = validate_config(config)
    result = {
        "decoder.embed_tokens.weight": [50272, 512],
        "decoder.embed_positions.weight": [2050, 1024],
        "decoder.project_in.weight": [1024, 512],
        "decoder.project_out.weight": [512, 1024],
    }
    for layer in range(components.layers):
        result.update(
            {
                f"decoder.layers.{layer}.{name}": shape
                for name, shape in layer_shapes(components).items()
            }
        )
    return result
