"""Transformer block module."""

import jax
from jaxtyping import Array, Float

from llm.layers.attention import Attention
from llm.layers.ffn import FFN
from llm.layers.norm import LayerNorm
from llm.utils import Module

DROPOUT_RATE = 0.1


def dropout(x: Float[Array, "..."], key: Array) -> Float[Array, "..."]:
    """Apply dropout."""
    mask = jax.random.bernoulli(key, 1 - DROPOUT_RATE, x.shape)
    return mask * x / (1 - DROPOUT_RATE)


class TransformerBlock(Module):
    """Single transformer block with pre-norm architecture."""

    norm1: LayerNorm
    attn: Attention
    norm2: LayerNorm
    ffn: FFN

    def __init__(self, id: str, key: Array):
        super().__init__(id, key)
        self.norm1 = LayerNorm(f"{id}.norm1", self.make_key("norm1"))
        self.attn = Attention(f"{id}.attn", self.make_key("attn"))
        self.norm2 = LayerNorm(f"{id}.norm2", self.make_key("norm2"))
        self.ffn = FFN(f"{id}.ffn", self.make_key("ffn"))

    def forward(
        self, x: Float[Array, "batch seq d_model"]
    ) -> Float[Array, "batch seq d_model"]:
        self.key, k1, k2 = jax.random.split(self.key, 3)
        x = x + dropout(self.attn(self.norm1(x)), k1)
        x = x + dropout(self.ffn(self.norm2(x)), k2)
        return x
