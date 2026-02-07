"""Normalization modules."""

import jax.numpy as jnp
from jaxtyping import Array, Float

from llm.constants import D_MODEL
from llm.utils import Module

EPS = 1e-6


class LayerNorm(Module):
    """Root Mean Square Layer Normalization."""

    scale: Float[Array, "d_model"]
    bias: Float[Array, "d_model"]

    def __init__(self, id: str, key: Array):
        super().__init__(id, key)
        self.scale = self.safetensor("scale", (D_MODEL,), lambda s, _: jnp.ones(s))
        self.bias = self.safetensor("bias", (D_MODEL,), lambda s, _: jnp.zeros(s))

    def forward(
        self, x: Float[Array, "batch seq d_model"]
    ) -> Float[Array, "batch seq d_model"]:
        rms = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + EPS)
        return (x / rms) * self.scale + self.bias
