"""Feed-forward network module."""

import jax
from jaxtyping import Array, Float

from llm.constants import D_MODEL
from llm.utils import Module, he_init


class FFN(Module):
    """Feed-forward network with GELU activation."""

    W1: Float[Array, "d_model d_ff"]
    W2: Float[Array, "d_ff d_model"]

    def __init__(self, id: str, key: Array):
        super().__init__(id, key)
        self.W1 = self.safetensor("W1", (D_MODEL, 4 * D_MODEL), he_init)
        self.W2 = self.safetensor("W2", (4 * D_MODEL, D_MODEL), he_init)

    def forward(
        self, x: Float[Array, "batch seq d_model"]
    ) -> Float[Array, "batch seq d_model"]:
        return jax.nn.gelu(x @ self.W1, approximate=True) @ self.W2
