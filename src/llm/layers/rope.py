"""Rotary Position Embeddings (RoPE)."""

from functools import lru_cache

import jax.numpy as jnp
from jaxtyping import Array, Float


@lru_cache(maxsize=1)
def precompute_freqs(
    d_head: int, max_seq_len: int, theta: float = 10000.0
) -> tuple[Float[Array, "max_seq d_head_half"], Float[Array, "max_seq d_head_half"]]:
    """Precompute cos/sin rotation frequencies for RoPE."""
    # theta^(-2i/d) for i in 0..d/2
    freqs = 1.0 / (theta ** (jnp.arange(0, d_head, 2) / d_head))
    positions = jnp.arange(max_seq_len)
    # [max_seq] x [d_head/2] -> [max_seq, d_head/2]
    angles = jnp.outer(positions, freqs)
    return jnp.cos(angles), jnp.sin(angles)


def apply_rope(
    x: Float[Array, "batch seq heads d_head"],
    cos: Float[Array, "seq d_head_half"],
    sin: Float[Array, "seq d_head_half"],
) -> Float[Array, "batch seq heads d_head"]:
    """Apply rotary embeddings to x."""
    # Split into pairs
    x1 = x[..., ::2]  # [batch, seq, heads, d_head/2]
    x2 = x[..., 1::2]  # [batch, seq, heads, d_head/2]

    # Reshape for broadcasting: [1, seq, 1, d_head/2]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]

    # Apply rotation
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos

    # Interleave back: [batch, seq, heads, d_head]
    return jnp.stack([out1, out2], axis=-1).reshape(x.shape)
