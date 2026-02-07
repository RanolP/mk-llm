"""Multi-head attention module."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Bool

from llm.constants import D_MODEL, MAX_SEQ_LEN
from llm.layers.rope import apply_rope, precompute_freqs
from llm.utils import Module, he_init

N_HEADS = 8
N_KV_HEADS = 4
D_HEAD = 64


CAUSAL_MASK: Bool[Array, "max_seq max_seq"] = jnp.tril(
    jnp.ones((MAX_SEQ_LEN, MAX_SEQ_LEN), dtype=bool)
)


class Attention(Module):
    """Multi-head attention with RoPE, GQA, and KV cache."""

    W_Q: Float[Array, "d_model n_heads*d_head"]
    W_K: Float[Array, "d_model n_kv_heads*d_head"]
    W_V: Float[Array, "d_model n_kv_heads*d_head"]
    W_O: Float[Array, "n_heads*d_head d_model"]
    kv_cache: (
        tuple[
            Float[Array, "batch n_kv_heads seq d_head"],
            Float[Array, "batch n_kv_heads seq d_head"],
        ]
        | None
    )

    def __init__(self, id: str, key: Array):
        super().__init__(id, key)
        assert N_HEADS % N_KV_HEADS == 0, "N_HEADS must be divisible by N_KV_HEADS"
        self.W_Q = self.safetensor("W_Q", (D_MODEL, N_HEADS * D_HEAD), he_init)
        self.W_K = self.safetensor("W_K", (D_MODEL, N_KV_HEADS * D_HEAD), he_init)
        self.W_V = self.safetensor("W_V", (D_MODEL, N_KV_HEADS * D_HEAD), he_init)
        self.W_O = self.safetensor("W_O", (N_HEADS * D_HEAD, D_MODEL), he_init)
        self.kv_cache = None

    def clear_cache(self):
        """Clear the KV cache."""
        self.kv_cache = None

    def forward(
        self, x: Float[Array, "batch seq d_model"]
    ) -> Float[Array, "batch seq d_model"]:
        batch, seq, _ = x.shape
        mask = CAUSAL_MASK[:seq, :seq]
        cos, sin = precompute_freqs(D_HEAD, MAX_SEQ_LEN)

        # --- KV Cache: position offset for RoPE ---
        start_pos = 0 if self.kv_cache is None else self.kv_cache[0].shape[2]

        # --- MHA: project Q, K, V ---
        q = (x @ self.W_Q).reshape(batch, seq, N_HEADS, D_HEAD)
        k = (x @ self.W_K).reshape(batch, seq, N_KV_HEADS, D_HEAD)
        v = (x @ self.W_V).reshape(batch, seq, N_KV_HEADS, D_HEAD)

        q = apply_rope(
            q, cos[start_pos : start_pos + seq], sin[start_pos : start_pos + seq]
        )
        k = apply_rope(
            k, cos[start_pos : start_pos + seq], sin[start_pos : start_pos + seq]
        )

        k = k.transpose(0, 2, 1, 3)  # [batch, N_KV_HEADS, seq, D_HEAD]
        v = v.transpose(0, 2, 1, 3)

        # --- KV Cache: concat with cached K, V ---
        if self.kv_cache is not None:
            k = jnp.concatenate([self.kv_cache[0], k], axis=2)
            v = jnp.concatenate([self.kv_cache[1], v], axis=2)
        self.kv_cache = (k, v)

        # --- GQA: repeat K, V heads to match Q heads ---
        n_rep = N_HEADS // N_KV_HEADS
        k = jnp.repeat(k, n_rep, axis=1)
        v = jnp.repeat(v, n_rep, axis=1)

        # --- MHA: scaled dot-product attention ---
        q = q.transpose(0, 2, 1, 3)  # [batch, N_HEADS, seq, D_HEAD]
        scores = q @ k.transpose(0, 1, 3, 2) / jnp.sqrt(D_HEAD)
        scores = jnp.where(mask, scores, -1e9)
        weights = jax.nn.softmax(scores, axis=-1)
        out = weights @ v

        # --- MHA: output projection ---
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq, N_HEADS * D_HEAD)
        return out @ self.W_O
