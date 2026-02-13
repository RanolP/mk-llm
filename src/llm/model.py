"""RanolP's Tiny Transformer model."""

import jax
import jax.numpy as jnp
import tiktoken
from jaxtyping import Array, Float, Int

from llm.constants import D_MODEL
from llm.layers import TransformerBlock, LayerNorm
from llm.utils import Module, he_init

tokenizer = tiktoken.get_encoding("gpt2")

N_LAYERS = 8


class RTT(Module):
    """RanolP's Tiny Transformer."""

    embed: Float[Array, "vocab d_model"]
    blocks: list[TransformerBlock]
    final_norm: LayerNorm

    def __init__(self, id: str, key: Array):
        super().__init__(id, key)
        vocab = tokenizer.n_vocab

        self.embed = self.safetensor("embed", (vocab, D_MODEL), he_init)
        self.blocks = [
            TransformerBlock(f"{id}.blocks.{i}", self.make_key(f"blocks.{i}"))
            for i in range(N_LAYERS)
        ]
        self.final_norm = LayerNorm(f"{id}.final_norm", self.make_key("final_norm"))

    def forward(self, x: Int[Array, "batch seq"]) -> Float[Array, "batch seq vocab"]:
        # [batch, seq] -> [batch, seq, d_model]
        x = self.embed[x]

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        # Weight tying: reuse embed as logit head
        return x @ self.embed.T

    def loss(self, tokens: Int[Array, "batch seq"]) -> Float[Array, ""]:
        # [batch, seq, vocab]
        logits = self.forward(tokens)[:, :-1, :]
        targets = tokens[:, 1:]
        log_probs = jax.nn.log_softmax(logits, axis=-1)

        batch, seq, _ = log_probs.shape
        batch_idx = jnp.arange(batch)[:, None]
        seq_idx = jnp.arange(seq)[None, :]
        target_log_probs = log_probs[batch_idx, seq_idx, targets]

        return -target_log_probs.mean()
