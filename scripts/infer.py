"""Inference script for text generation."""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp

from llm import Config, Transformer, tokenizer


def sample_token(logits: jnp.ndarray, key: jax.Array, temperature: float = 1.0, top_k: int = 0) -> int:
    """Sample a token from logits with temperature and optional top-k."""
    logits = logits / temperature

    if top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        indices = jnp.argsort(logits)[-top_k:]
        mask = jnp.zeros_like(logits, dtype=bool).at[indices].set(True)
        logits = jnp.where(mask, logits, -1e9)

    probs = jax.nn.softmax(logits)
    return int(jax.random.categorical(key, jnp.log(probs + 1e-9)))


def generate(
    model: Transformer,
    prompt: str,
    max_tokens: int,
    key: jax.Array,
    temperature: float = 1.0,
    top_k: int = 50,
) -> str:
    """Generate text continuation from a prompt."""
    tokens = tokenizer.encode(prompt)
    tokens = jnp.array(tokens)[None, :]

    for _ in range(max_tokens):
        if tokens.shape[1] >= model.config.max_seq_len:
            tokens = tokens[:, -model.config.max_seq_len + 1 :]

        logits = model(tokens)
        next_logits = logits[0, -1, :]

        key, sample_key = jax.random.split(key)
        next_token = sample_token(next_logits, sample_key, temperature, top_k)

        tokens = jnp.concatenate([tokens, jnp.array([[next_token]])], axis=1)

    return tokenizer.decode(tokens[0].tolist())


def main():
    parser = argparse.ArgumentParser(description="Generate text with the LLM")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint (.safetensors)")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Prompt for generation")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling (0 to disable)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--d-model", type=int, default=256, help="Model dimension")
    parser.add_argument("--n-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n-layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--seq-len", type=int, default=256, help="Max sequence length")
    args = parser.parse_args()

    config = Config(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.seq_len,
    )

    # Initialize model structure and load weights
    print(f"Loading checkpoint from {args.checkpoint}")
    model = Transformer("model", config, jax.random.key(0))
    model.load_safetensors(args.checkpoint)
    print(f"Loaded model: {config}")

    key = jax.random.key(args.seed)

    print(f"\nPrompt: {args.prompt}")
    print("-" * 40)

    output = generate(model, args.prompt, args.max_tokens, key, args.temperature, args.top_k)
    print(output)


if __name__ == "__main__":
    main()
