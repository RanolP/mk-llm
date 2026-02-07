"""Inference script for text generation."""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp

from llm import RTT, tokenizer
from llm.constants import MAX_SEQ_LEN


def sample_token(logits: jnp.ndarray, key: jax.Array, temperature: float = 1.0, top_k: int = 0) -> int:
    """Sample a token from logits with temperature and optional top-k."""
    # Greedy decoding when temperature is 0
    if temperature == 0:
        return int(jnp.argmax(logits))

    logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        top_values, _ = jax.lax.top_k(logits, top_k)
        threshold = top_values[-1]
        logits = jnp.where(logits >= threshold, logits, -jnp.inf)

    # categorical expects log probabilities
    log_probs = jax.nn.log_softmax(logits)
    return int(jax.random.categorical(key, log_probs))


def generate(
    model: RTT,
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
        if tokens.shape[1] >= MAX_SEQ_LEN:
            tokens = tokens[:, -MAX_SEQ_LEN + 1 :]

        logits = model.forward(tokens)
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
    args = parser.parse_args()

    # Initialize model structure and load weights
    print(f"Loading checkpoint from {args.checkpoint}")
    model = RTT("model", jax.random.key(0))
    model.load_safetensors(args.checkpoint)
    print("Loaded model")

    key = jax.random.key(args.seed)

    print(f"\nPrompt: {args.prompt}")
    print("-" * 40)

    output = generate(model, args.prompt, args.max_tokens, key, args.temperature, args.top_k)
    print(output)


if __name__ == "__main__":
    main()
