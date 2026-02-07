"""Training script for the LLM."""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

from llm import Config, Transformer, tokenizer


def load_text_data(path: Path, seq_len: int) -> jnp.ndarray:
    """Load and tokenize text file into training sequences."""
    text = path.read_text()
    tokens = tokenizer.encode(text)

    # Truncate to multiple of seq_len
    n_seqs = len(tokens) // seq_len
    tokens = tokens[: n_seqs * seq_len]

    return jnp.array(tokens).reshape(n_seqs, seq_len)


def create_batches(data: jnp.ndarray, batch_size: int, key: jax.Array):
    """Yield random batches from data."""
    n_seqs = data.shape[0]
    indices = jax.random.permutation(key, n_seqs)

    for i in range(0, n_seqs - batch_size + 1, batch_size):
        batch_indices = indices[i : i + batch_size]
        yield data[batch_indices]


def main():
    parser = argparse.ArgumentParser(description="Train the LLM")
    parser.add_argument("--data", type=Path, required=True, help="Path to training text file")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/model.safetensors"), help="Checkpoint path")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from checkpoint")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    parser.add_argument("--d-model", type=int, default=256, help="Model dimension")
    parser.add_argument("--n-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n-layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    key = jax.random.key(args.seed)

    config = Config(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.seq_len,
    )

    # Initialize model
    key, init_key = jax.random.split(key)
    model = Transformer("model", config, init_key)

    # Resume from checkpoint if specified
    if args.resume and args.resume.exists():
        print(f"Resuming from {args.resume}")
        model.load_safetensors(args.resume)
    else:
        print(f"Initialized new model: {config}")

    # Load data
    print(f"Loading data from {args.data}")
    data = load_text_data(args.data, args.seq_len)
    print(f"Loaded {data.shape[0]} sequences of length {args.seq_len}")

    # Get params as pytree for gradient computation
    params = model.state_dict()

    # Setup optimizer
    optimizer = optax.adamw(args.lr)
    opt_state = optimizer.init(params)

    # Pure loss function for JAX transformations
    def loss_fn(params: dict, batch: jnp.ndarray) -> jnp.ndarray:
        model.load_state_dict(params)
        return model.loss(batch)

    # JIT-compiled training step
    @jax.jit
    def train_step(params, batch, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # Training loop
    for epoch in range(args.epochs):
        key, epoch_key = jax.random.split(key)
        epoch_loss = 0.0
        n_batches = 0

        for batch in create_batches(data, args.batch_size, epoch_key):
            params, opt_state, loss = train_step(params, batch, opt_state)
            epoch_loss += float(loss)
            n_batches += 1

        avg_loss = epoch_loss / n_batches if n_batches > 0 else 0
        print(f"Epoch {epoch + 1}/{args.epochs} - Loss: {avg_loss:.4f}")

        # Save checkpoint
        model.load_state_dict(params)
        model.save_safetensors(args.checkpoint)
        print(f"Saved checkpoint to {args.checkpoint}")

    print("Training complete!")


if __name__ == "__main__":
    main()
