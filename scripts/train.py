"""Training script for the LLM."""

import argparse
import signal
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm


# Graceful shutdown on Ctrl+C
def signal_handler(sig, frame):
    print("\n\nInterrupted by user. Exiting...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

from llm import RTT, tokenizer
from llm.config import TrainConfig


def count_file_chars(path: Path) -> int:
    """Count characters in file without loading it all."""
    import os
    return os.path.getsize(path)


def load_shard(path: Path, seq_len: int, shard_idx: int, shard_size: int) -> tuple[jnp.ndarray, bool]:
    """Load and tokenize a single shard from file.

    Returns (data, is_last_shard).
    """
    # Estimate chars needed (~4 chars per token, with buffer)
    chars_per_seq = seq_len * 5
    start_char = shard_idx * shard_size * chars_per_seq
    chars_to_read = shard_size * chars_per_seq + 1000  # Extra buffer

    with open(path, "r") as f:
        f.seek(start_char)
        text = f.read(chars_to_read)

    if not text:
        return None, True

    tokens = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    n_seqs = min(len(tokens) // seq_len, shard_size)

    if n_seqs == 0:
        return None, True

    tokens = tokens[: n_seqs * seq_len]
    data = jnp.array(tokens).reshape(n_seqs, seq_len)

    is_last = n_seqs < shard_size
    return data, is_last


def create_batches(data: jnp.ndarray, batch_size: int, key: jax.Array):
    """Yield random batches from data."""
    n_seqs = data.shape[0]
    indices = jax.random.permutation(key, n_seqs)

    for i in range(0, n_seqs - batch_size + 1, batch_size):
        batch_indices = indices[i : i + batch_size]
        yield data[batch_indices]


def build_optimizer(cfg: TrainConfig, total_steps: int) -> optax.GradientTransformation:
    """Build optimizer from config."""
    opt_cfg = cfg.optimizer
    sched_cfg = opt_cfg.schedule

    # Build learning rate schedule
    if sched_cfg.type == "constant":
        schedule = sched_cfg.peak_lr
    elif sched_cfg.type == "warmup_cosine_decay":
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=sched_cfg.peak_lr,
            warmup_steps=sched_cfg.warmup_steps,
            decay_steps=total_steps,
            end_value=sched_cfg.end_lr,
        )
    else:
        raise ValueError(f"Unknown schedule type: {sched_cfg.type}")

    # Build optimizer chain
    return optax.chain(
        optax.clip_by_global_norm(opt_cfg.max_grad_norm),
        optax.adamw(
            learning_rate=schedule,
            weight_decay=opt_cfg.weight_decay,
            b1=opt_cfg.b1,
            b2=opt_cfg.b2,
            eps=opt_cfg.eps,
        ),
    ), schedule


def main():
    parser = argparse.ArgumentParser(description="Train the LLM")
    parser.add_argument("--config", type=Path, required=True, help="Path to config JSON file")
    args = parser.parse_args()

    # Load config
    cfg = TrainConfig.from_json(args.config)
    config_dir = args.config.parent
    print(f"Loaded config from {args.config}")

    key = jax.random.key(cfg.seed)

    # Initialize model
    key, init_key = jax.random.split(key)
    model = RTT("model", init_key)

    def resolve_path(p: str) -> Path:
        """Resolve path relative to config file."""
        path = Path(p)
        if not path.is_absolute():
            path = config_dir / path
        return path

    # Resume from checkpoint if specified
    if cfg.resume:
        resume_path = resolve_path(cfg.resume)
        if resume_path.exists():
            print(f"Resuming from {resume_path}")
            model.load_safetensors(resume_path)
        else:
            print(f"Warning: resume path {resume_path} not found, starting fresh")
    else:
        print("Initialized new model")

    # Data config
    data_path = resolve_path(cfg.data.path)
    seq_len = cfg.data.seq_len
    shard_size = cfg.data.shard_size if cfg.data.shard_size > 0 else 100000  # Default large
    print(f"Data: {data_path}")
    print(f"Shard size: {shard_size} sequences")

    # Estimate total shards from file size
    file_size = count_file_chars(data_path)
    estimated_tokens = file_size // 4  # ~4 chars per token
    estimated_seqs = estimated_tokens // seq_len
    estimated_shards = max(1, (estimated_seqs + shard_size - 1) // shard_size)
    print(f"Estimated: ~{estimated_seqs:,} sequences, ~{estimated_shards} shards")

    # Get params as pytree for gradient computation
    params = model.state_dict()

    # Calculate total training steps (estimate, will adjust as we go)
    n_micro_batches_per_shard = shard_size // cfg.data.batch_size
    n_steps_per_shard = n_micro_batches_per_shard // cfg.data.gradient_accumulation_steps
    estimated_total_steps = cfg.epochs * estimated_shards * n_steps_per_shard
    print(f"Estimated steps: ~{estimated_total_steps}")

    sched = cfg.optimizer.schedule
    if sched.type == "warmup_cosine_decay":
        print(f"LR schedule: warmup {sched.warmup_steps} steps → {sched.peak_lr:.2e} → cosine decay → {sched.end_lr:.2e}")
    else:
        print(f"LR: constant {sched.peak_lr:.2e}")

    # Build optimizer
    print("Building optimizer...")
    optimizer, schedule = build_optimizer(cfg, estimated_total_steps)
    print("Initializing optimizer state...")
    opt_state = optimizer.init(params)
    print("Optimizer ready")

    # Pure loss function for JAX transformations
    def loss_fn(params: dict, batch: jnp.ndarray) -> jnp.ndarray:
        model.load_state_dict(params)
        return model.loss(batch)

    # JIT-compiled gradient computation
    print("Compiling training functions (this may take a moment)...")

    @jax.jit
    def compute_grads(params, batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        return loss, grads

    # JIT-compiled optimizer step
    @jax.jit
    def apply_grads(params, grads, opt_state):
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    def accumulate_grads(g1, g2):
        """Add two gradient pytrees."""
        return jax.tree.map(lambda a, b: a + b, g1, g2)

    def scale_grads(grads, scale):
        """Scale gradients by a factor."""
        return jax.tree.map(lambda g: g * scale, grads)

    # Warm up JIT compilation with first shard
    print("Warming up JIT (first batch)...")
    first_shard, _ = load_shard(data_path, seq_len, 0, shard_size)
    key, warmup_key = jax.random.split(key)
    warmup_batch = next(create_batches(first_shard, cfg.data.batch_size, warmup_key))
    _ = compute_grads(params, warmup_batch)
    del first_shard, warmup_batch
    print("JIT compilation complete")

    # Training loop
    step = 0
    accum_steps = cfg.data.gradient_accumulation_steps
    checkpoint_path = resolve_path(cfg.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    effective_batch = cfg.data.batch_size * accum_steps
    print(f"Effective batch size: {effective_batch} ({cfg.data.batch_size} × {accum_steps} accumulation steps)")

    CHECKPOINT_INTERVAL = 180  # seconds
    last_checkpoint_time = time.monotonic()

    pbar = tqdm(total=estimated_total_steps, desc="Training", unit="step")

    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        epoch_batches = 0
        shard_idx = 0

        while True:
            # Load shard from file
            shard_data, is_last = load_shard(data_path, seq_len, shard_idx, shard_size)

            if shard_data is None:
                break

            key, shard_key = jax.random.split(key)
            accum_grads = None
            accum_loss = 0.0
            micro_steps = 0

            for batch in create_batches(shard_data, cfg.data.batch_size, shard_key):
                loss, grads = compute_grads(params, batch)
                accum_loss += float(loss)
                micro_steps += 1

                # Accumulate gradients
                if accum_grads is None:
                    accum_grads = grads
                else:
                    accum_grads = accumulate_grads(accum_grads, grads)

                # Apply update after accumulation steps
                if micro_steps == accum_steps:
                    # Average gradients
                    accum_grads = scale_grads(accum_grads, 1.0 / accum_steps)
                    params, opt_state = apply_grads(params, accum_grads, opt_state)

                    batch_loss = accum_loss / accum_steps
                    epoch_loss += batch_loss
                    epoch_batches += 1
                    step += 1

                    current_lr = schedule(step) if callable(schedule) else schedule
                    pbar.set_postfix(
                        loss=f"{batch_loss:.4f}",
                        lr=f"{current_lr:.2e}",
                        shard=f"{shard_idx + 1}",
                        epoch=f"{epoch + 1}/{cfg.epochs}",
                    )
                    pbar.update(1)

                    # Periodic checkpoint
                    now = time.monotonic()
                    if now - last_checkpoint_time >= CHECKPOINT_INTERVAL:
                        model.load_state_dict(params)
                        model.save_safetensors(checkpoint_path)
                        tqdm.write(f"Checkpoint saved at step {step} to {checkpoint_path}")
                        last_checkpoint_time = now

                    # Reset accumulation
                    accum_grads = None
                    accum_loss = 0.0
                    micro_steps = 0

            # Free shard memory
            del shard_data
            shard_idx += 1

            if is_last:
                break

        avg_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0
        tqdm.write(f"Epoch {epoch + 1}/{cfg.epochs} - Avg Loss: {avg_loss:.4f} ({shard_idx} shards)")

        # Save checkpoint after each epoch
        model.load_state_dict(params)
        model.save_safetensors(checkpoint_path)
        tqdm.write(f"Saved checkpoint to {checkpoint_path}")

    pbar.close()
    print("Training complete!")


if __name__ == "__main__":
    main()
