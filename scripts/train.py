"""Training script for the LLM."""

import argparse
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from safetensors.numpy import save_file, load_file
from tqdm import tqdm

from llm import RTT, tokenizer
from llm.config import TrainConfig
from llm.optim import AdamW, CosineDecay, Warmup, clip_grads


# --- Data loading ---


def load_shard(
    path: Path, seq_len: int, shard_idx: int, shard_size: int
) -> tuple[jnp.ndarray | None, bool]:
    """Load and tokenize a single shard. Returns (data, is_last_shard)."""
    chars_per_seq = seq_len * 5
    start_char = shard_idx * shard_size * chars_per_seq
    chars_to_read = shard_size * chars_per_seq + 1000

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
    return data, n_seqs < shard_size


def create_batches(data: jnp.ndarray, batch_size: int, key: jax.Array):
    """Yield shuffled batches from data."""
    n_seqs = data.shape[0]
    indices = jax.random.permutation(key, n_seqs)
    for i in range(0, n_seqs - batch_size + 1, batch_size):
        yield data[indices[i : i + batch_size]]


# --- Checkpointing ---


def save_training_state(
    path: Path, step: int, epoch: int, shard_idx: int,
    key: jax.Array, optimizer: AdamW,
) -> None:
    """Save training state alongside model checkpoint."""
    state: dict[str, np.ndarray] = {
        "meta.step": np.array(step, dtype=np.int32),
        "meta.epoch": np.array(epoch, dtype=np.int32),
        "meta.shard_idx": np.array(shard_idx, dtype=np.int32),
        "meta.rng_key": np.array(jax.random.key_data(key)),
    }
    opt_state = optimizer.serialize()
    for k, v in opt_state["mu"].items():
        state[f"mu.{k}"] = np.array(v)
    for k, v in opt_state["nu"].items():
        state[f"nu.{k}"] = np.array(v)
    save_file(state, str(path) + ".training.safetensors")


def load_training_state(
    path: Path,
) -> tuple[int, int, int, jax.Array, dict] | None:
    """Load training state. Returns (step, epoch, shard_idx, key, opt_state) or None."""
    train_path = str(path) + ".training.safetensors"
    if not Path(train_path).exists():
        return None
    state = load_file(train_path)
    mu = {k[3:]: jnp.array(v) for k, v in state.items() if k.startswith("mu.")}
    nu = {k[3:]: jnp.array(v) for k, v in state.items() if k.startswith("nu.")}
    return (
        int(state["meta.step"]),
        int(state["meta.epoch"]),
        int(state["meta.shard_idx"]),
        jax.random.wrap_key_data(state["meta.rng_key"]),
        {"mu": mu, "nu": nu},
    )


# --- Training ---


def estimate_steps(cfg: TrainConfig, data_path: Path) -> tuple[int, int, int]:
    """Estimate total sequences, shards, and training steps."""
    seq_len = cfg.data.seq_len
    shard_size = cfg.data.shard_size if cfg.data.shard_size > 0 else 100000
    file_size = os.path.getsize(data_path)
    estimated_seqs = file_size // 4 // seq_len
    estimated_shards = max(1, (estimated_seqs + shard_size - 1) // shard_size)
    micro_per_shard = shard_size // cfg.data.batch_size
    steps_per_shard = micro_per_shard // cfg.data.gradient_accumulation_steps
    total_steps = cfg.epochs * estimated_shards * steps_per_shard
    return estimated_seqs, estimated_shards, total_steps


def main():
    parser = argparse.ArgumentParser(description="Train the LLM")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = TrainConfig.from_json(args.config)
    config_dir = args.config.parent
    print(f"Loaded config from {args.config}")

    def resolve_path(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else config_dir / path

    # Model
    key = jax.random.key(cfg.seed)
    key, init_key = jax.random.split(key)
    model = RTT("model", init_key)

    # Auto-resume
    checkpoint_path = resolve_path(cfg.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    start_step, start_epoch, start_shard_idx = 0, 0, 0
    resumed_opt_state = None

    train_state = load_training_state(checkpoint_path)
    if train_state is not None:
        start_step, start_epoch, start_shard_idx, key, resumed_opt_state = train_state
        model.load_safetensors(checkpoint_path)
        print(f"Resumed: step={start_step}, epoch={start_epoch}, shard={start_shard_idx}")
    else:
        print("Starting fresh")

    # Data
    data_path = resolve_path(cfg.data.path)
    seq_len = cfg.data.seq_len
    shard_size = cfg.data.shard_size if cfg.data.shard_size > 0 else 100000
    accum_steps = cfg.data.gradient_accumulation_steps
    estimated_seqs, estimated_shards, estimated_total_steps = estimate_steps(cfg, data_path)
    print(f"Data: {data_path} (~{estimated_seqs:,} seqs, ~{estimated_shards} shards)")
    print(f"Batch: {cfg.data.batch_size} × {accum_steps} accum = {cfg.data.batch_size * accum_steps} effective")
    print(f"Steps: ~{estimated_total_steps}")

    # Optimizer + schedule
    params = model.state_dict()
    opt_cfg = cfg.optimizer
    sched = opt_cfg.schedule

    optimizer = AdamW(params, b1=opt_cfg.b1, b2=opt_cfg.b2, eps=opt_cfg.eps,
                      weight_decay=opt_cfg.weight_decay)
    if resumed_opt_state is not None:
        optimizer.deserialize(resumed_opt_state)
        optimizer.step = start_step

    decay = CosineDecay(peak_lr=sched.peak_lr, end_lr=sched.end_lr,
                        total_steps=estimated_total_steps - sched.warmup_steps)
    schedule = Warmup(decay, warmup_steps=sched.warmup_steps, peak_lr=sched.peak_lr)
    print(f"LR: warmup {sched.warmup_steps} → {sched.peak_lr:.2e} → cosine → {sched.end_lr:.2e}")

    # JIT compilation
    def loss_fn(params: dict, batch: jnp.ndarray) -> jnp.ndarray:
        model.load_state_dict(params)
        return model.loss(batch)

    @jax.jit
    def compute_grads(params: dict, batch: jnp.ndarray):
        return jax.value_and_grad(loss_fn)(params, batch)

    print("Warming up JIT...")
    first_shard, _ = load_shard(data_path, seq_len, 0, shard_size)
    key, warmup_key = jax.random.split(key)
    _ = compute_grads(params, next(create_batches(first_shard, cfg.data.batch_size, warmup_key)))
    del first_shard
    print("JIT ready")

    # Graceful shutdown
    shutdown_requested = False

    def on_sigint(_sig, _frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            print("\n\nForced exit.")
            sys.exit(1)
        shutdown_requested = True
        print("\n\nInterrupted — saving checkpoint...")

    signal.signal(signal.SIGINT, on_sigint)

    def save_checkpoint(step: int, epoch: int, shard_idx: int) -> None:
        model.load_state_dict(params)
        model.save_safetensors(checkpoint_path)
        save_training_state(checkpoint_path, step, epoch, shard_idx, key, optimizer)

    # Training loop
    step = start_step
    CHECKPOINT_INTERVAL = 180
    last_ckpt_time = time.monotonic()
    pbar = tqdm(total=estimated_total_steps, initial=start_step, desc="Training", unit="step")

    for epoch in range(start_epoch, cfg.epochs):
        epoch_loss, epoch_batches = 0.0, 0
        shard_idx = start_shard_idx if epoch == start_epoch else 0

        while not shutdown_requested:
            shard_data, is_last = load_shard(data_path, seq_len, shard_idx, shard_size)
            if shard_data is None:
                break

            key, shard_key = jax.random.split(key)
            accum_grads, accum_loss, micro_steps = None, 0.0, 0

            for batch in create_batches(shard_data, cfg.data.batch_size, shard_key):
                if shutdown_requested:
                    break

                loss, grads = compute_grads(params, batch)
                accum_loss += float(loss)
                micro_steps += 1
                accum_grads = grads if accum_grads is None else jax.tree.map(
                    lambda a, b: a + b, accum_grads, grads
                )

                if micro_steps == accum_steps:
                    # Average, clip, update
                    accum_grads = jax.tree.map(lambda g: g / accum_steps, accum_grads)
                    accum_grads = clip_grads(accum_grads, opt_cfg.max_grad_norm)
                    current_lr = schedule(step)
                    params = optimizer.update(params, accum_grads, current_lr)

                    batch_loss = accum_loss / accum_steps
                    epoch_loss += batch_loss
                    epoch_batches += 1
                    step += 1

                    pbar.set_postfix(loss=f"{batch_loss:.4f}", lr=f"{current_lr:.2e}",
                                     shard=f"{shard_idx+1}", epoch=f"{epoch+1}/{cfg.epochs}")
                    pbar.update(1)

                    if time.monotonic() - last_ckpt_time >= CHECKPOINT_INTERVAL:
                        save_checkpoint(step, epoch, shard_idx)
                        tqdm.write(f"Checkpoint at step {step}")
                        last_ckpt_time = time.monotonic()

                    accum_grads, accum_loss, micro_steps = None, 0.0, 0

            del shard_data
            shard_idx += 1
            if is_last:
                break

        if shutdown_requested:
            save_checkpoint(step, epoch, shard_idx)
            tqdm.write(f"Checkpoint at step {step} (shutdown)")
            break

        avg_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0
        tqdm.write(f"Epoch {epoch+1}/{cfg.epochs} - Loss: {avg_loss:.4f} ({shard_idx} shards)")
        save_checkpoint(step, epoch + 1, 0)

    pbar.close()
    if not shutdown_requested:
        print("Training complete!")


if __name__ == "__main__":
    main()
