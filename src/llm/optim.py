"""AdamW optimizer and LR schedule in pure JAX."""

import math
from typing import TypeAlias

import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

Params: TypeAlias = PyTree[Array]
Grads: TypeAlias = PyTree[Array]


class CosineDecay:
    """Cosine decay from peak_lr to end_lr over total_steps."""

    peak_lr: float
    end_lr: float
    total_steps: int

    def __init__(self, peak_lr: float, end_lr: float, total_steps: int):
        self.peak_lr = peak_lr
        self.end_lr = end_lr
        self.total_steps = total_steps

    def __call__(self, step: int) -> float:
        progress = min(step / max(self.total_steps, 1), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.end_lr + (self.peak_lr - self.end_lr) * cosine


class Warmup:
    """Linear warmup wrapper around any schedule."""

    warmup_steps: int
    peak_lr: float
    inner: CosineDecay

    def __init__(self, inner: CosineDecay, warmup_steps: int, peak_lr: float):
        self.inner = inner
        self.warmup_steps = warmup_steps
        self.peak_lr = peak_lr

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.peak_lr * step / max(self.warmup_steps, 1)
        return self.inner(step - self.warmup_steps)


class AdamW:
    """AdamW optimizer."""

    b1: float
    b2: float
    eps: float
    weight_decay: float
    step: int
    mu: PyTree[Array]
    nu: PyTree[Array]

    def __init__(
        self,
        params: Params,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.weight_decay = weight_decay
        self.step = 0
        self.mu = jax.tree.map(jnp.zeros_like, params)
        self.nu = jax.tree.map(jnp.zeros_like, params)

    def update(self, params: Params, grads: Grads, lr: float) -> Params:
        params, self.mu, self.nu = _adamw_step(
            params, grads, self.mu, self.nu,
            self.step, lr, self.b1, self.b2, self.eps, self.weight_decay,
        )
        self.step += 1
        return params

    def serialize(self) -> dict[str, PyTree[Array]]:
        return {"mu": self.mu, "nu": self.nu}

    def deserialize(self, state: dict[str, PyTree[Array]]) -> None:
        self.mu = state["mu"]
        self.nu = state["nu"]


@jax.jit
def clip_grads(grads: Grads, max_norm: float) -> Grads:
    """Clip gradients by global norm."""
    leaves = jax.tree.leaves(grads)
    sum_sq = jnp.zeros(())
    for g in leaves:
        sum_sq = sum_sq + jnp.sum(g**2)
    global_norm = jnp.sqrt(sum_sq)
    scale = jnp.minimum(1.0, max_norm / (global_norm + 1e-8))
    return jax.tree.map(lambda g: g * scale, grads)


@jax.jit
def _adamw_step(
    params: Params, grads: Grads,
    mu: PyTree[Array], nu: PyTree[Array],
    step: int, lr: float, b1: float, b2: float, eps: float, weight_decay: float,
) -> tuple[Params, PyTree[Array], PyTree[Array]]:
    mu = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, mu, grads)
    nu = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g**2, nu, grads)
    mu_hat = jax.tree.map(lambda m: m / (1 - b1 ** (step + 1)), mu)
    nu_hat = jax.tree.map(lambda v: v / (1 - b2 ** (step + 1)), nu)
    params = jax.tree.map(
        lambda p, m, v: p - lr * (m / (jnp.sqrt(v) + eps) + weight_decay * p),
        params, mu_hat, nu_hat,
    )
    return params, mu, nu
