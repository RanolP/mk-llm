"""Parameter initialization."""

from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

type Shape = tuple[int, ...]
type Initializer = Callable[[Shape, PRNGKeyArray], Float[Array, "..."]]


def he_init(shape: Shape, key: PRNGKeyArray) -> Float[Array, "..."]:
    """He (Kaiming) initialization. std = sqrt(2 / fan_in)"""
    fan_in = shape[0]
    std = jnp.sqrt(2.0 / fan_in)
    return jax.random.normal(key, shape) * std
