"""Base module class."""

from abc import ABC, abstractmethod
from pathlib import Path

import jax
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray
from safetensors.numpy import save_file, load_file

from llm.utils.init import Initializer, Shape


def _collect_modules(module: "Module") -> list["Module"]:
    """Recursively collect all modules in the tree."""
    modules = [module]
    for v in module.__dict__.values():
        if isinstance(v, Module):
            modules.extend(_collect_modules(v))
        elif isinstance(v, list):
            for x in v:
                if isinstance(x, Module):
                    modules.extend(_collect_modules(x))
    return modules


class Module(ABC):
    """Abstract base class for modules with safetensor support."""

    id: str
    key: PRNGKeyArray
    _tensors: dict[str, Array]

    def __init__(self, id: str, key: PRNGKeyArray):
        self.id = id
        self.key = key
        self._tensors = {}

    def make_key(self, name: str) -> PRNGKeyArray:
        """Derive a key for a named component."""
        return jax.random.fold_in(self.key, hash(name) & 0xFFFFFFFF)

    def safetensor(
        self, name: str, shape: Shape, initializer: Initializer
    ) -> Float[Array, "..."]:
        """Register tensor for serialization."""
        tensor = initializer(shape, self.make_key(name))
        self._tensors[name] = tensor
        return tensor

    def state_dict(self) -> dict[str, Array]:
        """Collect all tensors from this module tree."""
        state = {}
        for module in _collect_modules(self):
            for name, tensor in module._tensors.items():
                state[f"{module.id}.{name}"] = tensor
        return state

    def load_state_dict(self, state: dict[str, Array]) -> None:
        """Load tensors from a state dict."""
        import jax.numpy as jnp

        for module in _collect_modules(self):
            for name in module._tensors:
                key = f"{module.id}.{name}"
                if key in state:
                    module._tensors[name] = jnp.array(state[key])
                    setattr(module, name, module._tensors[name])

    def save_safetensors(self, path: Path | str) -> None:
        """Save module tree to safetensors file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np_state = {k: np.array(v) for k, v in self.state_dict().items()}
        save_file(np_state, path)

    def load_safetensors(self, path: Path | str) -> None:
        """Load module tree from safetensors file."""
        import jax.numpy as jnp

        state = load_file(path)
        jax_state = {k: jnp.array(v) for k, v in state.items()}
        self.load_state_dict(jax_state)

    @abstractmethod
    def forward(self, x: Array) -> Array:
        """Forward pass. Must be implemented by subclasses."""
        ...

    def __call__(self, x: Array) -> Array:
        """Call forward."""
        return self.forward(x)
