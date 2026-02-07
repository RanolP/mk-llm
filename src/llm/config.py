"""Training configuration with JSON schema support."""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal


@dataclass
class LRScheduleConfig:
    """Learning rate schedule configuration."""

    type: Literal["constant", "warmup_cosine_decay"] = "warmup_cosine_decay"
    peak_lr: float = 3e-4
    end_lr: float = 3e-5
    warmup_steps: int = 100


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""

    type: Literal["adamw"] = "adamw"
    weight_decay: float = 0.01
    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    schedule: LRScheduleConfig = field(default_factory=LRScheduleConfig)


@dataclass
class DataConfig:
    """Data configuration."""

    path: str = ""
    seq_len: int = 512
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    shard_size: int = 0  # 0 = load all at once, otherwise load this many sequences per shard


@dataclass
class TrainConfig:
    """Full training configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    epochs: int = 10
    seed: int = 42
    checkpoint: str = "checkpoints/model.safetensors"
    resume: str | None = None

    @classmethod
    def from_json(cls, path: Path) -> "TrainConfig":
        """Load config from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "TrainConfig":
        """Create config from dictionary."""
        data = data.copy()
        data.pop("$schema", None)  # Remove JSON schema reference

        if "data" in data:
            data["data"] = DataConfig(**data["data"])
        if "optimizer" in data:
            opt = data["optimizer"].copy()
            if "schedule" in opt:
                opt["schedule"] = LRScheduleConfig(**opt["schedule"])
            data["optimizer"] = OptimizerConfig(**opt)

        return cls(**data)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    def save_json(self, path: Path) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# JSON Schema for external validation and documentation
TRAIN_CONFIG_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "TrainConfig",
    "description": "Training configuration for RanolP's Tiny Transformer",
    "type": "object",
    "properties": {
        "data": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to training text file"},
                "seq_len": {"type": "integer", "default": 512, "minimum": 1},
                "batch_size": {"type": "integer", "default": 4, "minimum": 1},
            },
            "required": ["path"],
        },
        "optimizer": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["adamw"], "default": "adamw"},
                "weight_decay": {"type": "number", "default": 0.01, "minimum": 0},
                "b1": {"type": "number", "default": 0.9, "minimum": 0, "maximum": 1},
                "b2": {"type": "number", "default": 0.95, "minimum": 0, "maximum": 1},
                "eps": {"type": "number", "default": 1e-8, "minimum": 0},
                "max_grad_norm": {"type": "number", "default": 1.0, "minimum": 0},
                "schedule": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["constant", "warmup_cosine_decay"],
                            "default": "warmup_cosine_decay",
                        },
                        "peak_lr": {"type": "number", "default": 3e-4, "minimum": 0},
                        "end_lr": {"type": "number", "default": 3e-5, "minimum": 0},
                        "warmup_steps": {"type": "integer", "default": 100, "minimum": 0},
                    },
                },
            },
        },
        "epochs": {"type": "integer", "default": 10, "minimum": 1},
        "seed": {"type": "integer", "default": 42},
        "checkpoint": {"type": "string", "default": "checkpoints/model.safetensors"},
        "resume": {"type": ["string", "null"], "default": None},
    },
    "required": ["data"],
}


def save_schema(path: Path) -> None:
    """Save JSON schema to file."""
    with open(path, "w") as f:
        json.dump(TRAIN_CONFIG_SCHEMA, f, indent=2)
