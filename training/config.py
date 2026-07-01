"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import tomllib

REQUIRED_SECTIONS = ("run", "data", "tokenizer", "model", "training", "generation")


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    missing_sections = [section for section in REQUIRED_SECTIONS if section not in config]
    if missing_sections:
        joined = ", ".join(missing_sections)
        raise ValueError(f"missing required config section(s): {joined}")

    data = config["data"]
    tokenizer = config["tokenizer"]
    model = config["model"]
    training = config["training"]

    _require_positive_int(data, "block_size")
    _require_positive_int(data, "batch_size")
    _require_non_negative_int(data, "num_workers")
    _require_positive_int(tokenizer, "vocab_size")
    if bool(data.get("use_token_cache", False)):
        _require_non_empty_string(data, "train_cache_path")
        _require_non_empty_string(data, "validation_cache_path")

    _require_positive_int(model, "vocab_size")
    _require_positive_int(model, "max_sequence_length")
    _require_positive_int(model, "n_layers")
    _require_positive_int(model, "n_heads")
    _require_positive_int(model, "embedding_dim")

    if int(model["vocab_size"]) != int(tokenizer["vocab_size"]):
        raise ValueError("model.vocab_size must match tokenizer.vocab_size")
    if int(model["max_sequence_length"]) != int(data["block_size"]):
        raise ValueError("model.max_sequence_length must match data.block_size")
    if int(model["embedding_dim"]) % int(model["n_heads"]) != 0:
        raise ValueError("model.embedding_dim must be divisible by model.n_heads")
    if (int(model["embedding_dim"]) // int(model["n_heads"])) % 2 != 0:
        raise ValueError("attention head dimension must be even for RoPE")

    dropout = float(model.get("dropout", 0.0))
    if not 0.0 <= dropout < 1.0:
        raise ValueError("model.dropout must be in [0, 1)")

    _require_positive_int(training, "max_epochs")
    _require_positive_int(training, "max_steps")
    _require_positive_int(training, "gradient_accumulation_steps")
    _require_positive_int(training, "log_interval")
    _require_positive_int(training, "eval_interval")
    _require_positive_int(training, "checkpoint_interval")

    learning_rate = float(training["learning_rate"])
    if learning_rate <= 0.0:
        raise ValueError("training.learning_rate must be positive")


def _require_positive_int(section: dict[str, Any], key: str) -> None:
    if int(section[key]) <= 0:
        raise ValueError(f"{key} must be positive")


def _require_non_negative_int(section: dict[str, Any], key: str) -> None:
    if int(section[key]) < 0:
        raise ValueError(f"{key} must be non-negative")


def _require_non_empty_string(section: dict[str, Any], key: str) -> None:
    if not str(section.get(key, "")).strip():
        raise ValueError(f"{key} must be a non-empty string")
