"""Model configuration objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    max_sequence_length: int
    n_layers: int
    n_heads: int
    embedding_dim: int
    dropout: float = 0.1
    mlp_ratio: float = 4.0
    bias: bool = False
    rope_base: float = 10000.0
    tie_embeddings: bool = True

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ModelConfig":
        return cls(
            vocab_size=int(values["vocab_size"]),
            max_sequence_length=int(values["max_sequence_length"]),
            n_layers=int(values["n_layers"]),
            n_heads=int(values["n_heads"]),
            embedding_dim=int(values["embedding_dim"]),
            dropout=float(values.get("dropout", 0.1)),
            mlp_ratio=float(values.get("mlp_ratio", 4.0)),
            bias=bool(values.get("bias", False)),
            rope_base=float(values.get("rope_base", 10000.0)),
            tie_embeddings=bool(values.get("tie_embeddings", True)),
        )

    @property
    def head_dim(self) -> int:
        return self.embedding_dim // self.n_heads

    def validate(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if self.embedding_dim % self.n_heads != 0:
            raise ValueError("embedding_dim must be divisible by n_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
