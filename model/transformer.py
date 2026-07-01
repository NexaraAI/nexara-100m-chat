"""Decoder-only transformer used by Nexara."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first_half, second_half = x.chunk(2, dim=-1)
    return torch.cat((-second_half, first_half), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_sequence_length: int, base: float) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        positions = torch.arange(max_sequence_length, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        embedding = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", embedding.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", embedding.sin()[None, None, :, :], persistent=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_length = query.size(-2)
        cos = self.cos_cached[:, :, :sequence_length, :].to(dtype=query.dtype)
        sin = self.sin_cached[:, :, :sequence_length, :].to(dtype=query.dtype)
        return (
            (query * cos) + (_rotate_half(query) * sin),
            (key * cos) + (_rotate_half(key) * sin),
        )


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.dropout_probability = config.dropout

        self.qkv_projection = nn.Linear(
            config.embedding_dim,
            3 * config.embedding_dim,
            bias=config.bias,
        )
        self.output_projection = nn.Linear(
            config.embedding_dim,
            config.embedding_dim,
            bias=config.bias,
        )
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)
        self.rotary_embedding = RotaryEmbedding(
            head_dim=config.head_dim,
            max_sequence_length=config.max_sequence_length,
            base=config.rope_base,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, embedding_dim = x.shape

        qkv = self.qkv_projection(x)
        qkv = qkv.view(
            batch_size,
            sequence_length,
            3,
            self.n_heads,
            self.head_dim,
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        query, key = self.rotary_embedding(query, key)

        if hasattr(F, "scaled_dot_product_attention"):
            attention = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=self.dropout_probability if self.training else 0.0,
                is_causal=True,
            )
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = (query @ key.transpose(-2, -1)) * scale
            mask = torch.triu(
                torch.ones(sequence_length, sequence_length, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(mask, float("-inf"))
            probabilities = F.softmax(scores, dim=-1)
            probabilities = self.attention_dropout(probabilities)
            attention = probabilities @ value

        attention = (
            attention.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                sequence_length,
                embedding_dim,
            )
        )
        return self.residual_dropout(self.output_projection(attention))


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden_dim = int(config.embedding_dim * config.mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(config.embedding_dim, hidden_dim, bias=config.bias),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, config.embedding_dim, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.embedding_dim)
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = nn.LayerNorm(config.embedding_dim)
        self.feed_forward = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.feed_forward_norm(x))
        return x


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.final_norm = nn.LayerNorm(config.embedding_dim)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")

        sequence_length = input_ids.size(1)
        if sequence_length > self.config.max_sequence_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds "
                f"max_sequence_length {self.config.max_sequence_length}"
            )

        x = self.dropout(self.token_embedding(input_ids))
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
