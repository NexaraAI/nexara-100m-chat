"""Sampling helpers for autoregressive generation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_next_token(
    logits: torch.Tensor,
    previous_tokens: torch.Tensor | None = None,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    filtered = logits.clone()
    if previous_tokens is not None and repetition_penalty != 1.0:
        filtered = apply_repetition_penalty(filtered, previous_tokens, repetition_penalty)

    filtered = filtered / temperature
    if top_k > 0:
        filtered = apply_top_k(filtered, top_k)
    if top_p < 1.0:
        filtered = apply_top_p(filtered, top_p)

    probabilities = F.softmax(filtered, dim=-1)
    return torch.multinomial(probabilities, num_samples=1)


def apply_repetition_penalty(
    logits: torch.Tensor,
    previous_tokens: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    if penalty <= 0:
        raise ValueError("repetition penalty must be positive")

    for batch_index in range(logits.size(0)):
        unique_tokens = torch.unique(previous_tokens[batch_index])
        token_logits = logits[batch_index, unique_tokens]
        logits[batch_index, unique_tokens] = torch.where(
            token_logits < 0,
            token_logits * penalty,
            token_logits / penalty,
        )
    return logits


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    k = min(top_k, logits.size(-1))
    threshold = torch.topk(logits, k=k, dim=-1).values[:, [-1]]
    return logits.masked_fill(logits < threshold, float("-inf"))


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    remove_mask = cumulative_probabilities > top_p
    remove_mask[:, 1:] = remove_mask[:, :-1].clone()
    remove_mask[:, 0] = False

    sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    return filtered
