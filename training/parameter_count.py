"""Parameter count estimation without importing PyTorch."""

from __future__ import annotations

from typing import Any, Mapping


def estimate_transformer_parameters(model_config: Mapping[str, Any]) -> int:
    """Estimate trainable parameters for the configured decoder transformer."""

    vocab_size = int(model_config["vocab_size"])
    n_layers = int(model_config["n_layers"])
    embedding_dim = int(model_config["embedding_dim"])
    mlp_ratio = float(model_config.get("mlp_ratio", 4.0))
    bias = bool(model_config.get("bias", False))
    tie_embeddings = bool(model_config.get("tie_embeddings", True))

    hidden_dim = int(embedding_dim * mlp_ratio)
    bias_cost = 1 if bias else 0

    token_embedding = vocab_size * embedding_dim
    output_projection = 0 if tie_embeddings else vocab_size * embedding_dim

    qkv = embedding_dim * (3 * embedding_dim) + bias_cost * (3 * embedding_dim)
    attention_output = embedding_dim * embedding_dim + bias_cost * embedding_dim
    feed_forward = (
        embedding_dim * hidden_dim
        + bias_cost * hidden_dim
        + hidden_dim * embedding_dim
        + bias_cost * embedding_dim
    )
    layer_norms = 4 * embedding_dim
    transformer_layers = n_layers * (qkv + attention_output + feed_forward + layer_norms)
    final_norm = 2 * embedding_dim

    return token_embedding + output_projection + transformer_layers + final_norm
