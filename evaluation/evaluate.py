"""Evaluate a Nexara checkpoint on validation data."""

from __future__ import annotations

import argparse
import math

import torch
from torch.utils.data import DataLoader

from datasets.text import TokenBlockDataset
from model import DecoderOnlyTransformer, ModelConfig
from tokenizer import NexaraTokenizer
from training.checkpointing import load_checkpoint
from training.config import load_config
from training.train import evaluate_loss, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate validation perplexity.")
    parser.add_argument("--config", required=True, help="Path to a Nexara TOML config.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--device", default="auto", help="Device override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(args.device)

    tokenizer = NexaraTokenizer(config["tokenizer"]["path"])
    dataset = TokenBlockDataset(
        paths=[config["data"]["validation_path"]],
        tokenizer=tokenizer,
        block_size=int(config["data"]["block_size"]),
        text_key=str(config["data"].get("text_key", "text")),
        max_documents=int(config["data"].get("max_validation_documents", 0)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 0)),
    )

    model = DecoderOnlyTransformer(ModelConfig.from_mapping(config["model"])).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    validation_loss = evaluate_loss(
        model=model,
        loader=loader,
        device=device,
        max_batches=int(config["training"]["eval_batches"]),
    )
    perplexity = float("inf") if validation_loss > 50 else math.exp(validation_loss)
    print(f"validation_loss={validation_loss:.6f}")
    print(f"perplexity={perplexity:.6f}")


if __name__ == "__main__":
    main()
