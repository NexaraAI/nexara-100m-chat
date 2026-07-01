"""Forward pass benchmark for Nexara models."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch

from datasets.statistics import write_statistics
from model import DecoderOnlyTransformer, ModelConfig
from training.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Nexara forward passes.")
    parser.add_argument("--config", default="configs/stage1_tinystories.toml")
    parser.add_argument("--tiny", action="store_true", help="Use a CPU-sized test model.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", default="logs/benchmark/forward.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(1337)
    config = (
        tiny_config() if args.tiny else ModelConfig.from_mapping(load_config(args.config)["model"])
    )
    sequence_length = min(args.sequence_length, config.max_sequence_length)
    model = DecoderOnlyTransformer(config)
    model.eval()
    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(args.batch_size, sequence_length),
        dtype=torch.long,
    )
    targets = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(args.batch_size, sequence_length),
        dtype=torch.long,
    )

    with torch.no_grad():
        for _ in range(args.warmup):
            model(input_ids, targets)

        durations: list[float] = []
        last_loss = 0.0
        for _ in range(args.iterations):
            started = time.perf_counter()
            logits, loss = model(input_ids, targets)
            durations.append(time.perf_counter() - started)
            if loss is None:
                raise RuntimeError("benchmark forward pass did not return loss")
            last_loss = float(loss.detach().cpu())

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tiny": bool(args.tiny),
        "batch_size": args.batch_size,
        "sequence_length": sequence_length,
        "iterations": args.iterations,
        "parameter_count": model.count_parameters(),
        "logits_shape": list(logits.shape),
        "loss": last_loss,
        "average_seconds": sum(durations) / len(durations),
        "min_seconds": min(durations),
        "max_seconds": max(durations),
    }
    write_statistics(Path(args.output), report)
    print(json.dumps(report, indent=2, sort_keys=True))


def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=128,
        max_sequence_length=64,
        n_layers=2,
        n_heads=2,
        embedding_dim=32,
        dropout=0.0,
        tie_embeddings=True,
    )


if __name__ == "__main__":
    main()
