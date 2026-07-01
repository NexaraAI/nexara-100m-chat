"""Orchestrate Phase 1.4 Stage 1 model training on TinyStories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from training.config import load_config
from training.train import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Orchestrate Stage 1 training.")
    parser.add_argument(
        "--config", default="configs/stage1_tinystories.toml", help="Path to config file."
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Override maximum steps.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--device", default=None, help="Override device.")
    parser.add_argument("--output-dir", default=None, help="Override checkpoint output directory.")
    parser.add_argument("--log-dir", default=None, help="Override log directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    # Apply overrides
    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps
    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate
    if args.device is not None:
        config["run"]["device"] = args.device
    if args.output_dir is not None:
        config["run"]["output_dir"] = args.output_dir
    if args.log_dir is not None:
        config["run"]["log_dir"] = args.log_dir

    # Verify tokenizer and caches
    tokenizer_path = Path(config["tokenizer"]["path"])
    if not tokenizer_path.exists():
        print(f"Error: Tokenizer file not found at {tokenizer_path}", file=sys.stderr)
        print("Please train the tokenizer first using scripts/train_tokenizer.py", file=sys.stderr)
        sys.exit(1)

    if bool(config["data"].get("use_token_cache", False)):
        train_cache = Path(config["data"]["train_cache_path"])
        val_cache = Path(config["data"]["validation_cache_path"])
        if not train_cache.exists() or not val_cache.exists():
            print("Error: Token cache files not found.", file=sys.stderr)
            print(f"Expected train cache: {train_cache}", file=sys.stderr)
            print(f"Expected val cache: {val_cache}", file=sys.stderr)
            print("Please build caches first using scripts/cache_tokens.py", file=sys.stderr)
            sys.exit(1)
    else:
        train_path = Path(config["data"]["train_path"])
        val_path = Path(config["data"]["validation_path"])
        if not train_path.exists() or not val_path.exists():
            print("Error: Dataset paths not found.", file=sys.stderr)
            print(f"Expected train path: {train_path}", file=sys.stderr)
            print(f"Expected val path: {val_path}", file=sys.stderr)
            sys.exit(1)

    print("--- Configuration verified successfully. Starting training... ---")

    # Run training
    try:
        train(config)
        print("--- Training completed successfully. ---")

        # Save loss curve summary from metrics log
        log_dir = Path(config["run"]["log_dir"])
        metrics_file = log_dir / "train_metrics.jsonl"
        if metrics_file.exists():
            steps = []
            train_losses = []
            val_steps = []
            val_losses = []
            with metrics_file.open("r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if "train_loss" in data:
                            steps.append(data["step"])
                            train_losses.append(data["train_loss"])
                        if "validation_loss" in data:
                            val_steps.append(data["step"])
                            val_losses.append(data["validation_loss"])

            summary_path = log_dir / "loss_summary.json"
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "config": args.config,
                        "train_loss_curve": {"steps": steps, "losses": train_losses},
                        "val_loss_curve": {"steps": val_steps, "losses": val_losses},
                    },
                    f,
                    indent=2,
                )
            print(f"Wrote loss summary to {summary_path}")

    except Exception as e:
        print(f"Training failed with error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
