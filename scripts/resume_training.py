"""Resume Stage 1 model training from a checkpoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import torch

from training.checkpointing import find_latest_checkpoint, validate_checkpoint
from training.config import load_config
from training.train import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume Stage 1 training from checkpoint.")
    parser.add_argument(
        "--config", default="configs/stage1_tinystories.toml", help="Path to config file."
    )
    parser.add_argument(
        "--checkpoint", default="", help="Path to specific checkpoint to resume from (optional)."
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Override maximum steps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    checkpoint_path = None
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        output_dir = Path(config["run"]["output_dir"])
        checkpoint_path = find_latest_checkpoint(output_dir)

    if not checkpoint_path or not checkpoint_path.exists():
        print(f"Error: No checkpoint found to resume training.", file=sys.stderr)
        if not args.checkpoint:
            print(f"Looked in output directory: {config['run']['output_dir']}", file=sys.stderr)
        sys.exit(1)

    print(f"Found checkpoint: {checkpoint_path}")
    print("Validating checkpoint integrity...")
    if not validate_checkpoint(checkpoint_path):
        print(f"Error: Checkpoint {checkpoint_path} is invalid or corrupted.", file=sys.stderr)
        sys.exit(1)
    print("Checkpoint validated successfully.")

    # Load step/epoch metadata from checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    step = ckpt.get("step", 0)
    epoch = ckpt.get("epoch", 0)
    print(f"Resuming training from step {step}, epoch {epoch}")

    # Set resume_from in configuration
    config["training"]["resume_from"] = str(checkpoint_path)
    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps

    # Add log entry to resume history
    log_dir = Path(config["run"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    resume_log = log_dir / "resume_history.jsonl"
    with resume_log.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "checkpoint": str(checkpoint_path),
                    "resume_step": step,
                    "resume_epoch": epoch,
                }
            )
            + "\n"
        )

    print("--- Starting training resume... ---")
    try:
        train(config)
        print("--- Training completed successfully. ---")
    except Exception as e:
        print(f"Resuming training failed with error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
