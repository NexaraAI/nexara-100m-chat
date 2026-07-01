"""Export a training checkpoint to a clean model checkpoint and config for inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export training checkpoint for inference.")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to input training checkpoint .pt file."
    )
    parser.add_argument("--output-pt", help="Path to save clean state dict (.pt).")
    parser.add_argument("--output-json", help="Path to save model configuration (.json).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    print(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint at {ckpt_path} does not contain 'model_state_dict'")

    model_state = checkpoint["model_state_dict"]
    config = checkpoint.get("config", {})

    # Resolve output paths
    out_pt_path = Path(args.output_pt) if args.output_pt else ckpt_path.parent / "clean_model.pt"
    out_json_path = Path(args.output_json) if args.output_json else ckpt_path.parent / "config.json"

    out_pt_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure model state dict contains CPU tensors for portability
    model_state_cpu = {k: v.cpu() for k, v in model_state.items()}

    # Save clean model state dict + config package (fully compatible with generate.py)
    clean_package = {
        "model_state_dict": model_state_cpu,
        "config": config,
    }
    torch.save(clean_package, out_pt_path)
    print(f"Exported clean package to {out_pt_path}")

    # Save config.json
    with out_json_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Exported configuration to {out_json_path}")


if __name__ == "__main__":
    main()
