"""Evaluate a trained model checkpoint on validation loss, perplexity, generation, and save round-trips."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import torch

from inference.generate import generate_text
from model import DecoderOnlyTransformer, ModelConfig
from tokenizer import NexaraTokenizer
from training.checkpointing import load_checkpoint, save_checkpoint, validate_checkpoint
from training.config import load_config
from training.train import build_dataset, build_loader, evaluate_loss, resolve_device, safe_exp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file.")
    parser.add_argument(
        "--config", default="configs/stage1_tinystories.toml", help="Path to config file."
    )
    parser.add_argument("--device", default="auto", help="Override device.")
    parser.add_argument(
        "--eval-batches", type=int, default=100, help="Number of validation batches to evaluate."
    )
    parser.add_argument(
        "--output", default="logs/evaluation_report.json", help="Path to output JSON report."
    )
    return parser.parse_args()


def compare_state_dicts(dict1: dict[str, Any], dict2: dict[str, Any]) -> bool:
    if dict1.keys() != dict2.keys():
        return False
    for k in dict1:
        if not torch.equal(dict1[k].cpu(), dict2[k].cpu()):
            return False
    return True


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found at {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading checkpoint {checkpoint_path}...")
    if not validate_checkpoint(checkpoint_path):
        print(f"Warning: Checkpoint fails standard validation. Trying to load anyway...")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config") or load_config(args.config)

    # 1. Initialize tokenizer and validation loader
    tokenizer = NexaraTokenizer(config["tokenizer"]["path"])
    val_dataset = build_dataset(config, tokenizer, split="validation")
    val_loader = build_loader(config, val_dataset, shuffle=False)

    # 2. Initialize model and load weights
    model_config = ModelConfig.from_mapping(config["model"])
    model = DecoderOnlyTransformer(model_config).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()

    # 3. Evaluate validation loss & perplexity
    print("Evaluating validation loss...")
    val_loss = evaluate_loss(model, val_loader, device, max_batches=args.eval_batches)
    perplexity = safe_exp(val_loss)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Perplexity: {perplexity:.4f}")

    # 4. Generate text samples
    print("Generating sample texts...")
    sample_prompts = config["tokenizer"].get("sample_texts") or [
        "Once upon a time",
        "The little dog",
        "Sara wanted to",
    ]
    generations = {}
    generation_config = config.get("generation", {})
    for prompt in sample_prompts:
        try:
            sample_text = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                max_new_tokens=int(generation_config.get("max_new_tokens", 48)),
                temperature=float(generation_config.get("temperature", 0.8)),
                top_k=int(generation_config.get("top_k", 24)),
                top_p=float(generation_config.get("top_p", 0.95)),
                repetition_penalty=float(generation_config.get("repetition_penalty", 1.0)),
            )
            generations[prompt] = sample_text
            print(f"Prompt: {prompt}\nGeneration: {sample_text}\n")
        except Exception as e:
            generations[prompt] = f"Error during generation: {e}"
            print(f"Error generating for prompt '{prompt}': {e}")

    # 5. Verify checkpoint round-trip (save/reload, compare state dicts)
    print("Verifying checkpoint round-trip...")
    roundtrip_path = checkpoint_path.parent / f"roundtrip_temp.pt"
    # Create temporary optimizer and scaler to fulfill save_checkpoint args
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    save_checkpoint(
        roundtrip_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        step=checkpoint.get("step", 0),
        epoch=checkpoint.get("epoch", 0),
        config=config,
    )

    # Reload and compare
    reloaded_model = DecoderOnlyTransformer(model_config).to(device)
    load_checkpoint(roundtrip_path, reloaded_model, map_location=device)

    roundtrip_success = compare_state_dicts(model.state_dict(), reloaded_model.state_dict())
    print(f"Roundtrip match: {roundtrip_success}")

    if roundtrip_path.exists():
        roundtrip_path.unlink()

    # 6. Output JSON report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path),
        "step": checkpoint.get("step", 0),
        "epoch": checkpoint.get("epoch", 0),
        "metrics": {
            "validation_loss": val_loss,
            "validation_perplexity": perplexity,
        },
        "generations": generations,
        "roundtrip_verification": {
            "success": roundtrip_success,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote evaluation report to {output_path}")


if __name__ == "__main__":
    main()
