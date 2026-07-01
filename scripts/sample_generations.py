"""Generate text samples from a Nexara checkpoint for comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from datasets.statistics import write_statistics
from inference.generate import generate_text
from model import DecoderOnlyTransformer, ModelConfig
from tokenizer import NexaraTokenizer
from training.checkpointing import load_checkpoint
from training.train import resolve_device

DEFAULT_PROMPTS = [
    "Once upon a time",
    "The little dog",
    "Sara wanted to",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text samples from a checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint.")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer JSON.")
    parser.add_argument(
        "--prompts",
        nargs="*",
        default=None,
        help="Prompts to generate from. Defaults to built-in set.",
    )
    parser.add_argument(
        "--before-checkpoint",
        default="",
        help="Optional untrained checkpoint for comparison.",
    )
    parser.add_argument("--output", default="", help="Output JSON path.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--top-p", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = args.prompts if args.prompts else DEFAULT_PROMPTS
    results = generate_samples(
        checkpoint_path=Path(args.checkpoint),
        tokenizer_path=Path(args.tokenizer),
        prompts=prompts,
        before_checkpoint_path=Path(args.before_checkpoint) if args.before_checkpoint else None,
        device_name=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    output_path = Path(args.output) if args.output else None
    if output_path:
        write_statistics(output_path, results)
        print(f"wrote {output_path}")
    print(json.dumps(results, indent=2))


def generate_samples(
    checkpoint_path: Path,
    tokenizer_path: Path,
    prompts: list[str],
    before_checkpoint_path: Path | None = None,
    device_name: str = "auto",
    max_new_tokens: int = 48,
    temperature: float = 0.8,
    top_k: int = 24,
    top_p: float = 0.95,
) -> dict[str, Any]:
    """Generate text from multiple prompts using a trained checkpoint."""
    device = resolve_device(device_name)
    tokenizer = NexaraTokenizer(tokenizer_path)

    # Load the trained model
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    model_config = ModelConfig.from_mapping(config["model"])
    model = DecoderOnlyTransformer(model_config).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()

    generations_after: dict[str, str] = {}
    for prompt in prompts:
        text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=1.0,
        )
        generations_after[prompt] = text

    results: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "tokenizer": str(tokenizer_path),
        "prompts": prompts,
        "generations_after_training": generations_after,
    }

    # Optionally generate from an untrained/earlier checkpoint for comparison
    if before_checkpoint_path and before_checkpoint_path.exists():
        model_before = DecoderOnlyTransformer(model_config).to(device)
        load_checkpoint(before_checkpoint_path, model_before, map_location=device)
        model_before.eval()
        generations_before: dict[str, str] = {}
        for prompt in prompts:
            text = generate_text(
                model=model_before,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=1.0,
            )
            generations_before[prompt] = text
        results["generations_before_training"] = generations_before
        results["before_checkpoint"] = str(before_checkpoint_path)

    return results


if __name__ == "__main__":
    main()
