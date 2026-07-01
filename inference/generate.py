"""Text generation from a Nexara checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from inference.sampling import sample_next_token
from model import DecoderOnlyTransformer, ModelConfig
from tokenizer import NexaraTokenizer
from training.checkpointing import load_checkpoint
from training.config import load_config
from training.train import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--config", default="", help="Config path if checkpoint lacks config.")
    parser.add_argument("--prompt", required=True, help="Prompt text.")
    parser.add_argument("--device", default="auto", help="Device override.")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get("config") or load_config(args.config)

    tokenizer = NexaraTokenizer(config["tokenizer"]["path"])
    model = DecoderOnlyTransformer(ModelConfig.from_mapping(config["model"])).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    generation_config = config["generation"]
    text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        device=device,
        max_new_tokens=args.max_new_tokens or int(generation_config["max_new_tokens"]),
        temperature=coalesce(args.temperature, float(generation_config["temperature"])),
        top_k=coalesce(args.top_k, int(generation_config["top_k"])),
        top_p=coalesce(args.top_p, float(generation_config["top_p"])),
        repetition_penalty=coalesce(
            args.repetition_penalty,
            float(generation_config["repetition_penalty"]),
        ),
    )
    print(text)


@torch.no_grad()
def generate_text(
    model: DecoderOnlyTransformer,
    tokenizer: NexaraTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
) -> str:
    token_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        context = input_ids[:, -model.config.max_sequence_length :]
        logits, _ = model(context)
        next_token = sample_next_token(
            logits=logits[:, -1, :],
            previous_tokens=input_ids,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        input_ids = torch.cat((input_ids, next_token), dim=1)
        if int(next_token.item()) == tokenizer.eos_id:
            break

    return tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)


def coalesce(value: Any, default: Any) -> Any:
    return default if value is None else value


if __name__ == "__main__":
    main()
