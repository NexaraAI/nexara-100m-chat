"""Terminal chat interface for a trained Nexara checkpoint using a Markdown template."""

from __future__ import annotations

import argparse
import json
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
    parser = argparse.ArgumentParser(description="Chat with a Nexara checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file.")
    parser.add_argument(
        "--config", default="", help="Path to config file if checkpoint lacks config."
    )
    parser.add_argument("--device", default="auto", help="Device to run inference on.")
    parser.add_argument(
        "--max-new-tokens", type=int, default=128, help="Max new tokens to generate."
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k filtering.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p filtering.")
    parser.add_argument(
        "--repetition-penalty", type=float, default=1.15, help="Repetition penalty."
    )
    return parser.parse_args()


def coalesce(value: Any, default: Any) -> Any:
    return default if value is None else value


@torch.no_grad()
def chat_generate(
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
    prompt_len = len(token_ids)

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

    new_token_ids = input_ids[0][prompt_len:].tolist()
    return tokenizer.decode(new_token_ids, skip_special_tokens=True)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get("config") or load_config(args.config)

    # Resolution of tokenizer path: check local directory first, then fallback to config path
    checkpoint_dir = Path(args.checkpoint).parent
    tokenizer_path = checkpoint_dir / "nexara-bpe.json"
    if not tokenizer_path.exists():
        tokenizer_path = Path(config["tokenizer"]["path"])

    tokenizer = NexaraTokenizer(str(tokenizer_path))
    model = DecoderOnlyTransformer(ModelConfig.from_mapping(config["model"])).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    # Load defaults from generation_config.json if present
    gen_config_path = checkpoint_dir / "generation_config.json"
    gen_defaults = {}
    if gen_config_path.exists():
        try:
            with gen_config_path.open("r", encoding="utf-8") as f:
                gen_defaults = json.load(f)
        except Exception:
            pass

    max_new_tokens = coalesce(args.max_new_tokens, gen_defaults.get("max_new_tokens", 128))
    temperature = coalesce(args.temperature, gen_defaults.get("temperature", 0.7))
    top_k = coalesce(args.top_k, gen_defaults.get("top_k", 40))
    top_p = coalesce(args.top_p, gen_defaults.get("top_p", 0.95))
    repetition_penalty = coalesce(
        args.repetition_penalty, gen_defaults.get("repetition_penalty", 1.15)
    )

    print("==================================================")
    print("Nexara SFT Chat CLI. Type /exit or /quit to stop.")
    print(
        f"Sampling settings: Temp={temperature}, Top-K={top_k}, Top-P={top_p}, Rep-Penalty={repetition_penalty}"
    )
    print("==================================================")

    conversation_history = []

    while True:
        try:
            user_text = input("you> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat.")
            break

        if user_text in {"/exit", "/quit"}:
            break
        if not user_text:
            continue

        conversation_history.append({"role": "user", "content": user_text})

        # Format using chat template
        prompt = "### System:\nYou are Nexara, a helpful and polite AI assistant.\n\n"
        for turn in conversation_history:
            if turn["role"] == "user":
                prompt += f"### User:\n{turn['content']}\n\n"
            elif turn["role"] == "assistant":
                prompt += f"### Assistant:\n{turn['content']}\n\n"
        prompt += "### Assistant:\n"

        # Generate response
        response = chat_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        response_clean = response.strip()
        print(f"nexara> {response_clean}")
        print()

        # Save assistant turn in history
        conversation_history.append({"role": "assistant", "content": response_clean})


if __name__ == "__main__":
    main()
