"""Tiny end-to-end training smoke test."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.statistics import write_statistics
from datasets.token_cache import TokenCacheDataset, write_token_cache
from inference.generate import generate_text
from model import DecoderOnlyTransformer, ModelConfig
from scripts.smoke_fixtures import write_tiny_stories_jsonl
from tokenizer import NexaraTokenizer
from tokenizer.bpe import train_bpe_tokenizer
from training.checkpointing import load_checkpoint, save_checkpoint
from training.train import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny Nexara training smoke test.")
    parser.add_argument("--output-dir", default="logs/train_smoke")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    torch.manual_seed(1337)

    data_path = write_tiny_stories_jsonl(output_dir / "tiny_stories.jsonl", repeat=2)
    tokenizer_path = output_dir / "tokenizer.json"
    cache_path = output_dir / "tokens.bin"
    checkpoint_path = output_dir / "tiny_checkpoint.pt"

    train_bpe_tokenizer(
        input_paths=[data_path],
        output_path=tokenizer_path,
        vocab_size=128,
        text_key="text",
        min_frequency=1,
        overwrite=True,
    )
    tokenizer = NexaraTokenizer(tokenizer_path)
    write_token_cache(
        input_paths=[data_path],
        tokenizer=tokenizer,
        output_path=cache_path,
        block_size=32,
        text_key="text",
        overwrite=True,
    )
    dataset = TokenCacheDataset(cache_path, block_size=32)
    try:
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        input_ids, targets = next(iter(loader))
    finally:
        dataset.close()
    input_ids = input_ids.to(device)
    targets = targets.to(device)

    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        max_sequence_length=32,
        n_layers=2,
        n_heads=2,
        embedding_dim=32,
        dropout=0.0,
        tie_embeddings=True,
    )
    model = DecoderOnlyTransformer(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    before_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt="Once upon a time",
        device=device,
        max_new_tokens=16,
        temperature=0.8,
        top_k=16,
        top_p=0.95,
        repetition_penalty=1.0,
    )

    model.train()
    _, loss = model(input_ids, targets)
    if loss is None:
        raise RuntimeError("training smoke loss was not computed")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scaler=None,
        step=1,
        epoch=0,
        config={"model": model_config.__dict__, "tokenizer": {"path": str(tokenizer_path)}},
        metrics={"loss": float(loss.detach().cpu())},
    )
    reloaded = DecoderOnlyTransformer(model_config).to(device)
    checkpoint = load_checkpoint(checkpoint_path, reloaded, map_location=device)
    if int(checkpoint["step"]) != 1:
        raise RuntimeError("checkpoint step did not round-trip")
    reloaded.eval()
    after_text = generate_text(
        model=reloaded,
        tokenizer=tokenizer,
        prompt="Once upon a time",
        device=device,
        max_new_tokens=16,
        temperature=0.8,
        top_k=16,
        top_p=0.95,
        repetition_penalty=1.0,
    )
    if not after_text.strip():
        raise RuntimeError("generation smoke test produced empty text")

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "loss": float(loss.detach().cpu()),
        "checkpoint": str(checkpoint_path),
        "generation_before": before_text,
        "generation_after": after_text,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "dataset_sequences": len(dataset),
    }
    report_path = output_dir / "train_smoke.json"
    write_statistics(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
