"""Runtime smoke checks for the scheduled CI workflow."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from inference.sampling import sample_next_token
from model import DecoderOnlyTransformer, ModelConfig
from training.checkpointing import load_checkpoint, save_checkpoint


def main() -> None:
    output_dir = Path("logs/nightly")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "tiny_checkpoint.pt"

    config = ModelConfig(
        vocab_size=32,
        max_sequence_length=16,
        n_layers=2,
        n_heads=2,
        embedding_dim=32,
        dropout=0.0,
        tie_embeddings=True,
    )
    model = DecoderOnlyTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    input_ids = torch.randint(0, config.vocab_size, (2, 8), dtype=torch.long)
    targets = torch.randint(0, config.vocab_size, (2, 8), dtype=torch.long)

    logits, loss = model(input_ids, targets)
    if loss is None:
        raise RuntimeError("forward pass did not return a loss")
    loss.backward()
    optimizer.step()

    next_token = sample_next_token(
        logits[:, -1, :],
        previous_tokens=input_ids,
        temperature=0.8,
        top_k=8,
        top_p=0.9,
        repetition_penalty=1.05,
    )
    if next_token.shape != (2, 1):
        raise RuntimeError(f"unexpected sampled token shape: {tuple(next_token.shape)}")

    checkpoint_config = {"model": config.__dict__}
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scaler=None,
        step=1,
        epoch=0,
        config=checkpoint_config,
        metrics={"loss": float(loss.detach().cpu())},
    )

    reloaded = DecoderOnlyTransformer(config)
    checkpoint = load_checkpoint(checkpoint_path, reloaded, map_location="cpu")
    if int(checkpoint["step"]) != 1:
        raise RuntimeError("checkpoint step did not round-trip")

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "loss": float(loss.detach().cpu()),
        "logits_shape": list(logits.shape),
        "sampled_tokens": next_token.squeeze(1).tolist(),
        "checkpoint": str(checkpoint_path),
    }
    report_path = output_dir / "nightly_smoke.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
