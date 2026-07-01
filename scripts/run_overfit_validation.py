"""Run a tiny overfit validation experiment."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any

import torch
from torch.utils.data import DataLoader

from datasets.statistics import write_statistics
from datasets.token_cache import (
    TokenCacheDataset,
    collect_token_cache_statistics,
    write_token_cache,
)
from inference.generate import generate_text
from model import DecoderOnlyTransformer, ModelConfig
from scripts.smoke_fixtures import write_tiny_stories_jsonl_subset
from tokenizer import NexaraTokenizer
from tokenizer.bpe import train_bpe_tokenizer
from tokenizer.train_tokenizer import DEFAULT_SAMPLE_TEXTS, write_tokenizer_reports
from training.checkpointing import load_checkpoint, save_checkpoint
from training.train import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny overfit validation.")
    parser.add_argument("--output-dir", default="logs/overfit")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--subset-size", type=int, default=100)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--experiment-name", default="phase1_3_overfit_validation")
    parser.add_argument("--no-update-experiments", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_overfit_validation(
        output_dir=Path(args.output_dir),
        steps=args.steps,
        subset_size=args.subset_size,
        vocab_size=args.vocab_size,
        device_name=args.device,
        experiment_name=args.experiment_name,
        update_experiments=not args.no_update_experiments,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


def run_overfit_validation(
    output_dir: Path,
    steps: int = 300,
    subset_size: int = 100,
    vocab_size: int = 256,
    device_name: str = "auto",
    experiment_name: str = "phase1_3_overfit_validation",
    update_experiments: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    created_at = datetime.now(timezone.utc).isoformat()
    device = resolve_device(device_name)
    torch.manual_seed(1337)

    # --- Data preparation ---
    data_path = write_tiny_stories_jsonl_subset(
        output_dir / "tiny_stories_subset.jsonl",
        count=subset_size,
    )
    tokenizer_path = output_dir / "tokenizer.json"
    cache_path = output_dir / "tokens.bin"
    checkpoint_path = output_dir / "overfit_final.pt"

    train_bpe_tokenizer(
        input_paths=[data_path],
        output_path=tokenizer_path,
        vocab_size=vocab_size,
        text_key="text",
        min_frequency=1,
        overwrite=True,
    )
    tokenizer = NexaraTokenizer(tokenizer_path)
    report_paths = write_tokenizer_reports(
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
        input_paths=[data_path],
        config={
            "tokenizer": {"vocab_size": vocab_size, "min_frequency": 1},
            "data": {"text_key": "text"},
        },
        sample_texts=DEFAULT_SAMPLE_TEXTS,
    )
    write_token_cache(
        input_paths=[data_path],
        tokenizer=tokenizer,
        output_path=cache_path,
        block_size=64,
        text_key="text",
        overwrite=True,
    )
    cache_statistics = collect_token_cache_statistics(cache_path)
    write_statistics(cache_path.with_suffix(".stats.json"), cache_statistics)

    dataset = TokenCacheDataset(cache_path, block_size=64)
    try:
        loader = DataLoader(dataset, batch_size=8, shuffle=False)
        input_ids, targets = next(iter(loader))
    finally:
        dataset.close()
    input_ids = input_ids.to(device)
    targets = targets.to(device)

    # --- Model setup ---
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        max_sequence_length=64,
        n_layers=2,
        n_heads=2,
        embedding_dim=64,
        dropout=0.0,
        tie_embeddings=True,
    )
    model = DecoderOnlyTransformer(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.0)

    # --- Pre-training evaluation ---
    model.eval()
    initial_loss = batch_loss(model, input_ids, targets)
    generation_before = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt="Once upon a time",
        device=device,
        max_new_tokens=32,
        temperature=0.8,
        top_k=24,
        top_p=0.95,
        repetition_penalty=1.0,
    )

    # --- Training with gradient tracking ---
    loss_curve: list[dict[str, Any]] = [{"step": 0, "loss": initial_loss}]
    gradient_norms: list[dict[str, Any]] = []
    model.train()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(input_ids, targets)
        if loss is None:
            raise RuntimeError("overfit loss was not computed")
        loss.backward()
        grad_norm_before_clip = compute_gradient_norm(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grad_norm_after_clip = compute_gradient_norm(model)
        optimizer.step()
        step_loss = float(loss.detach().cpu())
        loss_curve.append({"step": step, "loss": step_loss})
        gradient_norms.append(
            {
                "step": step,
                "grad_norm_before_clip": grad_norm_before_clip,
                "grad_norm_after_clip": grad_norm_after_clip,
            }
        )

    # --- Post-training evaluation ---
    model.eval()
    final_loss = batch_loss(model, input_ids, targets)
    loss_curve.append({"step": steps + 1, "loss": final_loss, "kind": "final_eval"})
    if final_loss >= initial_loss:
        raise RuntimeError(
            f"overfit validation failed: final loss {final_loss:.4f} "
            f"did not improve initial loss {initial_loss:.4f}"
        )

    # --- Checkpoint save and reload ---
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scaler=None,
        step=steps,
        epoch=0,
        config={"model": model_config.__dict__, "tokenizer": {"path": str(tokenizer_path)}},
        metrics={"initial_loss": initial_loss, "final_loss": final_loss},
    )
    reloaded = DecoderOnlyTransformer(model_config).to(device)
    load_checkpoint(checkpoint_path, reloaded, map_location=device)
    reloaded.eval()
    generation_after = generate_text(
        model=reloaded,
        tokenizer=tokenizer,
        prompt="Once upon a time",
        device=device,
        max_new_tokens=32,
        temperature=0.8,
        top_k=24,
        top_p=0.95,
        repetition_penalty=1.0,
    )

    # --- Gradient statistics ---
    all_before = [g["grad_norm_before_clip"] for g in gradient_norms]
    all_after = [g["grad_norm_after_clip"] for g in gradient_norms]
    gradient_statistics = {
        "before_clip": {
            "min": min(all_before),
            "max": max(all_before),
            "mean": sum(all_before) / len(all_before),
        },
        "after_clip": {
            "min": min(all_after),
            "max": max(all_after),
            "mean": sum(all_after) / len(all_after),
        },
        "steps_tracked": len(gradient_norms),
    }

    # --- Write artifacts ---
    loss_curve_json = output_dir / "loss_curve.json"
    loss_curve_csv = output_dir / "loss_curve.csv"
    write_statistics(loss_curve_json, {"loss_curve": loss_curve})
    write_loss_curve_csv(loss_curve_csv, loss_curve, gradient_norms)
    loss_curve_png = write_loss_curve_plot(output_dir / "loss_curve.png", loss_curve)

    gradient_norms_path = output_dir / "gradient_norms.json"
    write_statistics(
        gradient_norms_path,
        {
            "gradient_norms": gradient_norms,
            "statistics": gradient_statistics,
        },
    )

    generations_path = output_dir / "generation_examples.json"
    generation_examples = {
        "prompt": "Once upon a time",
        "before_training": generation_before,
        "after_training": generation_after,
    }
    write_statistics(generations_path, generation_examples)

    duration_seconds = time.perf_counter() - started
    metrics = {
        "experiment_name": experiment_name,
        "created_at": created_at,
        "duration_seconds": duration_seconds,
        "dataset": "Synthetic TinyStories-like subset",
        "subset_size": subset_size,
        "hyperparameters": {
            "steps": steps,
            "learning_rate": 5e-3,
            "batch_size": 8,
            "block_size": 64,
            "vocab_size": tokenizer.vocab_size,
            "n_layers": model_config.n_layers,
            "n_heads": model_config.n_heads,
            "embedding_dim": model_config.embedding_dim,
            "dropout": model_config.dropout,
        },
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_delta": initial_loss - final_loss,
        "loss_curve": loss_curve,
        "gradient_statistics": gradient_statistics,
        "observations": [
            "Overfit validation passed: final loss decreased on a fixed batch.",
            "Gradient norms tracked per step to verify healthy optimization.",
            "Generated examples are qualitative smoke artifacts, not model-quality evidence.",
        ],
        "generation_examples": generation_examples,
        "artifacts": {
            "data": str(data_path),
            "tokenizer": str(tokenizer_path),
            "vocabulary_report": str(report_paths["vocabulary_report"]),
            "sample_encodings": str(report_paths["sample_encodings"]),
            "token_cache": str(cache_path),
            "token_cache_statistics": str(cache_path.with_suffix(".stats.json")),
            "checkpoint": str(checkpoint_path),
            "loss_curve_json": str(loss_curve_json),
            "loss_curve_csv": str(loss_curve_csv),
            "loss_curve_png": str(loss_curve_png) if loss_curve_png else "",
            "generation_examples": str(generations_path),
            "gradient_norms": str(gradient_norms_path),
        },
        "cache_statistics": cache_statistics,
    }
    metrics_path = output_dir / "metrics.json"
    write_statistics(metrics_path, metrics)
    if update_experiments:
        from scripts.generate_experiment_report import update_experiments_from_metrics

        update_experiments_from_metrics(metrics_path)
    return metrics


@torch.no_grad()
def batch_loss(
    model: DecoderOnlyTransformer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    _, loss = model(input_ids, targets)
    if loss is None:
        raise RuntimeError("loss was not computed")
    return float(loss.detach().cpu())


def compute_gradient_norm(model: torch.nn.Module) -> float:
    """Compute the L2 norm of all parameter gradients."""
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().norm(2).item() ** 2)
    return total**0.5


def write_loss_curve_csv(
    path: Path,
    loss_curve: list[dict[str, Any]],
    gradient_norms: list[dict[str, Any]] | None = None,
) -> Path:
    grad_lookup: dict[int, dict[str, Any]] = {}
    if gradient_norms:
        for entry in gradient_norms:
            grad_lookup[int(entry["step"])] = entry

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["step", "loss", "kind", "grad_norm_before_clip", "grad_norm_after_clip"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in loss_curve:
            step = int(row.get("step", 0))
            grad = grad_lookup.get(step, {})
            writer.writerow(
                {
                    "step": step,
                    "loss": row.get("loss", ""),
                    "kind": row.get("kind", "train"),
                    "grad_norm_before_clip": grad.get("grad_norm_before_clip", ""),
                    "grad_norm_after_clip": grad.get("grad_norm_after_clip", ""),
                }
            )
    return path


def write_loss_curve_plot(path: Path, loss_curve: list[dict[str, Any]]) -> Path | None:
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(path.parent / "matplotlib_cache"))
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    steps = [int(row["step"]) for row in loss_curve]
    losses = [float(row["loss"]) for row in loss_curve]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, losses, marker="o", linewidth=1.5, markersize=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Overfit Loss Curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
