"""Training loop for Nexara."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.text import TokenBlockDataset
from datasets.token_cache import StreamingTokenCacheDataset, TokenCacheDataset
from model import DecoderOnlyTransformer, ModelConfig
from tokenizer import NexaraTokenizer
from training.checkpointing import load_checkpoint, save_checkpoint, rotate_checkpoints
from training.config import load_config
from training.logger import JsonlMetricLogger
from training.parameter_count import estimate_transformer_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Nexara from scratch.")
    parser.add_argument("--config", required=True, help="Path to a Nexara TOML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(load_config(args.config))


def get_lr(step: int, max_steps: int, warmup_steps: int, base_lr: float, min_lr: float) -> float:
    # 1) linear warmup for warmup_steps
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    # 2) if step > max_steps, return min_lr
    if step > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min_lr
    decay_ratio = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (base_lr - min_lr)


@torch.no_grad()
def generate_samples_during_training(
    model: torch.nn.Module,
    tokenizer: NexaraTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 48,
) -> str:
    token_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    model.eval()
    for _ in range(max_new_tokens):
        context = input_ids[:, -model.config.max_sequence_length :]
        logits, _ = model(context)
        next_token_logits = logits[:, -1, :]
        probs = torch.softmax(next_token_logits / 0.8, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat((input_ids, next_token), dim=1)
        if int(next_token.item()) == tokenizer.eos_id:
            break

    return tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)


def train_step(
    model: DecoderOnlyTransformer,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    precision: str,
    gradient_accumulation_steps: int,
    grad_clip: float,
    is_update_step: bool,
) -> tuple[float, float, float]:
    """Perform a single training step. Returns (loss_val, grad_norm_before, grad_norm_after)."""
    with autocast_context(device, precision):
        _, loss = model(input_ids, targets)
        if loss is None:
            raise RuntimeError("training loss was not computed")
        scaled_loss = loss / gradient_accumulation_steps

    if scaler.is_enabled():
        scaler.scale(scaled_loss).backward()
    else:
        scaled_loss.backward()

    loss_val = float(loss.detach().cpu())
    grad_norm_before = 0.0
    grad_norm_after = 0.0

    if is_update_step:
        if scaler.is_enabled():
            scaler.unscale_(optimizer)

        # Calculate gradient norm before clipping
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        grad_norm_before = total_norm**0.5

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=grad_clip,
        )

        # Calculate gradient norm after clipping
        total_norm_after = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm_after += param_norm.item() ** 2
        grad_norm_after = total_norm_after**0.5

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return loss_val, grad_norm_before, grad_norm_after


def train(config: dict[str, Any]) -> None:
    seed_everything(int(config["run"].get("seed", 1337)))
    device = resolve_device(str(config["run"].get("device", "auto")))
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    tokenizer = NexaraTokenizer(config["tokenizer"]["path"])
    if tokenizer.vocab_size != int(config["model"]["vocab_size"]):
        raise ValueError(
            f"tokenizer vocab size {tokenizer.vocab_size} does not match "
            f"model vocab size {config['model']['vocab_size']}"
        )

    train_dataset = build_dataset(config, tokenizer, split="train")
    validation_dataset = build_dataset(config, tokenizer, split="validation")
    train_loader = build_loader(config, train_dataset, shuffle=True, device=device)
    validation_loader = build_loader(config, validation_dataset, shuffle=False, device=device)

    model_config = ModelConfig.from_mapping(config["model"])
    model = DecoderOnlyTransformer(model_config).to(device)
    estimated_parameters = estimate_transformer_parameters(config["model"])
    actual_parameters = model.count_parameters()
    print(f"estimated parameters: {estimated_parameters:,}")
    print(f"actual parameters: {actual_parameters:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        betas=(float(config["training"]["beta1"]), float(config["training"]["beta2"])),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    precision = str(config["training"].get("precision", "fp32"))
    use_amp = (device.type in {"cuda", "cpu"}) and precision in {"fp16", "bf16"}
    if device.type == "cpu" and precision == "fp16":
        use_amp = False

    scaler = torch.cuda.amp.GradScaler(
        enabled=use_amp and precision == "fp16" and device.type == "cuda"
    )

    start_epoch = 0
    global_step = 0
    resume_from = str(config["training"].get("resume_from", ""))
    if resume_from:
        checkpoint = load_checkpoint(resume_from, model, optimizer, scaler, map_location=device)
        start_epoch = int(checkpoint.get("epoch", 0))
        global_step = int(checkpoint.get("step", 0))

    # Compile the model if requested and CUDA is active
    if bool(config["training"].get("compile", False)):
        if hasattr(torch, "compile") and device.type == "cuda":
            print("Compiling model with torch.compile for extra speed...")
            model = torch.compile(model)

    log_dir = Path(config["run"]["log_dir"])
    output_dir = Path(config["run"]["output_dir"])
    metric_logger = JsonlMetricLogger(log_dir / "train_metrics.jsonl")
    gradient_accumulation_steps = int(config["training"]["gradient_accumulation_steps"])
    max_steps = int(config["training"]["max_steps"])

    warmup_steps = int(config["training"].get("warmup_steps", 1000))
    learning_rate = float(config["training"]["learning_rate"])
    min_lr = float(config["training"].get("min_lr", learning_rate * 0.1))
    early_stopping_patience = int(config["training"].get("early_stopping_patience", 5))

    best_val_loss = float("inf")
    patience_counter = 0

    start_time = time.time()
    last_log_time = start_time
    last_log_step = global_step

    optimizer.zero_grad(set_to_none=True)
    model.train()

    for epoch in range(start_epoch, int(config["training"]["max_epochs"])):
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}", leave=False)
        for micro_step, (input_ids, targets) in enumerate(progress, start=1):
            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            is_update_step = micro_step % gradient_accumulation_steps == 0

            if is_update_step:
                lr = get_lr(
                    global_step,
                    max_steps=max_steps,
                    warmup_steps=warmup_steps,
                    base_lr=learning_rate,
                    min_lr=min_lr,
                )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

            try:
                loss_val, grad_norm_before, grad_norm_after = train_step(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    input_ids=input_ids,
                    targets=targets,
                    device=device,
                    precision=precision,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    grad_clip=float(config["training"]["grad_clip"]),
                    is_update_step=is_update_step,
                )
            except Exception as e:
                raise e

            if math.isnan(loss_val) or math.isinf(loss_val):
                emergency_path = output_dir / f"emergency_nan_step_{global_step + 1}.pt"
                save_checkpoint(
                    emergency_path,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=global_step + 1,
                    epoch=epoch,
                    config=config,
                    metrics={"loss": loss_val},
                )
                raise RuntimeError(
                    f"NaN/Inf loss detected at step {global_step + 1}. "
                    f"Saved emergency checkpoint to {emergency_path}"
                )

            if not is_update_step:
                continue

            global_step += 1
            progress.set_postfix(step=global_step, loss=f"{loss_val:.4f}")

            if global_step % int(config["training"]["log_interval"]) == 0:
                current_time = time.time()
                elapsed = current_time - last_log_time
                steps_done = global_step - last_log_step

                # Calculate tokens per second
                tokens_per_step = (
                    int(config["data"]["batch_size"])
                    * int(config["training"]["gradient_accumulation_steps"])
                    * int(config["data"]["block_size"])
                )
                tokens_processed = steps_done * tokens_per_step
                tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0.0

                metric_logger.write(
                    {
                        "step": global_step,
                        "epoch": epoch,
                        "train_loss": loss_val,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "grad_norm_before": grad_norm_before,
                        "grad_norm_after": grad_norm_after,
                        "timestamp": current_time,
                        "tokens_per_sec": tokens_per_sec,
                    }
                )
                last_log_time = current_time
                last_log_step = global_step

            if global_step % int(config["training"]["eval_interval"]) == 0:
                validation_loss = evaluate_loss(
                    model=model,
                    loader=validation_loader,
                    device=device,
                    max_batches=int(config["training"]["eval_batches"]),
                )

                if validation_loss < best_val_loss:
                    best_val_loss = validation_loss
                    patience_counter = 0
                    save_checkpoint(
                        output_dir / "best.pt",
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        step=global_step,
                        epoch=epoch,
                        config=config,
                        metrics={
                            "validation_loss": validation_loss,
                            "best_validation_loss": best_val_loss,
                        },
                    )
                else:
                    patience_counter += 1

                metric_logger.write(
                    {
                        "step": global_step,
                        "epoch": epoch,
                        "validation_loss": validation_loss,
                        "validation_perplexity": safe_exp(validation_loss),
                        "best_validation_loss": best_val_loss,
                        "timestamp": time.time(),
                    }
                )

                save_checkpoint(
                    output_dir / "latest.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=global_step,
                    epoch=epoch,
                    config=config,
                    metrics={
                        "validation_loss": validation_loss,
                        "best_validation_loss": best_val_loss,
                    },
                )

                # Generate samples
                from inference.generate import generate_text

                sample_prompts = config["tokenizer"].get("sample_texts") or [
                    "Once upon a time",
                    "The little dog was",
                    "Lily wanted to",
                    "The sun was shining",
                    "Tom and his sister",
                    "There was a small bird",
                    "Ben went to school",
                    "The rabbit saw a flower",
                ]
                sample_results = {}
                generation_config = config.get("generation", {})
                for prompt in sample_prompts:
                    try:
                        sample_text = generate_text(
                            model=model,
                            tokenizer=tokenizer,
                            prompt=prompt,
                            device=device,
                            max_new_tokens=int(generation_config.get("max_new_tokens", 128)),
                            temperature=float(generation_config.get("temperature", 0.8)),
                            top_k=int(generation_config.get("top_k", 40)),
                            top_p=float(generation_config.get("top_p", 0.95)),
                            repetition_penalty=float(
                                generation_config.get("repetition_penalty", 1.1)
                            ),
                        )
                        sample_results[prompt] = sample_text
                    except Exception as e:
                        sample_results[prompt] = f"Error: {e}"

                sample_log_path = log_dir / f"samples_step_{global_step}.json"
                sample_log_path.parent.mkdir(parents=True, exist_ok=True)
                with sample_log_path.open("w", encoding="utf-8") as f:
                    json.dump(
                        {"step": global_step, "epoch": epoch, "samples": sample_results},
                        f,
                        indent=2,
                    )

                model.train()

                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {global_step} steps.")
                    save_checkpoint(
                        output_dir / "final.pt",
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        step=global_step,
                        epoch=epoch,
                        config=config,
                        metrics={"validation_loss": validation_loss},
                    )
                    return

            if global_step % int(config["training"]["checkpoint_interval"]) == 0:
                save_checkpoint(
                    output_dir / f"step_{global_step}.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=global_step,
                    epoch=epoch,
                    config=config,
                )
                keep_last_n = int(config["training"].get("keep_last_n", 5))
                rotate_checkpoints(output_dir, keep_last_n=keep_last_n)

            if global_step >= max_steps:
                save_checkpoint(
                    output_dir / "final.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=global_step,
                    epoch=epoch,
                    config=config,
                )
                return

    save_checkpoint(
        output_dir / "final.pt",
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        step=global_step,
        epoch=int(config["training"]["max_epochs"]) - 1,
        config=config,
    )


def build_dataset(
    config: dict[str, Any],
    tokenizer: NexaraTokenizer,
    split: str,
) -> TokenBlockDataset | TokenCacheDataset | StreamingTokenCacheDataset:
    data_config = config["data"]
    if split == "train":
        path = data_config["train_path"]
        cache_path = data_config.get("train_cache_path")
        max_documents = int(data_config.get("max_train_documents", 0))
    elif split == "validation":
        path = data_config["validation_path"]
        cache_path = data_config.get("validation_cache_path")
        max_documents = int(data_config.get("max_validation_documents", 0))
    else:
        raise ValueError(f"unknown split {split!r}")

    if bool(data_config.get("use_token_cache", False)):
        if not cache_path:
            raise ValueError(f"missing token cache path for split {split!r}")
        if bool(data_config.get("streaming", False)):
            return StreamingTokenCacheDataset(
                cache_path,
                block_size=int(data_config["block_size"]),
                use_mmap=bool(data_config.get("memory_map", True)),
            )
        return TokenCacheDataset(cache_path, block_size=int(data_config["block_size"]))

    return TokenBlockDataset(
        paths=[path],
        tokenizer=tokenizer,
        block_size=int(data_config["block_size"]),
        text_key=str(data_config.get("text_key", "text")),
        max_documents=max_documents,
    )


def seed_worker(worker_id: int) -> None:
    worker_seed = int(torch.initial_seed() % (2**32))
    import numpy as np
    import random

    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loader(
    config: dict[str, Any],
    dataset: TokenBlockDataset | TokenCacheDataset | StreamingTokenCacheDataset,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    data_config = config["data"]
    if isinstance(dataset, torch.utils.data.IterableDataset):
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=int(data_config["batch_size"]),
        shuffle=shuffle,
        num_workers=int(data_config.get("num_workers", 0)),
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker if int(data_config.get("num_workers", 0)) > 0 else None,
    )


@torch.no_grad()
def evaluate_loss(
    model: DecoderOnlyTransformer,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> float:
    model.eval()
    losses: list[float] = []
    for batch_index, (input_ids, targets) in enumerate(loader):
        if batch_index >= max_batches:
            break
        input_ids = input_ids.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        _, loss = model(input_ids, targets)
        if loss is None:
            raise RuntimeError("validation loss was not computed")
        losses.append(float(loss.detach().cpu()))
    if not losses:
        raise ValueError("validation loader produced no batches")
    return sum(losses) / len(losses)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available")
    return device


def autocast_context(device: torch.device, precision: str):
    if precision not in {"fp16", "bf16"}:
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=dtype)
    elif device.type == "cpu" and precision == "bf16":
        return torch.amp.autocast(device_type="cpu", dtype=dtype)
    return nullcontext()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_exp(value: float) -> float:
    if value > 50:
        return float("inf")
    return float(math.exp(value))


if __name__ == "__main__":
    main()
