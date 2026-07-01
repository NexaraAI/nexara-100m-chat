import argparse
import json
import math
import os
import random
import time
from pathlib import Path
import tomllib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import DecoderOnlyTransformer, ModelConfig
from tokenizer import NexaraTokenizer
from training.checkpointing import load_checkpoint, save_checkpoint
from training.sft.dataset import SFTDataset

# Evaluation prompts for validation checks during training
EVAL_PROMPTS = [
    ("Greeting", "Hi"),
    ("Identity", "What is your name?"),
    ("Identity Details", "Who developed you?"),
    ("Simple Explanation", "Why is the sky blue?"),
    ("Simple Math", "What is 3 + 4?"),
    ("Markdown List", "List three colors in a markdown list."),
    ("Web Refusal", "Can you search the web for today's news?"),
    ("Unknown Factual", "Who won the soccer match yesterday?"),
    ("Story Writing", "Write a short story about a little dog."),
    (
        "Multi-turn Memory",
        "### User:\nHello, my name is Alice.\n\n### Assistant:\nHello Alice!\n\n### User:\nWhat is my name?",
    ),
    ("Story Retention", "Once upon a time there was a rabbit"),
]


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def get_lr_multiplier(step: int, warmup_steps: int, max_steps: int) -> float:
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(progress * math.pi))


def run_evaluation_generations(model, tokenizer, device, step_num):
    print(f"\n=== SFT Generational Evaluation (Step {step_num}) ===")
    model.eval()

    results = {}
    with torch.no_grad():
        for category, prompt in EVAL_PROMPTS:
            # Construct SFT prompt
            if "### User:" in prompt:
                formatted_prompt = f"### System:\nYou are Nexara, a helpful and polite AI assistant.\n\n{prompt}\n\n### Assistant:\n"
            else:
                formatted_prompt = f"### System:\nYou are Nexara, a helpful and polite AI assistant.\n\n### User:\n{prompt}\n\n### Assistant:\n"

            input_ids = tokenizer.encode(formatted_prompt, add_bos=True)
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

            # Simple autoregressive decoding
            generated_ids = []
            for _ in range(80):  # limit length of generation
                logits, _ = model(input_tensor)
                next_token_logits = logits[0, -1, :]

                # Temperature scaling
                next_token_logits = next_token_logits / 0.7
                probs = torch.softmax(next_token_logits, dim=-1)

                # Sample
                next_token = torch.multinomial(probs, num_samples=1).item()

                if next_token == tokenizer.eos_id:
                    break

                generated_ids.append(next_token)
                input_tensor = torch.cat(
                    [input_tensor, torch.tensor([[next_token]], device=device)], dim=1
                )

            generated_text = tokenizer.decode(generated_ids)
            print(f"[{category}] Prompt: {prompt.strip()}")
            print(f"[{category}] Response: {generated_text.strip()}\n")
            results[category] = {"prompt": prompt, "response": generated_text}
    model.train()
    return results


def evaluate_loss(model, dataloader, device, eval_batches=10):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)
            _, loss = model(input_ids, targets)
            total_loss += loss.item()
            count += 1
            if count >= eval_batches:
                break
    model.train()
    return total_loss / max(1, count)


def main():
    parser = argparse.ArgumentParser(description="Run SFT instruction tuning for Nexara-0.2-chat.")
    parser.add_argument("--config", default="configs/stage2_sft.toml", help="Path to config file.")
    parser.add_argument("--overfit", action="store_true", help="Run in SFT overfit mode.")
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    # Seeds
    seed = config["run"].get("seed", 1337)
    torch.manual_seed(seed)
    random.seed(seed)

    # Device
    device = resolve_device(config["run"].get("device", "auto"))
    print(f"Using device: {device}")

    # Outputs
    output_dir = Path(config["run"]["output_dir"])
    log_dir = Path(config["run"]["log_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer
    tokenizer = NexaraTokenizer(config["tokenizer"]["path"])

    # Dataset
    train_path = config["data"]["train_path"]
    val_path = config["data"]["validation_path"]

    if args.overfit:
        # SFT Overfit mode uses a tiny 10-example subset
        print("Running in SFT OVERFIT MODE (10 examples)...")
        train_dataset = SFTDataset(
            train_path, tokenizer, max_seq_length=config["data"]["block_size"]
        )
        # Override SFTDataset examples
        train_dataset.examples = train_dataset.examples[:10]
        val_dataset = train_dataset

        # Override steps for overfit check
        config["training"]["max_steps"] = 80
        config["training"]["log_interval"] = 5
        config["training"]["eval_interval"] = 20
        config["training"]["checkpoint_interval"] = 40
        config["training"]["warmup_steps"] = 10
    else:
        print(f"Loading datasets: Train={train_path}, Val={val_path}")
        train_dataset = SFTDataset(
            train_path, tokenizer, max_seq_length=config["data"]["block_size"]
        )
        val_dataset = SFTDataset(val_path, tokenizer, max_seq_length=config["data"]["block_size"])

    train_loader = DataLoader(train_dataset, batch_size=config["data"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["data"]["batch_size"], shuffle=False)

    # Model configuration
    model_config = ModelConfig(
        vocab_size=config["model"]["vocab_size"],
        max_sequence_length=config["model"]["max_sequence_length"],
        n_layers=config["model"]["n_layers"],
        n_heads=config["model"]["n_heads"],
        embedding_dim=config["model"]["embedding_dim"],
        dropout=config["model"]["dropout"],
        mlp_ratio=config["model"]["mlp_ratio"],
        bias=config["model"]["bias"],
        rope_base=config["model"]["rope_base"],
        tie_embeddings=config["model"]["tie_embeddings"],
    )

    # Model
    model = DecoderOnlyTransformer(model_config)
    resume_from = config["training"]["resume_from"]
    if resume_from and Path(resume_from).exists():
        print(f"Loading base checkpoint: {resume_from}")
        load_checkpoint(resume_from, model, map_location=device)
    else:
        print(f"No checkpoint found at {resume_from}. Starting SFT from raw initialization.")

    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        betas=(config["training"]["beta1"], config["training"]["beta2"]),
        weight_decay=config["training"]["weight_decay"],
    )

    # Scaler for mixed precision
    use_amp = device.type == "cuda" and config["training"].get("precision") == "fp16"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Hyperparams
    max_steps = config["training"]["max_steps"]
    warmup_steps = config["training"]["warmup_steps"]
    base_lr = config["training"]["learning_rate"]
    grad_clip = config["training"]["grad_clip"]
    grad_accum_steps = config["training"]["gradient_accumulation_steps"]

    # Metrics log path
    metric_file = log_dir / ("sft_overfit_metrics.jsonl" if args.overfit else "sft_metrics.jsonl")
    metric_handle = metric_file.open("w", encoding="utf-8")

    # Training Loop
    step = 0
    epoch = 0
    t0 = time.time()

    # Initial before SFT generation
    if args.overfit:
        run_evaluation_generations(model, tokenizer, device, step)

    print(f"Starting SFT instruction tuning loop. Max steps: {max_steps}")

    while step < max_steps:
        epoch += 1
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)

            # Forward pass
            if use_amp:
                with torch.cuda.amp.autocast():
                    _, loss = model(input_ids, targets)
                scaled_loss = loss / grad_accum_steps
                scaler.scale(scaled_loss).backward()
            else:
                _, loss = model(input_ids, targets)
                scaled_loss = loss / grad_accum_steps
                scaled_loss.backward()

            # Optimizer step (simulating gradient accumulation)
            if (step + 1) % grad_accum_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

                optimizer.zero_grad()

                # Scheduler SFT learning rate
                lr_mult = get_lr_multiplier(step, warmup_steps, max_steps)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = base_lr * lr_mult

            step += 1

            # Log progress
            if step % config["training"]["log_interval"] == 0 or step == 1:
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                print(
                    f"Step {step}/{max_steps} | Epoch {epoch} | Loss: {loss.item():.4f} | LR: {optimizer.param_groups[0]['lr']:.7f} | Step Time: {dt:.2f}s"
                )

                # Write to metric log
                metric_handle.write(
                    json.dumps(
                        {
                            "step": step,
                            "train_loss": loss.item(),
                            "lr": optimizer.param_groups[0]["lr"],
                            "timestamp": time.time(),
                        }
                    )
                    + "\n"
                )
                metric_handle.flush()

            # SFT Evaluation and Checkpointing
            if step % config["training"]["eval_interval"] == 0 or step == max_steps:
                val_loss = evaluate_loss(
                    model, val_loader, device, eval_batches=config["training"]["eval_batches"]
                )
                val_ppl = math.exp(val_loss) if val_loss < 20 else 999.9
                print(
                    f"\n>>> Evaluation Step {step} | Val Loss: {val_loss:.4f} | Val Perplexity: {val_ppl:.4f}\n"
                )

                # Run sample generation checks
                generations = run_evaluation_generations(model, tokenizer, device, step)

                # Log val metrics
                metric_handle.write(
                    json.dumps(
                        {
                            "step": step,
                            "validation_loss": val_loss,
                            "validation_perplexity": val_ppl,
                            "generations": generations,
                            "timestamp": time.time(),
                        }
                    )
                    + "\n"
                )
                metric_handle.flush()

            # Checkpoint Save
            if step % config["training"]["checkpoint_interval"] == 0 or step == max_steps:
                ckpt_name = "sft_overfit_final.pt" if args.overfit else f"step_{step}.pt"
                ckpt_path = output_dir / ckpt_name
                save_checkpoint(
                    ckpt_path,
                    model,
                    optimizer,
                    scaler,
                    step,
                    epoch,
                    config,
                    metrics={"loss": loss.item()},
                )
                print(f"Saved SFT checkpoint to {ckpt_path}")

            if step >= max_steps:
                break

    metric_handle.close()

    # Save final best.pt checkpoint
    best_path = output_dir / "best.pt"
    save_checkpoint(
        best_path, model, optimizer, scaler, step, epoch, config, metrics={"loss": loss.item()}
    )
    print(f"Saved SFT checkpoint to {best_path}")
    print("SFT loop completed.")


if __name__ == "__main__":
    main()
