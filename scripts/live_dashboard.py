"""Live training dashboard for Nexara.

Displays real-time GPU/VRAM utilization, step progress, loss, learning rate, grad norm, tokens/sec, ETA, and sample generation outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any
import torch

# ANSI escape codes for coloring
COLOR_BLUE = "\033[94m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_CYAN = "\033[96m"
COLOR_MAGENTA = "\033[95m"
COLOR_RED = "\033[91m"
COLOR_BOLD = "\033[1m"
COLOR_RESET = "\033[0m"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live training dashboard for Nexara.")
    parser.add_argument(
        "--metrics-file",
        default="logs/stage1/train_metrics.jsonl",
        help="Path to train_metrics.jsonl",
    )
    parser.add_argument(
        "--sample-dir", default="logs/stage1", help="Directory where sample generations are saved"
    )
    parser.add_argument(
        "--refresh-rate", type=float, default=1.0, help="Refresh interval in seconds"
    )
    parser.add_argument(
        "--max-steps", type=int, default=100000, help="Total training steps for ETA calculation"
    )
    return parser.parse_args()


def get_gpu_status() -> tuple[str, str]:
    gpu_util = "N/A"
    vram_used = "N/A"

    # Try torch cuda
    if torch.cuda.is_available():
        try:
            device_idx = torch.cuda.current_device()
            mem_alloc = torch.cuda.memory_allocated(device_idx) / (1024**2)  # MB
            mem_reserved = torch.cuda.memory_reserved(device_idx) / (1024**2)  # MB
            vram_used = f"{mem_alloc:.1f} MB (alloc) / {mem_reserved:.1f} MB (reserved)"
        except Exception:
            pass

    # Try nvidia-smi command-line fallback
    try:
        res = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,nounits,noheader",
            ],
            stderr=subprocess.DEVNULL,
        )
        lines = res.decode("utf-8").strip().split("\n")
        if lines:
            gpu_u, mem_u, mem_t = lines[0].split(",")
            gpu_util = f"{gpu_u.strip()}%"
            if vram_used == "N/A":
                vram_used = f"{mem_u.strip()} MB / {mem_t.strip()} MB"
    except Exception:
        pass

    return gpu_util, vram_used


def parse_metrics(metrics_file: str | Path) -> dict[str, Any]:
    state = {
        "step": 0,
        "epoch": 0,
        "train_loss": None,
        "learning_rate": None,
        "grad_norm_before": None,
        "grad_norm_after": None,
        "validation_loss": None,
        "best_validation_loss": None,
        "tokens_per_sec": 0.0,
        "timestamp": None,
    }

    metrics_path = Path(metrics_file)
    if not metrics_path.exists():
        return state

    try:
        with metrics_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                for k in state:
                    if k in data:
                        state[k] = data[k]
    except Exception:
        pass
    return state


def get_latest_sample(sample_dir: str | Path) -> tuple[int | None, dict[str, str]]:
    p = Path(sample_dir)
    if not p.exists():
        return None, {}
    files = list(p.glob("samples_step_*.json"))
    if not files:
        return None, {}

    # Sort files by step number
    def get_step_from_path(path: Path) -> int:
        name = path.stem
        # format: samples_step_XXXX
        parts = name.split("_")
        if len(parts) >= 3 and parts[2].isdigit():
            return int(parts[2])
        return -1

    latest_file = max(files, key=get_step_from_path)
    try:
        with latest_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("step"), data.get("samples", {})
    except Exception:
        return None, {}


def calculate_eta(metrics_file: str | Path, current_step: int, max_steps: int) -> str:
    metrics_path = Path(metrics_file)
    if not metrics_path.exists() or current_step >= max_steps:
        return "00:00:00"

    try:
        with metrics_path.open("r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        valid_lines = [l for l in lines if "timestamp" in l and "step" in l]
        if len(valid_lines) < 2:
            return "N/A"

        # Use last 10 log entries for speed estimation
        recent = valid_lines[-10:]
        first = recent[0]
        last = recent[-1]

        step_diff = last["step"] - first["step"]
        time_diff = last["timestamp"] - first["timestamp"]

        if step_diff > 0 and time_diff > 0:
            steps_per_sec = step_diff / time_diff
            remaining_steps = max_steps - current_step
            remaining_seconds = remaining_steps / steps_per_sec

            hours = int(remaining_seconds // 3600)
            minutes = int((remaining_seconds % 3600) // 60)
            seconds = int(remaining_seconds % 60)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    except Exception:
        pass
    return "N/A"


def clear_screen() -> None:
    # Clear screen and move cursor to home position
    print("\033[H\033[J", end="")


def draw_dashboard(
    metrics_file: str | Path,
    sample_dir: str | Path,
    max_steps: int,
) -> None:
    gpu_util, vram_used = get_gpu_status()
    state = parse_metrics(metrics_file)
    step = state["step"]
    epoch = state["epoch"]

    train_loss = state["train_loss"]
    val_loss = state["validation_loss"]
    lr = state["learning_rate"]
    grad_norm = state["grad_norm_after"]
    tokens_per_sec = state["tokens_per_sec"]

    eta = calculate_eta(metrics_file, step, max_steps)
    sample_step, samples = get_latest_sample(sample_dir)

    # Formats
    loss_str = f"{train_loss:.4f}" if train_loss is not None else "N/A"
    val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
    lr_str = f"{lr:.6f}" if lr is not None else "N/A"
    grad_norm_str = f"{grad_norm:.4f}" if grad_norm is not None else "N/A"
    tokens_sec_str = f"{tokens_per_sec:,.1f}" if tokens_per_sec else "0.0"

    clear_screen()
    print("=" * 80)
    print(
        f"{COLOR_BOLD}{COLOR_BLUE}                    NEXARA TRAINING LIVE DASHBOARD{COLOR_RESET}"
    )
    print("=" * 80)
    print(f"  {COLOR_BOLD}{COLOR_MAGENTA}[System Status]{COLOR_RESET}")
    print(
        f"    GPU Utilization:  {COLOR_GREEN}{gpu_util:<15}{COLOR_RESET}   VRAM Usage: {COLOR_GREEN}{vram_used}{COLOR_RESET}"
    )
    print()
    print(f"  {COLOR_BOLD}{COLOR_MAGENTA}[Training Metrics]{COLOR_RESET}")
    print(
        f"    Step:             {COLOR_CYAN}{step:,} / {max_steps:,}{COLOR_RESET} (Epoch {epoch})"
    )
    print(
        f"    Train Loss:       {COLOR_GREEN}{loss_str:<15}{COLOR_RESET}   Val Loss:   {COLOR_GREEN}{val_loss_str}{COLOR_RESET}"
    )
    print(
        f"    Learning Rate:    {COLOR_YELLOW}{lr_str:<15}{COLOR_RESET}   Grad Norm:  {COLOR_YELLOW}{grad_norm_str}{COLOR_RESET}"
    )
    print(
        f"    Tokens/sec:       {COLOR_CYAN}{tokens_sec_str:<15}{COLOR_RESET}   ETA:        {COLOR_CYAN}{eta}{COLOR_RESET}"
    )
    print()
    print(
        f"  {COLOR_BOLD}{COLOR_MAGENTA}[Latest Sample Generation] (Step {sample_step or 'N/A'}){COLOR_RESET}"
    )
    if samples:
        first_prompt = list(samples.keys())[0]
        sample_output = samples[first_prompt]
        # Truncate lines for terminal safety
        sample_output_truncated = "\n".join(sample_output.split("\n")[:4])
        print(f"    {COLOR_BOLD}Prompt:{COLOR_RESET} {first_prompt}")
        print(
            f"    {COLOR_BOLD}Output:{COLOR_RESET} {COLOR_YELLOW}{sample_output_truncated}{COLOR_RESET}"
        )
    else:
        print("    No sample generations available yet.")
    print("=" * 80)
    print("Press Ctrl+C to exit.")


def main() -> None:
    args = parse_args()

    # Enable ANSI escape sequences on Windows
    if os.name == "nt":
        os.system("color")

    print("Initializing dashboard...")
    try:
        while True:
            draw_dashboard(args.metrics_file, args.sample_dir, args.max_steps)
            time.sleep(args.refresh_rate)
    except KeyboardInterrupt:
        print("\nExiting dashboard.")


if __name__ == "__main__":
    main()
