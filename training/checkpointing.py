"""Checkpoint save/load helpers."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    step: int,
    epoch: int,
    config: dict[str, Any],
    metrics: dict[str, Any] | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Convert model state dict to CPU tensors for portability
    model_state_cpu = {k: v.cpu() for k, v in model.state_dict().items()}

    # Convert optimizer state to CPU tensors for portability
    optimizer_state_cpu = {}
    if optimizer is not None:
        opt_state_dict = optimizer.state_dict()
        optimizer_state_cpu = {"state": {}, "param_groups": opt_state_dict["param_groups"]}
        for param_id, state in opt_state_dict["state"].items():
            param_state = {}
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    param_state[k] = v.cpu()
                else:
                    param_state[k] = v
            optimizer_state_cpu["state"][param_id] = param_state

    state = {
        "model_state_dict": model_state_cpu,
        "optimizer_state_dict": optimizer_state_cpu,
        "step": int(step),
        "epoch": int(epoch),
        "config": config,
        "metrics": metrics or {},
    }
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()

    # Save to a temporary file first, then atomically replace
    temp_path = output.with_suffix(output.suffix + ".tmp")
    torch.save(state, temp_path)
    temp_path.replace(output)
    return output


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location)

    # Clean keys if they were saved from a torch.compile model
    state_dict = checkpoint["model_state_dict"]
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            clean_state_dict[k[len("_orig_mod.") :]] = v
        else:
            clean_state_dict[k] = v

    model.load_state_dict(clean_state_dict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return checkpoint


def find_latest_checkpoint(directory: str | Path) -> Path | None:
    """Find the latest checkpoint file in a directory based on step suffix or modification time."""
    dir_path = Path(directory)
    if not dir_path.exists():
        return None

    # Check for latest.pt first
    latest = dir_path / "latest.pt"
    if latest.exists():
        return latest

    # Find step_*.pt files and sort by step
    checkpoints = []
    for p in dir_path.glob("*.pt"):
        if p.name in {"final.pt", "best.pt"}:
            continue
        match = re.match(r"^step_(\d+)\.pt$", p.name)
        if match:
            checkpoints.append((int(match.group(1)), p))

    if checkpoints:
        # Return checkpoint with highest step
        return sorted(checkpoints, key=lambda x: x[0])[-1][1]

    # Fallback to modification time for any *.pt file except final.pt/best.pt
    all_pt = [p for p in dir_path.glob("*.pt") if p.name not in {"final.pt", "best.pt"}]
    if all_pt:
        return max(all_pt, key=lambda p: p.stat().st_mtime)

    return None


def validate_checkpoint(path: str | Path) -> bool:
    """Validate checkpoint file format and key presence."""
    try:
        checkpoint = torch.load(path, map_location="cpu")
        required_keys = {"model_state_dict", "step", "epoch", "config"}
        return all(k in checkpoint for k in required_keys)
    except Exception:
        return False


def rotate_checkpoints(directory: str | Path, keep_last_n: int = 5) -> None:
    """Keep only the latest keep_last_n step_*.pt checkpoints. Never delete best.pt, latest.pt, or final.pt."""
    dir_path = Path(directory)
    if not dir_path.exists():
        return

    checkpoints = []
    for p in dir_path.glob("*.pt"):
        if p.name in {"best.pt", "latest.pt", "final.pt"}:
            continue
        match = re.match(r"^step_(\d+)\.pt$", p.name)
        if match:
            checkpoints.append((int(match.group(1)), p))

    if len(checkpoints) <= keep_last_n:
        return

    # Sort by step ascending (oldest first)
    sorted_checkpoints = sorted(checkpoints, key=lambda x: x[0])
    to_delete = sorted_checkpoints[:-keep_last_n]
    for step, p in to_delete:
        try:
            p.unlink()
        except Exception:
            pass
