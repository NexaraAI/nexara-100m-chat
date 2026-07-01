"""Plot loss curves and gradient norms from metrics JSON."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot loss and gradient curves.")
    parser.add_argument(
        "--metrics",
        required=True,
        help="Path to metrics.json or loss_curve.json.",
    )
    parser.add_argument("--output", default="loss_curve.png", help="Output PNG path.")
    parser.add_argument("--csv", default="", help="Optional CSV export path.")
    parser.add_argument(
        "--gradient-norms",
        default="",
        help="Path to gradient_norms.json for dual-axis plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics)
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    loss_curve = data.get("loss_curve", [])
    if not loss_curve:
        raise ValueError(f"no loss_curve found in {metrics_path}")

    gradient_norms: list[dict[str, Any]] = []
    if args.gradient_norms:
        grad_path = Path(args.gradient_norms)
        grad_data = json.loads(grad_path.read_text(encoding="utf-8"))
        gradient_norms = grad_data.get("gradient_norms", [])
    elif "gradient_statistics" in data:
        # Try sibling gradient_norms.json
        sibling = metrics_path.parent / "gradient_norms.json"
        if sibling.exists():
            grad_data = json.loads(sibling.read_text(encoding="utf-8"))
            gradient_norms = grad_data.get("gradient_norms", [])

    output_path = Path(args.output)
    plot_loss_and_gradients(output_path, loss_curve, gradient_norms)
    print(f"wrote {output_path}")

    if args.csv:
        csv_path = Path(args.csv)
        export_csv(csv_path, loss_curve, gradient_norms)
        print(f"wrote {csv_path}")


def plot_loss_and_gradients(
    output_path: Path,
    loss_curve: list[dict[str, Any]],
    gradient_norms: list[dict[str, Any]],
) -> Path:
    """Plot loss curve with optional gradient norm overlay."""
    os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / "matplotlib_cache"))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    steps = [int(row["step"]) for row in loss_curve]
    losses = [float(row["loss"]) for row in loss_curve]

    if gradient_norms:
        fig, (ax_loss, ax_grad) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    else:
        fig, ax_loss = plt.subplots(figsize=(8, 4))

    ax_loss.plot(steps, losses, color="#2196F3", linewidth=1.5, markersize=1.5)
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Overfit Loss Curve")
    ax_loss.grid(True, alpha=0.3)

    if gradient_norms:
        grad_steps = [int(g["step"]) for g in gradient_norms]
        norms_before = [float(g["grad_norm_before_clip"]) for g in gradient_norms]
        norms_after = [float(g["grad_norm_after_clip"]) for g in gradient_norms]
        ax_grad.plot(
            grad_steps,
            norms_before,
            color="#FF9800",
            linewidth=1.0,
            alpha=0.7,
            label="before clip",
        )
        ax_grad.plot(
            grad_steps,
            norms_after,
            color="#4CAF50",
            linewidth=1.0,
            alpha=0.7,
            label="after clip",
        )
        ax_grad.set_xlabel("Step")
        ax_grad.set_ylabel("Gradient Norm")
        ax_grad.set_title("Gradient Norms")
        ax_grad.legend(fontsize=8)
        ax_grad.grid(True, alpha=0.3)
    else:
        ax_loss.set_xlabel("Step")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def export_csv(
    path: Path,
    loss_curve: list[dict[str, Any]],
    gradient_norms: list[dict[str, Any]],
) -> Path:
    grad_lookup: dict[int, dict[str, Any]] = {}
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


if __name__ == "__main__":
    main()
