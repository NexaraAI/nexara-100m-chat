"""Generate and update markdown experiment reports from metrics JSON."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update EXPERIMENTS.md from metrics JSON.")
    parser.add_argument("--metrics", required=True, help="Path to a metrics JSON file.")
    parser.add_argument("--experiments", default="EXPERIMENTS.md")
    parser.add_argument(
        "--observation",
        action="append",
        default=None,
        help="Additional observation to include in the report.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = update_experiments_from_metrics(
        metrics_path=Path(args.metrics),
        experiments_path=Path(args.experiments),
        extra_observations=args.observation or [],
        dry_run=args.dry_run,
    )
    print(output)


def update_experiments_from_metrics(
    metrics_path: str | Path,
    experiments_path: str | Path = "EXPERIMENTS.md",
    extra_observations: list[str] | None = None,
    dry_run: bool = False,
) -> str:
    metrics_file = Path(metrics_path)
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    section = render_experiment_section(
        metrics,
        metrics_path=metrics_file,
        extra_observations=extra_observations or [],
    )
    experiments_file = Path(experiments_path)
    current = (
        experiments_file.read_text(encoding="utf-8")
        if experiments_file.exists()
        else "# Experiments\n"
    )
    updated = upsert_marked_section(
        current,
        experiment_name=experiment_name(metrics),
        section=section,
    )
    if not dry_run:
        write_text_atomic(experiments_file, updated)
    return updated


def render_experiment_section(
    metrics: dict[str, Any],
    metrics_path: Path,
    extra_observations: list[str],
) -> str:
    name = experiment_name(metrics)
    date = experiment_date(metrics)
    title = experiment_title(name)
    duration = float(metrics.get("duration_seconds", 0.0))
    observations = list(metrics.get("observations") or [])
    observations.extend(extra_observations)
    hyperparameters = metrics.get("hyperparameters") or {}
    artifacts = metrics.get("artifacts") or {}

    lines = [
        f"## {date}: {title}",
        "",
        f"- Name: `{name}`",
        f"- Dataset: {metrics.get('dataset', 'Not recorded')}",
        f"- Training duration: {duration:.2f} seconds",
        "- Hyperparameters:",
    ]
    if hyperparameters:
        lines.extend(f"  - `{key}`: `{value}`" for key, value in sorted(hyperparameters.items()))
    else:
        lines.append("  - Not recorded")

    lines.extend(
        [
            "- Losses:",
            f"  - Initial loss: {format_float(metrics.get('initial_loss'))}",
            f"  - Final loss: {format_float(metrics.get('final_loss'))}",
            f"  - Loss delta: {format_float(metrics.get('loss_delta'))}",
            f"- Metrics artifact: `{metrics_path}`",
        ]
    )
    if artifacts:
        lines.append("- Artifacts:")
        for key, value in sorted(artifacts.items()):
            if value:
                lines.append(f"  - `{key}`: `{value}`")
    lines.append("- Observations:")
    if observations:
        lines.extend(f"  - {observation}" for observation in observations)
    else:
        lines.append("  - Not recorded")
    lines.extend(["", ""])
    return "\n".join(lines)


def upsert_marked_section(current: str, experiment_name: str, section: str) -> str:
    start = marker(experiment_name, "start")
    end = marker(experiment_name, "end")
    marked = f"{start}\n{section}{end}\n"
    pattern = re.compile(
        rf"{re.escape(start)}.*?{re.escape(end)}\n?",
        flags=re.DOTALL,
    )
    if pattern.search(current):
        return pattern.sub(lambda _match: marked, current)
    if not current.endswith("\n"):
        current += "\n"
    return current.rstrip() + "\n\n" + marked


def experiment_name(metrics: dict[str, Any]) -> str:
    value = str(metrics.get("experiment_name") or metrics.get("name") or "").strip()
    return value or "unnamed_experiment"


def experiment_date(metrics: dict[str, Any]) -> str:
    created_at = str(metrics.get("created_at") or "").strip()
    if not created_at:
        return datetime.utcnow().date().isoformat()
    try:
        parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError:
        return created_at[:10]
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone()
    return parsed.date().isoformat()


def experiment_title(name: str) -> str:
    title = re.sub(r"phase(\d+)_(\d+)", r"Phase \1.\2", name)
    return title.replace("_", " ").title()


def format_float(value: Any) -> str:
    if value is None:
        return "Not recorded"
    return f"{float(value):.6f}"


def marker(experiment_name: str, side: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", experiment_name.strip())
    return f"<!-- experiment:{safe_name}:{side} -->"


def write_text_atomic(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(text, encoding="utf-8")
    temporary_path.replace(path)
    return path


if __name__ == "__main__":
    main()
