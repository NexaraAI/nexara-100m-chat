"""Estimate model parameters from a Nexara config."""

from __future__ import annotations

import argparse

from training.config import load_config
from training.parameter_count import estimate_transformer_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate configured model size.")
    parser.add_argument("--config", required=True, help="Path to a Nexara TOML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    parameters = estimate_transformer_parameters(config["model"])
    print(f"estimated_parameters={parameters}")


if __name__ == "__main__":
    main()
