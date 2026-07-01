"""Verify configured parameter counts against an expected range."""

from __future__ import annotations

import argparse

from training.config import load_config
from training.parameter_count import estimate_transformer_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify configured model parameter count.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--minimum", type=int, default=5_000_000)
    parser.add_argument("--maximum", type=int, default=10_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    parameters = estimate_transformer_parameters(config["model"])
    if parameters < args.minimum or parameters > args.maximum:
        raise SystemExit(
            "parameter count outside expected range: "
            f"{parameters} not in [{args.minimum}, {args.maximum}]"
        )
    print(f"parameter_count_ok value={parameters}")


if __name__ == "__main__":
    main()
