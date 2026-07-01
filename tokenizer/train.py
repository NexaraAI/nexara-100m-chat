"""Command line entry point for training the Nexara tokenizer."""

from __future__ import annotations

import argparse
from pathlib import Path

from tokenizer.bpe import train_bpe_tokenizer
from training.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer from local data.")
    parser.add_argument("--config", required=True, help="Path to a Nexara TOML config.")
    parser.add_argument(
        "--input",
        action="append",
        default=None,
        help="Input file. Can be passed multiple times. Defaults to config train path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    tokenizer_config = config["tokenizer"]
    data_config = config["data"]

    inputs = args.input or [data_config["train_path"]]
    output = train_bpe_tokenizer(
        input_paths=[Path(path) for path in inputs],
        output_path=tokenizer_config["path"],
        vocab_size=int(tokenizer_config["vocab_size"]),
        text_key=str(data_config.get("text_key", "text")),
        min_frequency=int(tokenizer_config.get("min_frequency", 2)),
    )
    print(f"saved tokenizer to {output}")


if __name__ == "__main__":
    main()
