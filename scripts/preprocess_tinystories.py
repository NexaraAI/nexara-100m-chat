"""Preprocess TinyStories raw text into JSONL training files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets.statistics import write_statistics
from datasets.tinystories import (
    TINYSTORIES_FILES,
    TinyStoriesPreprocessOptions,
    default_processed_path,
    default_raw_path,
    preprocess_tinystories_file,
)
from training.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess TinyStories raw text.")
    parser.add_argument("--config", default="configs/stage1_tinystories.toml")
    parser.add_argument("--variant", choices=sorted(TINYSTORIES_FILES), default=None)
    parser.add_argument("--raw-dir", default="datasets/raw")
    parser.add_argument("--output-dir", default="datasets/processed")
    parser.add_argument("--train-input", default="")
    parser.add_argument("--validation-input", default="")
    parser.add_argument("--min-characters", type=int, default=20)
    parser.add_argument("--max-characters", type=int, default=0)
    parser.add_argument("--keep-duplicates", action="store_true")
    parser.add_argument("--lowercase-dedupe", action="store_true")
    parser.add_argument("--preserve-paragraphs", action="store_true")
    parser.add_argument("--source-mode", choices=["auto", "delimiter", "line"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config) if args.config else {}
    tinystories_config: dict[str, Any] = config.get("tinystories", {})
    variant = args.variant or str(tinystories_config.get("variant", "original"))

    train_input = Path(
        args.train_input
        or tinystories_config.get("raw_train_path", "")
        or default_raw_path(args.raw_dir, variant, "train")
    )
    validation_input = Path(
        args.validation_input
        or tinystories_config.get("raw_validation_path", "")
        or default_raw_path(args.raw_dir, variant, "validation")
    )
    output_dir = Path(args.output_dir)
    data_config = config.get("data", {})
    train_output = (
        Path(data_config["train_path"])
        if "train_path" in data_config
        else default_processed_path(output_dir, "train")
    )
    validation_output = (
        Path(data_config["validation_path"])
        if "validation_path" in data_config
        else default_processed_path(output_dir, "validation")
    )

    options = TinyStoriesPreprocessOptions(
        min_characters=args.min_characters,
        max_characters=args.max_characters,
        deduplicate=not args.keep_duplicates,
        lowercase_dedupe=args.lowercase_dedupe,
        preserve_paragraphs=args.preserve_paragraphs,
        source_mode=args.source_mode,
    )
    report = {
        "variant": variant,
        "splits": [
            preprocess_tinystories_file(
                train_input,
                train_output,
                "train",
                options,
                overwrite=args.overwrite,
            ),
            preprocess_tinystories_file(
                validation_input,
                validation_output,
                "validation",
                options,
                overwrite=args.overwrite,
            ),
        ],
    }

    report_path = output_dir / f"tinystories_{variant}_preprocess_report.json"
    write_statistics(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
