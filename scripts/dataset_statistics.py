"""Compute statistics for JSONL or plain-text datasets."""

from __future__ import annotations

import argparse
import json

from datasets.statistics import (
    collect_dataset_statistics,
    collect_tokenized_dataset_statistics,
    write_statistics,
)
from datasets.token_cache import collect_token_cache_statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute text dataset statistics.")
    parser.add_argument("paths", nargs="+", help="Input JSONL or plain-text files.")
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--tokenizer", default="", help="Optional local tokenizer JSON path.")
    parser.add_argument("--block-size", type=int, default=0)
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Treat paths as binary token caches with JSON metadata.",
    )
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cache:
        statistics = {
            "caches": [collect_token_cache_statistics(path) for path in args.paths],
        }
    elif args.tokenizer:
        from tokenizer import NexaraTokenizer

        statistics = collect_tokenized_dataset_statistics(
            args.paths,
            tokenizer=NexaraTokenizer(args.tokenizer),
            text_key=args.text_key,
            block_size=args.block_size,
        )
    else:
        statistics = collect_dataset_statistics(args.paths, text_key=args.text_key)
    if args.output:
        write_statistics(args.output, statistics)
    print(json.dumps(statistics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
