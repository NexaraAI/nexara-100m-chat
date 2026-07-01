"""Build disk-backed token caches from processed text datasets."""

from __future__ import annotations

import argparse

from datasets.statistics import write_statistics
from datasets.token_cache import collect_token_cache_statistics, write_configured_token_cache
from tokenizer import NexaraTokenizer
from training.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Nexara token caches.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["train", "validation", "all"], default="all")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    tokenizer = NexaraTokenizer(config["tokenizer"]["path"])
    for split in _splits(args.split):
        result = write_cache_for_split(config, tokenizer, split, overwrite=args.overwrite)
        statistics = collect_token_cache_statistics(result.cache_path)
        statistics_path = result.cache_path.with_suffix(".stats.json")
        write_statistics(statistics_path, statistics)
        print(
            f"{split}: wrote {result.token_count} tokens "
            f"({result.sequence_count} sequences) to {result.cache_path}; "
            f"stats at {statistics_path}"
        )


def write_cache_for_split(
    config: dict,
    tokenizer: NexaraTokenizer,
    split: str,
    overwrite: bool = True,
):
    return write_configured_token_cache(config, tokenizer, split, overwrite=overwrite)


def _splits(split: str) -> list[str]:
    return ["train", "validation"] if split == "all" else [split]


if __name__ == "__main__":
    main()
