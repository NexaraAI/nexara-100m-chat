"""Tokenizer smoke test used by local validation and CI."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from datasets.statistics import write_statistics
from datasets.token_cache import collect_token_cache_statistics, write_token_cache
from scripts.smoke_fixtures import write_tiny_stories_jsonl
from tokenizer import NexaraTokenizer
from tokenizer.bpe import train_bpe_tokenizer
from tokenizer.train_tokenizer import DEFAULT_SAMPLE_TEXTS, write_tokenizer_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny tokenizer smoke test.")
    parser.add_argument("--output-dir", default="logs/tokenizer_smoke")
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = write_tiny_stories_jsonl(output_dir / "tiny_stories.jsonl", repeat=2)
    tokenizer_path = output_dir / "tokenizer.json"
    cache_path = output_dir / "tokens.bin"

    train_bpe_tokenizer(
        input_paths=[data_path],
        output_path=tokenizer_path,
        vocab_size=args.vocab_size,
        text_key="text",
        min_frequency=1,
        overwrite=True,
    )
    tokenizer = NexaraTokenizer(tokenizer_path)
    encoded = tokenizer.encode(DEFAULT_SAMPLE_TEXTS[0], add_bos=True, add_eos=True)
    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    if not encoded or "Lily" not in decoded:
        raise RuntimeError("tokenizer encode/decode smoke check failed")

    report_config = {
        "tokenizer": {"vocab_size": args.vocab_size, "min_frequency": 1},
        "data": {"text_key": "text"},
    }
    report_paths = write_tokenizer_reports(
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
        input_paths=[data_path],
        config=report_config,
        sample_texts=DEFAULT_SAMPLE_TEXTS,
    )
    cache_result = write_token_cache(
        input_paths=[data_path],
        tokenizer=tokenizer,
        output_path=cache_path,
        block_size=args.block_size,
        text_key="text",
        overwrite=True,
    )
    cache_statistics = collect_token_cache_statistics(cache_result.cache_path)
    write_statistics(cache_path.with_suffix(".stats.json"), cache_statistics)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path),
        "tokenizer_path": str(tokenizer_path),
        "configured_vocab_size": args.vocab_size,
        "actual_vocab_size": tokenizer.vocab_size,
        "sample_token_count": len(encoded),
        "vocabulary_report": str(report_paths["vocabulary_report"]),
        "sample_encodings": str(report_paths["sample_encodings"]),
        "cache_statistics": cache_statistics,
    }
    report_path = output_dir / "tokenizer_smoke.json"
    write_statistics(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
