"""Generate comprehensive dataset reports for TinyStories JSONL files."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

from datasets.statistics import write_statistics
from training.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dataset statistics report.")
    parser.add_argument("--config", default="configs/stage1_tinystories.toml")
    parser.add_argument("--train-path", default="")
    parser.add_argument("--validation-path", default="")
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--output-dir", default="logs/dataset_report")
    return parser.parse_args()


def count_jsonl_documents(path: Path, text_key: str = "text") -> list[str]:
    """Read all text values from a JSONL file."""
    texts = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                record = json.loads(line)
                texts.append(str(record.get(text_key, "")))
    return texts


def compute_text_statistics(texts: list[str]) -> dict[str, Any]:
    """Compute character and word count statistics for a list of texts."""
    if not texts:
        return {"count": 0}

    char_counts = [len(t) for t in texts]
    word_counts = [len(t.split()) for t in texts]

    return {
        "document_count": len(texts),
        "total_characters": sum(char_counts),
        "total_words": sum(word_counts),
        "characters": {
            "min": min(char_counts),
            "max": max(char_counts),
            "mean": round(mean(char_counts), 2),
            "median": round(median(char_counts), 2),
            "stdev": round(stdev(char_counts), 2) if len(char_counts) > 1 else 0.0,
        },
        "words": {
            "min": min(word_counts),
            "max": max(word_counts),
            "mean": round(mean(word_counts), 2),
            "median": round(median(word_counts), 2),
            "stdev": round(stdev(word_counts), 2) if len(word_counts) > 1 else 0.0,
        },
    }


def compute_token_statistics(
    texts: list[str],
    tokenizer: Any,
) -> dict[str, Any]:
    """Compute token count statistics using the provided tokenizer."""
    if not texts:
        return {"count": 0}

    token_counts = []
    for text in texts:
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        token_counts.append(len(ids))

    return {
        "document_count": len(texts),
        "total_tokens": sum(token_counts),
        "tokens_per_document": {
            "min": min(token_counts),
            "max": max(token_counts),
            "mean": round(mean(token_counts), 2),
            "median": round(median(token_counts), 2),
            "stdev": round(stdev(token_counts), 2) if len(token_counts) > 1 else 0.0,
        },
    }


def generate_report(
    config: dict[str, Any],
    train_path: str,
    validation_path: str,
    tokenizer_path: str,
    output_dir: str,
) -> dict[str, Any]:
    """Generate full dataset report."""
    data = config.get("data", {})
    train_file = Path(train_path or data.get("train_path", ""))
    validation_file = Path(validation_path or data.get("validation_path", ""))
    text_key = str(data.get("text_key", "text"))

    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "splits": {},
    }

    for split_name, split_path in [("train", train_file), ("validation", validation_file)]:
        if not split_path.exists():
            report["splits"][split_name] = {
                "path": str(split_path),
                "status": "file_not_found",
            }
            continue

        print(f"reading {split_name} split: {split_path}")
        texts = count_jsonl_documents(split_path, text_key=text_key)
        split_report: dict[str, Any] = {
            "path": str(split_path),
            "file_size_bytes": split_path.stat().st_size,
            "text_statistics": compute_text_statistics(texts),
        }

        tok_path = tokenizer_path or config.get("tokenizer", {}).get("path", "")
        if tok_path and Path(tok_path).exists():
            from tokenizer import NexaraTokenizer

            print(f"  computing token statistics with {tok_path}")
            tok = NexaraTokenizer(tok_path)
            split_report["token_statistics"] = compute_token_statistics(texts, tok)

        report["splits"][split_name] = split_report

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "dataset_report.json"
    write_statistics(report_path, report)
    print(f"wrote dataset report to {report_path}")
    return report


def main() -> None:
    args = parse_args()
    config = load_config(args.config) if args.config else {}
    report = generate_report(
        config=config,
        train_path=args.train_path,
        validation_path=args.validation_path,
        tokenizer_path=args.tokenizer_path,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
