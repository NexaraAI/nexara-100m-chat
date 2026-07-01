"""Generate token frequency and coverage reports from a trained tokenizer."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

from datasets.statistics import write_statistics
from tokenizer import NexaraTokenizer
from training.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate tokenizer analysis report.")
    parser.add_argument("--config", default="configs/stage1_tinystories.toml")
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--data-path", default="", help="JSONL file to analyze token frequencies.")
    parser.add_argument("--max-documents", type=int, default=10000)
    parser.add_argument("--output-dir", default="logs/tokenizer_report")
    parser.add_argument("--top-n", type=int, default=100, help="Top-N tokens by frequency.")
    parser.add_argument(
        "--sample-count",
        type=int,
        default=10,
        help="Number of real corpus samples to encode.",
    )
    return parser.parse_args()


def analyze_token_frequencies(
    tokenizer: NexaraTokenizer,
    data_path: Path,
    text_key: str = "text",
    max_documents: int = 10000,
) -> dict[str, Any]:
    """Analyze token frequency distribution on a corpus."""
    counter: Counter[int] = Counter()
    document_count = 0
    total_tokens = 0

    with data_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = str(record.get(text_key, ""))
            ids = tokenizer.encode(text, add_bos=False, add_eos=False)
            counter.update(ids)
            total_tokens += len(ids)
            document_count += 1
            if max_documents and document_count >= max_documents:
                break

    vocab_size = tokenizer.vocab_size
    used_tokens = len(counter)
    unused_tokens = vocab_size - used_tokens

    return {
        "documents_analyzed": document_count,
        "total_tokens": total_tokens,
        "unique_tokens_used": used_tokens,
        "unused_tokens": unused_tokens,
        "vocabulary_coverage_pct": round(used_tokens / vocab_size * 100, 2) if vocab_size else 0.0,
        "token_frequencies": counter,
    }


def collect_sample_encodings(
    tokenizer: NexaraTokenizer,
    data_path: Path,
    text_key: str = "text",
    sample_count: int = 10,
) -> list[dict[str, Any]]:
    """Encode real corpus samples for inspection."""
    samples = []
    with data_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = str(record.get(text_key, ""))
            if len(text) < 50:
                continue
            ids = tokenizer.encode(text, add_bos=True, add_eos=True)
            tokens = [tokenizer.id_to_token(tid) for tid in ids]
            samples.append(
                {
                    "text": text[:500],
                    "token_ids": ids[:100],
                    "tokens": tokens[:100],
                    "token_count": len(ids),
                    "compression_ratio": round(len(text) / len(ids), 3) if ids else 0.0,
                }
            )
            if len(samples) >= sample_count:
                break
    return samples


def generate_report(
    config: dict[str, Any],
    tokenizer_path: str,
    data_path: str,
    max_documents: int,
    output_dir: str,
    top_n: int,
    sample_count: int,
) -> dict[str, Any]:
    """Generate full tokenizer analysis report."""
    tok_path = tokenizer_path or config.get("tokenizer", {}).get("path", "")
    if not tok_path or not Path(tok_path).exists():
        raise FileNotFoundError(f"tokenizer not found at {tok_path}")

    tokenizer = NexaraTokenizer(tok_path)
    data_config = config.get("data", {})
    text_key = str(data_config.get("text_key", "text"))
    corpus_path = Path(data_path or data_config.get("train_path", ""))

    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tokenizer_path": str(tok_path),
        "vocab_size": tokenizer.vocab_size,
    }

    if corpus_path.exists():
        print(f"analyzing token frequencies on {corpus_path}")
        freq_result = analyze_token_frequencies(
            tokenizer, corpus_path, text_key=text_key, max_documents=max_documents
        )
        counter = freq_result.pop("token_frequencies")
        report["frequency_analysis"] = freq_result

        # Top-N tokens
        most_common = counter.most_common(top_n)
        report["top_tokens"] = [
            {
                "rank": i + 1,
                "token_id": tid,
                "token": tokenizer.id_to_token(tid),
                "count": count,
                "frequency_pct": round(count / freq_result["total_tokens"] * 100, 4),
            }
            for i, (tid, count) in enumerate(most_common)
        ]

        # Frequency distribution statistics
        all_counts = list(counter.values())
        if all_counts:
            report["frequency_distribution"] = {
                "min": min(all_counts),
                "max": max(all_counts),
                "mean": round(mean(all_counts), 2),
                "median": round(median(all_counts), 2),
            }

        # Sample encodings from real corpus
        print(f"collecting {sample_count} sample encodings")
        report["sample_encodings"] = collect_sample_encodings(
            tokenizer, corpus_path, text_key=text_key, sample_count=sample_count
        )
    else:
        report["corpus_status"] = f"file not found: {corpus_path}"

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "tokenizer_report.json"
    write_statistics(report_path, report)
    print(f"wrote tokenizer report to {report_path}")
    return report


def main() -> None:
    args = parse_args()
    config = load_config(args.config) if args.config else {}
    report = generate_report(
        config=config,
        tokenizer_path=args.tokenizer_path,
        data_path=args.data_path,
        max_documents=args.max_documents,
        output_dir=args.output_dir,
        top_n=args.top_n,
        sample_count=args.sample_count,
    )
    # Print summary to console (not the full report — it can be large)
    freq = report.get("frequency_analysis", {})
    print(f"\nvocab_size: {report.get('vocab_size', '?')}")
    print(f"unique tokens used: {freq.get('unique_tokens_used', '?')}")
    print(f"vocabulary coverage: {freq.get('vocabulary_coverage_pct', '?')}%")
    print(f"documents analyzed: {freq.get('documents_analyzed', '?')}")
    print(f"total tokens: {freq.get('total_tokens', '?')}")
    if report.get("sample_encodings"):
        ratios = [s["compression_ratio"] for s in report["sample_encodings"]]
        print(f"avg compression ratio: {mean(ratios):.3f} chars/token")


if __name__ == "__main__":
    main()
