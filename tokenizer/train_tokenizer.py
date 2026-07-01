"""Train the Nexara tokenizer and optionally build token caches."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
from typing import Any

from datasets.token_cache import write_configured_token_cache
from tokenizer import NexaraTokenizer
from tokenizer.bpe import SPECIAL_TOKENS
from tokenizer.bpe import train_bpe_tokenizer
from training.config import load_config

DEFAULT_SAMPLE_TEXTS = [
    "Once upon a time, Lily found a tiny red bird.",
    "Tom wanted to share his toy with Mia.",
    "The little dog ran home before the rain.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Nexara tokenizer pipeline.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--input",
        action="append",
        default=None,
        help="Input JSONL/text file. Defaults to config data.train_path.",
    )
    parser.add_argument("--build-cache", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--sample-text",
        action="append",
        default=None,
        help="Text to encode in the sample encoding report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data = config["data"]
    tokenizer_config = config["tokenizer"]
    inputs = [Path(path) for path in (args.input or [data["train_path"]])]
    output = Path(tokenizer_config["path"])

    if output.exists() and not args.overwrite:
        print(f"skipping existing tokenizer: {output}")
    else:
        output = train_bpe_tokenizer(
            input_paths=inputs,
            output_path=output,
            vocab_size=int(tokenizer_config["vocab_size"]),
            text_key=str(data.get("text_key", "text")),
            min_frequency=int(tokenizer_config.get("min_frequency", 2)),
            overwrite=args.overwrite,
        )
        print(f"saved tokenizer to {output}")

    tokenizer = NexaraTokenizer(output)
    metadata_path = write_tokenizer_metadata(output, inputs, config, tokenizer=tokenizer)
    print(f"saved tokenizer metadata to {metadata_path}")
    report_paths = write_tokenizer_reports(
        tokenizer=tokenizer,
        tokenizer_path=output,
        input_paths=inputs,
        config=config,
        sample_texts=args.sample_text or configured_sample_texts(config),
    )
    print(f"saved tokenizer vocabulary report to {report_paths['vocabulary_report']}")
    print(f"saved tokenizer sample encodings to {report_paths['sample_encodings']}")

    if args.build_cache:
        for split in ["train", "validation"]:
            result = write_configured_token_cache(
                config,
                tokenizer,
                split,
                overwrite=args.overwrite,
            )
            print(
                f"{split}: available {result.token_count} tokens "
                f"({result.sequence_count} sequences, "
                f"{result.average_tokens_per_document:.2f} avg tokens/document) "
                f"at {result.cache_path}"
            )


def write_tokenizer_metadata(
    tokenizer_path: str | Path,
    input_paths: list[Path],
    config: dict,
    tokenizer: NexaraTokenizer | None = None,
) -> Path:
    output = Path(tokenizer_path)
    metadata_path = output.with_suffix(".meta.json")
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tokenizer_path": str(output),
        "input_paths": [str(path) for path in input_paths],
        "vocab_size": int(config["tokenizer"]["vocab_size"]),
        "actual_vocab_size": tokenizer.vocab_size if tokenizer is not None else None,
        "min_frequency": int(config["tokenizer"].get("min_frequency", 2)),
        "text_key": str(config["data"].get("text_key", "text")),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    with temporary_metadata.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary_metadata.replace(metadata_path)
    return metadata_path


def write_tokenizer_reports(
    tokenizer: NexaraTokenizer,
    tokenizer_path: str | Path,
    input_paths: list[Path],
    config: dict[str, Any],
    sample_texts: list[str],
) -> dict[str, Path]:
    output = Path(tokenizer_path)
    vocabulary_path = output.with_suffix(".vocab_report.json")
    sample_path = output.with_suffix(".sample_encodings.json")

    vocabulary = tokenizer.vocabulary()
    ordered_vocabulary = sorted(vocabulary.items(), key=lambda item: item[1])
    token_lengths = [len(token) for token, _ in ordered_vocabulary]
    vocabulary_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tokenizer_path": str(output),
        "input_paths": [str(path) for path in input_paths],
        "configured_vocab_size": int(config["tokenizer"]["vocab_size"]),
        "actual_vocab_size": tokenizer.vocab_size,
        "min_frequency": int(config["tokenizer"].get("min_frequency", 2)),
        "text_key": str(config["data"].get("text_key", "text")),
        "special_tokens": {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS},
        "token_length_statistics": {
            "min": min(token_lengths) if token_lengths else 0,
            "max": max(token_lengths) if token_lengths else 0,
            "average": mean(token_lengths) if token_lengths else 0.0,
        },
        "vocabulary": [
            {
                "id": int(token_id),
                "token": token,
                "characters": len(token),
                "utf8_bytes": len(token.encode("utf-8")),
            }
            for token, token_id in ordered_vocabulary
        ],
    }
    write_json_atomic(vocabulary_path, vocabulary_payload)

    sample_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tokenizer_path": str(output),
        "samples": [
            encode_sample(tokenizer, sample_text)
            for sample_text in sample_texts
            if sample_text.strip()
        ],
    }
    write_json_atomic(sample_path, sample_payload)
    return {
        "vocabulary_report": vocabulary_path,
        "sample_encodings": sample_path,
    }


def encode_sample(tokenizer: NexaraTokenizer, text: str) -> dict[str, Any]:
    token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    return {
        "text": text,
        "token_ids": token_ids,
        "tokens": [tokenizer.id_to_token(token_id) for token_id in token_ids],
        "decoded": tokenizer.decode(token_ids, skip_special_tokens=True),
        "token_count": len(token_ids),
    }


def configured_sample_texts(config: dict[str, Any]) -> list[str]:
    values = config.get("tokenizer", {}).get("sample_texts", [])
    if isinstance(values, list) and all(isinstance(value, str) for value in values):
        return values or DEFAULT_SAMPLE_TEXTS
    return DEFAULT_SAMPLE_TEXTS


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    with temporary_output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary_output.replace(output)
    return output


if __name__ == "__main__":
    main()
