"""TinyStories download metadata and preprocessing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterator, Literal
from urllib.parse import quote

from datasets.statistics import TextStatistics

HF_DATASET_REPO = "roneneldan/TinyStories"
HF_DATASET_BASE_URL = f"https://huggingface.co/datasets/{HF_DATASET_REPO}"
END_OF_TEXT = "<|endoftext|>"

TINYSTORIES_FILES: dict[str, dict[str, str]] = {
    "original": {
        "train": "TinyStories-train.txt",
        "validation": "TinyStories-valid.txt",
    },
    "gpt4": {
        "train": "TinyStoriesV2-GPT4-train.txt",
        "validation": "TinyStoriesV2-GPT4-valid.txt",
    },
}

SplitName = Literal["train", "validation"]


@dataclass(frozen=True)
class TinyStoriesPreprocessOptions:
    min_characters: int = 20
    max_characters: int = 0
    deduplicate: bool = True
    lowercase_dedupe: bool = False
    preserve_paragraphs: bool = False
    source_mode: str = "auto"
    delimiter: str = END_OF_TEXT


def tiny_stories_filename(variant: str, split: SplitName) -> str:
    try:
        return TINYSTORIES_FILES[variant][split]
    except KeyError as exc:
        valid_variants = ", ".join(sorted(TINYSTORIES_FILES))
        message = f"unknown TinyStories variant {variant!r}; expected {valid_variants}"
        raise ValueError(message) from exc


def tiny_stories_url(variant: str, split: SplitName) -> str:
    filename = quote(tiny_stories_filename(variant, split))
    return f"{HF_DATASET_BASE_URL}/resolve/main/{filename}"


def default_raw_path(raw_dir: str | Path, variant: str, split: SplitName) -> Path:
    return Path(raw_dir) / tiny_stories_filename(variant, split)


def default_processed_path(processed_dir: str | Path, split: SplitName) -> Path:
    return Path(processed_dir) / f"tinystories_{split}.jsonl"


def iter_tinystories_records(
    path: str | Path,
    mode: str = "auto",
    delimiter: str = END_OF_TEXT,
) -> Iterator[str]:
    resolved_mode = detect_tinystories_mode(path, delimiter=delimiter) if mode == "auto" else mode
    if resolved_mode == "delimiter":
        yield from _iter_delimited_records(path, delimiter=delimiter)
    elif resolved_mode == "line":
        yield from _iter_line_records(path)
    else:
        raise ValueError("source mode must be 'auto', 'delimiter', or 'line'")


def detect_tinystories_mode(path: str | Path, delimiter: str = END_OF_TEXT) -> str:
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        sample = handle.read(1024 * 1024)
    return "delimiter" if delimiter in sample else "line"


def normalize_story(text: str, preserve_paragraphs: bool = False) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return ""

    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in normalized.split("\n")]
    lines = [line for line in lines if line]
    if preserve_paragraphs:
        return "\n".join(lines)
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def preprocess_tinystories_file(
    input_path: str | Path,
    output_path: str | Path,
    split: SplitName,
    options: TinyStoriesPreprocessOptions | None = None,
    overwrite: bool = True,
) -> dict[str, Any]:
    options = options or TinyStoriesPreprocessOptions()
    output = Path(output_path)
    if output.exists() and not overwrite:
        return {
            "split": split,
            "input_path": str(Path(input_path)),
            "output_path": str(output),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "skipped": True,
            "reason": "output already exists",
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(output.suffix + ".tmp")

    seen_hashes: set[str] = set()
    input_statistics = TextStatistics()
    output_statistics = TextStatistics()
    rejected_short = 0
    rejected_long = 0
    duplicate_count = 0

    with temporary_output.open("w", encoding="utf-8") as handle:
        for raw_story in iter_tinystories_records(
            input_path,
            mode=options.source_mode,
            delimiter=options.delimiter,
        ):
            input_statistics.update(raw_story)
            story = normalize_story(
                raw_story,
                preserve_paragraphs=options.preserve_paragraphs,
            )
            if len(story) < options.min_characters:
                rejected_short += 1
                continue
            if options.max_characters and len(story) > options.max_characters:
                rejected_long += 1
                continue

            if options.deduplicate:
                dedupe_text = story.casefold() if options.lowercase_dedupe else story
                digest = hashlib.sha1(dedupe_text.encode("utf-8")).hexdigest()
                if digest in seen_hashes:
                    duplicate_count += 1
                    continue
                seen_hashes.add(digest)

            output_statistics.update(story)
            handle.write(json.dumps({"text": story}, ensure_ascii=False) + "\n")

    temporary_output.replace(output)
    return {
        "split": split,
        "input_path": str(Path(input_path)),
        "output_path": str(output),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "options": {
            "min_characters": options.min_characters,
            "max_characters": options.max_characters,
            "deduplicate": options.deduplicate,
            "lowercase_dedupe": options.lowercase_dedupe,
            "preserve_paragraphs": options.preserve_paragraphs,
            "source_mode": options.source_mode,
            "delimiter": options.delimiter,
        },
        "input_statistics": input_statistics.to_dict(),
        "output_statistics": output_statistics.to_dict(),
        "rejected_short": rejected_short,
        "rejected_long": rejected_long,
        "duplicates_removed": duplicate_count,
    }


def _iter_delimited_records(path: str | Path, delimiter: str) -> Iterator[str]:
    buffer: list[str] = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            remaining = line
            while delimiter in remaining:
                before, remaining = remaining.split(delimiter, 1)
                buffer.append(before)
                story = "".join(buffer)
                if story.strip():
                    yield story
                buffer.clear()
            buffer.append(remaining)

    trailing = "".join(buffer)
    if trailing.strip():
        yield trailing


def _iter_line_records(path: str | Path) -> Iterator[str]:
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield stripped
