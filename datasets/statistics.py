"""Dataset statistics helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Iterable, Iterator, Sequence

from tokenizer.bpe import iter_text_records

WORD_RE = re.compile(r"\b[\w']+\b", re.UNICODE)


@dataclass
class TextStatistics:
    document_count: int = 0
    empty_document_count: int = 0
    character_count: int = 0
    byte_count: int = 0
    word_count: int = 0
    line_count: int = 0
    min_characters: int | None = None
    max_characters: int = 0
    min_words: int | None = None
    max_words: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def update(self, text: str) -> None:
        self.document_count += 1
        if not text:
            self.empty_document_count += 1

        characters = len(text)
        words = len(WORD_RE.findall(text))
        self.character_count += characters
        self.byte_count += len(text.encode("utf-8"))
        self.word_count += words
        self.line_count += text.count("\n") + 1 if text else 0
        self.min_characters = _min_or_value(self.min_characters, characters)
        self.max_characters = max(self.max_characters, characters)
        self.min_words = _min_or_value(self.min_words, words)
        self.max_words = max(self.max_words, words)

    def to_dict(self) -> dict[str, Any]:
        average_characters = (
            self.character_count / self.document_count if self.document_count else 0.0
        )
        average_words = self.word_count / self.document_count if self.document_count else 0.0
        payload: dict[str, Any] = {
            "document_count": self.document_count,
            "empty_document_count": self.empty_document_count,
            "character_count": self.character_count,
            "byte_count": self.byte_count,
            "word_count": self.word_count,
            "line_count": self.line_count,
            "min_characters": self.min_characters or 0,
            "max_characters": self.max_characters,
            "average_characters": average_characters,
            "min_words": self.min_words or 0,
            "max_words": self.max_words,
            "average_words": average_words,
        }
        payload.update(self.extra)
        return payload


@dataclass
class TokenStatistics:
    document_count: int = 0
    token_count: int = 0
    min_tokens: int | None = None
    max_tokens: int = 0
    block_size: int = 0

    def update(self, token_ids: Sequence[int]) -> None:
        count = len(token_ids)
        self.document_count += 1
        self.token_count += count
        self.min_tokens = _min_or_value(self.min_tokens, count)
        self.max_tokens = max(self.max_tokens, count)

    def to_dict(self) -> dict[str, Any]:
        average_tokens = self.token_count / self.document_count if self.document_count else 0.0
        sequence_count = max((self.token_count - 1) // self.block_size, 0) if self.block_size else 0
        average_sequence_length = self.block_size if sequence_count else 0.0
        return {
            "document_count": self.document_count,
            "token_count": self.token_count,
            "min_tokens": self.min_tokens or 0,
            "max_tokens": self.max_tokens,
            "average_tokens_per_document": average_tokens,
            "block_size": self.block_size,
            "sequence_count": sequence_count,
            "average_sequence_length": average_sequence_length,
            "remainder_tokens": (
                max(self.token_count - sequence_count * self.block_size, 0)
                if self.block_size
                else self.token_count
            ),
        }


def collect_text_statistics(texts: Iterable[str]) -> TextStatistics:
    statistics = TextStatistics()
    for text in texts:
        statistics.update(text)
    return statistics


def collect_dataset_statistics(
    paths: Sequence[str | Path],
    text_key: str = "text",
) -> dict[str, Any]:
    path_list = [str(Path(path)) for path in paths]
    statistics = collect_text_statistics(iter_text_records(paths, text_key=text_key))
    payload = statistics.to_dict()
    payload["paths"] = path_list
    payload["text_key"] = text_key
    return payload


def collect_tokenized_dataset_statistics(
    paths: Sequence[str | Path],
    tokenizer: Any,
    text_key: str = "text",
    block_size: int = 0,
    max_documents: int = 0,
    add_bos: bool = True,
    add_eos: bool = True,
) -> dict[str, Any]:
    text_statistics = TextStatistics()
    token_statistics = TokenStatistics(block_size=int(block_size))
    documents_seen = 0

    for text in iter_text_records(paths, text_key=text_key):
        if max_documents and documents_seen >= max_documents:
            break
        text_statistics.update(text)
        token_ids = tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)
        token_statistics.update(token_ids)
        documents_seen += 1

    payload = text_statistics.to_dict()
    payload["paths"] = [str(Path(path)) for path in paths]
    payload["text_key"] = text_key
    payload["token_statistics"] = token_statistics.to_dict()
    return payload


def write_statistics(path: str | Path, statistics: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    with temporary_output.open("w", encoding="utf-8") as handle:
        json.dump(statistics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary_output.replace(output)
    return output


def read_jsonl_texts(path: str | Path, text_key: str = "text") -> Iterator[str]:
    yield from iter_text_records([path], text_key=text_key)


def _min_or_value(current: int | None, value: int) -> int:
    return value if current is None else min(current, value)
