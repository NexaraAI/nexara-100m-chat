"""BPE tokenizer training and loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Sequence

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def _require_tokenizers() -> None:
    try:
        import tokenizers  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "HuggingFace Tokenizers is required. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc


def iter_text_records(paths: Sequence[str | Path], text_key: str = "text") -> Iterator[str]:
    """Yield text documents from JSONL or plain-text files."""

    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)

        with path.open("r", encoding="utf-8") as handle:
            if path.suffix.lower() == ".jsonl":
                for line_number, line in enumerate(handle, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        record = json.loads(stripped)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"invalid JSON on {path}:{line_number}") from exc
                    value = record.get(text_key)
                    if not isinstance(value, str):
                        raise ValueError(
                            f"expected string field {text_key!r} on {path}:{line_number}"
                        )
                    yield value
            else:
                for line in handle:
                    stripped = line.strip()
                    if stripped:
                        yield stripped


class NexaraTokenizer:
    """Small wrapper around a locally trained Tokenizers JSON file."""

    def __init__(self, path: str | Path) -> None:
        _require_tokenizers()
        from tokenizers import Tokenizer

        self.path = Path(path)
        self.tokenizer = Tokenizer.from_file(str(self.path))
        self.pad_id = self._token_id("<pad>")
        self.unk_id = self._token_id("<unk>")
        self.bos_id = self._token_id("<bos>")
        self.eos_id = self._token_id("<eos>")

    @property
    def vocab_size(self) -> int:
        return int(self.tokenizer.get_vocab_size())

    def vocabulary(self) -> dict[str, int]:
        return {token: int(token_id) for token, token_id in self.tokenizer.get_vocab().items()}

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = list(self.tokenizer.encode(text).ids)
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def token_to_id(self, token: str) -> int | None:
        token_id = self.tokenizer.token_to_id(token)
        return None if token_id is None else int(token_id)

    def id_to_token(self, token_id: int) -> str | None:
        return self.tokenizer.id_to_token(int(token_id))

    def decode(self, token_ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        ids = [int(token_id) for token_id in token_ids]
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def _token_id(self, token: str) -> int:
        token_id = self.tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"tokenizer is missing required special token {token!r}")
        return int(token_id)


def train_bpe_tokenizer(
    input_paths: Sequence[str | Path],
    output_path: str | Path,
    vocab_size: int,
    text_key: str = "text",
    min_frequency: int = 2,
    overwrite: bool = True,
) -> Path:
    """Train a byte-level BPE tokenizer from local text files."""

    _require_tokenizers()
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

    if vocab_size < len(SPECIAL_TOKENS):
        raise ValueError("vocab_size must leave room for special tokens")

    output = Path(output_path)
    if output.exists() and not overwrite:
        return output

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=int(vocab_size),
        min_frequency=int(min_frequency),
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    tokenizer.train_from_iterator(
        iter_text_records(input_paths, text_key=text_key),
        trainer=trainer,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    tokenizer.save(str(temporary_output))
    temporary_output.replace(output)
    return output
