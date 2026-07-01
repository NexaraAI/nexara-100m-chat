"""Disk-backed token cache for memory-efficient training."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

from tokenizer.bpe import iter_text_records

try:
    from torch.utils.data import IterableDataset as _TorchIterableDataset
except ImportError:
    _TorchIterableDataset = object  # type: ignore[assignment,misc]


TOKEN_DTYPE = "<u4"
TOKEN_ARRAY_CODE = "I"
TOKEN_CHUNK_SIZE = 1_000_000


@dataclass(frozen=True)
class TokenCacheResult:
    cache_path: Path
    metadata_path: Path
    document_count: int
    token_count: int
    sequence_count: int
    average_tokens_per_document: float
    average_sequence_length: float


def token_cache_metadata_path(cache_path: str | Path) -> Path:
    path = Path(cache_path)
    return path.with_suffix(".json")


def write_token_cache(
    input_paths: Sequence[str | Path],
    tokenizer: Any,
    output_path: str | Path,
    block_size: int,
    text_key: str = "text",
    max_documents: int = 0,
    add_bos: bool = True,
    add_eos: bool = True,
    overwrite: bool = True,
) -> TokenCacheResult:
    if block_size < 2:
        raise ValueError("block_size must be at least 2")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    metadata_path = token_cache_metadata_path(output)
    if output.exists() and metadata_path.exists() and not overwrite:
        metadata = load_token_cache_metadata(output)
        return TokenCacheResult(
            cache_path=output,
            metadata_path=metadata_path,
            document_count=int(metadata["document_count"]),
            token_count=int(metadata["token_count"]),
            sequence_count=int(metadata["sequence_count"]),
            average_tokens_per_document=float(metadata.get("average_tokens_per_document", 0.0)),
            average_sequence_length=float(metadata.get("average_sequence_length", 0.0)),
        )
    if output.exists() and not metadata_path.exists() and not overwrite:
        raise FileExistsError(f"{output} exists without metadata; rerun with overwrite enabled")

    document_count = 0
    token_count = 0
    pending = array(TOKEN_ARRAY_CODE)
    if pending.itemsize != 4:
        raise RuntimeError("platform array('I') is not 32 bits")

    with temporary_output.open("wb") as handle:
        for text in iter_text_records(input_paths, text_key=text_key):
            token_ids = list(tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos))
            _validate_token_ids(token_ids)
            pending.extend(token_ids)
            token_count += len(token_ids)
            document_count += 1

            if len(pending) >= TOKEN_CHUNK_SIZE:
                _write_uint32_chunk(handle, pending)
                pending = array(TOKEN_ARRAY_CODE)

            if max_documents and document_count >= max_documents:
                break

        if pending:
            _write_uint32_chunk(handle, pending)

    temporary_output.replace(output)
    sequence_count = max((token_count - 1) // block_size, 0)
    average_tokens_per_document = token_count / document_count if document_count else 0.0
    average_sequence_length = block_size if sequence_count else 0.0
    metadata = {
        "cache_path": str(output),
        "dtype": TOKEN_DTYPE,
        "block_size": int(block_size),
        "document_count": document_count,
        "token_count": token_count,
        "sequence_count": sequence_count,
        "average_tokens_per_document": average_tokens_per_document,
        "average_sequence_length": average_sequence_length,
        "remainder_tokens": max(token_count - sequence_count * block_size, 0),
        "input_paths": [str(Path(path)) for path in input_paths],
        "text_key": text_key,
        "add_bos": add_bos,
        "add_eos": add_eos,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    with temporary_metadata.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary_metadata.replace(metadata_path)

    return TokenCacheResult(
        cache_path=output,
        metadata_path=metadata_path,
        document_count=document_count,
        token_count=token_count,
        sequence_count=sequence_count,
        average_tokens_per_document=average_tokens_per_document,
        average_sequence_length=average_sequence_length,
    )


def write_configured_token_cache(
    config: dict[str, Any],
    tokenizer: Any,
    split: str,
    overwrite: bool = True,
) -> TokenCacheResult:
    data = config["data"]
    if split == "train":
        input_path = data["train_path"]
        output_path = data["train_cache_path"]
        max_documents = int(data.get("max_train_documents", 0))
    elif split == "validation":
        input_path = data["validation_path"]
        output_path = data["validation_cache_path"]
        max_documents = int(data.get("max_validation_documents", 0))
    else:
        raise ValueError(f"unknown split {split!r}")

    return write_token_cache(
        input_paths=[Path(input_path)],
        tokenizer=tokenizer,
        output_path=output_path,
        block_size=int(data["block_size"]),
        text_key=str(data.get("text_key", "text")),
        max_documents=max_documents,
        overwrite=overwrite,
    )


def load_token_cache_metadata(cache_path: str | Path) -> dict[str, Any]:
    metadata_path = token_cache_metadata_path(cache_path)
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_uint32_tokens(cache_path: str | Path) -> list[int]:
    raw = Path(cache_path).read_bytes()
    if len(raw) % 4 != 0:
        raise ValueError(f"token cache has invalid byte length: {len(raw)}")
    values = array(TOKEN_ARRAY_CODE)
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    return [int(value) for value in values]


class TokenCacheDataset:
    """Map-style dataset backed by a uint32 token memmap."""

    def __init__(self, cache_path: str | Path, block_size: int | None = None) -> None:
        self.cache_path = Path(cache_path)
        self.metadata = load_token_cache_metadata(self.cache_path)
        self.block_size = int(block_size or self.metadata["block_size"])
        if self.block_size != int(self.metadata["block_size"]):
            raise ValueError("requested block_size does not match token cache metadata")
        self.sequence_count = int(self.metadata["sequence_count"])
        self._tokens = None

    def __len__(self) -> int:
        return self.sequence_count

    def __getitem__(self, index: int):
        if index < 0 or index >= self.sequence_count:
            raise IndexError(index)

        import numpy as np
        import torch

        tokens = self._memmap()
        start = index * self.block_size
        chunk = np.asarray(tokens[start : start + self.block_size + 1], dtype=np.int64)
        return torch.from_numpy(chunk[:-1].copy()), torch.from_numpy(chunk[1:].copy())

    def _memmap(self):
        if self._tokens is None:
            import numpy as np

            self._tokens = np.memmap(self.cache_path, mode="r", dtype=TOKEN_DTYPE)
        return self._tokens

    def close(self) -> None:
        if self._tokens is not None and hasattr(self._tokens, "_mmap"):
            self._tokens._mmap.close()
        self._tokens = None


class StreamingTokenCacheDataset(_TorchIterableDataset):  # type: ignore[misc,valid-type]
    """Iterable dataset that streams fixed token blocks from a binary cache."""

    def __init__(
        self,
        cache_path: str | Path,
        block_size: int | None = None,
        use_mmap: bool = True,
    ) -> None:
        if _TorchIterableDataset is object:
            raise RuntimeError(
                "PyTorch is required for StreamingTokenCacheDataset. Install dependencies "
                "with `python -m pip install -r requirements.txt`."
            )
        self.cache_path = Path(cache_path)
        self.metadata = load_token_cache_metadata(self.cache_path)
        self.block_size = int(block_size or self.metadata["block_size"])
        if self.block_size != int(self.metadata["block_size"]):
            raise ValueError("requested block_size does not match token cache metadata")
        self.sequence_count = int(self.metadata["sequence_count"])
        self.use_mmap = bool(use_mmap)

    def __iter__(self):
        import numpy as np
        import torch

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        worker_count = worker_info.num_workers if worker_info is not None else 1
        tokens = open_token_array(self.cache_path, use_mmap=self.use_mmap)

        for index in range(worker_id, self.sequence_count, worker_count):
            start = index * self.block_size
            chunk = np.asarray(tokens[start : start + self.block_size + 1], dtype=np.int64)
            yield torch.from_numpy(chunk[:-1].copy()), torch.from_numpy(chunk[1:].copy())


def open_token_array(cache_path: str | Path, use_mmap: bool = True):
    import numpy as np

    if use_mmap:
        return np.memmap(cache_path, mode="r", dtype=TOKEN_DTYPE)
    return np.asarray(read_uint32_tokens(cache_path), dtype=np.uint32)


def iter_token_cache_blocks(
    cache_path: str | Path,
    block_size: int | None = None,
    use_mmap: bool = True,
):
    """Yield input/target token blocks from a binary cache without importing PyTorch."""

    metadata = load_token_cache_metadata(cache_path)
    resolved_block_size = int(block_size or metadata["block_size"])
    if resolved_block_size != int(metadata["block_size"]):
        raise ValueError("requested block_size does not match token cache metadata")
    sequence_count = int(metadata["sequence_count"])
    tokens = open_token_array(cache_path, use_mmap=use_mmap)
    for index in range(sequence_count):
        start = index * resolved_block_size
        chunk = tokens[start : start + resolved_block_size + 1]
        yield chunk[:-1], chunk[1:]


def collect_token_cache_statistics(cache_path: str | Path) -> dict[str, Any]:
    path = Path(cache_path)
    metadata = load_token_cache_metadata(path)
    token_count = int(metadata["token_count"])
    document_count = int(metadata["document_count"])
    sequence_count = int(metadata["sequence_count"])
    block_size = int(metadata["block_size"])
    file_size = path.stat().st_size if path.exists() else 0
    expected_size = token_count * 4
    return {
        "cache_path": str(path),
        "metadata_path": str(token_cache_metadata_path(path)),
        "dtype": metadata.get("dtype", TOKEN_DTYPE),
        "block_size": block_size,
        "document_count": document_count,
        "token_count": token_count,
        "sequence_count": sequence_count,
        "average_tokens_per_document": (token_count / document_count if document_count else 0.0),
        "average_sequence_length": block_size if sequence_count else 0.0,
        "remainder_tokens": max(token_count - sequence_count * block_size, 0),
        "file_size_bytes": file_size,
        "expected_file_size_bytes": expected_size,
        "file_size_matches_token_count": file_size == expected_size,
        "memory_mapping_supported": True,
        "streaming_supported": True,
    }


def _write_uint32_chunk(handle: Any, values: array[int]) -> None:
    if sys.byteorder != "little":
        values = array(TOKEN_ARRAY_CODE, values)
        values.byteswap()
    values.tofile(handle)


def _validate_token_ids(token_ids: Iterable[int]) -> None:
    for token_id in token_ids:
        if token_id < 0 or token_id > 0xFFFFFFFF:
            raise ValueError(f"token id {token_id} cannot be stored as uint32")
