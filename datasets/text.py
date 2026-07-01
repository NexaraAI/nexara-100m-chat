"""Text dataset loading for autoregressive language modeling."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset

from tokenizer.bpe import NexaraTokenizer, iter_text_records


class TokenBlockDataset(Dataset):
    """A contiguous-token dataset split into fixed-length next-token blocks."""

    def __init__(
        self,
        paths: Sequence[str | Path],
        tokenizer: NexaraTokenizer,
        block_size: int,
        text_key: str = "text",
        max_documents: int = 0,
    ) -> None:
        if block_size < 2:
            raise ValueError("block_size must be at least 2")

        self.block_size = int(block_size)
        token_ids: list[int] = []
        documents_seen = 0

        for text in iter_text_records(paths, text_key=text_key):
            if not text.strip():
                continue
            token_ids.extend(tokenizer.encode(text, add_bos=True, add_eos=True))
            documents_seen += 1
            if max_documents and documents_seen >= max_documents:
                break

        if len(token_ids) < self.block_size + 1:
            raise ValueError(
                "not enough tokens to create a dataset; "
                f"got {len(token_ids)}, need at least {self.block_size + 1}"
            )

        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.sequence_count = (len(self.tokens) - 1) // self.block_size

    def __len__(self) -> int:
        return self.sequence_count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= self.sequence_count:
            raise IndexError(index)

        start = index * self.block_size
        chunk = self.tokens[start : start + self.block_size + 1]
        return chunk[:-1], chunk[1:]
