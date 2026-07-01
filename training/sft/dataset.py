import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from tokenizer import NexaraTokenizer


class SFTDataset(Dataset):
    """Dataset for Supervised Fine-Tuning (SFT) with assistant-only loss masking."""

    def __init__(
        self, jsonl_path: str | Path, tokenizer: NexaraTokenizer, max_seq_length: int = 256
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.examples = []

        self.load_dataset()

    def load_dataset(self) -> None:
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"SFT dataset not found at {self.jsonl_path}")

        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                self.examples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        example = self.examples[idx]
        system_content = example.get("system", "")
        messages = example.get("messages", [])

        input_ids = []
        labels = []

        # 1. Format and tokenize system prompt
        if system_content:
            system_text = f"### System:\n{system_content}\n\n"
            system_ids = self.tokenizer.encode(system_text, add_bos=True)
            input_ids.extend(system_ids)
            labels.extend([-100] * len(system_ids))

        # 2. Format and tokenize dialogue turns
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                user_text = f"### User:\n{content}\n\n"
                # Add BOS if it hasn't been added yet (i.e. if system prompt was empty)
                add_bos = len(input_ids) == 0
                user_ids = self.tokenizer.encode(user_text, add_bos=add_bos)
                input_ids.extend(user_ids)
                labels.extend([-100] * len(user_ids))

            elif role == "assistant":
                prefix = "### Assistant:\n"
                prefix_ids = self.tokenizer.encode(prefix)
                content_ids = self.tokenizer.encode(content, add_eos=True)

                # Mask assistant prefix from loss
                input_ids.extend(prefix_ids)
                labels.extend([-100] * len(prefix_ids))

                # Keep assistant content for loss calculation
                input_ids.extend(content_ids)
                labels.extend(content_ids)

        # 3. Truncate if sequence exceeds max length
        if len(input_ids) > self.max_seq_length:
            input_ids = input_ids[: self.max_seq_length]
            labels = labels[: self.max_seq_length]

        # 4. Pad if sequence is shorter than max length
        pad_len = self.max_seq_length - len(input_ids)
        if pad_len > 0:
            input_ids.extend([self.tokenizer.pad_id] * pad_len)
            labels.extend([-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids[:-1], dtype=torch.long),
            "targets": torch.tensor(labels[1:], dtype=torch.long),
        }
