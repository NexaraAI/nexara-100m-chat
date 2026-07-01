import json
import unittest
from pathlib import Path
import torch

from tokenizer.bpe import NexaraTokenizer
from training.sft.dataset import SFTDataset


class TestSFTDataset(unittest.TestCase):
    def setUp(self):
        self.project_dir = Path(__file__).resolve().parent.parent
        self.tokenizer_path = self.project_dir / "tokenizer" / "nexara-bpe.json"
        self.tokenizer = NexaraTokenizer(self.tokenizer_path)

        # Create a temp JSONL file for testing
        self.temp_jsonl = self.project_dir / "tests" / "temp_sft_test.jsonl"
        self.test_data = [
            {
                "id": "test-1",
                "system": "You are a helpful AI.",
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello!"},
                ],
            }
        ]
        with self.temp_jsonl.open("w", encoding="utf-8") as f:
            for item in self.test_data:
                f.write(json.dumps(item) + "\n")

    def tearDown(self):
        if self.temp_jsonl.exists():
            self.temp_jsonl.unlink()

    def test_sft_dataset_masking(self):
        dataset = SFTDataset(self.temp_jsonl, self.tokenizer, max_seq_length=64)
        self.assertEqual(len(dataset), 1)

        batch = dataset[0]
        input_ids = batch["input_ids"]
        targets = batch["targets"]

        # Verify shapes
        self.assertEqual(input_ids.shape, (63,))
        self.assertEqual(targets.shape, (63,))

        # Decoded verification
        # Let's verify that only the assistant's content tokens are active in targets
        active_tokens = []
        for i in range(len(targets)):
            target_val = targets[i].item()
            if target_val != -100:
                active_tokens.append(target_val)

        decoded_active = self.tokenizer.decode(active_tokens)

        # Decoded active target should be exactly the assistant's response "Hello!"
        self.assertEqual(decoded_active.strip(), "Hello!")

        # The pad tokens at the end should be masked out with -100 in targets
        # Let's find where padding begins
        # Pad token ID is 0, let's verify if they have -100 in targets
        pad_indices = (input_ids == self.tokenizer.pad_id).nonzero(as_tuple=True)[0]
        for idx in pad_indices:
            self.assertEqual(targets[idx.item()].item(), -100)


if __name__ == "__main__":
    unittest.main()
