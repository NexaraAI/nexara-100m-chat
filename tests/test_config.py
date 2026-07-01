from pathlib import Path
import unittest

from training.config import load_config
from training.parameter_count import estimate_transformer_parameters

ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_stage1_config_is_valid(self) -> None:
        config = load_config(ROOT / "configs" / "stage1_tinystories.toml")
        self.assertEqual(config["model"]["vocab_size"], config["tokenizer"]["vocab_size"])
        self.assertEqual(config["model"]["max_sequence_length"], config["data"]["block_size"])

    def test_stage1_parameter_estimate_matches_target_band(self) -> None:
        config = load_config(ROOT / "configs" / "stage1_tinystories.toml")
        parameters = estimate_transformer_parameters(config["model"])
        self.assertGreaterEqual(parameters, 5_000_000)
        self.assertLessEqual(parameters, 10_000_000)


if __name__ == "__main__":
    unittest.main()
