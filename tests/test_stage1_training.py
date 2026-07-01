"""Tests for enhanced training loop, checkpointing, and resume/evaluate scripts."""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

from model import DecoderOnlyTransformer, ModelConfig
from training.checkpointing import find_latest_checkpoint, save_checkpoint, validate_checkpoint
from training.train import train_step, get_lr


class Stage1TrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        # Create a small model configuration for testing
        self.config = {
            "model": {
                "vocab_size": 256,
                "max_sequence_length": 64,
                "n_layers": 2,
                "n_heads": 2,
                "embedding_dim": 64,
                "dropout": 0.0,
                "mlp_ratio": 2.0,
                "bias": False,
                "rope_base": 10000.0,
                "tie_embeddings": True,
            },
            "training": {
                "learning_rate": 0.001,
                "beta1": 0.9,
                "beta2": 0.95,
                "weight_decay": 0.1,
                "grad_clip": 1.0,
                "gradient_accumulation_steps": 1,
            },
        }
        self.model_config = ModelConfig.from_mapping(self.config["model"])
        self.model = DecoderOnlyTransformer(self.model_config)

    def test_checkpoint_roundtrip_and_validation(self) -> None:
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        scaler = torch.cuda.amp.GradScaler(enabled=False)

        with tempfile.TemporaryDirectory() as directory:
            ckpt_dir = Path(directory)
            ckpt_path = ckpt_dir / "step_10.pt"

            # Save checkpoint
            save_checkpoint(
                ckpt_path,
                model=self.model,
                optimizer=optimizer,
                scaler=scaler,
                step=10,
                epoch=1,
                config=self.config,
                metrics={"loss": 2.5},
            )

            # Validate checkpoint format and key presence
            self.assertTrue(validate_checkpoint(ckpt_path))

            # Test find_latest_checkpoint
            latest = find_latest_checkpoint(ckpt_dir)
            self.assertEqual(latest, ckpt_path)

            # Reload and check matching state dict
            reloaded_model = DecoderOnlyTransformer(self.model_config)
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            reloaded_model.load_state_dict(checkpoint["model_state_dict"])

            for k in self.model.state_dict():
                self.assertTrue(
                    torch.equal(self.model.state_dict()[k], reloaded_model.state_dict()[k])
                )

    def test_get_lr_scheduler(self) -> None:
        # Test get_lr function logic
        warmup_steps = 10
        max_steps = 100
        base_lr = 0.001
        min_lr = 0.0001

        # Warmup phase (linear increase)
        self.assertAlmostEqual(get_lr(0, max_steps, warmup_steps, base_lr, min_lr), 0.0)
        self.assertAlmostEqual(get_lr(5, max_steps, warmup_steps, base_lr, min_lr), 0.0005)
        self.assertAlmostEqual(get_lr(10, max_steps, warmup_steps, base_lr, min_lr), base_lr)

        # Decay phase (cosine decrease)
        # At step 100, we should be at min_lr
        self.assertAlmostEqual(get_lr(100, max_steps, warmup_steps, base_lr, min_lr), min_lr)
        # In between, it should be between base_lr and min_lr
        lr_mid = get_lr(55, max_steps, warmup_steps, base_lr, min_lr)
        self.assertTrue(min_lr < lr_mid < base_lr)
        # Past max steps, it should stay at min_lr
        self.assertEqual(get_lr(120, max_steps, warmup_steps, base_lr, min_lr), min_lr)

    def test_train_step_gradient_accumulation(self) -> None:
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        scaler = torch.cuda.amp.GradScaler(enabled=False)

        input_ids = torch.randint(0, 256, (2, 16))
        targets = torch.randint(0, 256, (2, 16))

        # First step: microstep, no update
        loss_val, grad_norm_before, grad_norm_after = train_step(
            model=self.model,
            optimizer=optimizer,
            scaler=scaler,
            input_ids=input_ids,
            targets=targets,
            device=torch.device("cpu"),
            precision="fp32",
            gradient_accumulation_steps=2,
            grad_clip=1.0,
            is_update_step=False,
        )
        self.assertTrue(loss_val > 0)
        self.assertEqual(grad_norm_before, 0.0)
        self.assertEqual(grad_norm_after, 0.0)

        # Second step: update step
        loss_val, grad_norm_before, grad_norm_after = train_step(
            model=self.model,
            optimizer=optimizer,
            scaler=scaler,
            input_ids=input_ids,
            targets=targets,
            device=torch.device("cpu"),
            precision="fp32",
            gradient_accumulation_steps=2,
            grad_clip=1.0,
            is_update_step=True,
        )
        self.assertTrue(loss_val > 0)
        self.assertTrue(grad_norm_before >= 0.0)
        self.assertTrue(grad_norm_after >= 0.0)

    def test_checkpoint_rotation_policy(self) -> None:
        from training.checkpointing import rotate_checkpoints

        with tempfile.TemporaryDirectory() as directory:
            ckpt_dir = Path(directory)

            # Create a set of mock checkpoints
            (ckpt_dir / "best.pt").touch()
            (ckpt_dir / "latest.pt").touch()
            (ckpt_dir / "final.pt").touch()
            for step in range(500, 4500, 500):
                (ckpt_dir / f"step_{step}.pt").touch()

            # Total step checkpoints: 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000 (8 total)
            # Run rotation, keeping latest 5
            rotate_checkpoints(ckpt_dir, keep_last_n=5)

            # check that best.pt, latest.pt, final.pt are preserved
            self.assertTrue((ckpt_dir / "best.pt").exists())
            self.assertTrue((ckpt_dir / "latest.pt").exists())
            self.assertTrue((ckpt_dir / "final.pt").exists())

            # check that the latest 5 step checkpoints are kept:
            # 2000, 2500, 3000, 3500, 4000
            for step in [2000, 2500, 3000, 3500, 4000]:
                self.assertTrue((ckpt_dir / f"step_{step}.pt").exists())

            # check that older ones are deleted:
            # 500, 1000, 1500
            for step in [500, 1000, 1500]:
                self.assertFalse((ckpt_dir / f"step_{step}.pt").exists())

    def test_checkpoint_portability_cpu_save_load(self) -> None:
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        scaler = torch.amp.GradScaler("cuda", enabled=False)

        with tempfile.TemporaryDirectory() as directory:
            ckpt_dir = Path(directory)
            ckpt_path = ckpt_dir / "portability_test.pt"

            # Save
            save_checkpoint(
                ckpt_path,
                model=self.model,
                optimizer=optimizer,
                scaler=scaler,
                step=1,
                epoch=0,
                config=self.config,
            )

            # Load the checkpoint dict directly and verify all model and optimizer tensors are on CPU
            checkpoint = torch.load(ckpt_path, map_location="cpu")

            # Model state dict check
            for name, tensor in checkpoint["model_state_dict"].items():
                self.assertEqual(tensor.device.type, "cpu")

            # Optimizer state dict check
            for param_id, state in checkpoint["optimizer_state_dict"]["state"].items():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        self.assertEqual(v.device.type, "cpu")

    def test_export_checkpoint_script(self) -> None:
        import subprocess
        import sys

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        scaler = torch.amp.GradScaler("cuda", enabled=False)

        with tempfile.TemporaryDirectory() as directory:
            ckpt_dir = Path(directory)
            ckpt_path = ckpt_dir / "step_100.pt"

            save_checkpoint(
                ckpt_path,
                model=self.model,
                optimizer=optimizer,
                scaler=scaler,
                step=100,
                epoch=1,
                config=self.config,
            )

            # Run export_checkpoint.py script via subprocess
            out_pt = ckpt_dir / "clean_model.pt"
            out_json = ckpt_dir / "config.json"

            cmd = [
                sys.executable,
                "scripts/export_checkpoint.py",
                "--checkpoint",
                str(ckpt_path),
                "--output-pt",
                str(out_pt),
                "--output-json",
                str(out_json),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, f"export_checkpoint failed: {result.stderr}")

            # Verify exports exist and contain expected structures
            self.assertTrue(out_pt.exists())
            self.assertTrue(out_json.exists())

            clean_ckpt = torch.load(out_pt, map_location="cpu")
            self.assertIn("model_state_dict", clean_ckpt)
            self.assertIn("config", clean_ckpt)
            self.assertNotIn("optimizer_state_dict", clean_ckpt)  # Stripped!

            with out_json.open("r", encoding="utf-8") as f:
                json_config = json.load(f)
            self.assertEqual(json_config["model"]["vocab_size"], 256)

    def test_export_huggingface_script(self) -> None:
        import subprocess
        import sys

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        scaler = torch.amp.GradScaler("cuda", enabled=False)

        # We need a mock tokenizer file to pass the script check
        with tempfile.TemporaryDirectory() as directory:
            temp_dir = Path(directory)
            ckpt_path = temp_dir / "step_200.pt"

            save_checkpoint(
                ckpt_path,
                model=self.model,
                optimizer=optimizer,
                scaler=scaler,
                step=200,
                epoch=2,
                config=self.config,
            )

            mock_tokenizer_path = temp_dir / "mock_tokenizer.json"
            with mock_tokenizer_path.open("w", encoding="utf-8") as f:
                json.dump({"dummy": True}, f)

            hf_out_dir = temp_dir / "hf_package"

            # Run export_huggingface.py
            cmd = [
                sys.executable,
                "scripts/export_huggingface.py",
                "--checkpoint",
                str(ckpt_path),
                "--tokenizer",
                str(mock_tokenizer_path),
                "--output-dir",
                str(hf_out_dir),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, f"export_huggingface failed: {result.stderr}")

            # Verify HF exports
            self.assertTrue((hf_out_dir / "pytorch_model.bin").exists())
            self.assertTrue((hf_out_dir / "config.json").exists())
            self.assertTrue((hf_out_dir / "generation_config.json").exists())
            self.assertTrue((hf_out_dir / "tokenizer.json").exists())
            self.assertTrue((hf_out_dir / "tokenizer_config.json").exists())
            self.assertTrue((hf_out_dir / "configuration_nexara.py").exists())
            self.assertTrue((hf_out_dir / "modeling_nexara.py").exists())
            self.assertTrue((hf_out_dir / "README.md").exists())

            # Verify auto_map in config.json
            with (hf_out_dir / "config.json").open("r", encoding="utf-8") as f:
                hf_config = json.load(f)
            self.assertIn("auto_map", hf_config)
            self.assertEqual(
                hf_config["auto_map"]["AutoConfig"], "configuration_nexara.NexaraConfig"
            )
            self.assertEqual(
                hf_config["auto_map"]["AutoModelForCausalLM"], "modeling_nexara.NexaraForCausalLM"
            )

    def test_live_dashboard_telemetry_parsers(self) -> None:
        from scripts.live_dashboard import parse_metrics, calculate_eta

        with tempfile.TemporaryDirectory() as directory:
            metrics_path = Path(directory) / "metrics.jsonl"

            # Create a mock metrics history
            with metrics_path.open("w", encoding="utf-8") as f:
                f.write(json.dumps({"step": 100, "train_loss": 3.5, "timestamp": 1000.0}) + "\n")
                f.write(json.dumps({"step": 200, "train_loss": 3.0, "timestamp": 1010.0}) + "\n")
                f.write(json.dumps({"step": 300, "train_loss": 2.5, "timestamp": 1020.0}) + "\n")

            # Verify parser
            state = parse_metrics(metrics_path)
            self.assertEqual(state["step"], 300)
            self.assertEqual(state["train_loss"], 2.5)

            # Verify ETA calculation
            # 10 steps per second locally (from step 100 to 300 in 20s = 10 steps/sec)
            # Remaining steps to 1300 = 1000 steps.
            # 1000 steps / 10 steps/sec = 100 seconds = 00:01:40
            eta = calculate_eta(metrics_path, current_step=300, max_steps=1300)
            self.assertEqual(eta, "00:01:40")

    def test_train_long_orchestrator_argparse(self) -> None:
        import subprocess
        import sys

        # Test train_long help output and configuration overrides parsing
        cmd = [sys.executable, "scripts/train_long.py", "--help"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("--keep-last-n", result.stdout)
        self.assertIn("--no-resume", result.stdout)

    def test_benchmark_generation_statistics(self) -> None:
        from scripts.benchmark_generation import compute_statistics

        # Mock texts and token sequences
        texts = [
            "Once upon a time, there was a little dog. The dog liked to play.",
            "Lily wanted to eat a sweet apple. She found one on the tree.",
        ]
        tokens = [
            [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 2],
            [1, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 2],
        ]

        stats = compute_statistics(texts, tokens)

        self.assertIn("bigram_repetition_rate", stats)
        self.assertIn("trigram_repetition_rate", stats)
        self.assertIn("average_sentence_length_words", stats)
        self.assertIn("token_entropy", stats)
        self.assertIn("heuristic_coherence_score", stats)

        self.assertTrue(stats["average_sentence_length_words"] > 0)
        self.assertTrue(0.0 <= stats["bigram_repetition_rate"] <= 1.0)
        self.assertTrue(stats["token_entropy"] > 0)
        self.assertTrue(0.0 <= stats["heuristic_coherence_score"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
