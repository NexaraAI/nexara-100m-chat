"""Overfit validation tests for Nexara."""

from pathlib import Path
import importlib.util
import json
import tempfile
import unittest

HAS_RUNTIME_DEPS = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("tokenizers") is not None
)


@unittest.skipUnless(HAS_RUNTIME_DEPS, "requires torch and tokenizers")
class OverfitValidationTests(unittest.TestCase):
    def test_tiny_overfit_loss_decreases_and_writes_artifacts(self) -> None:
        from scripts.run_overfit_validation import run_overfit_validation

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "overfit"
            metrics = run_overfit_validation(
                output_dir=output_dir,
                steps=12,
                subset_size=20,
                vocab_size=160,
                device_name="cpu",
                experiment_name="test_tiny_overfit",
                update_experiments=False,
            )

            self.assertLess(metrics["final_loss"], metrics["initial_loss"])
            self.assertGreater(metrics["loss_delta"], 0.0)
            self.assertTrue(metrics["generation_examples"]["before_training"].strip())
            self.assertTrue(metrics["generation_examples"]["after_training"].strip())
            for key in [
                "loss_curve_json",
                "loss_curve_csv",
                "generation_examples",
                "checkpoint",
                "token_cache_statistics",
                "gradient_norms",
            ]:
                self.assertTrue(Path(metrics["artifacts"][key]).exists(), key)

    def test_extended_overfit_with_gradient_tracking(self) -> None:
        """Run 50 steps with gradient tracking and verify optimization health."""
        from scripts.run_overfit_validation import run_overfit_validation

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "overfit_extended"
            metrics = run_overfit_validation(
                output_dir=output_dir,
                steps=50,
                subset_size=25,
                vocab_size=160,
                device_name="cpu",
                experiment_name="test_extended_overfit",
                update_experiments=False,
            )

            # Loss must decrease overall
            self.assertLess(metrics["final_loss"], metrics["initial_loss"])

            # Loss trend: first 10% average must exceed last 10% average
            loss_curve = metrics["loss_curve"]
            train_losses = [
                entry["loss"] for entry in loss_curve if entry.get("kind", "train") == "train"
            ]
            n = len(train_losses)
            first_segment = train_losses[: max(1, n // 10)]
            last_segment = train_losses[-max(1, n // 10) :]
            self.assertGreater(
                sum(first_segment) / len(first_segment),
                sum(last_segment) / len(last_segment),
                "loss did not trend downward (first 10% avg should exceed last 10% avg)",
            )

            # Gradient norms must be finite and non-zero
            grad_stats = metrics["gradient_statistics"]
            self.assertGreater(grad_stats["before_clip"]["min"], 0.0)
            self.assertTrue(
                grad_stats["before_clip"]["max"] < float("inf"),
                "gradient norms should be finite",
            )
            self.assertGreater(grad_stats["after_clip"]["min"], 0.0)

            # Checkpoint exists and generation is non-empty
            self.assertTrue(Path(metrics["artifacts"]["checkpoint"]).exists())
            self.assertTrue(metrics["generation_examples"]["before_training"].strip())
            self.assertTrue(metrics["generation_examples"]["after_training"].strip())

    def test_checkpoint_load_and_generate(self) -> None:
        """Verify checkpoint round-trip and generation pipeline."""
        import torch
        from model import DecoderOnlyTransformer, ModelConfig
        from tokenizer import NexaraTokenizer
        from tokenizer.bpe import train_bpe_tokenizer
        from training.checkpointing import load_checkpoint, save_checkpoint
        from inference.generate import generate_text
        from scripts.smoke_fixtures import write_tiny_stories_jsonl

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            torch.manual_seed(42)

            data_path = write_tiny_stories_jsonl(output_dir / "data.jsonl", repeat=1)
            tokenizer_path = output_dir / "tokenizer.json"
            train_bpe_tokenizer(
                input_paths=[data_path],
                output_path=tokenizer_path,
                vocab_size=128,
                text_key="text",
                min_frequency=1,
                overwrite=True,
            )
            tokenizer = NexaraTokenizer(tokenizer_path)

            config = ModelConfig(
                vocab_size=tokenizer.vocab_size,
                max_sequence_length=32,
                n_layers=2,
                n_heads=2,
                embedding_dim=32,
                dropout=0.0,
                tie_embeddings=True,
            )
            model = DecoderOnlyTransformer(config)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

            # Save checkpoint
            checkpoint_path = output_dir / "ckpt.pt"
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scaler=None,
                step=0,
                epoch=0,
                config={"model": config.__dict__, "tokenizer": {"path": str(tokenizer_path)}},
            )

            # Reload checkpoint
            reloaded = DecoderOnlyTransformer(config)
            ckpt = load_checkpoint(checkpoint_path, reloaded, map_location="cpu")
            self.assertEqual(ckpt["step"], 0)
            reloaded.eval()

            # Generate from reloaded model
            text = generate_text(
                model=reloaded,
                tokenizer=tokenizer,
                prompt="Once upon",
                device=torch.device("cpu"),
                max_new_tokens=16,
                temperature=0.8,
                top_k=16,
                top_p=0.95,
                repetition_penalty=1.0,
            )
            self.assertTrue(text.strip(), "generation should produce non-empty text")

    def test_plot_loss_from_fixture(self) -> None:
        """Verify plot_loss produces a PNG from a metrics fixture."""
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            fixture = {
                "loss_curve": [
                    {"step": 0, "loss": 5.0},
                    {"step": 1, "loss": 4.5},
                    {"step": 2, "loss": 4.0},
                    {"step": 3, "loss": 3.5},
                    {"step": 4, "loss": 3.0},
                ],
            }
            metrics_path = output_dir / "metrics.json"
            metrics_path.write_text(json.dumps(fixture), encoding="utf-8")
            output_png = output_dir / "plot.png"

            from scripts.plot_loss import plot_loss_and_gradients

            result = plot_loss_and_gradients(output_png, fixture["loss_curve"], [])
            self.assertTrue(result.exists(), "plot PNG should be written")
            self.assertGreater(result.stat().st_size, 0, "plot PNG should not be empty")


if __name__ == "__main__":
    unittest.main()
