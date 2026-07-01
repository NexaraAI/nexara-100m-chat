from pathlib import Path
import json
import tempfile
import unittest

from scripts.generate_experiment_report import update_experiments_from_metrics


class ExperimentReportTests(unittest.TestCase):
    def test_update_experiments_from_metrics_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics_path = root / "metrics.json"
            experiments_path = root / "EXPERIMENTS.md"
            metrics_path.write_text(
                json.dumps(
                    {
                        "experiment_name": "tiny_test",
                        "created_at": "2026-06-20T00:00:00+00:00",
                        "duration_seconds": 1.25,
                        "dataset": "fixture",
                        "hyperparameters": {"steps": 2, "learning_rate": 0.01},
                        "initial_loss": 4.0,
                        "final_loss": 2.0,
                        "loss_delta": 2.0,
                        "observations": ["loss decreased"],
                    }
                ),
                encoding="utf-8",
            )
            experiments_path.write_text("# Experiments\n", encoding="utf-8")

            first = update_experiments_from_metrics(metrics_path, experiments_path)
            second = update_experiments_from_metrics(metrics_path, experiments_path)

        self.assertEqual(first, second)
        self.assertIn("## 2026-06-20: Tiny Test", second)
        self.assertEqual(second.count("experiment:tiny_test:start"), 1)


if __name__ == "__main__":
    unittest.main()
