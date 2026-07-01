# Runbook

This file exists so another engineer or agent can resume without relying on chat
history.

## Baseline Workflow

1. Install dependencies.
2. Download TinyStories raw text files.
3. Preprocess raw text into JSONL.
4. Train the tokenizer.
5. Build token caches.
6. Run a short model training smoke test.
7. Inspect validation loss and generations.
8. Update `PROGRESS.md`, `DECISIONS.md`, `EXPERIMENTS.md`, and `HANDOFF.md`.
9. Commit changes only after validation passes.

## Dependency Check

```powershell
python -c "import torch, tokenizers; print(torch.__version__)"
```

## Tokenizer Training

```powershell
python -m tokenizer.train_tokenizer --config configs/stage1_tinystories.toml --build-cache
```

Expected output:

```text
tokenizer/nexara-bpe.json
datasets/processed/tinystories_train_tokens.bin
datasets/processed/tinystories_validation_tokens.bin
```

Tokenizer reports are written beside the tokenizer JSON:

```text
tokenizer/nexara-bpe.meta.json
tokenizer/nexara-bpe.vocab_report.json
tokenizer/nexara-bpe.sample_encodings.json
```

## Dataset Preparation

```powershell
python -m scripts.download_tinystories --variant original --split all
python -m scripts.preprocess_tinystories --config configs/stage1_tinystories.toml
python -m scripts.dataset_statistics datasets/processed/tinystories_train.jsonl
python -m scripts.dataset_statistics datasets/processed/tinystories_train_tokens.bin --cache
```

## Smoke Training

For a smoke test, temporarily set these values in
`configs/stage1_tinystories.toml` before running `scripts.train_tokenizer
--build-cache`:

```toml
max_steps = 20
eval_interval = 10
checkpoint_interval = 20
max_train_documents = 1000
max_validation_documents = 200
```

Then run:

```powershell
python -m tokenizer.train_tokenizer --config configs/stage1_tinystories.toml --build-cache
python -m training.train --config configs/stage1_tinystories.toml
```

After the run, restore the config or record the changed config as a named
experiment.

## Phase 1.2 Runtime Validation

```powershell
python -m scripts.verify_parameter_count --config configs/stage1_tinystories.toml
python -m scripts.tokenizer_smoke --output-dir logs/tokenizer_smoke
python -m scripts.benchmark_forward --tiny --iterations 5 --output logs/benchmark/forward.json
python -m scripts.train_smoke --output-dir logs/train_smoke
python -m scripts.run_overfit_validation --output-dir logs/overfit --steps 40 --experiment-name phase1_2_tiny_overfit
python -m scripts.generate_experiment_report --metrics logs/overfit/metrics.json
```
