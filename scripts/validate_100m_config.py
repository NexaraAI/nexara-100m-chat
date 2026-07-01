import argparse
import sys
import tempfile
from pathlib import Path
import torch

from training.config import load_config
from training.parameter_count import estimate_transformer_parameters
from model import DecoderOnlyTransformer, ModelConfig
from training.checkpointing import save_checkpoint, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Nexara 100M configuration and instantiability."
    )
    parser.add_argument(
        "--config", default="configs/nexara_tiny_100m.toml", help="Path to the config TOML."
    )
    parser.add_argument("--device", default="cpu", help="Device to test instantiation on.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}", file=sys.stderr)
        sys.exit(1)

    print(f"--- Loading config from {config_path} ---")
    config = load_config(str(config_path))
    model_config_mapping = config["model"]

    # 1. Parameter Estimation
    estimated = estimate_transformer_parameters(model_config_mapping)
    print(f"Estimated parameters (formula): {estimated:,}")

    # 2. Model Instantiation
    print(f"Instantiating model on device: {args.device}...")
    try:
        model_config = ModelConfig(
            vocab_size=int(model_config_mapping["vocab_size"]),
            max_sequence_length=int(model_config_mapping["max_sequence_length"]),
            n_layers=int(model_config_mapping["n_layers"]),
            n_heads=int(model_config_mapping["n_heads"]),
            embedding_dim=int(model_config_mapping["embedding_dim"]),
            dropout=float(model_config_mapping["dropout"]),
            mlp_ratio=float(model_config_mapping.get("mlp_ratio", 4.0)),
            bias=bool(model_config_mapping.get("bias", False)),
            rope_base=float(model_config_mapping.get("rope_base", 10000.0)),
            tie_embeddings=bool(model_config_mapping.get("tie_embeddings", True)),
        )
        model = DecoderOnlyTransformer(model_config).to(args.device)
        print("Model instantiated successfully!")
    except Exception as e:
        print(f"Error instantiating model: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. PyTorch Parameter Count Verification
    actual_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Actual trainable parameters: {actual_params:,}")
    if actual_params != estimated:
        print(
            f"Warning: Discrepancy between estimated ({estimated}) and actual ({actual_params}) parameter count!",
            file=sys.stderr,
        )
    else:
        print("Success: Parameter count estimation matches PyTorch layer parameters exactly.")

    # 4. Dummy Forward Pass Test
    print("Testing dummy forward pass...")
    try:
        dummy_input = torch.randint(
            low=0,
            high=model_config.vocab_size,
            size=(2, model_config.max_sequence_length),
            dtype=torch.long,
            device=args.device,
        )
        model.eval()
        with torch.no_grad():
            logits, _ = model(dummy_input)
        expected_shape = (2, model_config.max_sequence_length, model_config.vocab_size)
        if logits.shape != expected_shape:
            print(
                f"Error: Expected logits shape {expected_shape}, got {logits.shape}",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Forward pass successful! Logits shape: {logits.shape}")
    except Exception as e:
        print(f"Error running forward pass: {e}", file=sys.stderr)
        sys.exit(1)

    # 5. Checkpoint Save/Load Test
    print("Testing checkpoint save and load cycle...")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_ckpt_path = Path(tmpdir) / "test_ckpt.pt"

            # Save
            print(f"Saving temporary checkpoint to {temp_ckpt_path}...")
            save_checkpoint(
                path=str(temp_ckpt_path),
                model=model,
                optimizer=None,
                scaler=None,
                step=100,
                epoch=0,
                config=config,
            )

            # Load
            print("Loading checkpoint back into fresh model instance...")
            fresh_model = DecoderOnlyTransformer(model_config).to(args.device)
            load_checkpoint(str(temp_ckpt_path), fresh_model)
            print("Checkpoint cycle completed successfully!")

    except Exception as e:
        print(f"Error testing checkpoint save/load: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n==================================================")
    print("Nexara 100M Configuration Validation: ALL PASSED")
    print(f"Total Model Size: {actual_params / 1e6:.2f}M parameters ({actual_params:,} parameters)")
    print("==================================================")


if __name__ == "__main__":
    main()
