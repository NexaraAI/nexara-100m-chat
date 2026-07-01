import argparse
import json
import tomllib
import torch
from pathlib import Path

from model import DecoderOnlyTransformer, ModelConfig
from tokenizer import NexaraTokenizer
from training.checkpointing import load_checkpoint
from inference.sampling import sample_next_token

TEST_PROMPTS = [
    "Hi",
    "What is your name?",
    "Who created you?",
    "Can you check today's weather?",
    "What is 3 + 4?",
    "List 3 fruits in markdown bullet points.",
    "Explain gravity simply.",
    "Once upon a time there was a rabbit",
]


@torch.no_grad()
def generate(
    model,
    tokenizer,
    device,
    prompt_text,
    max_new_tokens=128,
    temperature=0.7,
    top_k=40,
    top_p=0.95,
    repetition_penalty=1.15,
):
    formatted = f"### System:\nYou are Nexara, a helpful and polite AI assistant.\n\n### User:\n{prompt_text}\n\n### Assistant:\n"
    input_ids = tokenizer.encode(formatted, add_bos=True, add_eos=False)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    prompt_len = len(input_ids)

    for _ in range(max_new_tokens):
        context = input_tensor[:, -model.config.max_sequence_length :]
        logits, _ = model(context)
        next_token = sample_next_token(
            logits=logits[:, -1, :],
            previous_tokens=input_tensor,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        input_tensor = torch.cat((input_tensor, next_token), dim=1)
        if int(next_token.item()) == tokenizer.eos_id:
            break

    new_token_ids = input_tensor[0][prompt_len:].tolist()
    return tokenizer.decode(new_token_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(
        description="Test SFT model on custom prompts for chat release."
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to clean model or checkpoint file."
    )
    parser.add_argument("--config", default="configs/stage2_sft.toml", help="Path to config file.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    )

    checkpoint_dir = Path(args.checkpoint).parent
    tokenizer_path = checkpoint_dir / "nexara-bpe.json"
    if not tokenizer_path.exists():
        tokenizer_path = Path(config["tokenizer"]["path"])

    tokenizer = NexaraTokenizer(str(tokenizer_path))

    model_config = ModelConfig(
        vocab_size=config["model"]["vocab_size"],
        max_sequence_length=config["model"]["max_sequence_length"],
        n_layers=config["model"]["n_layers"],
        n_heads=config["model"]["n_heads"],
        embedding_dim=config["model"]["embedding_dim"],
        dropout=0.0,
        mlp_ratio=config["model"]["mlp_ratio"],
        bias=config["model"]["bias"],
        rope_base=config["model"]["rope_base"],
        tie_embeddings=config["model"]["tie_embeddings"],
    )

    model = DecoderOnlyTransformer(model_config)
    print(f"Loading checkpoint {args.checkpoint} on {device}")
    load_checkpoint(args.checkpoint, model, map_location=device)
    model = model.to(device)
    model.eval()

    # Load defaults from generation_config.json if present
    gen_config_path = checkpoint_dir / "generation_config.json"
    temperature = 0.7
    top_k = 40
    top_p = 0.95
    repetition_penalty = 1.15
    max_new_tokens = 128

    if gen_config_path.exists():
        try:
            with gen_config_path.open("r", encoding="utf-8") as f:
                gen_defaults = json.load(f)
                temperature = gen_defaults.get("temperature", temperature)
                top_k = gen_defaults.get("top_k", top_k)
                top_p = gen_defaults.get("top_p", top_p)
                repetition_penalty = gen_defaults.get("repetition_penalty", repetition_penalty)
                max_new_tokens = gen_defaults.get("max_new_tokens", max_new_tokens)
        except Exception:
            pass

    print("\n=== Running Chat Release Prompt Tests (Sampled Decoding) ===")
    print(
        f"Settings: Temp={temperature}, Top-K={top_k}, Top-P={top_p}, Rep-Penalty={repetition_penalty}"
    )

    for prompt in TEST_PROMPTS:
        response = generate(
            model,
            tokenizer,
            device,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        print(f"\nPrompt: {prompt}")
        print(f"Response: {response.strip()}")
        print("-" * 50)


if __name__ == "__main__":
    main()
