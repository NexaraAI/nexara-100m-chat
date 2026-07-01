import argparse
import tomllib
import torch
from pathlib import Path

from model import DecoderOnlyTransformer, ModelConfig
from tokenizer import NexaraTokenizer
from training.checkpointing import load_checkpoint

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


def generate(
    model, tokenizer, device, prompt_text, max_new_tokens=80, temperature=0.7, sample=True
):
    formatted = f"### System:\nYou are Nexara, a helpful and polite AI assistant.\n\n### User:\n{prompt_text}\n\n### Assistant:\n"
    input_ids = tokenizer.encode(formatted, add_bos=True)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    generated_ids = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(input_tensor)
            next_token_logits = logits[0, -1, :]

            if sample and temperature > 0:
                next_token_logits = next_token_logits / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = torch.argmax(next_token_logits, dim=-1).item()

            if next_token == tokenizer.eos_id:
                break

            generated_ids.append(next_token)
            input_tensor = torch.cat(
                [input_tensor, torch.tensor([[next_token]], device=device)], dim=1
            )

    return tokenizer.decode(generated_ids)


def main():
    parser = argparse.ArgumentParser(description="Test SFT model on custom prompts.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file.")
    parser.add_argument("--config", default="configs/stage2_sft.toml", help="Path to config file.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    )
    tokenizer = NexaraTokenizer(config["tokenizer"]["path"])

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

    print("\n=== Testing Prompts (Greedy) ===")
    for prompt in TEST_PROMPTS:
        response = generate(model, tokenizer, device, prompt, sample=False)
        print(f"Prompt: {prompt}")
        print(f"Response: {response.strip()}")
        print("-" * 50)

    print("\n=== Testing Prompts (Sampled, Temp=0.7) ===")
    for prompt in TEST_PROMPTS:
        # Run three samples for diversity check
        print(f"Prompt: {prompt}")
        for i in range(2):
            response = generate(model, tokenizer, device, prompt, temperature=0.7, sample=True)
            print(f"  [Sample {i+1}]: {response.strip()}")
        print("-" * 50)


if __name__ == "__main__":
    main()
