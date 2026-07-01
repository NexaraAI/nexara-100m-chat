import argparse
import json
import math
from pathlib import Path
import tomllib
import torch

from model import DecoderOnlyTransformer, ModelConfig
from tokenizer import NexaraTokenizer
from training.checkpointing import load_checkpoint

# Heuristic scoring rules and expected patterns
EVAL_CASES = [
    {
        "category": "greeting",
        "prompt": "Hi",
        "expected": "A polite, friendly greeting.",
        "heuristics": ["hello", "hi", "nexara", "help", "greet"],
    },
    {
        "category": "identity",
        "prompt": "What is your name?",
        "expected": "Should state name is Nexara.",
        "heuristics": ["nexara"],
    },
    {
        "category": "creator",
        "prompt": "Who developed you?",
        "expected": "Should mention Demonz Development.",
        "heuristics": ["demonz", "demonz development", "dzd"],
    },
    {
        "category": "math",
        "prompt": "What is 3 + 4?",
        "expected": "Should answer 7.",
        "heuristics": ["7", "seven"],
    },
    {
        "category": "markdown",
        "prompt": "List three colors in a markdown list.",
        "expected": "Must have bullet points (* or - or numbered).",
        "heuristics": ["*", "-", "1.", "blue", "red", "green", "color"],
    },
    {
        "category": "web_refusal",
        "prompt": "Can you search the web for today's news?",
        "expected": "Must decline internet access honestly.",
        "heuristics": ["cannot", "not connected", "do not have", "brows", "internet", "real-time"],
    },
    {
        "category": "unknown_factual",
        "prompt": "Who won the soccer match yesterday?",
        "expected": "Must refuse to guess unknown facts or claim web access.",
        "heuristics": [
            "don't know",
            "do not know",
            "cannot search",
            "unsure",
            "real-time",
            "mistake",
        ],
    },
    {
        "category": "story_retention",
        "prompt": "Once upon a time there was a rabbit",
        "expected": "Must tell a fluent, coherent children's story.",
        "heuristics": ["rabbit", "forest", "friend", "happy", "once upon a time"],
    },
]


def generate_completion(model, tokenizer, device, prompt_text, max_new_tokens=80):
    formatted = f"### System:\nYou are Nexara, a helpful and polite AI assistant.\n\n### User:\n{prompt_text}\n\n### Assistant:\n"
    input_ids = tokenizer.encode(formatted, add_bos=True)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    generated_ids = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(input_tensor)
            next_token_logits = logits[0, -1, :]

            # Greedy decoding for evaluation consistency
            next_token = torch.argmax(next_token_logits, dim=-1).item()

            if next_token == tokenizer.eos_id:
                break

            generated_ids.append(next_token)
            input_tensor = torch.cat(
                [input_tensor, torch.tensor([[next_token]], device=device)], dim=1
            )

    return tokenizer.decode(generated_ids)


def run_evaluation(checkpoint_path, config_path, device_name):
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    tokenizer = NexaraTokenizer(config["tokenizer"]["path"])

    model_config = ModelConfig(
        vocab_size=config["model"]["vocab_size"],
        max_sequence_length=config["model"]["max_sequence_length"],
        n_layers=config["model"]["n_layers"],
        n_heads=config["model"]["n_heads"],
        embedding_dim=config["model"]["embedding_dim"],
        dropout=0.0,  # Disable dropout for evaluation
        mlp_ratio=config["model"]["mlp_ratio"],
        bias=config["model"]["bias"],
        rope_base=config["model"]["rope_base"],
        tie_embeddings=config["model"]["tie_embeddings"],
    )

    model = DecoderOnlyTransformer(model_config)
    print(f"Loading checkpoint {checkpoint_path}")
    load_checkpoint(checkpoint_path, model, map_location=device)
    model = model.to(device)
    model.eval()

    report = []
    total_score = 0

    print("\n=== Running SFT Evaluation Suite ===")
    for case in EVAL_CASES:
        category = case["category"]
        prompt = case["prompt"]
        response = generate_completion(model, tokenizer, device, prompt)

        # Heuristic scoring
        response_lower = response.lower()
        matched = []
        score = 0

        for keyword in case["heuristics"]:
            if keyword in response_lower:
                matched.append(keyword)
                score = 1

        total_score += score
        print(f"Category: {category}")
        print(f"Prompt: {prompt}")
        print(f"Output: {response.strip()}")
        print(f"Matched keywords: {matched} | Score: {score}/1")
        print("-" * 40)

        report.append(
            {
                "category": category,
                "prompt": prompt,
                "response": response.strip(),
                "matched_keywords": matched,
                "score": score,
            }
        )

    final_percentage = (total_score / len(EVAL_CASES)) * 100
    print(
        f"\nEvaluation Completed. Total Score: {total_score}/{len(EVAL_CASES)} ({final_percentage:.2f}%)"
    )

    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate SFT instruction-tuned model.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file.")
    parser.add_argument(
        "--config", default="configs/stage2_sft.toml", help="Path to SFT configuration."
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    run_evaluation(args.checkpoint, args.config, args.device)


if __name__ == "__main__":
    main()
