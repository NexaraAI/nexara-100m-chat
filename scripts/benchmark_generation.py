"""Benchmark generation script for Nexara models.

Generates text using benchmark prompts and parameters, and computes quantitative quality metrics.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
from pathlib import Path
import torch

from inference.generate import generate_text
from model import DecoderOnlyTransformer, ModelConfig
from tokenizer import NexaraTokenizer
from training.checkpointing import load_checkpoint
from training.config import load_config
from training.train import resolve_device

BENCHMARK_PROMPTS = [
    "Once upon a time",
    "The little dog was",
    "Lily wanted to",
    "The sun was shining",
    "Tom and his sister",
    "There was a small bird",
    "Ben went to school",
    "The rabbit saw a flower",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run generation benchmarks on Nexara.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file.")
    parser.add_argument(
        "--config", default="", help="Path to training config if checkpoint lacks it."
    )
    parser.add_argument("--output", help="Path to save benchmark JSON output.")
    parser.add_argument("--device", default="auto", help="Device override.")

    # Generation parameter overrides
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def compute_statistics(texts: list[str], tokenized_sequences: list[list[int]]) -> dict[str, any]:
    """Compute generation quality statistics over generated text and token sequences."""
    total_words = 0
    total_sentences = 0
    sentence_lengths = []

    punctuation_counts = collections.defaultdict(int)
    punctuation_targets = {".", ",", "!", "?"}

    total_bigrams = 0
    repeated_bigrams = 0
    total_trigrams = 0
    repeated_trigrams = 0

    token_counts = collections.defaultdict(int)
    total_tokens = 0

    coherence_scores = []

    for text, tokens in zip(texts, tokenized_sequences):
        # 1. Token entropy
        for tok in tokens:
            token_counts[tok] += 1
            total_tokens += 1

        # Clean words
        words = [w.lower() for w in re.findall(r"\b\w+\b", text)]
        total_words += len(words)

        # 2. Word n-grams repetition
        if len(words) >= 2:
            bigrams = list(zip(words[:-1], words[1:]))
            total_bigrams += len(bigrams)
            unique_bigrams = len(set(bigrams))
            repeated_bigrams += len(bigrams) - unique_bigrams

        if len(words) >= 3:
            trigrams = list(zip(words[:-2], words[1:-1], words[2:]))
            total_trigrams += len(trigrams)
            unique_trigrams = len(set(trigrams))
            repeated_trigrams += len(trigrams) - unique_trigrams

        # 3. Sentences and sentence lengths
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        total_sentences += len(sentences)
        for s in sentences:
            s_words = re.findall(r"\b\w+\b", s)
            sentence_lengths.append(len(s_words))

        # 4. Punctuation usage
        for char in text:
            if char in punctuation_targets:
                punctuation_counts[char] += 1

        # 5. Heuristic coherence score
        # Start capitalization check
        cap_score = 0.0
        if sentences:
            cap_ok = sum(1 for s in sentences if s[0].isupper())
            cap_score = cap_ok / len(sentences)

        # Ending punctuation check
        ends_punc = 1.0 if text and text[-1] in punctuation_targets else 0.0

        # Word frequency sanity (avoiding infinite loops)
        word_freq_ok = 1.0
        if words:
            word_counts = collections.Counter(words)
            max_word_ratio = word_counts.most_common(1)[0][1] / len(words)
            if max_word_ratio > 0.2:  # If any word takes up >20% of text
                word_freq_ok = max(0.0, 1.0 - (max_word_ratio - 0.2) * 2)

        coherence = 0.3 * cap_score + 0.3 * ends_punc + 0.4 * word_freq_ok
        coherence_scores.append(coherence)

    # Summarize stats
    avg_sentence_len = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0.0
    bigram_rep_rate = repeated_bigrams / total_bigrams if total_bigrams > 0 else 0.0
    trigram_rep_rate = repeated_trigrams / total_trigrams if total_trigrams > 0 else 0.0

    # Calculate token entropy
    entropy = 0.0
    if total_tokens > 0:
        for count in token_counts.values():
            p = count / total_tokens
            entropy -= p * math.log2(p)

    avg_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0

    punc_density = {}
    for k, v in punctuation_counts.items():
        punc_density[k] = (v / total_words * 100) if total_words > 0 else 0.0

    return {
        "bigram_repetition_rate": bigram_rep_rate,
        "trigram_repetition_rate": trigram_rep_rate,
        "average_sentence_length_words": avg_sentence_len,
        "token_entropy": entropy,
        "heuristic_coherence_score": avg_coherence,
        "punctuation_counts": dict(punctuation_counts),
        "punctuation_density_per_100_words": punc_density,
        "total_words_generated": total_words,
        "total_tokens_generated": total_tokens,
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    config = checkpoint.get("config") or load_config(args.config)

    tokenizer = NexaraTokenizer(config["tokenizer"]["path"])
    model = DecoderOnlyTransformer(ModelConfig.from_mapping(config["model"])).to(device)
    load_checkpoint(ckpt_path, model, map_location=device)
    model.eval()

    print(f"Loaded model and tokenizer. Running benchmarks on {len(BENCHMARK_PROMPTS)} prompts...")

    results = {}
    texts = []
    tokenized_sequences = []

    for prompt in BENCHMARK_PROMPTS:
        print(f"Generating for prompt: '{prompt}'...")
        text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        texts.append(text)

        # Keep track of tokens
        tokens = tokenizer.encode(text, add_bos=True, add_eos=True)
        tokenized_sequences.append(tokens)

        results[prompt] = {
            "output": text,
            "length_tokens": len(tokens),
            "length_words": len(re.findall(r"\b\w+\b", text)),
        }

    stats = compute_statistics(texts, tokenized_sequences)

    output_data = {
        "checkpoint": str(ckpt_path),
        "parameters": {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "max_new_tokens": args.max_new_tokens,
        },
        "statistics": stats,
        "generations": results,
    }

    # Resolve output path
    out_path = Path(args.output) if args.output else ckpt_path.parent / "benchmark_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nBenchmark completed successfully!")
    print(f"Saved benchmark results to {out_path}")
    print(f"Average Sentence Length: {stats['average_sentence_length_words']:.2f} words")
    print(f"Bigram Repetition Rate: {stats['bigram_repetition_rate']:.2%}")
    print(f"Token Entropy: {stats['token_entropy']:.4f}")
    print(f"Heuristic Coherence Score: {stats['heuristic_coherence_score']:.2%}")


if __name__ == "__main__":
    main()
