# TinyStories Notes

TinyStories is the initial training target for Nexara Stage 1.

The original paper is "TinyStories: How Small Can Language Models Be and Still
Speak Coherent English?" by Ronen Eldan and Yuanzhi Li:

- <https://arxiv.org/abs/2305.07759>
- <https://huggingface.co/datasets/roneneldan/TinyStories>

Key takeaways for Nexara:

- TinyStories is a synthetic English short-story dataset designed around simple
  vocabulary and simple plots.
- The Hugging Face dataset card currently lists the default split as about
  2.12M train rows and 22k validation rows.
- The paper reports that models below 10M parameters can produce coherent short
  stories when trained on this narrower distribution.
- The original work uses small context windows and reports useful comparisons
  across width, depth, and model size.
- Nexara should use the dataset, not the published pretrained TinyStories model
  weights or tokenizer weights.

## Local Data Expectation

The download script supports these Hugging Face raw text variants:

- `original`: `TinyStories-train.txt`, `TinyStories-valid.txt`
- `gpt4`: `TinyStoriesV2-GPT4-train.txt`, `TinyStoriesV2-GPT4-valid.txt`

The Stage 1 config currently uses the original variant. Raw files should be:

```text
datasets/raw/TinyStories-train.txt
datasets/raw/TinyStories-valid.txt
```

Preprocessing writes JSONL files:

```text
datasets/processed/tinystories_train.jsonl
datasets/processed/tinystories_validation.jsonl
```

Each row contains a `text` field.

## Stage 1 Training Notes

- Keep the first context length at 256 tokens to control memory usage.
- Train an 8192-token BPE tokenizer from the local train split.
- Build uint32 token caches after tokenizer training so the training dataloader
  can use memory-mapped token blocks instead of loading the full corpus into
  Python memory.
- Keep `memory_map = true` for normal Stage 1 training. Enable `streaming = true`
  only when ordered iterable reads are preferred over shuffled random access.
- Inspect tokenizer reports, sample encodings, and token-cache statistics before
  launching a long run.
- Dataset scripts skip completed outputs by default and use temporary files so
  interrupted runs can be resumed without trusting partially written final
  artifacts.
- Start with short smoke runs before full training.
- Save validation loss, perplexity, and sample generations after each meaningful
  run.
