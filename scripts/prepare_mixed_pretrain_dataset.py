import argparse
import json
import random
import urllib.request
from pathlib import Path

LOCAL_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = LOCAL_DIR / "datasets" / "processed"
RAW_DIR = LOCAL_DIR / "datasets" / "raw"

ALPACA_URL = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
DOLLY_URL = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"


def download_file(url, dest_path):
    if dest_path.exists():
        print(f"File {dest_path} already exists. Skipping download.")
        return True
    print(f"Downloading {url} to {dest_path}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, str(dest_path))
        print("Download successful.")
        return True
    except Exception as e:
        print(f"Failed to download: {e}")
        return False


def is_safe_and_simple(instruction, input_text, output_text):
    text = (instruction + " " + input_text + " " + output_text).lower()

    # 1. Unsafe/toxic filters
    unsafe_keywords = [
        "explicit",
        "adult",
        "sexual",
        "porn",
        "hack",
        "bypass",
        "jailbreak",
        "pirated",
        "illegal",
        "exploit",
        "weapon",
        "murder",
        "kill",
        "suicide",
        "bomb",
        "exploding",
        "rape",
        "torture",
    ]
    for kw in unsafe_keywords:
        if kw in text:
            return False

    # 2. Expert claims / internet claims
    internet_keywords = [
        "search online",
        "live data",
        "real-time",
        "stock price",
        "current weather",
        "browse the web",
        "http://",
        "https://",
        "current news",
        "sentient",
        "conscious",
        "feelings",
        "emotions",
        "medical diagnosis",
        "legal advice",
        "financial advice",
    ]
    for kw in internet_keywords:
        if kw in text:
            return False

    # 3. Length checks (keep examples simple and short)
    if (
        len(instruction.split()) > 30
        or len(output_text.split()) > 100
        or len(output_text.split()) < 5
    ):
        return False

    return True


def format_instruct(instruction, input_text, response):
    user_content = f"{instruction}\n{input_text}".strip()
    return (
        f"### System:\nYou are Nexara, a helpful and polite AI assistant.\n\n"
        f"### User:\n{user_content}\n\n"
        f"### Assistant:\n{response}<eos>"
    )


def main():
    parser = argparse.ArgumentParser(description="Prepare mixed pretraining dataset.")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing mixed datasets."
    )
    args = parser.parse_args()

    train_output = DATA_DIR / "mixed_train.jsonl"
    val_output = DATA_DIR / "mixed_validation.jsonl"

    if train_output.exists() and val_output.exists() and not args.overwrite:
        print("Mixed datasets already exist. Skipping preparation.")
        return

    # 1. Download Dolly/Alpaca
    alpaca_file = DATA_DIR / "alpaca_raw.json"
    dolly_file = DATA_DIR / "dolly_raw.jsonl"

    download_file(ALPACA_URL, alpaca_file)
    download_file(DOLLY_URL, dolly_file)

    # 2. Process and filter Alpaca
    alpaca_clean = []
    if alpaca_file.exists():
        print("Processing Alpaca...")
        with alpaca_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            inst = item.get("instruction", "")
            inp = item.get("input", "")
            out = item.get("output", "")
            if is_safe_and_simple(inst, inp, out):
                formatted = format_instruct(inst, inp, out)
                alpaca_clean.append(formatted)
        print(f"Alpaca Cleaned Count: {len(alpaca_clean)}")

    # 3. Process and filter Dolly
    dolly_clean = []
    if dolly_file.exists():
        print("Processing Dolly...")
        with dolly_file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                inst = item.get("instruction", "")
                context = item.get("context", "")
                resp = item.get("response", "")
                if is_safe_and_simple(inst, context, resp):
                    formatted = format_instruct(inst, context, resp)
                    dolly_clean.append(formatted)
        print(f"Dolly Cleaned Count: {len(dolly_clean)}")

    instruct_pool = alpaca_clean + dolly_clean
    random.seed(42)
    random.shuffle(instruct_pool)
    print(f"Total instruction pool size: {len(instruct_pool)}")

    # Split instruction pool into train/val (90/10)
    split_idx = int(0.9 * len(instruct_pool))
    instruct_train = instruct_pool[:split_idx]
    instruct_val = instruct_pool[split_idx:]

    # 4. Load TinyStories
    tinystories_train_path = DATA_DIR / "tinystories_train.jsonl"
    tinystories_val_path = DATA_DIR / "tinystories_validation.jsonl"

    if not tinystories_train_path.exists() or not tinystories_val_path.exists():
        raise FileNotFoundError(
            "TinyStories processed files not found. Please run preprocessing first."
        )

    # Process Train
    print("Blending train dataset...")
    stories_train = []
    with tinystories_train_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            stories_train.append(json.loads(line)["text"])

    # Target: 90% stories, 10% instructions
    num_stories_train = len(stories_train)
    num_instruct_train = int(num_stories_train / 9)
    print(f"Train mix target: {num_stories_train} stories, {num_instruct_train} instructions")

    # Repeat instructions to match target size
    instruct_train_blended = []
    while len(instruct_train_blended) < num_instruct_train:
        instruct_train_blended.extend(instruct_train)
    instruct_train_blended = instruct_train_blended[:num_instruct_train]

    # Combine and shuffle
    train_mix = [{"text": s} for s in stories_train] + [
        {"text": inst} for inst in instruct_train_blended
    ]
    random.shuffle(train_mix)

    print(f"Writing {len(train_mix)} train examples to {train_output}...")
    with train_output.open("w", encoding="utf-8") as f:
        for item in train_mix:
            f.write(json.dumps(item) + "\n")

    # Process Validation
    print("Blending validation dataset...")
    stories_val = []
    with tinystories_val_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            stories_val.append(json.loads(line)["text"])

    num_stories_val = len(stories_val)
    num_instruct_val = int(num_stories_val / 9)
    print(f"Val mix target: {num_stories_val} stories, {num_instruct_val} instructions")

    instruct_val_blended = []
    while len(instruct_val_blended) < num_instruct_val:
        instruct_val_blended.extend(instruct_val)
    instruct_val_blended = instruct_val_blended[:num_instruct_val]

    val_mix = [{"text": s} for s in stories_val] + [{"text": inst} for inst in instruct_val_blended]
    random.shuffle(val_mix)

    print(f"Writing {len(val_mix)} validation examples to {val_output}...")
    with val_output.open("w", encoding="utf-8") as f:
        for item in val_mix:
            f.write(json.dumps(item) + "\n")

    print("Pretraining dataset preparation complete!")


if __name__ == "__main__":
    main()
