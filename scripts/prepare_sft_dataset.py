import argparse
import json
import random
import urllib.request
import hashlib
from pathlib import Path

# Paths
LOCAL_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = LOCAL_DIR / "datasets" / "processed"
IDENTITY_FILE = DATA_DIR / "nexara_identity.jsonl"
VALIDATION_STORIES_FILE = DATA_DIR / "tinystories_validation.jsonl"

TRAIN_OUTPUT_500 = DATA_DIR / "sft_train_500.jsonl"
TRAIN_OUTPUT_3000 = DATA_DIR / "sft_train_3000.jsonl"
TRAIN_OUTPUT_10000 = DATA_DIR / "sft_train_10000.jsonl"
TRAIN_OUTPUT_30000 = DATA_DIR / "sft_train_30000.jsonl"
TRAIN_OUTPUT_50000 = DATA_DIR / "sft_train_50000.jsonl"
TRAIN_OUTPUT_100000 = DATA_DIR / "sft_train_100000.jsonl"
TRAIN_OUTPUT_170000 = DATA_DIR / "sft_train_170000.jsonl"
VAL_OUTPUT = DATA_DIR / "sft_val.jsonl"
RAW_POOL_OUTPUT = DATA_DIR / "sft_raw_pool_100k.jsonl"

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


def calculate_quality_score(instruction, input_text, output_text):
    """Heuristic quality scoring from 0.0 to 1.0."""
    score = 1.0

    # Penalize too short/long responses
    word_count = len(output_text.split())
    if word_count < 10:
        score -= 0.3
    elif word_count > 120:
        score -= 0.2

    # Penalize capital letter ratios if broken
    if word_count > 0:
        caps = sum(1 for c in output_text if c.isupper())
        cap_ratio = caps / len(output_text) if len(output_text) > 0 else 0
        if cap_ratio > 0.25:  # Too many capital letters
            score -= 0.3

    # Penalize repetitive text structures
    lines = [l.strip().lower() for l in output_text.split("\n") if l.strip()]
    if len(lines) > 0:
        unique_lines = set(lines)
        if len(unique_lines) / len(lines) < 0.7:  # High line repetition
            score -= 0.4

    return max(0.0, min(1.0, score))


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

    # 2. Expert claims / internet claims / capability mismatch
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
        "powerful AI",
        "medical diagnosis",
        "legal advice",
        "financial advice",
    ]
    for kw in internet_keywords:
        if kw in text:
            return False

    # 3. Relaxed math/code filters: Allow basic code and math, but block extremely advanced topics
    advanced_math_code = [
        "integrate",
        "derivative",
        "vector space",
        "theorem",
        "lemma",
        "calculus",
        "differential equation",
    ]
    for kw in advanced_math_code:
        if kw in text:
            return False

    # 4. Relaxed length checks (block_size is 512, which fits ~400 words easily)
    if (
        len(instruction.split()) > 60
        or len(output_text.split()) > 250
        or len(output_text.split()) < 5
    ):
        return False

    return True


def deduplicate_dataset(dataset):
    """Filter out duplicate entries based on md5 hash of instruction + input."""
    seen_hashes = set()
    deduped = []

    for item in dataset:
        # Construct message content hash
        content = ""
        for m in item["messages"]:
            if m["role"] == "user":
                content += m["content"]
        h = hashlib.md5(content.encode("utf-8")).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            deduped.append(item)

    print(f"Deduplication: Reduced dataset from {len(dataset)} to {len(deduped)} entries.")
    return deduped


def balance_sources(dataset, max_per_source=1500):
    """Ensure no source dominates the final training set."""
    source_counts = {}
    balanced = []

    for item in dataset:
        src = item["source"]
        source_counts[src] = source_counts.get(src, 0) + 1
        if source_counts[src] <= max_per_source:
            balanced.append(item)

    print("Source distribution after balancing:")
    for k, v in source_counts.items():
        print(f"  {k}: {min(v, max_per_source)} (original: {v})")

    return balanced


def validate_format(dataset):
    """Schema validation check."""
    valid = []
    for item in dataset:
        if "messages" not in item or "system" not in item:
            continue
        msg_valid = True
        for m in item["messages"]:
            if "role" not in m or "content" not in m:
                msg_valid = False
                break
        if msg_valid:
            valid.append(item)
    return valid


def generate_synthetic_conversations(count):
    print(f"Generating {count} synthetic multi-turn conversations...")
    random.seed(123)

    greetings = ["Hello!", "Hi there!", "Good day!", "Hey!", "Greetings!"]
    user_questions = [
        (
            "What is your favorite animal?",
            "I like dogs and rabbits! They are very friendly and soft.",
        ),
        (
            "Do you like reading books?",
            "I love stories! Reading stories helps me learn about interesting adventures.",
        ),
        (
            "What color do you like?",
            "I think green and blue are very nice colors, like the grass and the sky.",
        ),
        ("Can you help me write?", "Yes! I can help you write short, fun stories or simple poems."),
        (
            "Do you eat food?",
            "No, I am an AI assistant, so I don't eat. But I like stories about tasty food!",
        ),
        (
            "What is the sun?",
            "The sun is a big, hot star in the sky that gives us light and warmth.",
        ),
        (
            "Why is the grass green?",
            "Grass is green because of a special substance called chlorophyll, which helps it capture sunlight.",
        ),
        (
            "Tell me about trees.",
            "Trees are tall plants with trunks and leaves. They provide shade and fresh air.",
        ),
    ]

    follow_ups = [
        ("Why do you like them?", "Because they are playful and make people happy!"),
        (
            "Can you tell me a story about them?",
            "Sure! Once upon a time, there was a little animal who loved to explore...",
        ),
        ("That sounds nice.", "Thank you! I like keeping things simple and pleasant."),
        ("What else do you like?", "I like listening to stories and helping you learn new things."),
        ("Is it hot?", "Yes, the sun is extremely hot, but it is very far away from Earth."),
    ]

    dialogues = []
    for i in range(count):
        user_greet = random.choice(greetings)
        asst_greet = f"Hello! I am Nexara, a small AI assistant. How can I help you today?"

        q1, a1 = random.choice(user_questions)
        q2, a2 = random.choice(follow_ups)

        item = {
            "id": f"nexara-synthetic-{i:04d}",
            "source": "synthetic_conversations",
            "type": "multi_turn",
            "system": "You are Nexara, a helpful and polite AI assistant.",
            "messages": [
                {"role": "user", "content": user_greet},
                {"role": "assistant", "content": asst_greet},
                {"role": "user", "content": q1},
                {"role": "assistant", "content": a1},
                {"role": "user", "content": q2},
                {"role": "assistant", "content": a2},
            ],
            "quality_score": 0.95,
            "license": "CC0",
            "safety_flags": [],
        }
        dialogues.append(item)

    return dialogues


def generate_math_sft(count):
    print(f"Generating {count} math SFT examples...")
    random.seed(789)
    examples = []

    algebra_templates = [
        ("Solve for x: x + {a} = {b}", "{b} - {a}", "x = {ans}"),
        ("Solve for x: x - {a} = {b}", "{b} + {a}", "x = {ans}"),
        ("Solve for y: {a}y = {b}", "{b} // {a}", "y = {ans}"),
    ]

    trig_values = [
        ("sin(0)", "0"),
        ("sin(90)", "1"),
        ("cos(0)", "1"),
        ("cos(90)", "0"),
        ("tan(0)", "0"),
        ("tan(45)", "1"),
    ]

    for i in range(count):
        category = random.choice(["add", "sub", "mul", "div", "algebra", "trig"])

        if category == "add":
            a = random.randint(0, 99)
            b = random.randint(0, 99)
            q = f"What is {a} + {b}?"
            ans = f"{a} + {b} = {a + b}."
        elif category == "sub":
            a = random.randint(10, 99)
            b = random.randint(0, a)
            q = f"What is {a} - {b}?"
            ans = f"{a} - {b} = {a - b}."
        elif category == "mul":
            a = random.randint(0, 12)
            b = random.randint(0, 12)
            q = f"What is {a} * {b}?"
            ans = f"{a} * {b} = {a * b}."
        elif category == "div":
            b = random.randint(1, 12)
            ans_val = random.randint(0, 12)
            a = b * ans_val
            q = f"What is {a} / {b}?"
            ans = f"{a} / {b} = {ans_val}."
        elif category == "algebra":
            tmpl, math_expr, ans_tmpl = random.choice(algebra_templates)
            if "y" in tmpl:
                a = random.randint(2, 9)
                ans_val = random.randint(1, 12)
                b = a * ans_val
            else:
                a = random.randint(1, 50)
                b = random.randint(51, 100)
                ans_val = eval(math_expr.format(a=a, b=b))
            q = tmpl.format(a=a, b=b)
            ans = ans_tmpl.format(ans=ans_val)
        else:  # trig
            expr, val = random.choice(trig_values)
            q = f"What is {expr}?"
            ans = f"{expr} = {val}."

        item = {
            "id": f"nexara-math-{i:04d}",
            "source": "synthetic_math",
            "type": "single_turn",
            "system": "You are Nexara, a helpful and polite AI assistant.",
            "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": ans}],
            "quality_score": 1.0,
            "license": "MIT",
            "safety_flags": [],
        }
        examples.append(item)
    return examples


def generate_structured_tasks_sft(count):
    print(f"Generating {count} structured task SFT examples...")
    random.seed(543)
    examples = []

    markdown_templates = [
        (
            "Create a markdown table of three fruits and their colors.",
            "| Fruit | Color |\n| :--- | :--- |\n| Apple | Red |\n| Banana | Yellow |\n| Grape | Purple |",
        ),
        ("Format this list as a markdown list: cat, dog, rabbit", "* Cat\n* Dog\n* Rabbit"),
        ("Write a markdown header with a subtitle.", "# Main Title\n## Subtitle"),
        (
            "Convert this to a markdown table: Name: Alice, Score: 95. Name: Bob, Score: 88.",
            "| Name | Score |\n| :--- | :---: |\n| Alice | 95 |\n| Bob | 88 |",
        ),
    ]

    json_templates = [
        (
            "Fix the syntax of this JSON block: {name: 'Nexara', age: 1}",
            '{\n  "name": "Nexara",\n  "age": 1\n}',
        ),
        ('Format this JSON: {"status":"ok","code":200}', '{\n  "status": "ok",\n  "code": 200\n}'),
        (
            'Check if this is valid JSON: {"a": 1',
            "No, this is invalid JSON because it is missing a closing curly brace `}`.",
        ),
    ]

    grammar_templates = [
        (
            "Correct the grammar: Me goes to school yesterday.",
            "Corrected: I went to school yesterday.",
        ),
        (
            "Rephrase this sentence to be polite: Give me water.",
            "Rephrased: Could you please give me some water?",
        ),
        ("Fix spelling mistakes: The rabit jumpd high.", "Corrected: The rabbit jumped high."),
    ]

    code_templates = [
        ("Write a Python script to print hello world.", '```python\nprint("Hello, World!")\n```'),
        (
            "Write a simple Python function to add two numbers.",
            "```python\ndef add(a, b):\n    return a + b\n```",
        ),
        (
            "Create a Python loop to print numbers from 1 to 5.",
            "```python\nfor i in range(1, 6):\n    print(i)\n```",
        ),
    ]

    categories = [markdown_templates, json_templates, grammar_templates, code_templates]

    for i in range(count):
        cat = random.choice(categories)
        q, ans = random.choice(cat)

        item = {
            "id": f"nexara-task-{i:04d}",
            "source": "synthetic_tasks",
            "type": "single_turn",
            "system": "You are Nexara, a helpful and polite AI assistant.",
            "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": ans}],
            "quality_score": 1.0,
            "license": "MIT",
            "safety_flags": [],
        }
        examples.append(item)
    return examples


def extract_tinystories_sft(count):
    print(f"Extracting {count} SFT story-retention examples from validation set...")
    if not VALIDATION_STORIES_FILE.exists():
        print(f"Validation file {VALIDATION_STORIES_FILE} not found. Skipping story retention.")
        return []

    stories = []
    with VALIDATION_STORIES_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            stories.append(json.loads(line)["text"])

    random.seed(456)
    selected_stories = random.sample(stories, min(count, len(stories)))

    prompts = [
        "Tell me a short children's story.",
        "Write a simple story for kids.",
        "Tell me a story.",
        "Can you share a short story?",
        "Please tell me a fun story.",
    ]

    examples = []
    for i, story in enumerate(selected_stories):
        story_words = story.split()
        trimmed_story = " ".join(story_words[:80]) + "."

        prompt = random.choice(prompts)
        item = {
            "id": f"nexara-tinystories-{i:04d}",
            "source": "tinystories_validation_sft",
            "type": "single_turn",
            "system": "You are Nexara, a helpful and polite AI assistant.",
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": trimmed_story},
            ],
            "quality_score": 1.0,
            "license": "Apache-2.0",
            "safety_flags": [],
        }
        examples.append(item)

    return examples


def load_local_identity():
    print("Loading local identity examples...")
    if not IDENTITY_FILE.exists():
        print(f"Identity file {IDENTITY_FILE} not found!")
        return []
    examples = []
    with IDENTITY_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            examples.append(json.loads(line))
    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Curate and filter SFT candidate pools and SFT datasets."
    )
    parser.add_argument(
        "--build-raw-pool", action="store_true", help="Build the future 100k raw candidate pool."
    )
    parser.add_argument("--dedup", action="store_true", default=True, help="Enable deduplication.")
    parser.add_argument(
        "--balance", action="store_true", default=True, help="Enable source balancing."
    )
    args = parser.parse_args()

    # Setup download paths
    alpaca_file = DATA_DIR / "alpaca_raw.json"
    dolly_file = DATA_DIR / "dolly_raw.jsonl"

    # Download datasets
    download_file(ALPACA_URL, alpaca_file)
    download_file(DOLLY_URL, dolly_file)

    # Process Alpaca
    alpaca_cleaned = []
    if alpaca_file.exists():
        print("Processing Alpaca...")
        with alpaca_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            inst = item.get("instruction", "")
            inp = item.get("input", "")
            out = item.get("output", "")
            if is_safe_and_simple(inst, inp, out):
                full_user = f"{inst}\n{inp}".strip()
                score = calculate_quality_score(inst, inp, out)
                alpaca_cleaned.append(
                    {
                        "source": "stanford_alpaca_filtered",
                        "type": "single_turn",
                        "system": "You are Nexara, a helpful and polite AI assistant.",
                        "messages": [
                            {"role": "user", "content": full_user},
                            {"role": "assistant", "content": out},
                        ],
                        "quality_score": score,
                        "license": "CC-BY-4.0",
                        "safety_flags": [],
                    }
                )
        print(f"Alpaca Cleaned Count: {len(alpaca_cleaned)}")

    # Process Dolly
    dolly_cleaned = []
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
                    full_user = f"{inst}\n{context}".strip()
                    score = calculate_quality_score(inst, context, resp)
                    dolly_cleaned.append(
                        {
                            "source": "dolly_15k_filtered",
                            "type": "single_turn",
                            "system": "You are Nexara, a helpful and polite AI assistant.",
                            "messages": [
                                {"role": "user", "content": full_user},
                                {"role": "assistant", "content": resp},
                            ],
                            "quality_score": score,
                            "license": "CC-BY-SA-3.0",
                            "safety_flags": [],
                        }
                    )
        print(f"Dolly Cleaned Count: {len(dolly_cleaned)}")

    # Generate larger pool or standard training sets
    if args.build_raw_pool:
        print("=== Building future 100k raw SFT candidate pool ===")
        # Mix in huge synthetics and all clean ones
        huge_synth = generate_synthetic_conversations(5000)
        raw_pool = alpaca_cleaned + dolly_cleaned + huge_synth

        # Apply standard future filters optionally
        if args.dedup:
            raw_pool = deduplicate_dataset(raw_pool)
        if args.balance:
            raw_pool = balance_sources(raw_pool, max_per_source=40000)

        raw_pool = validate_format(raw_pool)

        with RAW_POOL_OUTPUT.open("w", encoding="utf-8") as f:
            for idx, item in enumerate(raw_pool):
                item["id"] = f"nexara-raw-pool-{idx:06d}"
                f.write(json.dumps(item) + "\n")
        print(f"Saved {len(raw_pool)} raw SFT candidates to {RAW_POOL_OUTPUT}")
        return

    # Standard Curriculum SFT Blending
    random.seed(999)
    random.shuffle(alpaca_cleaned)
    random.shuffle(dolly_cleaned)

    synthetic_pool = generate_synthetic_conversations(60000)
    identity_pool = load_local_identity()
    random.shuffle(identity_pool)

    tinystories_pool = extract_tinystories_sft(10000)
    math_pool = generate_math_sft(35000)
    task_pool = generate_structured_tasks_sft(35000)

    # Build standard splits
    def distribute(pool, train_count, val_count):
        train_items = pool[:train_count]
        val_items = pool[train_count : train_count + val_count]
        return train_items, val_items

    t_alpaca, v_alpaca = distribute(alpaca_cleaned, 42000, 800)
    t_dolly, v_dolly = distribute(dolly_cleaned, 11500, 500)
    t_synth, v_synth = distribute(synthetic_pool, 50000, 2000)

    # Upsample identity examples (repeat 15 times for train)
    t_ident = (identity_pool[:180]) * 15
    random.shuffle(t_ident)
    v_ident = identity_pool[180:200]

    t_story, v_story = distribute(tinystories_pool, 8000, 800)
    t_math, v_math = distribute(math_pool, 30000, 2000)
    t_task, v_task = distribute(task_pool, 30000, 2000)

    # 170k split
    train_170000_set = t_alpaca + t_dolly + t_synth + t_ident + t_story + t_math + t_task
    val_set = v_alpaca + v_dolly + v_synth + v_ident + v_story + v_math + v_task

    # Downsample smaller splits from the 170,000 pool
    train_100000_set = train_170000_set[:100000]
    train_50000_set = train_100000_set[:50000]

    # For the 30,000 split, we do a dedicated high-quality, balanced blend to ensure high identity ratio
    t_alpaca_30k = alpaca_cleaned[:10000]
    t_dolly_30k = dolly_cleaned[:4000]
    t_synth_30k = synthetic_pool[:6000]
    t_story_30k = tinystories_pool[:1000]
    t_math_30k = math_pool[:4000]
    t_task_30k = task_pool[:4000]
    train_30000_set = (
        t_alpaca_30k + t_dolly_30k + t_synth_30k + t_ident + t_story_30k + t_math_30k + t_task_30k
    )

    train_10000_set = train_30000_set[:10000]
    train_3000_set = train_10000_set[:3000]

    # Shuffle
    random.shuffle(train_170000_set)
    random.shuffle(train_100000_set)
    random.shuffle(train_50000_set)
    random.shuffle(train_30000_set)
    random.shuffle(train_10000_set)
    random.shuffle(train_3000_set)
    random.shuffle(val_set)

    # Trim to exact size
    train_170000_set = train_170000_set[:170000]
    train_100000_set = train_100000_set[:100000]
    train_50000_set = train_50000_set[:50000]
    train_30000_set = train_30000_set[:30000]
    train_10000_set = train_10000_set[:10000]
    train_3000_set = train_3000_set[:3000]
    val_set = val_set[:3000]  # Set validation set to exactly 3,000 for SFT

    # Create 500-example curriculum subset (select highest quality score from the 3000-set)
    train_3000_set_sorted = sorted(train_3000_set, key=lambda x: x["quality_score"], reverse=True)
    train_500_set = train_3000_set_sorted[:500]
    random.shuffle(train_500_set)

    # Write train splits
    with TRAIN_OUTPUT_170000.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(train_170000_set):
            item["id"] = f"nexara-sft-train-{idx:06d}"
            f.write(json.dumps(item) + "\n")

    with TRAIN_OUTPUT_100000.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(train_100000_set):
            item["id"] = f"nexara-sft-train-{idx:06d}"
            f.write(json.dumps(item) + "\n")

    with TRAIN_OUTPUT_50000.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(train_50000_set):
            item["id"] = f"nexara-sft-train-{idx:05d}"
            f.write(json.dumps(item) + "\n")

    with TRAIN_OUTPUT_30000.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(train_30000_set):
            item["id"] = f"nexara-sft-train-{idx:05d}"
            f.write(json.dumps(item) + "\n")

    with TRAIN_OUTPUT_10000.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(train_10000_set):
            item["id"] = f"nexara-sft-train-{idx:05d}"
            f.write(json.dumps(item) + "\n")

    with TRAIN_OUTPUT_3000.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(train_3000_set):
            item["id"] = f"nexara-sft-train-{idx:04d}"
            f.write(json.dumps(item) + "\n")

    with TRAIN_OUTPUT_500.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(train_500_set):
            item["id"] = f"nexara-sft-train-{idx:04d}"
            f.write(json.dumps(item) + "\n")

    with VAL_OUTPUT.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(val_set):
            item["id"] = f"nexara-sft-val-{idx:04d}"
            f.write(json.dumps(item) + "\n")

    print(f"Successfully generated curriculum splits:")
    print(f"  Train Phase 2.4-A (Smoke SFT): {len(train_500_set)} examples to {TRAIN_OUTPUT_500}")
    print(f"  Train Phase 2.4-B (Full SFT): {len(train_3000_set)} examples to {TRAIN_OUTPUT_3000}")
    print(
        f"  Train Phase 2.4-C (Expanded SFT): {len(train_10000_set)} examples to {TRAIN_OUTPUT_10000}"
    )
    print(
        f"  Train Phase 2.4-D (Target SFT): {len(train_30000_set)} examples to {TRAIN_OUTPUT_30000}"
    )
    print(
        f"  Train Phase 2.4-E (Massive SFT): {len(train_50000_set)} examples to {TRAIN_OUTPUT_50000}"
    )
    print(
        f"  Train Phase 2.4-F (Gigantic SFT): {len(train_100000_set)} examples to {TRAIN_OUTPUT_100000}"
    )
    print(
        f"  Train Phase 2.4-G (Maximum SFT): {len(train_170000_set)} examples to {TRAIN_OUTPUT_170000}"
    )
    print(f"  Val Set: {len(val_set)} examples to {VAL_OUTPUT}")


if __name__ == "__main__":
    main()
