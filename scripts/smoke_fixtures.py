"""Small TinyStories-like fixtures for runtime smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

TINY_STORIES = [
    "Once upon a time, Lily found a tiny red bird in the garden. She gave it water and watched it fly home.",
    "Tom had a blue toy truck. He shared it with Mia, and they built a small road in the sand.",
    "A little dog named Pip ran to the porch before the rain. Pip was happy to be warm and dry.",
    "Sara saw a bright kite stuck in a tree. Her dad lifted her up, and Sara helped the kite fly again.",
    "Ben wanted the biggest apple. He gave half to his sister, and the apple tasted even better.",
    "Nora made a paper boat. It floated down the stream until a frog hopped beside it.",
    "The sleepy cat heard a bell at the door. It was Sam with a bowl of milk and a gentle smile.",
    "Mia lost her yellow hat at the park. A kind boy found it and gave it back before lunch.",
    "Jake built a tall tower with blocks. When it fell, he laughed and started again.",
    "Emma drew a picture of the sun. She put it on the fridge and smiled every morning.",
    "The rabbit hid behind a bush. Leo crept close and the rabbit wiggled its nose at him.",
    "Zoe had a shiny coin. She tossed it into the fountain and made a secret wish.",
    "Dad read a story about a dragon. Max pretended to be the brave knight until bedtime.",
    "Lucy picked flowers in the meadow. She made a crown and wore it all day long.",
    "Sam found a seashell at the beach. He held it to his ear and heard the ocean inside.",
    "A tiny ladybug landed on Ava's hand. She counted its spots before it flew away.",
    "Oliver made a snowman with a carrot nose. The next day the sun melted it into a puddle.",
    "Ella sang a song to her teddy bear. The teddy seemed to smile when she finished.",
    "Two kittens played with a ball of yarn. They rolled it across the room until Mom laughed.",
    "Henry planted a seed in a pot. Every day he watered it and watched the green sprout grow.",
    "The wind blew Anna's balloon into the sky. She waved goodbye and promised to get a new one.",
    "Grandma baked cookies with chocolate chips. The whole house smelled warm and sweet.",
    "A frog jumped into the pond with a big splash. The ducks quacked and swam the other way.",
    "Rosie carried her lunch box to school. Inside she had a sandwich, an apple, and a note from Mom.",
    "The stars came out one by one. Dad pointed at the brightest star and told a story about it.",
]


def write_tiny_stories_jsonl(path: str | Path, repeat: int = 1) -> Path:
    """Write TINY_STORIES to JSONL, optionally repeating the full set."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for _ in range(repeat):
            for story in TINY_STORIES:
                handle.write(json.dumps({"text": story}, ensure_ascii=False) + "\n")
    return output


def write_tiny_stories_jsonl_subset(
    path: str | Path,
    count: int = 100,
) -> Path:
    """Write *count* TinyStories-like examples to JSONL, wrapping if needed."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index in range(count):
            story = TINY_STORIES[index % len(TINY_STORIES)]
            handle.write(json.dumps({"text": story}, ensure_ascii=False) + "\n")
    return output
