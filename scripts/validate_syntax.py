"""Parse project Python files without importing project dependencies."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

DEFAULT_ROOTS = (
    "datasets",
    "tokenizer",
    "model",
    "training",
    "evaluation",
    "inference",
    "scripts",
    "tests",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Python syntax with ast.parse.")
    parser.add_argument("roots", nargs="*", default=list(DEFAULT_ROOTS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = python_files(args.roots)
    for path in files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    print(f"syntax_ok files={len(files)}")


def python_files(roots: list[str]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        path = Path(root)
        if path.is_file() and path.suffix == ".py":
            files.append(path)
        elif path.is_dir():
            files.extend(path.rglob("*.py"))
    return sorted(path for path in files if "__pycache__" not in path.parts)


if __name__ == "__main__":
    main()
