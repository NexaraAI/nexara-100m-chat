"""Generate lightweight markdown reports for repository automation."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import re

BADGE_START = "<!-- badges:start -->"
BADGE_END = "<!-- badges:end -->"


def main() -> None:
    update_experiment_summary()
    update_readme_badges()


def update_experiment_summary() -> Path:
    experiments_path = Path("EXPERIMENTS.md")
    if experiments_path.exists():
        experiments = experiments_path.read_text(encoding="utf-8")
        headings = re.findall(r"^## (.+)$", experiments, flags=re.MULTILINE)
    else:
        headings = []
    output = Path("docs/EXPERIMENT_SUMMARY.md")
    output.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Experiment Summary",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Entries",
        "",
    ]
    if headings:
        lines.extend(f"- {heading}" for heading in headings)
    else:
        lines.append("- No experiments documented yet.")
    lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def update_readme_badges() -> None:
    repository = os.environ.get("GITHUB_REPOSITORY", "")
    if not repository:
        return

    readme = Path("README.md")
    text = readme.read_text(encoding="utf-8")
    badge_block = "\n".join(
        [
            BADGE_START,
            f"[![CI](https://github.com/{repository}/actions/workflows/ci.yml/badge.svg)]"
            f"(https://github.com/{repository}/actions/workflows/ci.yml)",
            f"[![Docs](https://github.com/{repository}/actions/workflows/docs.yml/badge.svg)]"
            f"(https://github.com/{repository}/actions/workflows/docs.yml)",
            f"[![Nightly](https://github.com/{repository}/actions/workflows/nightly.yml/badge.svg)]"
            f"(https://github.com/{repository}/actions/workflows/nightly.yml)",
            BADGE_END,
        ]
    )

    if BADGE_START in text and BADGE_END in text:
        pattern = re.compile(f"{re.escape(BADGE_START)}.*?{re.escape(BADGE_END)}", re.S)
        updated = pattern.sub(badge_block, text)
    else:
        updated = text.replace("# Nexara\n", f"# Nexara\n\n{badge_block}\n", 1)
    readme.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    main()
