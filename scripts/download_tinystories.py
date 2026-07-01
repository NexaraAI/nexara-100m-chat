"""Download TinyStories raw text files from Hugging Face."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time
import urllib.request

from datasets.tinystories import (
    TINYSTORIES_FILES,
    default_raw_path,
    tiny_stories_filename,
    tiny_stories_url,
)


@dataclass(frozen=True)
class DownloadTarget:
    variant: str
    split: str
    filename: str
    url: str
    output_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download TinyStories raw text files.")
    parser.add_argument("--variant", choices=sorted(TINYSTORIES_FILES), default="original")
    parser.add_argument(
        "--split",
        choices=["train", "validation", "all"],
        default="all",
        help="Split to download.",
    )
    parser.add_argument("--output-dir", default="datasets/raw")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = build_download_targets(args.variant, args.split, args.output_dir)
    if args.dry_run:
        for target in targets:
            print(f"{target.split}: {target.url} -> {target.output_path}")
        return

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "variant": args.variant,
        "targets": [],
    }
    for target in targets:
        output = Path(target.output_path)
        start_time = time.monotonic()
        bytes_written = download_file(
            target.url,
            output,
            overwrite=args.overwrite,
            resume=not args.no_resume,
            timeout=args.timeout,
        )
        elapsed = time.monotonic() - start_time
        checksum = compute_sha256(output)
        record = asdict(target)
        record["bytes"] = bytes_written
        record["sha256"] = checksum
        record["download_seconds"] = round(elapsed, 2)
        manifest["targets"].append(record)
        print(f"sha256({output.name}): {checksum}")

    manifest_path = Path(args.output_dir) / f"tinystories_{args.variant}_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with temporary_manifest.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary_manifest.replace(manifest_path)
    print(f"wrote manifest to {manifest_path}")


def build_download_targets(
    variant: str,
    split: str,
    output_dir: str | Path,
) -> list[DownloadTarget]:
    splits = ["train", "validation"] if split == "all" else [split]
    targets = []
    for split_name in splits:
        filename = tiny_stories_filename(variant, split_name)  # type: ignore[arg-type]
        targets.append(
            DownloadTarget(
                variant=variant,
                split=split_name,
                filename=filename,
                url=tiny_stories_url(variant, split_name),  # type: ignore[arg-type]
                output_path=str(
                    default_raw_path(output_dir, variant, split_name)  # type: ignore[arg-type]
                ),
            )
        )
    return targets


def download_file(
    url: str,
    output_path: str | Path,
    overwrite: bool = False,
    resume: bool = True,
    timeout: int = 60,
) -> int:
    output = Path(output_path)
    if output.exists() and not overwrite:
        print(f"skipping existing file: {output}")
        return output.stat().st_size

    output.parent.mkdir(parents=True, exist_ok=True)
    partial_output = output.with_suffix(output.suffix + ".part")
    if overwrite and partial_output.exists():
        partial_output.unlink()

    existing_bytes = partial_output.stat().st_size if resume and partial_output.exists() else 0
    headers = {"Range": f"bytes={existing_bytes}-"} if existing_bytes else {}
    request = urllib.request.Request(url, headers=headers)

    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = getattr(response, "status", response.getcode())
        mode = "ab" if existing_bytes and status == 206 else "wb"
        if existing_bytes and status != 206:
            existing_bytes = 0
        total = _expected_total_bytes(response, existing_bytes)
        with partial_output.open(mode) as handle:
            downloaded = existing_bytes
            chunk_size = 1024 * 1024
            last_report = time.monotonic()
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if now - last_report >= 2.0 or not chunk:
                    if total:
                        pct = downloaded / total * 100
                        print(
                            f"\r  {downloaded:,} / {total:,} bytes ({pct:.1f}%)",
                            end="",
                            flush=True,
                        )
                    else:
                        print(f"\r  {downloaded:,} bytes", end="", flush=True)
                    last_report = now
            print()  # newline after progress

    size = partial_output.stat().st_size
    if total and size != total:
        raise IOError(f"downloaded {size} bytes, expected {total} bytes for {url}")

    partial_output.replace(output)
    size = output.stat().st_size
    print(f"downloaded {size} bytes to {output}")
    return size


def _expected_total_bytes(response, existing_bytes: int) -> int:
    content_range = response.headers.get("Content-Range", "")
    match = re.match(r"bytes \d+-\d+/(\d+)$", content_range)
    if match:
        return int(match.group(1))

    content_length = int(response.headers.get("Content-Length", "0") or "0")
    if content_length and getattr(response, "status", response.getcode()) == 206:
        return existing_bytes + content_length
    return content_length


def compute_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 hex digest of a file."""
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


def verify_checksum(
    path: str | Path,
    expected_sha256: str,
) -> bool:
    """Verify file SHA-256 matches expected value."""
    actual = compute_sha256(path)
    return actual == expected_sha256


if __name__ == "__main__":
    main()
