"""Download complete MIRACL corpus shards for selected languages."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def download(languages: list[str], output: Path) -> str:
    patterns = [f"miracl-corpus-v1.0-{language}/docs-*.jsonl.gz" for language in languages]
    return snapshot_download(
        repo_id="miracl/miracl-corpus",
        repo_type="dataset",
        local_dir=str(output),
        allow_patterns=patterns,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", default=["ar", "bn", "de", "en", "es", "fa", "fi", "fr", "hi", "id", "ja", "ko", "ru", "sw", "te", "th", "yo", "zh"])
    parser.add_argument("--output", type=Path, default=Path("data/miracl_full"))
    args = parser.parse_args()
    location = download(args.languages, args.output)
    print(f"Downloaded MIRACL corpus shards to {location}")


if __name__ == "__main__":
    main()
