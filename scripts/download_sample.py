"""Download a small real sample from the official MIRACL corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


API_URL = "https://datasets-server.huggingface.co/rows"
MAX_ROWS_PER_REQUEST = 100


def download(languages: list[str], per_language: int, output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for language in languages:
            for offset in range(0, per_language, MAX_ROWS_PER_REQUEST):
                length = min(MAX_ROWS_PER_REQUEST, per_language - offset)
                query = urlencode({
                    "dataset": "miracl/miracl-corpus",
                    "config": language,
                    "split": "train",
                    "offset": offset,
                    "length": length,
                })
                with urlopen(f"{API_URL}?{query}", timeout=60) as response:
                    payload = json.load(response)
                rows = payload.get("rows", [])
                for item in rows:
                    row = item["row"]
                    handle.write(json.dumps({
                        "docid": row["docid"],
                        "title": row["title"],
                        "text": row["text"],
                        "language": language,
                    }, ensure_ascii=False) + "\n")
                    count += 1
                if len(rows) < length:
                    break
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", default=["en", "hi", "es", "ar"])
    parser.add_argument("--profile", choices=["cloud", "local"], default="cloud")
    parser.add_argument("--per-language", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    defaults = {"cloud": (12, Path("data/sample_corpus.jsonl")), "local": (1000, Path("data/local_corpus.jsonl"))}
    default_per_language, default_output = defaults[args.profile]
    per_language = args.per_language or default_per_language
    output = args.output or default_output
    if per_language < 1:
        parser.error("--per-language must be at least 1")
    count = download(args.languages, per_language, output)
    print(f"Wrote {count} MIRACL passages to {output}")


if __name__ == "__main__":
    main()
