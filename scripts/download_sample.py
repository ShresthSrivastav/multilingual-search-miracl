"""Download a small real sample from the official MIRACL corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


API_URL = "https://datasets-server.huggingface.co/rows"


def download(languages: list[str], per_language: int, output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for language in languages:
            query = urlencode({
                "dataset": "miracl/miracl-corpus",
                "config": language,
                "split": "train",
                "offset": 0,
                "length": per_language,
            })
            with urlopen(f"{API_URL}?{query}", timeout=60) as response:
                payload = json.load(response)
            for item in payload.get("rows", []):
                row = item["row"]
                handle.write(json.dumps({
                    "docid": row["docid"],
                    "title": row["title"],
                    "text": row["text"],
                    "language": language,
                }, ensure_ascii=False) + "\n")
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", default=["en", "hi", "es", "ar"])
    parser.add_argument("--per-language", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("data/sample_corpus.jsonl"))
    args = parser.parse_args()
    if args.per_language < 1:
        parser.error("--per-language must be at least 1")
    count = download(args.languages, args.per_language, args.output)
    print(f"Wrote {count} MIRACL passages to {args.output}")


if __name__ == "__main__":
    main()
