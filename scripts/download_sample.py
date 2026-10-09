"""Download a small real sample from the official MIRACL corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen


API_URL = "https://datasets-server.huggingface.co/rows"
SIZE_URL = "https://datasets-server.huggingface.co/size"
MAX_ROWS_PER_REQUEST = 100


def get_row_count(language: str) -> int:
    query = urlencode({"dataset": "miracl/miracl-corpus", "config": language, "split": "train"})
    for attempt in range(4):
        try:
            with urlopen(f"{SIZE_URL}?{query}", timeout=60) as response:
                payload = json.load(response)
            break
        except HTTPError as exc:
            if exc.code != 429 or attempt == 3:
                raise
            time.sleep(2 ** attempt)
    return int(payload["size"]["config"]["num_rows"])


def download(languages: list[str], per_language: int, output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    temporary_output = output.with_name(output.name + ".tmp")
    with temporary_output.open("w", encoding="utf-8") as handle:
        for language in languages:
            total_rows = get_row_count(language)
            window_count = min(4 if per_language <= 100 else 10, per_language)
            rows_per_window = min(MAX_ROWS_PER_REQUEST, (per_language + window_count - 1) // window_count)
            offsets = [
                0 if window_count == 1 else round((total_rows - rows_per_window) * index / (window_count - 1))
                for index in range(window_count)
            ]
            for window_index, offset in enumerate(offsets):
                length = min(rows_per_window, per_language - window_index * rows_per_window)
                if length <= 0:
                    break
                query = urlencode({
                    "dataset": "miracl/miracl-corpus",
                    "config": language,
                    "split": "train",
                    "offset": offset,
                    "length": length,
                })
                for attempt in range(4):
                    try:
                        with urlopen(f"{API_URL}?{query}", timeout=60) as response:
                            payload = json.load(response)
                        break
                    except HTTPError as exc:
                        if exc.code != 429 or attempt == 3:
                            raise
                        time.sleep(2 ** attempt)
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
                time.sleep(0.5)
    temporary_output.replace(output)
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
