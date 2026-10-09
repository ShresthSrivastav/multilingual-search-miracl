"""Download official MIRACL development topics and positive qrels for local evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlopen


BASE_URL = "https://huggingface.co/datasets/miracl/miracl/resolve/main/miracl-v1.0-{language}"


def read_lines(url: str) -> list[str]:
    with urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8").splitlines()


def download(languages: list[str], topics_path: Path, qrels_path: Path) -> tuple[int, int]:
    topics_path.parent.mkdir(parents=True, exist_ok=True)
    topic_count = 0
    qrel_count = 0
    with topics_path.open("w", encoding="utf-8") as topics, qrels_path.open("w", encoding="utf-8") as qrels:
        for language in languages:
            topic_url = f"{BASE_URL.format(language=language)}/topics/topics.miracl-v1.0-{language}-dev.tsv"
            qrels_url = f"{BASE_URL.format(language=language)}/qrels/qrels.miracl-v1.0-{language}-dev.tsv"
            for line in read_lines(topic_url):
                if not line.strip():
                    continue
                qid, query = line.split("\t", 1)
                topics.write(json.dumps({"language": language, "qid": qid, "query": query}, ensure_ascii=False) + "\n")
                topic_count += 1
            for line in read_lines(qrels_url):
                if not line.strip():
                    continue
                qid, _, docid, relevance = line.split("\t")
                if int(relevance) > 0:
                    qrels.write(json.dumps({"language": language, "qid": qid, "docid": docid}, ensure_ascii=False) + "\n")
                    qrel_count += 1
    return topic_count, qrel_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", default=["ar", "bn", "de", "en", "es", "fa", "fi", "fr", "hi", "id", "ja", "ko", "ru", "sw", "te", "th", "yo", "zh"])
    parser.add_argument("--topics", type=Path, default=Path("data/miracl_dev_topics.jsonl"))
    parser.add_argument("--qrels", type=Path, default=Path("data/miracl_dev_qrels.jsonl"))
    args = parser.parse_args()
    topics, qrels = download(args.languages, args.topics, args.qrels)
    print(f"Wrote {topics} official dev topics and {qrels} positive qrels")


if __name__ == "__main__":
    main()
