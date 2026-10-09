"""Build or resume a disk-backed vector and SQLite FTS index from MIRACL."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import re
import sqlite3
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from search_engine import SUPPORTED_LANGUAGES


def corpus_shards(source: Path, languages: list[str]):
    if source.is_file():
        for language in languages:
            yield language, source
        return
    for path in sorted(source.rglob("docs-*.jsonl.gz")):
        match = re.search(r"miracl-corpus-v1\.0-([a-z]{2})", str(path))
        if match and match.group(1) in languages:
            yield match.group(1), path


def read_records(language: str, path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                actual_language = str(item.get("language", language))
                if actual_language == language:
                    yield {"docid": str(item["docid"]), "title": str(item["title"]), "text": str(item["text"]), "language": language}


def build(source: Path, output: Path, languages: list[str], model_name: str, batch_size: int):
    shards = list(corpus_shards(source, languages))
    if not shards:
        raise ValueError(f"No MIRACL document shards found under {source}")
    output.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(model_name)
    dimension = model.get_sentence_embedding_dimension()
    manifest_path = output / "manifest.json"
    inputs = [[language, str(path.resolve())] for language, path in shards]
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("model") != model_name or manifest.get("dimension") != dimension or manifest.get("inputs") != inputs:
            raise ValueError("Existing index uses a different model or source list; choose a new output folder")
    else:
        manifest = {"model": model_name, "dimension": dimension, "count": 0, "languages": {}, "shards": [], "inputs": inputs}
    connection = sqlite3.connect(output / "lexical.sqlite")
    connection.execute("CREATE VIRTUAL TABLE IF NOT EXISTS passages USING fts5(docid UNINDEXED, title, text, language UNINDEXED)")
    connection.execute("DELETE FROM passages WHERE rowid > ?", (manifest["count"],))
    connection.commit()

    for shard_number, (language, path) in enumerate(shards):
        count = sum(1 for _ in read_records(language, path))
        if not count:
            continue
        key = f"{language}-{shard_number:03d}"
        vectors_name = f"{key}-vectors.npy"
        previous = next((item for item in manifest["shards"] if item["vectors"] == vectors_name), None)
        if previous:
            if previous["count"] != count:
                raise ValueError(f"Source shard changed since indexing: {path}")
            continue
        if manifest["shards"] and manifest["shards"][-1]["start"] + manifest["shards"][-1]["count"] != manifest["count"]:
            raise ValueError("Index manifest has inconsistent shard offsets")
        vectors = np.lib.format.open_memmap(output / vectors_name, mode="w+", dtype=np.float16, shape=(count, dimension))
        rows, passages = [], []
        for record_index, record in enumerate(read_records(language, path)):
            passages.append(f"passage: {record['title']}. {record['text']}")
            global_index = manifest["count"] + record_index
            rows.append((global_index + 1, record["docid"], record["title"], record["text"], language))
            if len(passages) >= batch_size:
                vectors[record_index + 1 - len(passages):record_index + 1] = model.encode(passages, normalize_embeddings=True, show_progress_bar=False).astype(np.float16)
                connection.executemany("INSERT INTO passages(rowid,docid,title,text,language) VALUES(?,?,?,?,?)", rows)
                connection.commit()
                rows.clear(); passages.clear()
            if record_index and record_index % 10000 == 0:
                print(f"{language}: {record_index:,}/{count:,}", flush=True)
        if passages:
            start = count - len(passages)
            vectors[start:count] = model.encode(passages, normalize_embeddings=True, show_progress_bar=False).astype(np.float16)
            connection.executemany("INSERT INTO passages(rowid,docid,title,text,language) VALUES(?,?,?,?,?)", rows)
            connection.commit()
        vectors.flush()
        manifest["shards"].append({"language": language, "start": manifest["count"], "count": count, "vectors": vectors_name})
        manifest["count"] += count
        manifest["languages"][language] = manifest["languages"].get(language, 0) + count
        temporary_manifest = manifest_path.with_suffix(".tmp")
        temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary_manifest.replace(manifest_path)
        print(f"Indexed {language}: {count:,} passages", flush=True)

    connection.close()
    print(f"Index complete: {manifest['count']:,} passages in {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/miracl_full"), help="Downloaded gzip shards folder or JSONL corpus file")
    parser.add_argument("--output", type=Path, default=Path("data/miracl_index"))
    parser.add_argument("--languages", nargs="+", default=list(SUPPORTED_LANGUAGES))
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size < 1 or any(language not in SUPPORTED_LANGUAGES for language in args.languages):
        parser.error("batch size must be positive and languages must be supported MIRACL codes")
    build(args.source, args.output, args.languages, args.model, args.batch_size)


if __name__ == "__main__":
    main()
