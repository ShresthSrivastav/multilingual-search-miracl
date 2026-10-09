"""Disk-backed exact vector search plus SQLite full-text candidates."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sqlite3

import numpy as np

from search_engine import Document


class LocalIndex:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        self.db_path = self.directory / "lexical.sqlite"

    @property
    def count(self) -> int:
        return int(self.manifest["count"])

    @property
    def languages(self) -> dict[str, int]:
        return self.manifest["languages"]

    def _documents(self, global_indexes: list[int]) -> dict[int, Document]:
        if not global_indexes:
            return {}
        marks = ",".join("?" for _ in global_indexes)
        connection = sqlite3.connect(self.db_path)
        try:
            rows = connection.execute(
                f"SELECT rowid,docid,title,text,language FROM passages WHERE rowid IN ({marks})",
                [index + 1 for index in global_indexes],
            ).fetchall()
        finally:
            connection.close()
        return {row[0] - 1: Document(row[1], row[2], row[3], row[4]) for row in rows}

    def vector_search(self, query: np.ndarray, language: str = "all", limit: int = 100):
        best: list[tuple[int, int, float]] = []
        for shard_id, shard in enumerate(self.manifest["shards"]):
            if language != "all" and shard["language"] != language:
                continue
            vectors = np.load(self.directory / shard["vectors"], mmap_mode="r")
            for start in range(0, len(vectors), 32768):
                scores = np.asarray(vectors[start:start + 32768], dtype=np.float32) @ query
                take = min(limit, len(scores))
                indexes = np.argpartition(-scores, take - 1)[:take]
                best.extend((shard_id, start + int(i), float(scores[i])) for i in indexes)
                if len(best) > 4 * limit:
                    best = sorted(best, key=lambda row: -row[2])[:limit]
        best.sort(key=lambda row: -row[2])
        global_indexes = [self.manifest["shards"][sid]["start"] + idx for sid, idx, _ in best[:limit]]
        documents = self._documents(global_indexes)
        return [(documents[index], score) for index, (_, _, score) in zip(global_indexes, best[:limit])]

    def keyword_search(self, query: str, language: str = "all", limit: int = 100):
        terms = re.findall(r"\w+", query, flags=re.UNICODE)
        if not terms:
            return []
        match = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
        language_sql = "" if language == "all" else " AND language = ?"
        params: tuple = (match, limit) if language == "all" else (match, language, limit)
        connection = sqlite3.connect(self.db_path)
        try:
            rows = connection.execute(
                "SELECT rowid, bm25(passages) FROM passages WHERE passages MATCH ?" + language_sql + " ORDER BY bm25(passages) LIMIT ?",
                params,
            ).fetchall()
        finally:
            connection.close()
        return [(int(rowid) - 1, -float(score)) for rowid, score in rows]

    def document_by_global_index(self, index: int) -> Document:
        documents = self._documents([index])
        if index not in documents:
            raise IndexError(index)
        return documents[index]
