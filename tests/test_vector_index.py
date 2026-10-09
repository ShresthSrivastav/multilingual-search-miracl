import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np

from vector_index import LocalIndex


class LocalIndexTests(unittest.TestCase):
    def test_vector_and_fts_retrieval(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = [
                {"docid": "1", "title": "Solar", "text": "Energy from sunlight", "language": "en"},
                {"docid": "2", "title": "Andorra", "text": "Catalan is the official language", "language": "es"},
            ]
            shards = []
            for index, (record, vector) in enumerate(zip(records, ([1, 0], [0, 1]))):
                language = record["language"]
                np.save(root / f"{language}-vectors.npy", np.asarray([vector], dtype=np.float16))
                shards.append({"language": language, "start": index, "count": 1, "vectors": f"{language}-vectors.npy"})
            database = sqlite3.connect(root / "lexical.sqlite")
            try:
                database.execute("CREATE VIRTUAL TABLE passages USING fts5(docid UNINDEXED,title,text,language UNINDEXED)")
                database.executemany("INSERT INTO passages(rowid,docid,title,text,language) VALUES(?,?,?,?,?)", [(1, "1", "Solar", records[0]["text"], "en"), (2, "2", "Andorra", records[1]["text"], "es")])
                database.commit()
            finally:
                database.close()
            (root / "manifest.json").write_text(json.dumps({
                "count": 2,
                "languages": {"en": 1, "es": 1},
                "shards": shards,
            }), encoding="utf-8")

            index = LocalIndex(root)
            self.assertEqual(index.vector_search(np.asarray([0, 1]), "es")[0][0].docid, "2")
            lexical = index.keyword_search("Andorra", "es")
            self.assertEqual(index.document_by_global_index(lexical[0][0]).title, "Andorra")


if __name__ == "__main__":
    unittest.main()
