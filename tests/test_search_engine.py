import unittest

import numpy as np

from search_engine import Document, diversify_ranked, filter_documents, keyword_rank, rank_embeddings


class SearchEngineTests(unittest.TestCase):
    def test_cosine_ranking(self):
        ranked = rank_embeddings([1.0, 0.0], [[0.0, 1.0], [1.0, 0.0], [0.7, 0.7]], top_k=2)
        self.assertEqual([index for index, _ in ranked], [1, 2])

    def test_language_filter(self):
        documents = [
            Document("1", "A", "a", "en"),
            Document("2", "B", "b", "hi"),
        ]
        self.assertEqual([item.docid for item in filter_documents(documents, "hi")], ["2"])
        self.assertEqual(len(filter_documents(documents, "all")), 2)

    def test_invalid_top_k(self):
        with self.assertRaises(ValueError):
            rank_embeddings(np.ones(2), np.ones((2, 2)), top_k=0)

    def test_keyword_baseline(self):
        documents = [
            Document("1", "Solar energy", "Solar power comes from sunlight.", "en"),
            Document("2", "Ocean", "The ocean contains salt water.", "en"),
        ]
        self.assertEqual(keyword_rank("solar power", documents, top_k=1)[0][0], 0)

    def test_diversifies_article_titles(self):
        documents = [
            Document("1", "Same article", "first", "en"),
            Document("2", "Same article", "second", "en"),
            Document("3", "Different article", "third", "en"),
        ]
        ranked = diversify_ranked([(0, 1.0), (1, 0.9), (2, 0.8)], documents, top_k=2)
        self.assertEqual([index for index, _ in ranked], [0, 2])


if __name__ == "__main__":
    unittest.main()
