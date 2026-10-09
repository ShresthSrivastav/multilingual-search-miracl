import unittest

import numpy as np

from search_engine import Document, detect_language, diversify_ranked, extractive_answer, filter_documents, hybrid_rank, keyword_rank, rank_embeddings


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

    def test_hybrid_combines_both_rankings(self):
        ranked = hybrid_rank([(0, 0.9), (1, 0.8)], [(2, 1.0), (0, 0.7)], top_k=3)
        self.assertEqual([index for index, _ in ranked], [0, 2, 1])

    def test_diversifies_article_titles(self):
        documents = [
            Document("1", "Same article", "first", "en"),
            Document("2", "Same article", "second", "en"),
            Document("3", "Different article", "third", "en"),
        ]
        ranked = diversify_ranked([(0, 1.0), (1, 0.9), (2, 0.8)], documents, top_k=2)
        self.assertEqual([index for index, _ in ranked], [0, 2])

    def test_extractive_answer_uses_source_text(self):
        text = "Solar power comes from sunlight. Oceans contain salt water."
        answer = extractive_answer("Where does solar power come from?", text)
        self.assertIn("Solar power comes from sunlight.", answer)
        self.assertNotIn("generated", answer)

    def test_detects_query_language(self):
        self.assertEqual(detect_language("भारत की राजधानी"), "hi")
        self.assertEqual(detect_language("ما هو الماء؟"), "ar")
        self.assertEqual(detect_language("¿Qué es Andorra?"), "es")
        self.assertEqual(detect_language("What is anarchism?"), "en")
        self.assertEqual(detect_language("\u00bfQu\u00e9 idioma oficial se habla?"), "es")
        self.assertEqual(detect_language("Was ist die Hauptstadt von Deutschland?"), "de")
        self.assertEqual(detect_language("\u092d\u093e\u0930\u0924 \u0915\u0940 \u0930\u093e\u091c\u0927\u093e\u0928\u0940"), "hi")


if __name__ == "__main__":
    unittest.main()
