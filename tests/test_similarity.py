import unittest
from src.embedding_utils import EmbeddingUtils

class TestEmbeddingUtils(unittest.TestCase):
    def setUp(self):
        self.embedding_utils = EmbeddingUtils()

    def test_generate_embedding(self):
        text = "This is a test sentence."
        embedding = self.embedding_utils.generate_embedding(text)
        self.assertEqual(len(embedding.shape), 1)  # Ensure it's a 1D vector
        self.assertGreater(len(embedding), 0)  # Embedding should have a non-zero length

    def test_calculate_similarity(self):
        text1 = "This is a test sentence."
        text2 = "This is a similar test sentence."
        embedding1 = self.embedding_utils.generate_embedding(text1)
        embedding2 = self.embedding_utils.generate_embedding(text2)
        similarity = self.embedding_utils.calculate_similarity(embedding1, embedding2)
        self.assertGreater(similarity, 0.8)  # Similar sentences should have high similarity

    def test_find_best_match(self):
        query = "Find the best match for this query."
        cache = [
            "This is a random sentence.",
            "This is another random query.",
            "Find the best match for this query.",
        ]
        query_embedding = self.embedding_utils.generate_embedding(query)
        cache_embeddings = [self.embedding_utils.generate_embedding(text) for text in cache]
        best_match_index = self.embedding_utils.find_best_match(query_embedding, cache_embeddings, threshold=0.8)
        self.assertEqual(best_match_index, 2)  # Index 2 should be the best match

if __name__ == "__main__":
    unittest.main()
