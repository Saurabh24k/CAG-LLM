import unittest
from src.cache_manager import CacheManager

class TestCacheManager(unittest.TestCase):
    def setUp(self):
        self.cache = CacheManager(max_cache_size=2)

    def test_add_and_get_from_cache(self):
        self.cache.add_to_cache("query1", "response1")
        self.assertEqual(self.cache.get_from_cache("query1"), "response1")

    def test_cache_eviction(self):
        self.cache.add_to_cache("query1", "response1")
        self.cache.add_to_cache("query2", "response2")
        self.cache.add_to_cache("query3", "response3")  # This should evict "query1"
        self.assertIsNone(self.cache.get_from_cache("query1"))
        self.assertEqual(self.cache.get_from_cache("query2"), "response2")
        self.assertEqual(self.cache.get_from_cache("query3"), "response3")

    def test_clear_cache(self):
        self.cache.add_to_cache("query1", "response1")
        self.cache.clear_cache()
        self.assertIsNone(self.cache.get_from_cache("query1"))

if __name__ == "__main__":
    unittest.main()
