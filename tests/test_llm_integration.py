import unittest
from src.generation_model import LLMIntegration

class TestLLMIntegration(unittest.TestCase):
    def setUp(self):
        # Initialize the LLM Integration module
        self.llm_integration = LLMIntegration()

    def test_generate_response(self):
        query = "What is machine learning?"
        response = self.llm_integration.generate_response(query)
        self.assertIsInstance(response, str)  # Response should be a string
        self.assertGreater(len(response), 0)  # Response should not be empty

    def test_cache_functionality(self):
        query = "What is artificial intelligence?"
        # First query (cache miss)
        response1 = self.llm_integration.generate_response(query)
        self.assertIn("Cache Miss!", response1)

        # Second query (cache hit)
        response2 = self.llm_integration.generate_response(query)
        self.assertIn("Cache Hit!", response2)

if __name__ == "__main__":
    unittest.main()
