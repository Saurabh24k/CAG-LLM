import subprocess
import numpy as np
from src.cache_manager import CacheManager
from src.embedding_utils import EmbeddingUtils
from dotenv import load_dotenv
import os

load_dotenv()

class LLMIntegration:
    def __init__(self, cache_size=100, model_name="llama3.2:latest", similarity_threshold=0.8):
        self.cache_manager = CacheManager(max_cache_size=cache_size)
        self.embedding_utils = EmbeddingUtils()
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold

    def query_llm(self, prompt):
        """ Queries the LLM for generating a response. """
        try:
            result = subprocess.run(
                ["ollama", "run", self.model_name],
                input=prompt,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                print(f"[ERROR] Query failed: {result.stderr}")
                return "Error: Unable to generate response."
        except Exception as e:
            print(f"[EXCEPTION] LLM query failed: {e}")
            return "Error: LLM query failed."

    def generate_response(self, query):
        """ Generate a response with caching and similarity checking. """
        query_key = self.cache_manager.normalize_key(query)
        
        # Check exact cache match
        cached_response = self.cache_manager.get_from_cache(query_key)
        if cached_response:
            return f"Cache Hit! {cached_response}"

        # Generate and store embedding
        query_embedding = self.embedding_utils.generate_embedding(query)

        # Check for approximate match using embeddings
        best_match_key = self._find_best_match(query_embedding)
        if best_match_key:
            cached_response = self.cache_manager.get_from_cache(best_match_key)
            return f"Cache Hit! {cached_response}"

        # Query the LLM and cache response
        response = self.query_llm(query)
        self.cache_manager.add_to_cache(query_key, response, embedding=query_embedding)
        return f"Cache Miss! {response}"

    def _find_best_match(self, query_embedding):
        """ Find the best match using cosine similarity on embeddings. """
        best_match_key = None
        highest_similarity = 0

        for key in self.cache_manager.cache:
            cached_embedding = self.cache_manager.get_embedding(key)
            if cached_embedding is not None and cached_embedding.size > 0:
                similarity = self.embedding_utils.calculate_similarity(
                    np.array(query_embedding), np.array(cached_embedding)
                )
                if similarity > highest_similarity and similarity >= self.similarity_threshold:
                    highest_similarity = similarity
                    best_match_key = key

        return best_match_key
