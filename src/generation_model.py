import requests
import numpy as np
import os
import time
import warnings
from dotenv import load_dotenv
from cache_manager import CacheManager
from embedding_utils import EmbeddingUtils

# Suppress PyTorch Warnings
warnings.filterwarnings("ignore", message="Tried to instantiate class '__path__._path'")

# Load environment variables
load_dotenv()

class LLMIntegration:
    def __init__(self, api_key=None, cache_size=100, similarity_threshold=0.8):
        """Initialize the LLM Integration with API Key, Cache, and Embedding Utilities."""
        self.cache_manager = CacheManager(max_cache_size=cache_size)
        self.embedding_utils = EmbeddingUtils()
        self.similarity_threshold = similarity_threshold

        # API Key Handling (priority to passed key)
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        if not self.api_key:
            raise ValueError("HUGGINGFACE_API_KEY is missing. Please add it to secrets.toml or .env file.")

        # Default model URL
        self.api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"

    def query_llm(self, prompt):
        """Query the Hugging Face API for a generated response with retry logic."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        retries = 3
        wait_time = 3  # Exponential backoff delay

        for attempt in range(retries):
            try:
                response = requests.post(self.api_url, json={"inputs": prompt}, headers=headers)
                response.raise_for_status()
                response_json = response.json()
                
                # Handling Response Structure
                if isinstance(response_json, list) and 'generated_text' in response_json[0]:
                    return response_json[0]['generated_text']
                else:
                    return "Error: Unexpected response format from Hugging Face API."

            except requests.exceptions.RequestException as e:
                if response.status_code == 503:
                    print(f"[WARNING] Service Unavailable. Retrying... ({attempt + 1}/{retries})")
                    time.sleep(wait_time)
                    wait_time *= 2
                else:
                    print(f"[ERROR] API Error: {e}")
                    return "**Error: Unable to fetch a response from the API.**"
        return "**Error: Unable to fetch a response after retries.**"

    def generate_response(self, query):
        """Generate a response with cache checking and similarity matching."""
        query_key = self.cache_manager.normalize_key(query)

        # Check for cache match
        cached_response = self.cache_manager.get_from_cache(query_key)
        if cached_response:
            return f"Cache Hit! {cached_response}"

        # Generate query embedding
        query_embedding = self.embedding_utils.generate_embedding(query)

        # Check for approximate match
        best_match_key = self._find_best_match(query_embedding)
        if best_match_key:
            cached_response = self.cache_manager.get_from_cache(best_match_key)
            return f"Cache Hit! {cached_response}"

        # If no cache match, query the API
        response = self.query_llm(query)
        
        # ✅ Only cache successful responses
        if response and "Error" not in response:
            self.cache_manager.add_to_cache(query_key, response, embedding=query_embedding)
            return f"Cache Miss! {response}"
        else:
            return "**Error: Could not generate a response.**"

    def _find_best_match(self, query_embedding):
        """Find the best match in the cache using similarity checking."""
        best_match_key = None
        highest_similarity = 0

        for key in self.cache_manager.cache:
            cached_embedding = self.cache_manager.get_embedding(key)
            if cached_embedding is not None:
                similarity = self.embedding_utils.calculate_similarity(query_embedding, cached_embedding)
                if similarity > highest_similarity and similarity >= self.similarity_threshold:
                    best_match_key = key
                    highest_similarity = similarity
        return best_match_key
