from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class EmbeddingUtils:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the embedding utility with a pre-trained model.
        
        Args:
        - model_name (str): Name of the sentence-transformers model.
        """
        self.model = SentenceTransformer(model_name)

    def generate_embedding(self, text):
        """
        Generate embedding for a given text.
        
        Args:
        - text (str): Input text to generate embedding for.
        
        Returns:
        - np.ndarray: Embedding vector.
        """
        return self.model.encode([text])[0]  # Encode returns a list; we extract the first item

    def calculate_similarity(self, embedding1, embedding2):
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
        - embedding1 (np.ndarray): First embedding vector.
        - embedding2 (np.ndarray): Second embedding vector.
        
        Returns:
        - float: Cosine similarity score.
        """
        return cosine_similarity([embedding1], [embedding2])[0][0]

    def find_best_match(self, query_embedding, cache_embeddings, threshold=0.8):
        """
        Find the best match for a query embedding from a list of cached embeddings.
        
        Args:
        - query_embedding (np.ndarray): Embedding of the input query.
        - cache_embeddings (list of np.ndarray): List of cached embeddings.
        - threshold (float): Minimum similarity score to consider a match.
        
        Returns:
        - int: Index of the best match if above threshold, otherwise -1.
        """
        if not cache_embeddings:
            return -1  # No cached embeddings to compare

        similarities = cosine_similarity([query_embedding], cache_embeddings)[0]
        best_match_index = np.argmax(similarities)
        best_match_score = similarities[best_match_index]

        if best_match_score >= threshold:
            return best_match_index
        return -1
