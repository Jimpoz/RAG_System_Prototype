# semantic_cache.py
# This class implements a semantic cache to reduce latency.

from typing import Dict, Any, List, Optional
import time

# We need an embedding model, same as the query pipeline
# from sentence_transformers import SentenceTransformer
import numpy as np # For similarity calculation

class SemanticCache:
    """
    Implements a semantic cache to store and retrieve answers for
    similar queries, reducing latency, as described in section 3.3.
    """
    
    # Use the same model as the query pipeline for comparison
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    
    def __init__(self, similarity_threshold: float = 0.95, max_size: int = 1000):
        # self.embedding_model = SentenceTransformer(self.EMBEDDING_MODEL_NAME)
        self.embedding_model = "all-MiniLM-L6-v2_model_placeholder"
        
        # Cache stores vectors and their corresponding responses
        self.cache_vectors = []
        self.cache_responses = []
        
        self.threshold = similarity_threshold
        self.max_size = max_size
        
        print(f"Semantic Cache initialized (Threshold: {self.threshold}, Max Size: {self.max_size}).")

    def _get_embedding(self, query: str) -> np.ndarray:
        """Helper to get a query embedding."""
        # return self.embedding_model.encode(query)
        return np.random.rand(384) # Placeholder (dim=384 for MiniLM)
        
    def check_cache(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Checks the cache for a semantically similar query.
        
        Args:
            query: The incoming user query.

        Returns:
            The cached response if a similar query is found, else None.
        """
        if not self.cache_vectors:
            return None # Cache is empty
            
        query_vector = self._get_embedding(query)
        
        # Convert cache to numpy array for efficient calculation
        cache_matrix = np.array(self.cache_vectors)
        
        # Calculate cosine similarity
        # (dot product of normalized vectors)
        query_norm = query_vector / np.linalg.norm(query_vector)
        cache_norms = cache_matrix / np.linalg.norm(cache_matrix, axis=1, keepdims=True)
        
        similarities = np.dot(cache_norms, query_norm)
        
        # Find the best match
        best_match_idx = np.argmax(similarities)
        best_score = similarities[best_match_idx]
        
        if best_score >= self.threshold:
            print(f"CACHE HIT! (Score: {best_score:.4f})")
            return self.cache_responses[best_match_idx]
        
        print("CACHE MISS.")
        return None

    def add_to_cache(self, query: str, response: Dict[str, Any]):
        """
        Adds a new query and its response to the cache.
        
        Args:
            query: The user query string.
            response: The final RAG response object.
        """
        print("Adding to cache...")
        
        if len(self.cache_vectors) >= self.max_size:
            # Simple FIFO eviction strategy
            self.cache_vectors.pop(0)
            self.cache_responses.pop(0)
            
        query_vector = self._get_embedding(query)
        self.cache_vectors.append(query_vector)
        self.cache_responses.append(response)

# Example usage (if run as a script)
if __name__ == "__main__":
    cache = SemanticCache(similarity_threshold=0.9)
    
    dummy_response = {"answer": "Metformin is 500mg.", "citations": []}
    
    # 1. Add an item
    cache.add_to_cache("What is metformin dosage?", dummy_response)
    
    # 2. Check for a similar query (should be a HIT)
    print("\nChecking for: 'How much metformin should I take?'")
    result = cache.check_cache("How much metformin should I take?")
    
    # 3. Check for a different query (should be a MISS)
    print("\nChecking for: 'What is paracetamol?'")