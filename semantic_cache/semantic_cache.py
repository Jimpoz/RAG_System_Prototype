from typing import Dict, Any, List, Optional
import time
from sentence_transformers import SentenceTransformer, util
import numpy as np

class SemanticCache:
    """
    Implements a semantic cache to store and retrieve answers for
    similar queries, reducing latency, as described in section 3.3.
    """
    
    EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
    EMBEDDING_DIM = 1024
    
    def __init__(self, embedder: SentenceTransformer, similarity_threshold: float = 0.95, max_size: int = 1000):
        
        # 1. Use the embedder from the pipeline
        self.embedding_model = embedder
        
        # 2. Cache stores vectors and their corresponding responses
        # Initialize as an empty array with the correct shape
        self.cache_vectors = np.empty((0, self.EMBEDDING_DIM), dtype=np.float32)
        self.cache_responses = []
        
        self.threshold = similarity_threshold
        self.max_size = max_size
        
        print(f"Semantic Cache initialized (Threshold: {self.threshold}, Max Size: {self.max_size}).")

    def _get_embedding(self, query: str) -> np.ndarray:
        """Helper to get a query embedding."""
        # Use the real model
        vector = self.embedding_model.encode(query, normalize_embeddings=True)
        return vector.astype(np.float32)
        
    def check_cache(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Checks the cache for a semantically similar query.
        """
        if self.cache_vectors.shape[0] == 0:
            return None # Cache is empty
            
        query_vector = self._get_embedding(query)
        
        # Calculate cosine similarity (dot product of normalized vectors)
        # The embedder and self.cache_vectors are already normalized
        similarities = np.dot(self.cache_vectors, query_vector)
        
        # Find the best match
        best_match_idx = np.argmax(similarities)
        best_score = similarities[best_match_idx]
        
        if best_score >= self.threshold:
            print(f"--- SEMANTIC CACHE HIT! (Score: {best_score:.4f}) ---")
            # Return a copy of the cached response
            return self.cache_responses[best_match_idx].copy()
        
        print("--- SEMANTIC CACHE MISS ---")
        return None

    def add_to_cache(self, query: str, response: Dict[str, Any]):
        """
        Adds a new query and its response to the cache.
        """
        print("--- Adding response to semantic cache... ---")
        
        # Get the query vector to store
        query_vector = self._get_embedding(query)
        
        if len(self.cache_responses) >= self.max_size:
            # Simple FIFO eviction strategy
            self.cache_vectors = self.cache_vectors[1:]
            self.cache_responses.pop(0)
            
        # Add new item
        # Use vstack to add the new vector row
        self.cache_vectors = np.vstack([self.cache_vectors, query_vector])
        self.cache_responses.append(response)

# Example usage (if run as a script)
if __name__ == "__main__":
    
    # 1. Load the real embedder model
    try:
        model = SentenceTransformer(SemanticCache.EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"Could not load model: {e}")
        model = None

    if model:
        cache = SemanticCache(embedder=model, similarity_threshold=0.9)
        
        dummy_response = {"answer": "Metformin is 500mg.", "citations": []}
        
        # 2. Add an item
        cache.add_to_cache("What is metformin dosage?", dummy_response)
        
        # 3. Check for a similar query (should be a HIT)
        print("\nChecking for: 'How much metformin should I take?'")
        result = cache.check_cache("How much metformin should I take?")
        if result:
            print(f"Found cached answer: {result['answer']}")
        
        # 4. Check for a different query (should be a MISS)
        print("\nChecking for: 'What is paracetamol?'")
        result = cache.check_cache("What is paracetamol?")
        if not result:
            print("Correctly missed cache.")