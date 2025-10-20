from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer

class TextEmbedder:
    """
    Generates 1024-dimension vector embeddings for text chunks using
    the BAAI/bge-m3 model.
    """
    
    MODEL_NAME = "BAAI/bge-m3"
    DIMENSIONS = 1024
    CONTEXT_WINDOW = 8192

    def __init__(self):
        print(f"Loading embedding model: {self.MODEL_NAME}...")
        # 1. Load the model
        try:
            self.model = SentenceTransformer(self.MODEL_NAME)
            print("Embedding model loaded successfully.")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            print("Please ensure 'sentence-transformers' is installed and you have an internet connection.")
            self.model = None

    def embed_chunk(self, chunk_text: str) -> np.ndarray:
        """
        Generates an embedding for a single text chunk.
        """
        if not self.model:
            raise ValueError("Embedding model is not loaded.")
            
        # 2. Use the model to encode
        vector = self.model.encode(chunk_text, normalize_embeddings=True)
        return vector.astype(np.float32)

    def embed_batch(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generates embeddings for a batch of chunk objects.
        This is the main method used in an ingestion pipeline.
        """
        if not self.model:
            raise ValueError("Embedding model is not loaded.")
            
        print(f"Embedding batch of {len(chunks)} chunks...")
        
        texts_to_embed = [chunk['text'] for chunk in chunks]
        
        # 3. Generate embeddings in a real batch
        vectors = self.model.encode(
            texts_to_embed, 
            show_progress_bar=True,
            normalize_embeddings=True # Normalize for fast similarity search
        )
        
        # Add the numpy vector to each chunk object
        for i, chunk in enumerate(chunks):
            chunk['vector'] = vectors[i].astype(np.float32)

        print("Batch embedding complete.")
        return chunks