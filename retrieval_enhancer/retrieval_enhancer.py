from typing import List, Dict, Any
import json
from rank_bm25 import BM25Okapi 
from sentence_transformers import CrossEncoder 
import numpy as np

class RetrievalEnhancer:
    """
    Implements advanced retrieval strategies (Hybrid Search, Reranking)
    to improve accuracy, as described in section 3.1.
    """
    
    def __init__(self):
        # 1. Initialize a real cross-encoder for reranking
        model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
        print(f"Loading reranker model: {model_name}...")
        try:
            self.reranker = CrossEncoder(model_name)
            print("Reranker loaded successfully.")
        except Exception as e:
            print(f"Error loading reranker: {e}")
            self.reranker = None

        # 2. Initialize keyword search index
        self.keyword_index: BM25Okapi = None
        self.chunk_corpus: List[Dict] = []
        
        print("Retrieval Enhancer initialized.")

    def build_keyword_index(self, chunks: List[Dict[str, Any]]):
        """
        Builds the BM25 keyword index from the provided text chunks.
        This must be called once after chunks are created.
        """
        print(f"Building keyword (BM25) index from {len(chunks)} chunks...")
        self.chunk_corpus = chunks # Store the original chunks
        
        # BM25 requires a list of tokenized documents
        tokenized_corpus = [chunk['text'].split() for chunk in chunks]
        self.keyword_index = BM25Okapi(tokenized_corpus)
        print("Keyword index built successfully.")

    def keyword_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """
        Performs a BM25 keyword search.
        """
        if not self.keyword_index:
            print("Warning: Keyword index not built. Skipping keyword search.")
            return []
            
        tokenized_query = query.split()
        
        # Get scores for all documents
        doc_scores = self.keyword_index.get_scores(tokenized_query)
        
        # Get the indices of the top-k scores
        top_k_indices = np.argsort(doc_scores)[::-1][:k]
        
        # Map indices back to the original chunks
        results = [self.chunk_corpus[i] for i in top_k_indices if doc_scores[i] > 0]
        return results

    def fuse_results(self, semantic_results: List[Dict], keyword_results: List[Dict], k: int = 60) -> List[Dict]:
        """
        Combines two lists of search results using Reciprocal Rank Fusion (RRF).
        Gives a balanced score to documents that appear in both lists.
        """
        ranks = {}
        doc_map = {} 

        for i, chunk in enumerate(semantic_results):
            chunk_id = chunk['metadata'].get('chunk_index')
            if chunk_id is None: continue
            
            doc_map[chunk_id] = chunk
            ranks[chunk_id] = 1.0 / (i + k) 

        for i, chunk in enumerate(keyword_results):
            chunk_id = chunk['metadata'].get('chunk_index')
            if chunk_id is None: continue

            doc_map[chunk_id] = chunk
            ranks[chunk_id] = ranks.get(chunk_id, 0) + (1.0 / (i + k))

        # Sort by the fused RRF score
        sorted_ranks = sorted(ranks.items(), key=lambda item: item[1], reverse=True)
        
        fused_list = [doc_map[chunk_id] for chunk_id, score in sorted_ranks]
        return fused_list

    def rerank_results(self, query: str, chunks: List[Dict]) -> List[Dict]:
        """
        Uses a powerful cross-encoder to re-rank the retrieved chunks.
        """
        if not self.reranker or not chunks:
            print("Reranker not loaded or no chunks to rerank. Skipping.")
            return chunks
            
        print(f"Reranking {len(chunks)} results with cross-encoder...")
        
        # 1. Create pairs of (query, chunk_text)
        sentence_pairs = [(query, chunk['text']) for chunk in chunks]
        
        # 2. Get real scores from the cross-encoder
        scores = self.reranker.predict(sentence_pairs)
        
        # 3. Combine scores with chunks and sort
        scored_chunks = list(zip(scores, chunks))
        scored_chunks.sort(key=lambda x: x[0], reverse=True) # Sort by score, descending
        
        # 4. Unwrap the sorted chunks
        reranked_chunks = [chunk for score, chunk in scored_chunks]
        
        print("Reranking complete.")
        return reranked_chunks