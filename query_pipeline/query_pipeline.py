import numpy as np
from typing import List, Dict, Any
from llama_cpp import Llama
import torch  
from sentence_transformers import SentenceTransformer, util
import os
from dotenv import load_dotenv
load_dotenv() 

class QueryPipeline:
    """
    Implements the 5-step RAG pipeline that takes a user query and
    returns a citable answer.
    This version works entirely in-memory.
    """
    
    EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
    
    LLM_MODEL_NAME = "Llama 3.1"
    # Path to your local GGUF model
    LLM_MODEL_PATH = os.environ.get("LLAMA_PATH")

    def __init__(self, chunks: List[Dict[str, Any]]):
        
        print(f"Loading embedding model: {self.EMBEDDING_MODEL_NAME}...")
        self.embedding_model = SentenceTransformer(self.EMBEDDING_MODEL_NAME)
        
        print(f"Loading {len(chunks)} embedded chunks into memory...")
        self.chunks = chunks
        
        self.chunk_vectors = np.array([chunk['vector'] for chunk in self.chunks], dtype=np.float32)
        
        print(f"Loading LLM from: {self.LLM_MODEL_PATH}...")
        self.llm = Llama(
            model_path=self.LLM_MODEL_PATH,
            chat_format="llama-3", 
            n_ctx=4096,            # Context window size
            n_gpu_layers=-1,       # Offload all layers to GPU (set to 0 for CPU)
            verbose=False         
        )
        
        print("Query Pipeline initialized.")
        print(f"  - Query Embedder: {self.EMBEDDING_MODEL_NAME}")
        print(f"  - Vector DB: In-Memory ({len(self.chunks)} chunks)")
        print(f"  - LLM: {self.LLM_MODEL_NAME}")

    def run(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Runs the full RAG pipeline on a user query.
        """
        print(f"\nReceived query: \"{query}\"")
        
        # Step 1: Query Embedding
        query_vector = self._embed_query(query)
        
        # Step 2: Semantic Search (Retrieval)
        retrieved_chunks = self._semantic_search(query_vector, top_k)
        
        # Step 3: Prompt Augmentation (Build Chat Messages)
        chat_messages = self._create_chat_messages(query, retrieved_chunks)
        
        # Step 4: LLM Response Generation
        raw_llm_response = self._generate_response(chat_messages)
        
        # Step 5: Citation Parsing and Formatting
        final_output = self._parse_and_format_citations(raw_llm_response, retrieved_chunks)
        
        return final_output

    def _embed_query(self, query: str) -> np.ndarray:
        """
        Step 1: Converts the user's query into a vector.
        """
        print("Step 1: Embedding query...")
        vector = self.embedding_model.encode(query).astype(np.float32)
        return vector

    def _semantic_search(self, query_vector: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """
        Step 2: Retrieves the top-k most relevant chunks from our local chunks.
        """
        print(f"Step 2: Performing local semantic search (top-k={top_k})...")
        
        similarities = util.cos_sim(query_vector, self.chunk_vectors)[0]
        
        # Ensure k is not larger than the number of chunks
        k = min(top_k, len(self.chunks))

        # This finds the 'k' highest values and their indices from the tensor
        top_k_scores, top_k_indices = torch.topk(similarities, k=k)
        
        results = [self.chunks[i] for i in top_k_indices.tolist()]
        
        print(f"Found {len(results)} relevant chunks.")
        return results

    def _create_chat_messages(self, query: str, context_chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Step 3: Constructs the chat messages for the Llama 3.1 Instruct model.
        """
        print("Step 3: Building chat messages...")
        
        context_parts = []
        for i, chunk in enumerate(context_chunks):
            metadata = chunk['metadata']
            context_parts.append(
                f"Context {i+1} (Source: {metadata.get('source_file', 'N/A')}, Page: {metadata.get('page', 'N/A')}):\n"
                f"{chunk['text']}"
            )
        context = "\n\n".join(context_parts)
        
        system_prompt = """You are a helpful assistant. You must answer the user's question based *only* on the context provided below.
                            If the context does not contain the answer, you must state that the answer is not found in the provided documents.
                            For every piece of information you use, you *must* cite the context number it came from, like [Context 1], [Context 2], etc."""
        
        user_prompt = f"""---
                        CONTEXT:
                        {context}
                        ---

                        USER QUESTION:
                        {query}
                        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return messages

    def _generate_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Step 4: Sends the prompt to the LLM (Llama 3.1) to get an answer.
        """
        print("Step 4: Generating response from LLM...")
        
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=512,        
            temperature=0.1,       
            stop=["<|eot_id|>", "<|end_of_turn|>"] 
        )
        
        raw_response_text = response['choices'][0]['message']['content']
        print(f"LLM Raw Response: {raw_response_text}")
        return raw_response_text

    def _parse_and_format_citations(self, raw_response: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Step 5: Parses the LLM's raw text to create the final cited output.
        """
        print("Step 5: Formatting final answer...")
        
        citations = []
        for i, chunk in enumerate(chunks):
            metadata = chunk['metadata']
            citations.append({
                "id": f"Context {i+1}",
                "text": chunk['text'],
                "source": metadata.get('source_file', 'N/A'),
                "page": metadata.get('page', 'N/A')
            })

        return {
            "answer": raw_response,
            "citations": citations 
        }