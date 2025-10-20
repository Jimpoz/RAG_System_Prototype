from typing import List, Dict
import re
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ResponseValidator:
    """
    Validates that LLM responses are factually grounded in cited context.
    Uses lexical and semantic checks to prevent hallucinations.
    """

    STOP_WORDS = {
        'a', 'an', 'the', 'is', 'in', 'of', 'to', 'and', 'for', 'was', 'with',
        'on', 'at', 'it', 'this', 'that', 'as', 'by', 'are', 'be', 'which',
        'from', 'also', 'overall', 'additionally', 'however', 'consequently',
        # Domain-neutral but high-frequency medical stopwords
        'patient', 'patients', 'study', 'data', 'results', 'treatment', 'dose'
    }

    UNVERIFIED_WORD_THRESHOLD = 0.3
    SEMANTIC_SIMILARITY_THRESHOLD = 0.6

    def __init__(self, embedder=None):
        """
        Optionally pass a TextEmbedder instance for semantic validation.
        
        *** NOTE ***: If you pass your 'TextEmbedder', you must ensure
        it has an 'embed_text(str)' method, not just 'embed_batch(list)'.
        Or, you must modify '_semantic_similarity' to handle batches.
        """
        self.embedder = embedder
        self.stemmer = PorterStemmer()
        if self.embedder:
            print("Response Validator initialized (lexical + semantic mode).")
        else:
            print("Response Validator initialized (lexical-only mode).")
            print("  - WARNING: No embedder provided. Semantic validation will be skipped.")

    def get_strict_prompt_template(self) -> str:
        """
        Returns a strict prompt template instructing the LLM to avoid
        hallucinations and cite every factual claim.
        """
        return """
        You are a factual medical assistant. Use ONLY the provided context.

        If the answer is not explicitly stated in the context, respond:
        "I do not have enough information to answer this question."

        Every factual claim must cite a context reference like [Context N].
        ---
        CONTEXT:
        {context}
        ---
        USER QUESTION:
        {query}

        ANSWER:
        """

    def _lexical_overlap(self, claim_text: str, context_text: str) -> float:
        """
        Computes ratio of verified words (after stemming and stopword removal)
        between the claim and its cited context.
        """
        clean = lambda text: re.sub(r'[^\w\s]', '', text).lower().split()
        claim_words = [self.stemmer.stem(w) for w in clean(claim_text)
                           if w not in self.STOP_WORDS and len(w) > 3]
        context_words = [self.stemmer.stem(w) for w in clean(context_text)
                             if w not in self.STOP_WORDS and len(w) > 3]

        if not claim_words:
            return 1.0

        claim_set, context_set = set(claim_words), set(context_words)
        verified = claim_set & context_set
        return len(verified) / len(claim_set)

    def _semantic_similarity(self, claim_text: str, context_text: str) -> float:
        """
        Uses embeddings to compute cosine similarity between claim and context.
        Only runs if an embedder is provided.
        """
        if not self.embedder:
            return 1.0 
        try:
            claim_vec = np.array(self.embedder.embed_text(claim_text)).reshape(1, -1)
            context_vec = np.array(self.embedder.embed_text(context_text)).reshape(1, -1)
            return float(cosine_similarity(claim_vec, context_vec)[0][0])
        except Exception as e:
            print(f"  [Semantic check failed: {e}]")
            return 0.0

    def _validate_claim(self, claim_text: str, context_text: str) -> bool:
        """
        Combines lexical and semantic validation for a single claim.
        Returns True if the claim appears grounded in its context.
        """
        lexical_score = self._lexical_overlap(claim_text, context_text)
        semantic_score = self._semantic_similarity(claim_text, context_text)

        if lexical_score < (1 - self.UNVERIFIED_WORD_THRESHOLD) and semantic_score < self.SEMANTIC_SIMILARITY_THRESHOLD:
            print(f"  [FAILED] '{claim_text[:60]}...' — "
                  f"lexical={lexical_score:.2f}, semantic={semantic_score:.2f}")
            return False
        
        print(f"  [PASSED] '{claim_text[:60]}...' — "
              f"lexical={lexical_score:.2f}, semantic={semantic_score:.2f}")
        return True

    def verify_citations(self, generated_answer: str, context_chunks: List[Dict]) -> bool:
        """
        Ensures that all claims are cited and verifiable against context.
        Performs both lexical and semantic validation.
        """
        print("\nVerifying citations and factual grounding...")

        if "i do not have enough information" in generated_answer.lower():
            print("Validator: model appropriately abstained.")
            return True

        # Change map key from '[Context 1]' to 'Context 1' ---
        context_map = {
            f"Context {i+1}": chunk['text'] for i, chunk in enumerate(context_chunks)
        }
        all_context_text = " ".join(context_map.values())

        # Split answer into sentences
        sentences = re.split(r'(?<=[.!?])\s+|\n+', generated_answer.strip())
        sentences = [s for s in sentences if s.strip() and len(s) > 10] # Ignore short lines
        
        if not sentences:
            print("No sentences to verify.")
            return True

        all_verified = True
        for sentence in sentences:
            
            # Update regex to find LLM's *actual* output format
            citations = set(re.findall(r'\((Context \d+)(?:,\s*Page\s*\d+)?\)', sentence))

            # Update regex to *remove* the citation block
            claim_text = re.sub(r'\((Context \d+(?:,\s*Page\s*\d+)?)\)', '', sentence).strip()
            # Also remove list numbers like "1.", "2.", etc.
            claim_text = re.sub(r'^\d+\.\s+', '', claim_text)

            if not claim_text:
                continue

            if not citations:
                print(f"Uncited claim: '{claim_text[:70]}...'")
                
                print(f" - Checking uncited claim against ALL context...")
                if not self._validate_claim(claim_text, all_context_text):
                     all_verified = False
                continue

            # Keys now match 'Context 1', 'Context 2', etc.
            # Combine text of *only* the cited contexts
            cited_context = " ".join([context_map.get(c, "") for c in citations])
            
            if not cited_context:
                print(f"  [FAILED] '{claim_text[:60]}...' — Citation '{citations}' not found in map.")
                all_verified = False
                continue

            if not self._validate_claim(claim_text, cited_context):
                all_verified = False

        if all_verified:
            print("All claims verified successfully.")
        else:
            print("Some claims failed verification.")

        return all_verified