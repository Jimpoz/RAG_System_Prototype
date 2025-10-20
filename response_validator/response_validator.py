# response_validator.py
# This class provides tools to prevent LLM hallucinations.

from typing import List, Dict, Any

class ResponseValidator:
    """
    Implements strategies to prevent hallucinations, such as stricter
    prompting and citation verification, as described in section 3.2.
    """
    def __init__(self):
        print("Response Validator initialized.")

    def get_strict_prompt_template(self) -> str:
        """
        Returns a "stricter" prompt template that explicitly instructs
        the LLM to not answer if the context is insufficient.
        """
        
        # This prompt is more forceful than the one in 2.5
        strict_prompt = """
You are a factual medical assistant. Answer the user's question using ONLY
the provided context. Do not use any outside knowledge.

If the answer to the question is not contained within the context, you
MUST respond with the exact sentence:
"I do not have enough information to answer this question."

You must cite the source and page for every claim you make, taken from the
context.

---
CONTEXT:
{context}
---

USER QUESTION:
{query}

ANSWER:
"""
        return strict_prompt

    def verify_citations(self, generated_answer: str, context_chunks: List[Dict]) -> bool:
        """
        A post-processing step to verify that statements in the answer
        map back to the retrieved text chunks.
        
        Args:
            generated_answer: The text answer from the LLM.
            context_chunks: The list of chunks provided to the LLM.

        Returns:
            True if all claims are verified, False otherwise.
        """
        print("Verifying citations...")
        
        # This is a very complex NLP task (Natural Language Inference).
        # A simple placeholder checks if keywords from the answer
        # are present in the context.
        
        all_context_text = " ".join([c['text'] for c in context_chunks]).lower()
        answer_words = set(generated_answer.lower().split())
        
        unverified_words = []
        for word in answer_words:
            if word not in all_context_text and len(word) > 4: # Ignore stop words
                unverified_words.append(word)

        if len(unverified_words) > 0:
            print(f"Verification FAILED. Unverified terms: {unverified_words}")
            return False
        
        print("Verification PASSED. All claims appear grounded in context.")
        return True

# Example usage (if run as a script)
if __name__ == "__main__":
    validator = ResponseValidator()
    
    print("\n--- Strict Prompt Template ---")
    print(validator.get_strict_prompt_template())

    print("\n--- Testing Verification (FAIL) ---")
    context = [{"text": "Metformin is a drug.", "metadata": {}}]
    answer = "Metformin is a useful drug developed in France."
    validator.verify_citations(answer, context)
    
    print("\n--- Testing Verification (PASS) ---")
    context = [{"text": "Metformin is a drug.", "metadata": {}}]
    answer = "Metformin is a drug."
    validator.verify_citations(answer, context)