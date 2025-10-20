from typing import List, Dict, Any

class SemanticChunker:
    MODEL_NAME = "BAAI/bge-m3"
    MAX_TOKENS = 8192
    
    # --- UPDATED ---
    # Use a LARGER semantic unit size (in words) as a fallback
    # 100 words is too small. Let's aim for ~500 words, or ~2000 tokens.
    # We will use this ONLY if a paragraph is larger than MAX_TOKENS.
    FALLBACK_WINDOW_SIZE_WORDS = 500
    FALLBACK_OVERLAP_RATIO = 0.15

    def __init__(self):
        print(f"Semantic Chunker initialized for {self.MODEL_NAME} (Max Tokens: {self.MAX_TOKENS}).")
        print(f"Chunking by paragraph, with a fallback window of {self.FALLBACK_WINDOW_SIZE_WORDS} words.")

    def _split_oversized_paragraph(self, text_content: str, current_page_number: str, 
                                     section_title: str, source_file: str, 
                                     chunks: List[Dict[str, Any]], paragraph_index: int):
        """
        Fallback function to split a single paragraph that is too large
        using a sliding window of WORDS.
        """
        print(f"Warning: Paragraph {paragraph_index} on page {current_page_number} is too large. Splitting with sliding window...")
        words = text_content.split()
        
        overlap_words = int(self.FALLBACK_WINDOW_SIZE_WORDS * self.FALLBACK_OVERLAP_RATIO)
        step_size = self.FALLBACK_WINDOW_SIZE_WORDS - overlap_words

        i = 0
        while i < len(words):
            chunk_words = words[i:i + self.FALLBACK_WINDOW_SIZE_WORDS]
            chunk_text = " ".join(chunk_words)
            token_count = len(chunk_words)

            if token_count == 0:
                i += step_size
                continue
            
            # This chunk is still part of the *same* oversized paragraph
            chunks.append({
                "text": f"## {section_title} (Paragraph {paragraph_index}, Part {i//step_size + 1})\n{chunk_text}",
                "metadata": {
                    "source_file": source_file,
                    "section": section_title,
                    "page": current_page_number,
                    "token_count": token_count,
                    "chunk_index": len(chunks)
                }
            })
            i += step_size

    def chunk_document(self, structured_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        print("Starting semantic chunking by PARAGRAPH...")
        chunks = []
        
        doc_data = structured_data.get("data", {})
        source_file = structured_data.get("metadata", {}).get("source_file", "unknown")

        for section_title, content in doc_data.items():
            
            page_content_list = []
            
            if isinstance(content, str):
                page_content_list.append( (content, "1") )
            
            elif isinstance(content, list):
                # Handle "pages" section
                for page_string in content:
                    if not isinstance(page_string, str):
                        continue

                    # Extract Page Number
                    current_page_number = "unknown"
                    page_lines = page_string.split('\n')
                    first_line = page_lines[0].strip()
                    
                    if first_line.startswith("--- Page ") and first_line.endswith(" ---"):
                        current_page_number = first_line.split(" ")[2]
                        page_content = "\n".join(page_lines[1:])
                    else:
                        page_content = page_string
                    
                    page_content_list.append( (page_content, current_page_number) )
            else:
                print(f"Skipping section '{section_title}': unsupported content type ({type(content)}).")
                continue

            for page_text, page_num in page_content_list:
                # Split this single page's content into paragraphs
                page_paragraphs = page_text.split('\n\n')
                
                # Loop through paragraphs
                for i, para_text in enumerate(page_paragraphs):
                    para_text = para_text.strip()
                    
                    if not para_text or (para_text.startswith("--- Page ") and para_text.endswith(" ---")):
                        continue
                    
                    chunk_text = f"## {section_title}\n{para_text}"
                    token_count = len(chunk_text.split())

                    if token_count == 0:
                        continue
                    
                    # If oversized paragraph, use fallback splitting
                    if token_count > self.MAX_TOKENS:
                        self._split_oversized_paragraph(
                            text_content=para_text,
                            current_page_number=page_num,
                            section_title=section_title,
                            source_file=source_file,
                            chunks=chunks,
                            paragraph_index=i
                        )
                    else:
                        # if standard sized then just add normally
                        chunks.append({
                            "text": chunk_text,
                            "metadata": {
                                "source_file": source_file,
                                "section": section_title,
                                "token_count": token_count,
                                "page": page_num,
                                "paragraph_index_on_page": i
                            }
                        })

        print(f"Successfully reated {len(chunks)} semantic chunks.")
        return chunks