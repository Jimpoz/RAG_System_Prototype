# RAG System Prototype

**Author:** Jinpeng Zhang  

---

## Overview

This project presents the design of a **Retrieval-Augmented Generation (RAG)** system that takes a user query and returns a **human-language answer with accurate citations** from stored PDF documents (medical reports, trial protocols, etc...).

**Goal:**  
`Query → Human-language Response + Citations (highlighted snippets)`

---

## System Architecture

The system is divided into the following core components:

### 1. PDF Ingestion
Extracting structured data from unstructured PDFs using a multi-stage process:
- **Structured Parsing:** `PyMuPDF4LLM` for semantic Markdown/JSON output.  
- **OCR Fallback:** `pytesseract` for scanned or image-based PDFs.  
- **Schema Normalization:** NLP-based mapping to a consistent JSON schema.  
- **Validation:** Regex and semantic checks for key sections.  
- **Final Output:** Machine-readable data with metadata (page, section, source).

### 2. Chunking Strategy
The system employs **semantic chunking** based on document structure (e.g., paragraphs, sections).  
Using the **BAAI/bge-m3** embedding model (8192-token window) enables:
- Preservation of full medical context  
- Higher-quality embeddings  
- Improved retrieval accuracy  

### 3. Embedding
**Model:** `BAAI/bge-m3`  
- **Performance:** State-of-the-art on Massive Text Embedding Benchmark
- **Context Window:** 8192 tokens  
- **Vector Size:** 1024 dimensions  

**Process:**
1. Input structured text chunk  
2. Pass to embedding model  
3. Output 1024-dimension vector representation  

### 4. Query Pipeline & Response Generation
**Steps:**
1. **Query Embedding:** Convert user question into semantic vector.  
2. **Semantic Search:** Retrieve top-k relevant chunks via Supabase `pgvector`.  
3. **Prompt Augmentation:** Construct a detailed, context-rich prompt.  
4. **LLM Generation:** Produce answer using a model such as **Llama 3.1**.  
5. **Citation Formatting:** Link each claim to its source and highlight in the output.  

**Example Prompt:**
“You are a helpful medical assistant. Using only the context below, answer the user's question. You must cite the source for every claim you make, referencing the ‘source’ and ‘page’.”

---

### Preventing Hallucinations
LLMs often can generate responses that are not true or reliable therefore in order to prevent these hallucinations we can implement:
- **Stricter Prompting:** Instruct LLMs to answer “I do not have enough information” when uncertain.  
- **Citation Verification:** Ensure all generated claims map back to retrieved context.  
