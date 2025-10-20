import os
import json
from datetime import datetime
import sys
from pdf_ingestion.pdf_ingestor import PDFIngestor
from chunking.semantic_chunker import SemanticChunker
from embedding.text_embedder import TextEmbedder
from query_pipeline.query_pipeline import QueryPipeline


if __name__ == "__main__":
    # Step 1: Ingest PDF
    ingestor = PDFIngestor()
    # PDF file to test
    pdf_file = r"C:\Users\jimpo\Downloads\HERALD_CV-NCOV-004-Protocol_test.pdf"

    try:
        # Check if file exists *before* trying to ingest
        if not os.path.exists(pdf_file):
            raise FileNotFoundError(f"Error: The file '{pdf_file}' was not found.")
        
        document_data = ingestor.ingest_pdf(pdf_file)
        print("\nSuccessfully ingested PDF document.")
    
    except FileNotFoundError as e:
        print(e)
        print("Please check the path and try again.")
        sys.exit()  # <-- Exit script if file is not found
    except Exception as e:
        print(f"An error occurred during PDF ingestion: {e}")
        sys.exit()  # <-- Exit script if ingestion fails

    # Save the structured data to a JSON file and add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json_path = os.path.join("outputs", f"ingested_{timestamp}.json")
    os.makedirs("outputs", exist_ok=True)
    try:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(document_data, f, ensure_ascii=False, indent=4)
        print(f"Successfully saved ingested data to {output_json_path}")
    except NameError:
        print("Error: 'document_data' does not exist. Cannot save to JSON.")
        sys.exit()
    except Exception as e:
        print(f"An error occurred while saving the JSON file: {e}")
        sys.exit()
        
    # Step 2: Chunk Document from the latest json file in outputs/
    chunker = SemanticChunker()
    output_dir = "outputs"
    
    # Find the latest JSON file in outputs/
    json_files = [f for f in os.listdir(output_dir) if f.endswith(".json")]
    
    if not json_files:
        print("Error: No JSON files found in outputs/ directory for chunking.")
        sys.exit()  # <-- Exit if no JSON files are found
    else:
        latest_file = max(json_files, key=lambda f: os.path.getmtime(os.path.join(output_dir, f)))
        latest_file_path = os.path.join(output_dir, latest_file)
        print(f"\nLoading {latest_file_path} for chunking...")    
        try:
            with open(latest_file_path, "r", encoding="utf-8") as f:
                structured_data = json.load(f)
            
            chunks = chunker.chunk_document(structured_data)
            
            # save chunks to a file for inspection and add timestamp
            chunks_output_path = os.path.join("outputs", f"chunks_{timestamp}.json")
            with open(chunks_output_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=4)
            
            print(f"\nCreated {len(chunks)} semantic chunks and saved to {chunks_output_path}.")
        
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {latest_file_path}.")
            sys.exit()
        except Exception as e:
            print(f"An error occurred during chunking: {e}")
            sys.exit()


    # Step 3: Embed Chunks
    embedder = TextEmbedder()
    try:
        embedded_chunks = embedder.embed_batch(chunks)
        
        # Save embedded chunks to a file for inspection
        embedded_output_path = os.path.join("outputs", f"embedded_chunks_{timestamp}.json")
        with open(embedded_output_path, "w", encoding="utf-8") as f:
            json.dump(embedded_chunks, f, ensure_ascii=False, indent=4)
        
        print(f"\nEmbedded chunks and saved to {embedded_output_path}.")
    except Exception as e:
        print(f"An error occurred during embedding: {e}")

    # Step 4: Test Query Pipeline
    print("\n--- Initializing Query Pipeline (This will load the LLM) ---")
    try:
        # --- FIX 1: Pass your REAL embedded_chunks to the pipeline ---
        pipeline = QueryPipeline(embedded_chunks)
        
        # --- FIX 2: Ask a REAL question about your PDF ---
        test_query = "Summarize this file"
        # Another good test: "What does ICH-E6 stand for?"
        
        # --- FIX 3: Call .run() and print the dictionary ---
        response_dict = pipeline.run(test_query, top_k=3)
        
        print("\n--- Final RAG Output ---")
        
        # This is how you print the final RAG answer
        print(f"Query: {test_query}")
        print(f"Answer: {response_dict.get('answer', 'No answer found.')}")
        
        print("\nCitations (Contexts provided to LLM):")
        if response_dict.get('citations'):
            for citation in response_dict['citations']:
                print(f"  --- {citation.get('id')} ---")
                print(f"    Source: {citation.get('source', 'N/A')}")
                print(f"    Page: {citation.get('page', 'N/A')}")
                print(f"    Text: {citation.get('text', '')[:150]}...")
        else:
            print("  No citations provided.")
            
    except Exception as e:
        print(f"An error occurred during querying: {e}")