# The following test script only runs the end-to-end RAG pipeline locally

import os
import json
from datetime import datetime
import sys
from pdf_ingestion.pdf_ingestor import PDFIngestor
from chunking.semantic_chunker import SemanticChunker
from embedding.text_embedder import TextEmbedder
from query_pipeline.query_pipeline import QueryPipeline
from response_validator.response_validator import ResponseValidator 
# from supabase import create_client, Client   

if __name__ == "__main__":

    # try:
    #     supabase_url = os.environ.get("SUPABASE_URL")
    #     supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
    #     
    #     if not supabase_url or not supabase_key:
    #         raise EnvironmentError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in environment.")
    #         
    #     supabase: Client = create_client(supabase_url, supabase_key)
    #     print("Successfully connected to Supabase.")
    # except Exception as e:
    #     print(f"Error connecting to Supabase: {e}")
    #     sys.exit()
    # --- (END SUPABASE LOGIC) ---

    # Step 1: Ingest PDF
    ingestor = PDFIngestor()
    pdf_file = r"C:\Users\jimpo\Downloads\HERALD_CV-NCOV-004-Protocol_test.pdf" # Update with your PDF path

    try:
        if not os.path.exists(pdf_file):
            raise FileNotFoundError(f"Error: The file '{pdf_file}' was not found.")
        
        document_data = ingestor.ingest_pdf(pdf_file)
        print("\nSuccessfully ingested PDF document.")
    
    except FileNotFoundError as e:
        print(e)
        sys.exit()
    except Exception as e:
        print(f"An error occurred during PDF ingestion: {e}")
        sys.exit()

    # Save ingested data locally (optional)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json_path = os.path.join("outputs", f"ingested_{timestamp}.json")
    os.makedirs("outputs", exist_ok=True)
    try:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(document_data, f, ensure_ascii=False, indent=4)
        print(f"Successfully saved ingested data to {output_json_path}")
    except Exception as e:
        print(f"Could not save local JSON: {e}")

    # Step 2: Chunk Document
    chunker = SemanticChunker()
    try:
        chunks = chunker.chunk_document(document_data)
        
        chunks_output_path = os.path.join("outputs", f"chunks_{timestamp}.json")
        with open(chunks_output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=4)
        
        print(f"\nCreated {len(chunks)} semantic chunks and saved to {chunks_output_path}.")
    except Exception as e:
        print(f"An error occurred during chunking: {e}")
        sys.exit()

    # Step 3: Embed Chunks
    embedder = TextEmbedder()
    try:
        embedded_chunks = embedder.embed_batch(chunks)
        
        # Convert numpy arrays to lists for JSON serialization
        chunks_to_save_locally = []
        for chunk in embedded_chunks:
            local_chunk = chunk.copy()
            if 'vector' in local_chunk and hasattr(local_chunk['vector'], 'tolist'):
                local_chunk['vector'] = local_chunk['vector'].tolist()
            chunks_to_save_locally.append(local_chunk)

        embedded_output_path = os.path.join("outputs", f"embedded_chunks_{timestamp}.json")
        with open(embedded_output_path, "w", encoding="utf-8") as f:
            json.dump(chunks_to_save_locally, f, ensure_ascii=False, indent=4)
        
        print(f"Embedded {len(embedded_chunks)} chunks and saved to {embedded_output_path}.")
        
    except Exception as e:
        print(f"An error occurred during embedding: {e}")
        sys.exit()

    # --- (2. SUPABASE LOGIC COMMENTED OUT) ---
    # print("\n--- Saving Embeddings to Supabase ---")
    # data_to_insert = []
    # for chunk in embedded_chunks:
    #     data_to_insert.append({
    #         'content': chunk.get('text'),
    #         'metadata': chunk.get('metadata'),
    #         'embedding': chunk.get('vector') 
    #     })
    # try:
    #     if data_to_insert:
    #         # ... supabase insert logic ...
    #         print(f"(SKIPPED) Would have upserted {len(data_to_insert)} chunks to Supabase.")
    #     else:
    #         print("No embedded chunks to insert.")
    # except Exception as e:
    #     print(f"Error inserting data into Supabase: {e}")
    #     sys.exit()
    # --- (END SUPABASE LOGIC) ---


    # Step 4: Test Query Pipeline
    print("\n--- Initializing Query Pipeline (In-Memory) ---")
    try:
        pipeline = QueryPipeline(embedded_chunks)
        
        # (Supabase pipeline commented out)
        # pipeline = QueryPipeline(supabase) 
        
        validator = ResponseValidator() # VALIDATOR
        
        test_query = "Summarize this file"
        response_dict = pipeline.run(test_query, top_k=3)

        print("\n--- VALIDATOR STEP ---")
        
        # The 'citations' key in the response_dict holds the
        # context chunks that were sent to the LLM.
        is_verified = validator.verify_citations(
            response_dict.get('answer', ''),
            response_dict.get('citations', [])
        )
        
        if not is_verified:
            print("WARNING: Response may be a hallucination. Review carefully.")
        else:
            print("Response appears grounded in context.")
            
        # --- (END VALIDATION STEP) ---

        print("\n--- Final RAG Output ---")
        print(f"Query: {test_query}")
        print(f"Answer: {response_dict.get('answer', 'No answer found.')}")

        if response_dict.get("citations"):
            citations_output_path = os.path.join("outputs", f"citations_{timestamp}.json")
            with open(citations_output_path, "w", encoding="utf-8") as f:
                json.dump(response_dict["citations"], f, ensure_ascii=False, indent=4)
            print(f"\nCitations saved successfully: {citations_output_path}")

            for citation in response_dict["citations"]:
                print(f"  --- {citation.get('id')} ---")
                print(f"    Source: {citation.get('source', 'N/A')}")
                print(f"    Page: {citation.get('page', 'N/A')}")
                print(f"    Text: {citation.get('text', '')[:150]}...")
        else:
            print("  No citations provided.")
            
    except Exception as e:
        print(f" An error occurred during querying: {e}")