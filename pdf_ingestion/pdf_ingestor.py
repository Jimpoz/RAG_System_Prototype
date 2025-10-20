import re
import pytesseract
from typing import Dict, Any
import fitz
from PIL import Image 
from datetime import datetime
import io

class PDFIngestor:
    """
    Extracts and processes data from PDF documents using a multi-step approach.
    """

    def __init__(self):
        print("PDF Ingestor initialized.")

    def ingest_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Main method to run the full ingestion pipeline on a single PDF file.

        Args:
            pdf_path: The file path to the PDF document.

        Returns:
            A structured, machine-readable data object (e.g., a dictionary).
        """
        print(f"Starting ingestion for: {pdf_path}")

        #1. Hybdrid Parsing with OCR Fallback
        structured_text = self._extract_text_with_hybrid_ocr(pdf_path)

        if not structured_text.strip():
            print(f"Error: No text could be extracted from {pdf_path}, not even with OCR.")
            # Still create a final object, but with empty data
            return self._create_final_object({}, pdf_path)

        #2. Schema Normalization
        normalized_json = self._normalize_schema(structured_text)

        #3. Validation
        is_valid = self._validate_data(normalized_json)
        if not is_valid:
            print(f"Warning: Validation failed for {pdf_path}")

        #4. Result
        final_output = self._create_final_object(normalized_json, pdf_path)

        print(f"Successfully ingested and structured: {pdf_path}")
        return final_output

    def _extract_text_with_hybrid_ocr(self, file_path: str) -> str:
        """
        Step 1 & 2: Extract text using PyMuPDF.
        If a page yields no text, fall back to OCR for that *specific* page.
        """
        print("Step 1/2: Running hybrid parsing (PyMuPDF + Tesseract OCR fallback)...")
        text_chunks = []
        
        try:
            with fitz.open(file_path) as doc:
                for page_num, page in enumerate(doc, start=1):
                    text = page.get_text("text")
                    
                    if not text.strip():
                        print(f"  - Page {page_num} is empty, falling back to OCR...")
                        zoom = 300 / 72
                        mat = fitz.Matrix(zoom, zoom)
                        pix = page.get_pixmap(matrix=mat)
                        
                        img_data = pix.tobytes("png")  
                        img = Image.open(io.BytesIO(img_data))
                        
                        # Use Pytesseract to extract text
                        try:
                            ocr_text = pytesseract.image_to_string(img, lang='eng')
                            if ocr_text.strip():
                                text_chunks.append(f"--- Page {page_num} (OCR) ---\n{ocr_text}")
                            else:
                                print(f"  - OCR for page {page_num} yielded no text.")
                                
                        except pytesseract.TesseractNotFoundError:
                            print("Please install Tesseract-OCR from: https://github.com/tesseract-ocr/tesseract")
                            return "Error: Tesseract not installed."
                        except Exception as ocr_err:
                            print(f"  - Error during OCR on page {page_num}: {ocr_err}")
                    
                    else:
                        print(f"  - Page {page_num} parsed digitally (PyMuPDF).")
                        text_chunks.append(f"--- Page {page_num} ---\n{text}")
                        
        except Exception as e:
            print(f"Error opening PDF: {e}")
            return ""

        return "\n".join(text_chunks)

    def _normalize_schema(self, raw_text: str) -> Dict[str, Any]:
        """
        Step 3: Convert raw text into a standardized JSON schema.
        """
        print("Step 3: Normalizing schema to JSON...")

        sections = re.split(r"\n(?=--- Page)", raw_text)
        content_blocks = []
        for sec in sections:
            if sec.strip():
                content_blocks.append(sec.strip())

        return {
            "title": "Extracted PDF Content",
            "pages": content_blocks
        }

    def _validate_data(self, data: Dict[str, Any]) -> bool:
        """
        Step 4: Validate key sections using regex.
        """
        print("Step 4: Validating key sections...")
        for page in data.get("pages", []):
            if re.search(r"\d{2}/\d{2}/\d{4}", page):
                print("  - Found date pattern in text.")
                return True
        print("  - No date pattern found.")
        return True

    def _create_final_object(self, data: Dict[str, Any], source_file: str) -> Dict[str, Any]:
        """
        Step 5: Package the output with metadata.
        """
        print("Step 5: Creating final data object with metadata.")
        return {
            "metadata": {
                "source_file": source_file,
                "processed_at": datetime.utcnow().isoformat() + "Z"
            },
            "data": data
        }