"""
PDF Parser using PyMuPDF (Fitz) for RAG-Anything.
Extracts text (chunked by paragraphs) and images (saved as PNGs).
"""

import os
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any, Generator

class PDFParser:
    """
    Parses PDF files, extracting text chunks and images.
    """

    def __init__(self, output_dir: str = "parsed_content"):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def parse_pdf(self, pdf_path: str) -> Generator[Dict[str, Any], None, None]:
        """
        Parse a PDF file and yield chunks (text or image).
        
        Args:
            pdf_path: Path to the PDF file.
            
        Yields:
            Dictionary containing chunk data.
        """
        doc = fitz.open(pdf_path)
        last_text_content = ""
        
        for page_num, page in enumerate(doc):
            # 1. Extract Text (Paragraphs)
            blocks = page.get_text("blocks")
            # blocks structure: (x0, y0, x1, y1, "lines in block", block_no, block_type)
            
            for block in blocks:
                block_type = block[6]
                
                if block_type == 0: # Text
                    text = block[4].strip()
                    if text:
                        chunk = {
                            "type": "text",
                            "content": text,
                            "page": page_num + 1,
                            "context": "" # Text chunks don't strictly need context from previous, but could have it
                        }
                        last_text_content = text # Update context for subsequent images
                        yield chunk
                        
                elif block_type == 1: # Image
                    # Note: get_text("blocks") might not give full image control, 
                    # so we iterate over images separately or handle them here if possible.
                    # Better approach for images in PyMuPDF is get_images() or iterating drawing commands.
                    # However, blocks with type 1 are images.
                    pass

            # 2. Extract Images (Robust method)
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                image_filename = f"page_{page_num + 1}_img_{img_index + 1}.{image_ext}"
                image_path = self.images_dir / image_filename
                
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                
                chunk = {
                    "type": "image",
                    "content": str(image_path.absolute()),
                    "page": page_num + 1,
                    "context": last_text_content # Context from the text immediately preceding
                }
                yield chunk

        doc.close()

if __name__ == "__main__":
    # Simple test
    import sys
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
        parser = PDFParser()
        for chunk in parser.parse_pdf(pdf_file):
            print(f"Type: {chunk['type']}, Page: {chunk['page']}")
            if chunk['type'] == 'text':
                print(f"Content: {chunk['content'][:50]}...")
            else:
                print(f"File: {chunk['content']}")
                print(f"Context: {chunk['context'][:50]}...")
            print("-" * 20)