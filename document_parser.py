# document_parser.py

from pdf2image import convert_from_path
import PyPDF2
from PIL import Image
import os

class DocumentParser:
    """Parse PDFs into sequential chunks with multimodal content"""
    
    def __init__(self):
        self.supported_types = ['text', 'image', 'table']
    
    def parse_pdf(self, pdf_path):
        """
        Parse PDF into sequential chunks
        
        Returns:
            List of dicts: [
                {'type': 'text', 'content': '...', 'position': 0, 'page': 0},
                {'type': 'image', 'content': Image, 'position': 1, 'page': 0},
                ...
            ]
        """
        chunks = []
        position = 0
        
        # Step 1: Extract text with PyPDF2
        with open(pdf_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                
                # Split into paragraphs (simple approach)
                paragraphs = text.split('\n\n')
                
                for para in paragraphs:
                    if para.strip():
                        chunks.append({
                            'type': 'text',
                            'content': para.strip(),
                            'position': position,
                            'page': page_num
                        })
                        position += 1
        
        # Step 2: Extract images
        # Note: This is simplified. In practice you'd need better positioning logic
        try:
            images = convert_from_path(pdf_path, dpi=150)
            
            for page_num, image in enumerate(images):
                # Very rough estimate of where image should go
                # In real implementation, use pdfplumber or similar for accurate positioning
                insert_pos = self._estimate_image_position(chunks, page_num)
                
                chunks.insert(insert_pos, {
                    'type': 'image',
                    'content': image,
                    'position': insert_pos,
                    'page': page_num
                })
                
                # Update positions after insertion
                for i in range(insert_pos + 1, len(chunks)):
                    chunks[i]['position'] = i
                    
        except Exception as e:
            print(f"Warning: Could not extract images: {e}")
        
        return chunks
    
    def _estimate_image_position(self, chunks, page_num):
        """Rough estimate of where to insert image in chunk sequence"""
        # Find chunks from this page
        page_chunks = [i for i, c in enumerate(chunks) if c['page'] == page_num]
        
        if page_chunks:
            # Insert in middle of page chunks
            return page_chunks[len(page_chunks) // 2]
        else:
            # Fallback
            return len(chunks)
    
    def save_images_from_chunks(self, chunks, output_dir="extracted_images"):
        """Save extracted images to disk and replace content with paths"""
        os.makedirs(output_dir, exist_ok=True)
        
        for chunk in chunks:
            if chunk['type'] == 'image' and isinstance(chunk['content'], Image.Image):
                # Save image
                img_path = os.path.join(
                    output_dir, 
                    f"image_pos_{chunk['position']}_page_{chunk['page']}.png"
                )
                chunk['content'].save(img_path)
                
                # Replace content with path
                chunk['content'] = img_path
                chunk['image_path'] = img_path
        
        return chunks


# Usage example
if __name__ == "__main__":
    parser = DocumentParser()
    
    # Parse a PDF
    chunks = parser.parse_pdf("example.pdf")
    
    # Save images to disk
    chunks = parser.save_images_from_chunks(chunks)
    
    print(f"Extracted {len(chunks)} chunks")
    print(f"Text chunks: {sum(1 for c in chunks if c['type'] == 'text')}")
    print(f"Image chunks: {sum(1 for c in chunks if c['type'] == 'image')}")