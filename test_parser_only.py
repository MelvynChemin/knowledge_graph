
from parser import PDFParser
import sys

def test_parser(pdf_path):
    print(f"Testing parser on: {pdf_path}")
    parser = PDFParser()
    
    counts = {"text": 0, "image": 0}
    
    for chunk in parser.parse_pdf(pdf_path):
        counts[chunk['type']] += 1
        print(f"Found chunk: Type={chunk['type']}, Page={chunk['page']}")
        if chunk['type'] == 'image':
            print(f"  Image path: {chunk['content']}")
            print(f"  Context length: {len(chunk['context'])}")
            
    print("\nSummary:")
    print(f"Text chunks: {counts['text']}")
    print(f"Image chunks: {counts['image']}")

if __name__ == "__main__":
    pdf_path = "./images/sample.pdf"
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    test_parser(pdf_path)
