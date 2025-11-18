# test_multimodal.py

import os
from pipeline import (
    TextualGraphBuilder, 
    MultimodalGraphBuilder,
    process_text_chunk,
    process_multimodal_chunk,
    process_full_document,
    get_surrounding_context
)
from document_parser import DocumentParser
from multimodal_processing import MultimodalProcessor
from neo4j_lightrag_storage import Neo4jLightRAG
from PIL import Image, ImageDraw, ImageFont

# ============================================================================
# TEST DATA GENERATION
# ============================================================================

def create_test_image(filename, text="Test Chart"):
    """Create a simple test image with text"""
    img = Image.new('RGB', (400, 300), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw some simple chart-like elements
    draw.rectangle([50, 50, 350, 250], outline='black', width=2)
    draw.line([50, 150, 350, 150], fill='blue', width=2)
    draw.line([200, 50, 200, 250], fill='blue', width=2)
    
    # Add text
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    draw.text((150, 120), text, fill='black', font=font)
    
    # Save
    os.makedirs("test_data", exist_ok=True)
    img.save(filename)
    print(f"✅ Created test image: {filename}")
    return filename

def create_test_chunks():
    """Create test chunks simulating a parsed document"""
    
    # Create test images
    img1 = create_test_image("test_data/figure1.png", "Accuracy: 95%")
    img2 = create_test_image("test_data/figure2.png", "Model Comparison")
    
    chunks = [
        {
            'type': 'text',
            'content': 'Dr. Sarah Chen is a cardiologist at Stanford Medical Center.',
            'position': 0,
            'page': 0
        },
        {
            'type': 'text',
            'content': 'In 2024, she published groundbreaking research on using AI to diagnose arrhythmias. Her models achieved 95% accuracy.',
            'position': 1,
            'page': 0
        },
        {
            'type': 'image',
            'content': img1,
            'position': 2,
            'page': 0
        },
        {
            'type': 'text',
            'content': 'As shown in the figure, the AI model outperformed traditional methods significantly.',
            'position': 3,
            'page': 0
        },
        {
            'type': 'text',
            'content': 'Dr. Chen collaborates with Dr. Michael Torres, a data scientist at MIT.',
            'position': 4,
            'page': 1
        },
        {
            'type': 'image',
            'content': img2,
            'position': 5,
            'page': 1
        },
        {
            'type': 'text',
            'content': 'The research was funded by the National Heart Institute and could revolutionize cardiac care.',
            'position': 6,
            'page': 1
        }
    ]
    
    return chunks

# ============================================================================
# TEST 1: TEXT-ONLY PROCESSING
# ============================================================================

def test_text_processing():
    """Test basic text knowledge graph extraction"""
    print("\n" + "="*60)
    print("TEST 1: TEXT-ONLY PROCESSING")
    print("="*60 + "\n")
    
    neo4j = Neo4jLightRAG(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="testpass"
    )
    
    text_builder = TextualGraphBuilder()
    
    test_text = """Dr. Sarah Chen is a cardiologist at Stanford Medical Center who specializes in treating heart disease. 
    In 2024, she published groundbreaking research on using AI to diagnose arrhythmias early. Her work showed that 
    machine learning models can detect irregular heartbeats with 95% accuracy."""
    
    print("📝 Processing text chunk...")
    triples, key_values = process_text_chunk(test_text, 1, neo4j, text_builder)
    
    print("\n✅ Extracted Entities:")
    for entity in triples.get('entities', []):
        print(f"  - {entity['name']} ({entity['type']})")
    
    print("\n✅ Extracted Relationships:")
    for rel in triples.get('relationships', []):
        print(f"  - {rel['source']} -[{rel['relation']}]-> {rel['target']}")
    
    print("\n✅ Test 1 Complete! Check Neo4j at http://localhost:7474")
    return neo4j

# ============================================================================
# TEST 2: CONTEXT EXTRACTION
# ============================================================================

def test_context_extraction():
    """Test extracting context around multimodal elements"""
    print("\n" + "="*60)
    print("TEST 2: CONTEXT EXTRACTION")
    print("="*60 + "\n")
    
    chunks = create_test_chunks()
    
    # Test context extraction for image at position 2
    image_idx = 2
    context = get_surrounding_context(chunks, image_idx, delta=2)
    
    print(f"📷 Image at position {image_idx}")
    print(f"📝 Surrounding context (delta=2):")
    print("-" * 60)
    print(context)
    print("-" * 60)
    
    print("\n✅ Test 2 Complete!")

# ============================================================================
# TEST 3: MULTIMODAL PROCESSING (WITHOUT VLM)
# ============================================================================

def test_multimodal_basic():
    """Test multimodal processing without actual VLM calls"""
    print("\n" + "="*60)
    print("TEST 3: MULTIMODAL PROCESSING (MOCK VLM)")
    print("="*60 + "\n")
    
    neo4j = Neo4jLightRAG(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="testpass"
    )
    
    chunks = create_test_chunks()
    mm_builder = MultimodalGraphBuilder()
    
    # Process just the first image with context
    image_idx = 2
    image_chunk = chunks[image_idx]
    context = get_surrounding_context(chunks, image_idx, delta=2)
    
    print(f"📷 Processing image at position {image_idx}")
    print(f"📝 Context length: {len(context)} chars")
    
    mm_info = process_multimodal_chunk(
        image_chunk,
        image_idx,
        context,
        neo4j,
        mm_builder
    )
    
    print(f"\n✅ Created multimodal anchor: MM_Anchor_{image_idx}")
    print(f"   Detailed description: {mm_info['detailed_description'][:100]}...")
    
    print("\n✅ Test 3 Complete! Check Neo4j for MM_Anchor nodes")
    return neo4j

# ============================================================================
# TEST 4: FULL DOCUMENT PROCESSING
# ============================================================================

def test_full_document():
    """Test processing entire document with text and images"""
    print("\n" + "="*60)
    print("TEST 4: FULL DOCUMENT PROCESSING")
    print("="*60 + "\n")
    
    neo4j = Neo4jLightRAG(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="testpass"
    )
    
    chunks = create_test_chunks()
    
    print(f"📄 Processing document with {len(chunks)} chunks:")
    print(f"   - Text chunks: {sum(1 for c in chunks if c['type'] == 'text')}")
    print(f"   - Image chunks: {sum(1 for c in chunks if c['type'] == 'image')}")
    
    process_full_document(chunks, neo4j, delta=2)
    
    print("\n✅ Test 4 Complete! Full document processed")
    print("   Check Neo4j at http://localhost:7474")
    print("   Query: MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50")
    
    return neo4j

# ============================================================================
# TEST 5: MULTIMODAL PROCESSOR WITH REAL VLM (if available)
# ============================================================================

def test_vlm_processing():
    """Test actual VLM processing if llava is available"""
    print("\n" + "="*60)
    print("TEST 5: VLM PROCESSING (requires llava model)")
    print("="*60 + "\n")
    
    try:
        processor = MultimodalProcessor(vlm_model="llava:7b-v1.5-q4_1")
        
        # Create test image
        img_path = create_test_image("test_data/vlm_test.png", "AI Accuracy: 95%")
        
        context = """This section discusses AI model performance in medical diagnosis. 
        The research shows that machine learning can detect arrhythmias with high accuracy."""
        
        print("🤖 Calling VLM (this may take a moment)...")
        result = processor.process_image_with_context(img_path, context)
        
        print("\n📝 Detailed Description:")
        print("-" * 60)
        print(result['detailed_description'])
        print("-" * 60)
        
        print("\n📊 Entity Summary:")
        print("-" * 60)
        print(result['entity_summary'])
        print("-" * 60)
        
        # Try extracting entities
        entities = processor.extract_entities_from_description(
            result['detailed_description']
        )
        
        print("\n✅ Extracted Entities from VLM description:")
        for entity in entities.get('entities', []):
            print(f"   - {entity['name']} ({entity['type']})")
        
        print("\n✅ Test 5 Complete!")
        
    except Exception as e:
        print(f"⚠️  VLM test skipped: {e}")
        print("   Make sure you have llava model: ollama pull llava:7b")

# ============================================================================
# TEST 6: DOCUMENT PARSER TEST
# ============================================================================

def test_document_parser():
    """Test PDF parsing (if you have a test PDF)"""
    print("\n" + "="*60)
    print("TEST 6: DOCUMENT PARSER")
    print("="*60 + "\n")
    
    # Check if test PDF exists
    test_pdf = "test_data/sample.pdf"
    
    if not os.path.exists(test_pdf):
        print("⚠️  No test PDF found at test_data/sample.pdf")
        print("   Skipping this test. To run it:")
        print("   1. Place a PDF at test_data/sample.pdf")
        print("   2. Run this test again")
        return
    
    parser = DocumentParser()
    
    print(f"📄 Parsing PDF: {test_pdf}")
    chunks = parser.parse_pdf(test_pdf)
    
    print(f"\n✅ Extracted {len(chunks)} chunks:")
    print(f"   - Text chunks: {sum(1 for c in chunks if c['type'] == 'text')}")
    print(f"   - Image chunks: {sum(1 for c in chunks if c['type'] == 'image')}")
    
    # Save images
    chunks = parser.save_images_from_chunks(chunks, "test_data/extracted")
    
    print(f"\n✅ Saved images to test_data/extracted/")
    
    # Show first few chunks
    print("\n📝 First 3 chunks:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n   Chunk {i} ({chunk['type']}):")
        if chunk['type'] == 'text':
            print(f"   {chunk['content'][:100]}...")
        else:
            print(f"   Image at position {chunk['position']}")

# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all tests in sequence"""
    print("\n" + "="*60)
    print("RUNNING ALL RAG-ANYTHING TESTS")
    print("="*60)
    
    tests = [
        ("Text Processing", test_text_processing),
        ("Context Extraction", test_context_extraction),
        ("Multimodal Basic", test_multimodal_basic),
        ("Full Document", test_full_document),
        ("VLM Processing", test_vlm_processing),
        ("Document Parser", test_document_parser),
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            test_func()
            results[name] = "✅ PASSED"
        except Exception as e:
            results[name] = f"❌ FAILED: {str(e)}"
            print(f"\n❌ Test failed: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for name, result in results.items():
        print(f"{name}: {result}")
    
    print("\n" + "="*60)
    print("To view results in Neo4j:")
    print("1. Open http://localhost:7474")
    print("2. Run: MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50")
    print("3. Look for MM_Anchor nodes (multimodal anchors)")
    print("="*60)

def run_single_test(test_name):
    """Run a specific test"""
    tests = {
        "text": test_text_processing,
        "context": test_context_extraction,
        "multimodal": test_multimodal_basic,
        "full": test_full_document,
        "vlm": test_vlm_processing,
        "parser": test_document_parser,
    }
    
    if test_name in tests:
        tests[test_name]()
    else:
        print(f"Unknown test: {test_name}")
        print(f"Available tests: {', '.join(tests.keys())}")

# ============================================================================
# INTERACTIVE MENU
# ============================================================================

def interactive_menu():
    """Interactive test menu"""
    while True:
        print("\n" + "="*60)
        print("RAG-ANYTHING TEST SUITE")
        print("="*60)
        print("1. Test Text Processing Only")
        print("2. Test Context Extraction")
        print("3. Test Multimodal Basic (mock VLM)")
        print("4. Test Full Document Processing")
        print("5. Test VLM Processing (requires llava)")
        print("6. Test Document Parser (requires PDF)")
        print("7. Run ALL tests")
        print("8. Exit")
        print("="*60)
        
        choice = input("\nSelect test (1-8): ").strip()
        
        if choice == "1":
            test_text_processing()
        elif choice == "2":
            test_context_extraction()
        elif choice == "3":
            test_multimodal_basic()
        elif choice == "4":
            test_full_document()
        elif choice == "5":
            test_vlm_processing()
        elif choice == "6":
            test_document_parser()
        elif choice == "7":
            run_all_tests()
        elif choice == "8":
            print("Goodbye!")
            break
        else:
            print("Invalid choice, try again")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Check if specific test requested
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        run_single_test(test_name)
    else:
        # Run interactive menu
        interactive_menu()