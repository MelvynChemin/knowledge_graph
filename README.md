# Knowledge Graph RAG System
Idexing part of the Rag-Anything Paper implementation.
A multimodal knowledge graph construction system that extracts entities and relationships from PDFs and images, storing them in Neo4j for retrieval-augmented generation (RAG) applications.

## Overview

This system processes documents and images to build a structured knowledge graph, enabling semantic search and relationship-based querying. It combines LLM-based entity extraction with vision-language models for multimodal content understanding.

## Features

- **PDF Processing**: Extracts text and images from PDF documents with paragraph-level chunking
- **Image Processing**: Analyzes standalone images using vision-language models
- **Entity Extraction**: Automatically identifies entities (people, organizations, concepts) and their relationships
- **Multimodal Graph**: Creates anchor nodes for images with linked entities
- **Neo4j Storage**: Stores knowledge graphs in Neo4j for efficient querying and visualization
- **Context-Aware**: Maintains contextual information between chunks for improved accuracy

## Architecture


```
Input File
    ↓
File Type Detection
    ↓
┌─────────────┬──────────────┐
│   PDF       │    Image     │
└─────────────┴──────────────┘
      ↓              ↓
  PDFParser    Direct Processing
      ↓              ↓
┌─────────────┬──────────────┐
│   Text      │    Image     │
└─────────────┴──────────────┘
      ↓              ↓
KnowledgeGraph  MultimodalProcessor
  Extractor          ↓
      ↓         VLM Analysis
      ↓              ↓
   Neo4j ←──────────┘
```

## Installation

### Prerequisites

- Python 3.8+
- Neo4j 4.0+ (running locally or remotely)
- Ollama (for LLM inference)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd knowledge_graph
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install pymupdf neo4j ollama pillow
```

4. Configure Neo4j:
   - Start Neo4j server
   - Update credentials in `pipeline.py` (default: `neo4j`/`yourpassword`)

5. Configure Ollama:
   - Install Ollama from [ollama.ai](https://ollama.ai)
   - Pull required models:
```bash
ollama pull gemma3:1b
ollama pull llava:7b-v1.5-q4_1
```

## Usage

### Process a PDF Document

```bash
python pipeline.py path/to/document.pdf
```

This will:
- Extract text chunks and images
- Generate entities and relationships from text
- Analyze images with vision-language models
- Store everything in Neo4j

### Process a Single Image

```bash
python pipeline.py path/to/image.jpg
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`

### Query the Knowledge Graph

Open Neo4j Browser at `http://localhost:7474` and run Cypher queries:

```cypher
// View all nodes and relationships
MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50

// Find specific entities
MATCH (n:Person) RETURN n

// View multimodal content
MATCH (n:MultimodalAnchor)-[r]-(e) RETURN n, r, e

// Search by entity name
MATCH (n {name: "Obama"}) RETURN n
```

## Project Structure

```
knowledge_graph/
├── parser.py                 # PDF parsing with PyMuPDF
├── pipeline.py              # Main orchestration and routing
├── multimodal_processing.py # Image analysis with VLM
├── neo4j_lightrag_storage.py # Neo4j operations
├── chat.py                  # LLM utilities and prompt templates
├── test_parser_only.py      # Parser testing utilities
└── images/                  # Sample input files
```

## Core Components

### PDFParser (`parser.py`)
- Extracts text blocks and images from PDFs
- Chunks text by paragraphs
- Saves images to `parsed_content/images/`
- Provides context from preceding chunks

### KnowledgeGraphBuilder (`pipeline.py`)
- Orchestrates the extraction pipeline
- Routes text to entity extraction
- Routes images to multimodal processing
- Manages Neo4j storage

### MultimodalProcessor (`multimodal_processing.py`)
- Analyzes images using vision-language models
- Generates detailed descriptions
- Extracts entities from visual content
- Creates contextual summaries

### Neo4jLightRAG (`neo4j_lightrag_storage.py`)
- Manages Neo4j connections
- Creates entity and relationship nodes
- Handles label sanitization
- Provides query utilities

## Configuration

### Neo4j Connection

Edit `pipeline.py`:
```python
builder = KnowledgeGraphBuilder(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="your_password"
)
```

### LLM Models

Edit `pipeline.py` and `multimodal_processing.py`:
```python
# Text extraction model
builder = KnowledgeGraphBuilder(model="gemma3:1b")

# Vision-language model
multimodal = MultimodalProcessor(vlm_model="llava:7b-v1.5-q4_1")
```

## Output

### Extracted Chunks
JSON files are saved to the project root:
- `knowledge_graph_data_chunk_*.json`: Entity and relationship data

### Images
Extracted images are saved to:
- `parsed_content/images/`: All extracted images from PDFs

### Neo4j Graph
Nodes are created with the following labels:
- `Person`, `Organization`, `Concept`, `Event`, etc. (from text)
- `MultimodalAnchor`: Anchor nodes for images
- Relationships: `BELONGS_TO`, custom relationship types

## Examples

### Example 1: Medical Research Paper

```bash
python pipeline.py medical_research.pdf
```

Extracts:
- Researchers and institutions
- Medical conditions and treatments
- Research findings and relationships
- Diagrams and charts with context

### Example 2: Political Analysis Image

```bash
python pipeline.py election_chart.jpg
```

Extracts:
- Political figures
- Events and dates
- Visual elements (charts, graphs)
- Contextual relationships

## Troubleshooting

### Neo4j Connection Issues
- Verify Neo4j is running: `neo4j status`
- Check credentials in `pipeline.py`
- Ensure port 7687 is accessible

### Ollama Model Issues
- Verify Ollama is running: `ollama list`
- Pull missing models: `ollama pull <model-name>`
- Check Ollama server: `http://localhost:11434`

### Import Errors
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`

## License

MIT License

## Contributing

Contributions are welcome. Please open an issue or submit a pull request.

## Acknowledgments

- PyMuPDF for PDF processing
- Neo4j for graph storage
- Ollama for local LLM inference
