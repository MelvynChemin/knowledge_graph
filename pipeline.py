"""
Knowledge Graph Builder

A system for constructing knowledge graphs from text using LLM-based entity and relationship extraction.

Architecture:
1. Chunking: Split documents into manageable pieces
2. Entity Extraction: Identify key entities (people, places, organizations, etc.)
3. Relationship Extraction: Find connections between entities
4. Graph Storage: Store in Neo4j graph database
5. Indexing: Create searchable key-value index for entities
"""

import json
from typing import Dict, List, Tuple, Any
import ollama
from chat import PromptTemplate, ChatOllamaMini
from neo4j_lightrag_storage import load_lightrag_data, Neo4jLightRAG


class PromptTemplates:
    """Storage for prompt templates used in knowledge graph extraction"""
    
    @staticmethod
    def get_entity_extraction_prompt() -> PromptTemplate:
        """
        Prompt for extracting entities and relationships from text.
        
        Returns entities as concrete nouns and relationships as action verbs
        connecting two entities.
        """
        return PromptTemplate.from_messages([
            ("system",
             """You are an expert knowledge graph builder. Extract entities and relationships from text.

**ENTITIES** are concrete things (nouns):
- People: Dr. Sarah Chen, Dr. Michael Torres
- Organizations: Stanford Medical Center, MIT, National Heart Institute  
- Medical Conditions: Heart Disease, Arrhythmias
- Technologies: AI, Machine Learning Models
- Concepts: Research, Diagnostic Tools

**RELATIONSHIPS** connect two entities with action verbs:
- works_at, specializes_in, researches, collaborates_with, develops, funds, diagnoses, detects

**RULES:**
1. Extract only entities explicitly mentioned in the text
2. Do NOT extract properties (like "cardiologist" or "95%") as separate entities
3. Ensure relationship direction is correct (who does what to whom)
4. Output as JSON array of triples
**ADDITIONAL RULES:**
- DO NOT extract percentages, numbers, or statistics as entities
- Ensure relationship directions are correct (check who does what to whom)
- Include funding organizations explicitly mentioned
- Extract all people and organizations mentioned by name

**EXAMPLE OUTPUT FORMAT:**
```json
{
  "entities": [
    {"name": "Dr. Sarah Chen", "type": "Person"},
    {"name": "Stanford Medical Center", "type": "Organization"}
  ],
  "relationships": [
    {"source": "Dr. Sarah Chen", "relation": "works_at", "target": "Stanford Medical Center"},
    {"source": "Dr. Sarah Chen", "relation": "specializes_in", "target": "Heart Disease"}
  ]
}
```

Extract entities and relationships from the following text."""),
            ("user", "{text}"),          
        ])
    
    @staticmethod
    def get_index_generation_prompt() -> PromptTemplate:
        """
        Prompt for generating searchable key-value index for entities.
        
        Creates summaries that include context and relationships for each entity.
        """
        return PromptTemplate.from_messages([
            ("system",
            """You are creating a searchable index for a knowledge graph database.

For each entity, generate key-value pairs:

**ENTITY INDEX:**
- Key: The entity name (e.g., "Dr. Sarah Chen")
- Value: A 2-3 sentence summary containing:
  * What the entity is
  * Key facts and context from the text
  * Related entities and relationships

**ENTITY INDEX RULES:**
- Only include facts explicitly stated in the text
- Do not add general knowledge or hallucinate details

**EXAMPLE OUTPUT:**
```json
{
  "entity_index": [
    {
      "key": "Dr. Sarah Chen",
      "value": "Cardiologist at Stanford Medical Center who specializes in treating heart disease. In 2024, published research on AI diagnosis of arrhythmias achieving 95% accuracy. Collaborates with Dr. Michael Torres from MIT."
    },
    {
      "key": "Arrhythmias", 
      "value": "Irregular heartbeats that can be diagnosed using AI/machine learning with 95% accuracy according to 2024 research by Dr. Sarah Chen."
    }
  ]
}
```

Generate the key-value index from the provided entities, relationships, and original text."""),
            ("user", "Entities and Relationships:{question} Original Text:{text}"), 
        ])


class KnowledgeGraphExtractor:
    """Handles LLM-based extraction of entities and relationships from text"""
    
    def __init__(self, model: str = "gemma3:1b", temperature: float = 0.0, 
                 base_url: str = "http://localhost:11434"):
        """
        Initialize the extractor with LLM configuration.
        
        Args:
            model: Ollama model name to use
            temperature: LLM temperature (0.0 for deterministic)
            base_url: Ollama server URL
        """
        self.llm = ChatOllamaMini(model=model, temperature=temperature, base_url=base_url)
        self.entity_prompt = PromptTemplates.get_entity_extraction_prompt()
        self.index_prompt = PromptTemplates.get_index_generation_prompt()
    
    def extract_entities_and_relationships(self, text: str) -> str:
        """
        Extract entities and relationships from text using LLM.
        
        Args:
            text: Input text to analyze
            
        Returns:
            JSON string containing entities and relationships
        """
        messages = self.entity_prompt.format(text=text)
        return self.llm.invoke(messages)
    
    def generate_entity_index(self, triples: str, original_text: str) -> str:
        """
        Generate searchable key-value index for entities.
        
        Args:
            triples: JSON string of extracted entities/relationships
            original_text: Original source text for context
            
        Returns:
            JSON string containing entity index
        """
        messages = self.index_prompt.format(question=triples, text=original_text)
        return self.llm.invoke(messages)
    
    def extract_complete_knowledge_graph(self, text: str) -> Tuple[str, str]:
        """
        Complete pipeline: extract entities, relationships, and generate index.
        
        Args:
            text: Input text to process
            
        Returns:
            Tuple of (triples_json, key_values_json)
        """
        triples = self.extract_entities_and_relationships(text)
        key_values = self.generate_entity_index(triples, text)
        return triples, key_values


class KnowledgeGraphBuilder:
    """Main class for building and storing knowledge graphs"""
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str,
                 model: str = "gemma3:1b"):
        """
        Initialize the knowledge graph builder.
        
        Args:
            neo4j_uri: Neo4j database URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            model: Ollama model to use for extraction
        """
        self.extractor = KnowledgeGraphExtractor(model=model)
        self.neo4j = Neo4jLightRAG(
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password
        )
    
    @staticmethod
    def clean_code_fence(s: str) -> str:
        """
        Remove markdown code fences from LLM output.
        
        Args:
            s: String potentially containing ```json ... ```
            
        Returns:
            Clean JSON string
        """
        s = s.strip()
        if s.startswith("```"):
            lines = s.splitlines()
            lines = lines[1:]  # Drop first line (```json or ```)
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]  # Drop closing ```
            s = "\n".join(lines).strip()
        return s
    
    def save_extraction_results(self, triples: str, key_values: str, 
                               name: str) -> Tuple[Dict, Dict]:
        """
        Clean, parse, and save extraction results to JSON file.
        
        Args:
            triples: Raw triples JSON from LLM
            key_values: Raw key-values JSON from LLM
            name: Identifier for the output file
            
        Returns:
            Tuple of (triples_dict, key_values_dict)
        """
        triples_json = self.clean_code_fence(triples)
        key_values_json = self.clean_code_fence(key_values)
        
        triples_dict = json.loads(triples_json)
        key_values_dict = json.loads(key_values_json)
        
        # Save to file for debugging/archiving
        output_data = {
            "triples": triples_dict,
            "key_values": key_values_dict
        }
        with open(f"knowledge_graph_data_{name}.json", "w") as f:
            json.dump(output_data, f, indent=2)
        
        return triples_dict, key_values_dict
    
    def process_chunk(self, text: str, chunk_id: int) -> bool:
        """
        Complete pipeline for processing a text chunk into knowledge graph.
        
        Steps:
        1. Extract entities and relationships
        2. Generate entity index
        3. Save results to JSON
        4. Load into Neo4j database
        
        Args:
            text: Text chunk to process
            chunk_id: Unique identifier for this chunk
            
        Returns:
            True if successful
        """
        # Extract knowledge graph data
        triples, key_values = self.extractor.extract_complete_knowledge_graph(text)
        
        # Save and parse results
        saved_triples, saved_key_values = self.save_extraction_results(
            triples, key_values, f"chunk_{chunk_id}"
        )
        
        # Load into Neo4j
        load_lightrag_data(
            self.neo4j,
            saved_triples['entities'],
            saved_triples['relationships'],
            saved_key_values['entity_index']
        )
        
        return True
    
    def create_multimodal_graph(self, image_info: Dict[str, Any], 
                               chunk_id: int) -> str:
        """
        Create multimodal knowledge graph with image anchor nodes.
        
        Structure:
        [Image_Anchor] <-belongs_to- [Entity1]
        [Image_Anchor] <-belongs_to- [Entity2]
        
        Args:
            image_info: Dictionary with 'image_path' and 'detailed_description'
            chunk_id: Unique identifier for this image
            
        Returns:
            Name of the created anchor node
        """
        # Create anchor node for the image
        anchor_name = f"Image_{chunk_id}"
        self.neo4j.create_entity(
            entity_name=anchor_name,
            entity_type="MultimodalAnchor",
            properties={
                "modality": "image",
                "image_path": image_info['image_path'],
                "detailed_description": image_info['detailed_description']
            }
        )
        
        # Extract entities from image description
        triples, _ = self.extractor.extract_complete_knowledge_graph(
            image_info['detailed_description']
        )
        triples_clean = self.clean_code_fence(triples)
        entities_from_image = json.loads(triples_clean)
        
        # Create entity nodes and link to anchor
        for entity in entities_from_image['entities']:
            self.neo4j.create_entity(
                entity_name=entity['name'],
                entity_type=entity['type']
            )
            
            # Create belongs_to relationship
            self.neo4j.create_relationship(
                source=entity['name'],
                target=anchor_name,
                relation_type="BELONGS_TO"
            )
        
        return anchor_name



from parser import PDFParser
from multimodal_processing import MultimodalProcessor

def process_pdf_document(pdf_path: str, builder: KnowledgeGraphBuilder, multimodal: MultimodalProcessor):
    """
    Process a PDF document, routing text to the graph builder and images to the multimodal processor.
    
    Args:
        pdf_path: Path to the PDF file.
        builder: KnowledgeGraphBuilder instance.
        multimodal: MultimodalProcessor instance.
    """
    parser = PDFParser()
    print(f"🚀 Starting PDF processing: {pdf_path}")
    
    chunk_counter = 0
    
    for chunk in parser.parse_pdf(pdf_path):
        chunk_counter += 1
        chunk_id = f"pdf_chunk_{chunk_counter}"
        
        if chunk['type'] == 'text':
            print(f"  📄 Processing Text Chunk {chunk_counter} (Page {chunk['page']})...")
            try:
                builder.process_chunk(chunk['content'], chunk_id)
            except Exception as e:
                print(f"    ❌ Error processing text chunk: {e}")
                
        elif chunk['type'] == 'image':
            print(f"  🖼️ Processing Image Chunk {chunk_counter} (Page {chunk['page']})...")
            try:
                image_info = multimodal.extract_image_info(chunk['content'], chunk['context'])
                builder.create_multimodal_graph(image_info, chunk_counter)
            except Exception as e:
                print(f"    ❌ Error processing image chunk: {e}")

    print("✅ PDF processing complete!")


def process_image_document(image_path: str, builder: KnowledgeGraphBuilder, multimodal: MultimodalProcessor):
    """
    Process a single image document directly.
    
    Args:
        image_path: Path to the image file.
        builder: KnowledgeGraphBuilder instance.
        multimodal: MultimodalProcessor instance.
    """
    print(f"🚀 Starting Image processing: {image_path}")
    
    try:
        # For single image input, we pass empty context as there is no preceding text
        image_info = multimodal.extract_image_info(image_path, surrounding_context="")
        
        # Use a generic ID or derive from filename
        import os
        image_name = os.path.basename(image_path)
        chunk_id = f"image_{image_name}"
        
        builder.create_multimodal_graph(image_info, chunk_id)
        print("✅ Image processing complete!")
        
    except Exception as e:
        print(f"    ❌ Error processing image: {e}")


def main():
    """
    Main execution function demonstrating knowledge graph construction.
    """
    # Initialize the knowledge graph builder
    # Note: You might need to adjust credentials or make them configurable
    builder = KnowledgeGraphBuilder(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="yourpassword" 
    )
    
    # Initialize Multimodal Processor
    multimodal = MultimodalProcessor()
    
    # Example usage with a PDF or Image
    # Check if a file path is provided via command line, otherwise use a default or prompt
    import sys
    import os
    
    file_path = "./images/sample.pdf" # Default
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        
    if os.path.exists(file_path):
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            process_pdf_document(file_path, builder, multimodal)
        elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
            process_image_document(file_path, builder, multimodal)
        else:
            print(f"⚠️ Unsupported file type: {file_ext}")
            print("Supported types: .pdf, .jpg, .jpeg, .png")
            
    else:
        print(f"⚠️ File not found: {file_path}")
        print("Usage: python pipeline.py <path_to_file>")
        
        # Fallback to original text-only test if no file found
        print("\nRunning fallback text-only test...")
        text1 = """Dr. Sarah Chen is a cardiologist at Stanford Medical Center who specializes in treating heart disease. In 2024, she published
groundbreaking research on using AI to diagnose arrhythmias early. Her work showed that machine learning models can detect
irregular heartbeats with 95% accuracy. Dr. Chen collaborates with Dr. Michael Torres, a data scientist at MIT, to develop these AI
diagnostic tools. The research was funded by the National Heart Institute and could revolutionize cardiac care."""
        builder.process_chunk(text1, chunk_id=1)

if __name__ == "__main__":
    main()



"""
Visualization:
**Neo4j Browser** (Built-in, Best Option)
   - Open http://localhost:7474 after starting Neo4j
   - Run Cypher queries to visualize
   - Example: MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 25
    MATCH (n)
    DETACH DELETE n
"""
