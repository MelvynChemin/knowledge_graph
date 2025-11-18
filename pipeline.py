# pipeline.py - Refactored

import ollama
from chat import PromptTemplate, ChatOllamaMini
import json
from neo4j import GraphDatabase
from neo4j_lightrag_storage import load_lightrag_data, Neo4jLightRAG

# ============================================================================
# TEXTUAL GRAPH BUILDER CLASS
# ============================================================================

class TextualGraphBuilder:
    """Handles extraction and processing of text-based knowledge graphs"""
    
    def __init__(self, model="gemma3:1b", temperature=0.0):
        self.llm = ChatOllamaMini(model=model, temperature=temperature)
        self._setup_prompts()
    
    def _setup_prompts(self):
        """Initialize the prompts for entity/relationship extraction"""
        
        # Prompt for extracting entities and relationships
        self.entity_prompt = PromptTemplate.from_messages([
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
        
        # Prompt for generating searchable index
        self.index_prompt = PromptTemplate.from_messages([
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
    
    def extract_entities_and_relations(self, text):
        """Extract entities and relationships from text"""
        messages = self.entity_prompt.format(text=text)
        response = self.llm.invoke(messages)
        return response
    
    def generate_index(self, entities_and_relations, original_text):
        """Generate searchable index summaries"""
        messages = self.index_prompt.format(
            question=entities_and_relations, 
            text=original_text
        )
        response = self.llm.invoke(messages)
        return response
    
    def process_chunk(self, text, chunk_id):
        """Full processing pipeline for a text chunk"""
        # Step 1: Extract entities and relationships
        triples = self.extract_entities_and_relations(text)
        
        # Step 2: Generate index
        key_values = self.generate_index(triples, text)
        
        # Step 3: Clean and parse
        triples_json = self._clean_code_fence(triples)
        key_values_json = self._clean_code_fence(key_values)
        
        triples_dict = json.loads(triples_json)
        key_values_dict = json.loads(key_values_json)
        
        # Step 4: Save to file (optional)
        self._save_to_file(triples_dict, key_values_dict, f"text_chunk_{chunk_id}")
        
        return triples_dict, key_values_dict
    
    def _clean_code_fence(self, s):
        """Remove markdown code fences from LLM output"""
        s = s.strip()
        if s.startswith("```"):
            lines = s.splitlines()
            lines = lines[1:]  # drop first line
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]  # drop last line
            s = "\n".join(lines).strip()
        return s
    
    def _save_to_file(self, triples, key_values, name):
        """Save extracted data to JSON file"""
        with open(f"knowledge_graph_data_{name}.json", "w") as f:
            json.dump({
                "triples": triples, 
                "key_values": key_values
            }, f, indent=2)


# ============================================================================
# MULTIMODAL GRAPH BUILDER CLASS
# ============================================================================

class MultimodalGraphBuilder:
    """Handles extraction and processing of multimodal content (images, tables, etc)"""
    
    def __init__(self, vlm_model="llava:7b", temperature=0.0):
        self.vlm_model = vlm_model
        self.temperature = temperature
    
    def process_image(self, image_element, position, surrounding_context):
        """
        Process an image with its surrounding text context
        
        Args:
            image_element: Image data or path
            position: Position in document
            surrounding_context: Text from nearby chunks
        """
        # Generate detailed description
        detailed_desc = self._generate_detailed_description(
            image_element, 
            surrounding_context
        )
        
        # Generate entity summary for graph
        entity_summary = self._generate_entity_summary(
            image_element,
            surrounding_context
        )
        
        return {
            "detailed_description": detailed_desc,
            "entity_summary": entity_summary,
            "position": position,
            "modality": "image"
        }
    
    def process_table(self, table_data, position, surrounding_context):
        """Process a table with its context"""
        # TODO: implement table processing
        pass
    
    def _generate_detailed_description(self, image_element, context):
        """Generate comprehensive image description using VLM"""
        prompt = f"""Analyze this image in the context of the surrounding document.

**Surrounding text:**
{context}

**Task:** Provide a comprehensive description of the image that:
1. Identifies all objects, people, text, and visual elements
2. Explains relationships between elements  
3. Notes colors, lighting, and visual style
4. References connections to the surrounding text when relevant

Respond in 2-3 paragraphs."""
        
        # Call VLM (simplified - you'll need actual implementation)
        # response = self._call_vlm(image_element, prompt)
        # For now return placeholder
        response = f"[VLM Description based on context: {context[:100]}...]"
        return response
    
    def _generate_entity_summary(self, image_element, context):
        """Extract entities from image for graph construction"""
        prompt = f"""Based on this image and its document context, extract key entities for a knowledge graph.

**Context:**
{context}

**Output JSON:**
{{
  "entity_name": "[descriptive name for this image]",
  "entity_type": "image", 
  "description": "concise summary",
  "key_entities": ["entity1", "entity2", "entity3"]
}}

Extract entities that are visually present or strongly implied."""
        
        # Call VLM
        # response = self._call_vlm(image_element, prompt)
        # Placeholder
        response = '{"entity_name": "Figure_1", "entity_type": "image", "key_entities": []}'
        return response
    
    def _call_vlm(self, image_element, prompt):
        """Call vision-language model (placeholder for now)"""
        # TODO: Implement actual VLM call
        # This would use ollama with llava or similar
        return "[VLM response placeholder]"


# ============================================================================
# MAIN PROCESSING FUNCTIONS
# ============================================================================

def process_text_chunk(text, chunk_id, neo4j, text_builder):
    """Process a single text chunk and load into Neo4j"""
    triples_dict, key_values_dict = text_builder.process_chunk(text, chunk_id)
    
    # Load into Neo4j
    load_lightrag_data(
        neo4j, 
        triples_dict['entities'], 
        triples_dict['relationships'], 
        key_values_dict['entity_index']
    )
    
    return triples_dict, key_values_dict


def process_multimodal_chunk(element, chunk_id, context, neo4j, mm_builder):
    """Process a multimodal element (image/table) and create anchor node"""
    
    if element['type'] == 'image':
        mm_info = mm_builder.process_image(
            element['content'],
            chunk_id,
            context
        )
    elif element['type'] == 'table':
        mm_info = mm_builder.process_table(
            element['content'],
            chunk_id, 
            context
        )
    else:
        return None
    
    # Create anchor node in Neo4j
    anchor_name = f"MM_Anchor_{chunk_id}"
    
    neo4j.create_entity(
        entity_name=anchor_name,
        entity_type="MultimodalAnchor",
        properties={
            "modality": mm_info['modality'],
            "position": mm_info['position'],
            "detailed_description": mm_info['detailed_description']
        }
    )
    
    # Extract entities from description and link to anchor
    # TODO: parse entity_summary and create belongs_to relationships
    
    return mm_info


def process_full_document(chunks, neo4j, delta=2):
    """
    Process entire document with text and multimodal chunks
    
    Args:
        chunks: List of chunk dicts with 'type', 'content', 'position'
        neo4j: Neo4j handler
        delta: Context window size for multimodal elements
    """
    text_builder = TextualGraphBuilder()
    mm_builder = MultimodalGraphBuilder()
    
    for i, chunk in enumerate(chunks):
        if chunk['type'] == 'text':
            # Regular text processing
            process_text_chunk(
                chunk['content'], 
                chunk['position'], 
                neo4j,
                text_builder
            )
            
        elif chunk['type'] in ['image', 'table']:
            # Get surrounding context
            context = get_surrounding_context(chunks, i, delta)
            
            # Process multimodal element
            process_multimodal_chunk(
                chunk,
                chunk['position'],
                context,
                neo4j,
                mm_builder
            )


def get_surrounding_context(chunks, index, delta):
    """Extract text context around a multimodal element"""
    context_parts = []
    
    # Get chunks before
    start_idx = max(0, index - delta)
    for i in range(start_idx, index):
        if chunks[i]['type'] == 'text':
            context_parts.append(chunks[i]['content'])
    
    # Get chunks after
    end_idx = min(len(chunks), index + delta + 1)
    for i in range(index + 1, end_idx):
        if chunks[i]['type'] == 'text':
            context_parts.append(chunks[i]['content'])
    
    return "\n\n".join(context_parts)


# ============================================================================
# MAIN EXECUTION (testing)
# ============================================================================

if __name__ == "__main__":
    
    neo4j = Neo4jLightRAG(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="yourpassword"
    )
    
    # Initialize builders
    text_builder = TextualGraphBuilder()
    
    # Test text chunks
    Text1 = """Dr. Sarah Chen is a cardiologist at Stanford Medical Center who specializes in treating heart disease. In 2024, she published
groundbreaking research on using AI to diagnose arrhythmias early. Her work showed that machine learning models can detect
irregular heartbeats with 95% accuracy. Dr. Chen collaborates with Dr. Michael Torres, a data scientist at MIT, to develop these AI
diagnostic tools. The research was funded by the National Heart Institute and could revolutionize cardiac care."""
    
    Text2 = """MIT is a leading research university located in Cambridge, Massachusetts. Founded in 1861, MIT has been at the forefront of scientific research"""
    
    # Process text chunks
    process_text_chunk(Text1, 1, neo4j, text_builder)
    process_text_chunk(Text2, 2, neo4j, text_builder)
    
    print("✅ Processed text chunks and loaded into Neo4j")
    
    # For multimodal processing, you'd use:
    # process_full_document(parsed_chunks, neo4j)