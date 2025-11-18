# multimodal_processing.py

import ollama
import base64
from PIL import Image
from io import BytesIO
import json

class MultimodalProcessor:
    """Process multimodal content with VLMs"""
    
    def __init__(self, vlm_model="llava:7b-v1.5-q4_1"):
        self.vlm_model = vlm_model
        self.client = ollama.Client()
    
    def process_image_with_context(self, image_path, surrounding_context):
        """
        Process image with surrounding text context
        
        Returns dict with:
            - detailed_description: Comprehensive description for retrieval
            - entity_summary: Structured entities for graph
        """
        # Load and encode image
        image_b64 = self._encode_image(image_path)
        
        # Generate detailed description
        detail_prompt = self._build_detail_prompt(surrounding_context)
        detailed_desc = self._call_vlm(image_b64, detail_prompt)
        
        # Generate entity summary
        entity_prompt = self._build_entity_prompt(surrounding_context)
        entity_summary = self._call_vlm(image_b64, entity_prompt)
        
        return {
            "detailed_description": detailed_desc,
            "entity_summary": entity_summary,
            "image_path": image_path
        }
    
    def _encode_image(self, image_path):
        """Encode image to base64 for VLM"""
        if isinstance(image_path, str):
            with open(image_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        elif isinstance(image_path, Image.Image):
            buffered = BytesIO()
            image_path.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def _build_detail_prompt(self, context):
        """Build prompt for detailed description"""
        return f"""Analyze this image in the context of the surrounding document.

**Surrounding text:**
{context}

**Task:** Provide a comprehensive description of the image that:
1. Identifies all objects, people, text, and visual elements
2. Explains relationships between elements  
3. Notes any charts, graphs, diagrams with their key data
4. References connections to the surrounding text when relevant

Respond in 2-3 detailed paragraphs."""
    
    def _build_entity_prompt(self, context):
        """Build prompt for entity extraction"""
        return f"""Based on this image and its document context, extract key entities for a knowledge graph.

**Context:**
{context}

**Output as JSON:**
{{
  "entity_name": "[descriptive name like 'Figure_2_Results' or 'Chart_Accuracy']",
  "entity_type": "image",
  "description": "brief summary of what image shows",
  "key_entities": ["entity1", "entity2", "entity3"]
}}

Extract only entities that are visually present or strongly implied by the image."""
    
    def _call_vlm(self, image_b64, prompt):
        """Call vision-language model"""
        try:
            response = self.client.chat(
                model=self.vlm_model,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [image_b64]
                }]
            )
            return response['message']['content']
        except Exception as e:
            print(f"VLM call failed: {e}")
            return "[VLM processing failed]"
    
    def extract_entities_from_description(self, description_text):
        """
        Use LLM to extract entities from VLM description
        (for creating graph nodes from the detailed description)
        """
        prompt = f"""Extract entities and relationships from this image description:

{description_text}

Output JSON format:
{{
  "entities": [
    {{"name": "entity1", "type": "type1"}},
    {{"name": "entity2", "type": "type2"}}
  ],
  "relationships": [
    {{"source": "entity1", "relation": "relation", "target": "entity2"}}
  ]
}}"""
        
        response = ollama.chat(
            model="gemma3:1b",
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        content = response['message']['content']
        
        # Clean code fences
        content = content.strip()
        if content.startswith("```"):
            lines = content.splitlines()
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines).strip()
        
        try:
            return json.loads(content)
        except:
            return {"entities": [], "relationships": []}


# Usage example
if __name__ == "__main__":
    processor = MultimodalProcessor(vlm_model="llava:7b-v1.5-q4_1")
    
    # Process an image
    context = "This section discusses the accuracy of AI models in detecting arrhythmias."
    
    result = processor.process_image_with_context(
        "extracted_images/image_pos_5_page_2.png",
        context
    )
    
    print("Detailed Description:")
    print(result['detailed_description'])
    print("\nEntity Summary:")
    print(result['entity_summary'])