# New file: multimodal_processing.py

from PIL import Image
import base64
from io import BytesIO

import ollama

class MultimodalProcessor:
    """Extract and process non-text content from documents"""
    
    def __init__(self, vlm_model="llava:7b-v1.5-q4_1"):
        self.vlm_model = vlm_model
    
    def extract_image_info(self, image_path, surrounding_context):
        """
        Generate two representations for an image:
        1. Detailed description (for retrieval)
        2. Entity summary (for graph construction)
        """
        # Encode image for VLM
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()
        
        # Prompt 1: Detailed description
        detail_prompt = f"""Analyze this image in detail, considering the surrounding context.

        Context from document: {surrounding_context}

        Provide a comprehensive description including:
        - Main objects and their relationships
        - Visual elements (charts, diagrams, etc.)
        - How this image relates to the surrounding text
        - Any technical details or data shown

        Respond in 2-3 paragraphs."""

        detailed_desc = self._call_vlm(image_b64, detail_prompt)
        
        # Prompt 2: Entity summary for graph
        entity_prompt = f"""Based on this image, extract key entities for a knowledge graph.

        Context: {surrounding_context}
        detailed description: {detailed_desc}   

        Output JSON format:
        {{
        "entity_name": "Figure_X_Title",
        "entity_type": "image",
        "key_entities": ["entity1", "entity2"]
        }}"""
        
        entity_summary = self._call_vlm(image_b64, entity_prompt)
        
        return {
            "detailed_description": detailed_desc,
            "entity_summary": entity_summary,
            "image_path": image_path
        }
    
    def extract_table_info(self, table_data, surrounding_context):
        """Extract structured info from tables"""
        # Convert table to text representation
        table_text = self._table_to_text(table_data)
        
        prompt = f"""Analyze this table considering the context.

        Table:
        {table_text}

        Context: {surrounding_context}

        Extract:
        1. What the table shows
        2. Key data points
        3. Column/row meanings
        4. Relationships to surrounding text"""
        
        # Use regular LLM for tables
        return self._call_llm(prompt)
    
    def _call_vlm(self, image_b64, prompt):
        """Call vision-language model"""
        response = ollama.chat(
            model=self.vlm_model,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [image_b64]
            }]
        )
        return response['message']['content']
    
#Test
def main():
    processor = MultimodalProcessor()
    image_info = processor.extract_image_info(
        image_path="./images/presidentielles.jpg",
        surrounding_context="This image shows the French presidential election results."
    )
    print("Detailed Description:\n", image_info['detailed_description'])
    print("Entity Summary:\n", image_info['entity_summary'])

if __name__ == "__main__":
    main()