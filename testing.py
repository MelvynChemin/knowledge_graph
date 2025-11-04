#Idea of building a knowledge graph
#Step 1 : Chunking the pdf
#Step 2 : For each chunks, the prinipal node of the graph is the chunk itself 
#Step 3 : To create the nodes of the graph we Uses an LLM to identify important things (entities) like people, places, events
#Step 4 : To create the edges of the graph we leverage LLM to extract relationships between the entities
#Step 5 : Store the graph in a graph database like neo4j


# Import necessary libraries
# from PyPDF2 import PdfReader
# For now this project is not using langchain
# query_gemma3_1b.py


Text = """Dr. Sarah Chen is a cardiologist at Stanford Medical Center who specializes in treating heart disease. In 2024, she published
groundbreaking research on using AI to diagnose arrhythmias early. Her work showed that machine learning models can detect
irregular heartbeats with 95% accuracy. Dr. Chen collaborates with Dr. Michael Torres, a data scientist at MIT, to develop these AI
diagnostic tools. The research was funded by the National Heart Institute and could revolutionize cardiac care."""

import ollama
from chat import PromptTemplate, ChatOllamaMini

# IMPROVED PROMPT 1: Extract entities and relationships
prompt = PromptTemplate.from_messages(
    [
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
        ("user", "{question}"),          
    ]
)

messages = prompt.format(question=Text)
llm = ChatOllamaMini(model="gemma3:1b", temperature=0.0, base_url="http://localhost:11434")
triples = llm.invoke(messages)
print("Extracted Triples:", triples)

# IMPROVED PROMPT 2: Generate key-value index
prompt2 = PromptTemplate.from_messages(
    [
        ("system",
        """You are creating a searchable index for a knowledge graph database.

For each entity and relationship, generate key-value pairs:

**ENTITY INDEX:**
- Key: The entity name (e.g., "Dr. Sarah Chen")
- Value: A 2-3 sentence summary containing:
  * What the entity is
  * Key facts and context from the text
  * Related entities and relationships

**RELATIONSHIP INDEX:**  
- Key: Natural language search terms/themes that users might query (e.g., "AI cardiac diagnosis", "Stanford MIT collaboration")
- Value: A 1-2 sentence summary explaining:
  * How the entities are connected
  * Important details (dates, metrics, outcomes)

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
  ],
  "relationship_index": [
    {
      "key": "AI cardiac diagnosis, machine learning arrhythmia detection",
      "value": "AI diagnostic tools developed by Dr. Sarah Chen and Dr. Michael Torres can detect irregular heartbeats and diagnose arrhythmias with 95% accuracy, potentially revolutionizing cardiac care."
    },
    {
      "key": "Stanford MIT research collaboration, academic partnership",
      "value": "Dr. Sarah Chen at Stanford Medical Center collaborates with Dr. Michael Torres, a data scientist at MIT, to develop AI-powered cardiac diagnostic tools."
    }
  ]
}
```

Generate the key-value index from the provided entities, relationships, and original text."""),
        ("user", "Entities and Relationships:{question} Original Text:{text}"), 
    ]          
)

messages2 = prompt2.format(question=triples, text=Text)
llm2 = ChatOllamaMini(model="gemma3:1b", temperature=0.0, base_url="http://localhost:11434")
key_values = llm2.invoke(messages2)
print("Key-Value Pairs for Graph DB:", key_values)