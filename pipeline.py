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



import ollama
from chat import PromptTemplate, ChatOllamaMini
import json
from neo4j import GraphDatabase
from neo4j_lightrag_storage import load_lightrag_data, Neo4jLightRAG

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
    ]
)

# messages = prompt.format(question=Text)
# llm = ChatOllamaMini(model="gemma3:1b", temperature=0.0, base_url="http://localhost:11434")
# triples = llm.invoke(messages)
# print("Extracted Triples:", triples)

# IMPROVED PROMPT 2: Generate key-value index
prompt2 = PromptTemplate.from_messages(
    [
        ("system",
        """You are creating a searchable index for a knowledge graph database.

For each entity, generate key-value pairs:

**ENTITY INDEX:**
- Key: The entity name (e.g., "Dr. Sarah Chen")
- Value: A 2-3 sentence summary containing:
  * What the entity is
  * Key facts and context from the text
  * Related entities and relationships

*

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
  ],

```

Generate the key-value index from the provided entities, relationships, and original text."""),
        ("user", "Entities and Relationships:{question} Original Text:{text}"), 
    ]          
)

# messages2 = prompt2.format(question=triples, text=Text)
# llm2 = ChatOllamaMini(model="gemma3:1b", temperature=0.0, base_url="http://localhost:11434")
# key_values = llm2.invoke(messages2)
# print("Key-Value Pairs for Graph DB:", key_values)

def get_knowledge_graph_data(text):
    llm = ChatOllamaMini(model="gemma3:1b", temperature=0.0, base_url="http://localhost:11434")
    messages = prompt.format(text=text)
    triples = llm.invoke(messages)

    messages2 = prompt2.format(question=triples, text=text)
    key_values = llm.invoke(messages2)

    return triples, key_values

def clean_code_fence(s: str) -> str:
    s = s.strip()
    # Handle ```json ... ``` or ``` ... ```
    if s.startswith("```"):
        lines = s.splitlines()
        # drop first line (```json or ```)
        lines = lines[1:]
        # if last line is ``` drop it
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s



def handle_output(triples, key_values, name):
  triples_json = clean_code_fence(triples)
  key_values_json = clean_code_fence(key_values)
  # print("Type of triples:", type(triples))
  # print("First 200 chars of triples:", str(triples)[:200])
  triples_dict = json.loads(triples_json)
  key_values_dict = json.loads(key_values_json)

  with open(f"knowledge_graph_data_{name}.json", "w") as f:
      json.dump({"triples": triples_dict, "key_values": key_values_dict}, f, indent=2)
      #put in variables now the triples and key_values

  saved_triples = triples_dict
  saved_key_values = key_values_dict
  return saved_triples, saved_key_values


def chunk_parsing(text, chunk_id, neo4j):
  triples, key_values = get_knowledge_graph_data(text)
  saved_triples, saved_key_values = handle_output(triples, key_values, f"chunk_{chunk_id}")
  load_lightrag_data(neo4j, saved_triples['entities'], saved_triples['relationships'], saved_key_values['entity_index'])
  return 1






neo4j = Neo4jLightRAG(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="yourpassword"  # CHANGE THIS!
    )


Text = """Dr. Sarah Chen is a cardiologist at Stanford Medical Center who specializes in treating heart disease. In 2024, she published
groundbreaking research on using AI to diagnose arrhythmias early. Her work showed that machine learning models can detect
irregular heartbeats with 95% accuracy. Dr. Chen collaborates with Dr. Michael Torres, a data scientist at MIT, to develop these AI
diagnostic tools. The research was funded by the National Heart Institute and could revolutionize cardiac care."""
Text2 = """Mit is a leading research university located in Cambridge, Massachusetts. Founded in 1861, MIT has been at the forefront of scientific research """
chunk_parsing(Text, 1)
chunk_parsing(Text2, 2)




# triples, key_values = get_knowledge_graph_data(Text)
# # print("Extracted Triples:", triples)
# # print("Key-Value Pairs for Graph DB:", key_values)
# #Save the results to a file json for now

# saved_triples, saved_key_values = handle_output(triples, key_values, "text1")
# #To instert in graph database like neo4j : do a query for each entity is it already in the graph if yes then connect the relationship if it exist or go to the next entity if not create the entity node and then connect the relationship


# triples2, key_values2 = get_knowledge_graph_data(Text2)
# saved_triples2, saved_key_values2 = handle_output(triples2, key_values2, "text2")





# load_lightrag_data(neo4j, saved_triples['entities'], saved_triples['relationships'], saved_key_values['entity_index'])
# print('Sucessfully loaded knowledge graph1 data into Neo4j! open http://localhost:7474 to view it.')
# load_lightrag_data(neo4j, saved_triples2['entities'], saved_triples2['relationships'], saved_key_values2['entity_index'])
# print('Sucessfully loaded knowledge graph2 data into Neo4j! open http://localhost:7474 to view it.')


# if(1==0):
#     #Debug
#     print("saved_triples:", saved_triples)
#     print("saved_key_values:", saved_key_values)
#     print("saved_triples2:", saved_triples2)
#     print("saved_key_values2:", saved_key_values2)





# print(saved_triples['entities'])
# print(saved_triples['relationships'])
# print(saved_key_values['entity_index'])
# triples2, key_values2 = get_knowledge_graph_data(Text2)
# print("Extracted Triples:", triples2)
# print("Key-Value Pairs for Graph DB:", key_values2)

