"""
LightRAG Knowledge Graph Storage and Visualization with Neo4j
Demonstrates how to store extracted entities and relationships in Neo4j
and visualize them using free tools.
"""

import json
from neo4j import GraphDatabase
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network

# ============================================================================
# PART 1: NEO4J SETUP AND CONNECTION
# ============================================================================

class Neo4jLightRAG:
    """Class to handle Neo4j operations for LightRAG knowledge graphs"""
    
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="yourpassword"):
        """
        Initialize Neo4j connection
        
        Args:
            uri: Neo4j database URI (default: local instance)
            user: Username (default: neo4j)
            password: Password you set during Neo4j installation
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        """Close the database connection"""
        self.driver.close()
    
    def clear_database(self):
        """Clear all nodes and relationships (use with caution!)"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("✅ Database cleared")
    
    def create_entity(self, entity_name, entity_type, properties=None):
        """
        Create an entity node in Neo4j
        
        Args:
            entity_name: Name of the entity
            entity_type: Type (Person, Organization, Technology, etc.)
            properties: Additional properties as dictionary
        """
        with self.driver.session() as session:
            if properties is None:
                properties = {}
            
            # Create node with dynamic label based on entity_type
            query = f"""
            MERGE (e:{entity_type} {{name: $name}})
            SET e += $properties
            RETURN e
            """
            session.run(query, name=entity_name, properties=properties)
            print(f"✅ Created entity: {entity_name} ({entity_type})")
    
    def create_relationship(self, source, target, relation_type, properties=None):
        """
        Create a relationship between two entities
        
        Args:
            source: Source entity name
            target: Target entity name
            relation_type: Type of relationship (works_at, collaborates_with, etc.)
            properties: Additional properties as dictionary
        """
        with self.driver.session() as session:
            if properties is None:
                properties = {}
            
            # Use MERGE to avoid duplicates
            query = f"""
            MATCH (s {{name: $source}})
            MATCH (t {{name: $target}})
            MERGE (s)-[r:{relation_type}]->(t)
            SET r += $properties
            RETURN r
            """
            result = session.run(query, source=source, target=target, properties=properties)
            if result.single():
                print(f"✅ Created relationship: {source} -[{relation_type}]-> {target}")
            else:
                print(f"⚠️  Could not create relationship (entities may not exist)")
    
    def add_entity_index(self, entity_name, summary):
        """
        Add the key-value index summary to an entity
        
        Args:
            entity_name: Name of the entity
            summary: Text summary for retrieval
        """
        with self.driver.session() as session:
            query = """
            MATCH (e {name: $name})
            SET e.index_summary = $summary
            RETURN e
            """
            session.run(query, name=entity_name, summary=summary)
            print(f"✅ Added index summary to: {entity_name}")
    
    def add_relationship_index(self, relation_key, relation_value):
        """
        Create a special 'RelationshipIndex' node for theme-based searching
        
        Args:
            relation_key: Search theme/topic
            relation_value: Summary of the relationship
        """
        with self.driver.session() as session:
            query = """
            MERGE (idx:RelationshipIndex {key: $key})
            SET idx.summary = $value
            RETURN idx
            """
            session.run(query, key=relation_key, value=relation_value)
            print(f"✅ Added relationship index: {relation_key}")
    
    def query_entity(self, entity_name):
        """Retrieve an entity and its summary"""
        with self.driver.session() as session:
            query = """
            MATCH (e {name: $name})
            RETURN e.name as name, labels(e) as type, e.index_summary as summary
            """
            result = session.run(query, name=entity_name)
            return result.single()
    
    def query_relationships(self, entity_name):
        """Get all relationships for an entity"""
        with self.driver.session() as session:
            query = """
            MATCH (e {name: $name})-[r]-(other)
            RETURN e.name as entity, type(r) as relationship, other.name as connected_to
            """
            result = session.run(query, name=entity_name)
            return [record.data() for record in result]
    
    def search_by_theme(self, search_term):
        """
        Search relationship indexes by theme/topic
        
        Args:
            search_term: Keywords to search for
        """
        with self.driver.session() as session:
            query = """
            MATCH (idx:RelationshipIndex)
            WHERE idx.key CONTAINS $term OR idx.summary CONTAINS $term
            RETURN idx.key as theme, idx.summary as summary
            """
            result = session.run(query, term=search_term)
            return [record.data() for record in result]
    
    def get_full_graph(self):
        """Retrieve the entire graph for visualization"""
        with self.driver.session() as session:
            query = """
            MATCH (n)-[r]->(m)
            RETURN n.name as source, type(r) as relationship, m.name as target,
                   labels(n) as source_type, labels(m) as target_type
            """
            result = session.run(query)
            return [record.data() for record in result]
    def entity_exists(self, entity_name):
        """
        Check if an entity with the given name already exists in the database
        
        Args:
            entity_name: Name of the entity to check
            
        Returns:
            bool: True if entity exists, False otherwise
        """
        with self.driver.session() as session:
            query = """
                MATCH (n {name: $name})
                RETURN count(n) > 0 as exists
            """
            result = session.run(query, name=entity_name)
            record = result.single()
            return record["exists"] if record else False

# ============================================================================
# PART 2: LOAD YOUR EXTRACTED DATA INTO NEO4J
# ============================================================================
def sanitize_label(label):
    """Convert a label to a valid Neo4j label format"""
    # Replace spaces and special characters with underscores
    return label.replace(' ', '_').replace('-', '_')

def load_lightrag_data(neo4j_handler, entities_json, relationships_json, 
                       entity_index_json):
    """
    Load all extracted LightRAG data into Neo4j
    
    Args:
        neo4j_handler: Neo4jLightRAG instance
        entities_json: List of entity dictionaries
        relationships_json: List of relationship dictionaries
        entity_index_json: List of entity index key-value pairs
    """
    print("\n" + "="*60)
    print("LOADING DATA INTO NEO4J")
    print("="*60 + "\n")
    
    # Step 1: Create all entities
    print("📥 Creating entities...")
    for entity in entities_json:
        entity_name = sanitize_label(entity["name"])
        entity_type = sanitize_label(entity["type"])
        if neo4j_handler.entity_exists(entity_name):
            print(f"⏭️  Entity already exists: {entity_name} ({entity_type})")
        else:
            neo4j_handler.create_entity(
                entity_name=entity_name,
                entity_type=entity_type
            )

    # Step 2: Create all relationships
    print("\n📥 Creating relationships...")
    for rel in relationships_json:
        neo4j_handler.create_relationship(
            source=sanitize_label(rel["source"]),
            target=sanitize_label(rel["target"]),
            relation_type=rel["relation"].upper()  # Neo4j convention: uppercase
        )
    
    # Step 3: Add entity index summaries
    print("\n📥 Adding entity index summaries...")
    for idx in entity_index_json:
        neo4j_handler.add_entity_index(
            entity_name=sanitize_label(idx["key"]),
            summary=idx["value"]
        )
    
    # Step 4: Add relationship index (theme-based search)
    # print("\n📥 Adding relationship indexes...")
    # for idx in relationship_index_json:
    #     neo4j_handler.add_relationship_index(
    #         relation_key=idx["key"],
    #         relation_value=idx["value"]
    #     )
    
    print("\n✅ All data loaded successfully!")



if __name__ == "__main__":
    
    # Sample data (replace with your actual extracted data)
    entities = [
        {"name": "Dr. Sarah Chen", "type": "Person"},
        {"name": "Stanford Medical Center", "type": "Organization"},
        {"name": "Dr. Michael Torres", "type": "Person"},
        {"name": "MIT", "type": "Organization"},
        {"name": "National Heart Institute", "type": "Organization"},
        {"name": "Heart Disease", "type": "MedicalCondition"},
        {"name": "Arrhythmias", "type": "MedicalCondition"},
        {"name": "AI Diagnostic Tools", "type": "Technology"},
    ]
    
    relationships = [
        {"source": "Dr. Sarah Chen", "relation": "works_at", "target": "Stanford Medical Center"},
        {"source": "Dr. Sarah Chen", "relation": "specializes_in", "target": "Heart Disease"},
        {"source": "Dr. Sarah Chen", "relation": "researches", "target": "Arrhythmias"},
        {"source": "Dr. Sarah Chen", "relation": "collaborates_with", "target": "Dr. Michael Torres"},
        {"source": "Dr. Sarah Chen", "relation": "develops", "target": "AI Diagnostic Tools"},
        {"source": "Dr. Michael Torres", "relation": "works_at", "target": "MIT"},
        {"source": "Dr. Michael Torres", "relation": "develops", "target": "AI Diagnostic Tools"},
        {"source": "AI Diagnostic Tools", "relation": "diagnoses", "target": "Arrhythmias"},
        {"source": "National Heart Institute", "relation": "funds", "target": "Dr. Sarah Chen"},
    ]
    
    entity_index = [
        {
            "key": "Dr. Sarah Chen",
            "value": "Cardiologist at Stanford Medical Center specializing in heart disease treatment. Published 2024 research demonstrating AI can diagnose arrhythmias with 95% accuracy."
        },
        {
            "key": "AI Diagnostic Tools",
            "value": "Machine learning-based technology that detects irregular heartbeats with 95% accuracy. Developed by Dr. Sarah Chen and Dr. Michael Torres."
        }
    ]
    
    relationship_index = [
        {
            "key": "AI cardiac diagnosis, arrhythmia detection technology",
            "value": "Machine learning models can detect arrhythmias with 95% accuracy, potentially revolutionizing cardiac care."
        },
        {
            "key": "Stanford MIT collaboration, cross-institutional research",
            "value": "Dr. Sarah Chen from Stanford collaborates with Dr. Michael Torres from MIT to develop AI diagnostic tools."
        }
    ]
    
    # Initialize Neo4j connection
    # IMPORTANT: Change password to match your Neo4j installation
    neo4j = Neo4jLightRAG(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="yourpassword"  # CHANGE THIS!
    )
    
    try:
        # Optional: Clear existing data
        # neo4j.clear_database()
        
        # Load all data
        load_lightrag_data(neo4j, entities, relationships, 
                          entity_index, relationship_index)
        
        # Example queries
        print("\n" + "="*60)
        print("EXAMPLE QUERIES")
        print("="*60 + "\n")
        
        # Query 1: Get entity information
        print("Query: Who is Dr. Sarah Chen?")
        result = neo4j.query_entity("Dr. Sarah Chen")
        if result:
            print(f"  Name: {result['name']}")
            print(f"  Type: {result['type']}")
            print(f"  Summary: {result['summary']}")
        
        # Query 2: Get relationships
        print("\nQuery: What are Dr. Sarah Chen's relationships?")
        relationships = neo4j.query_relationships("Dr. Sarah Chen")
        for rel in relationships:
            print(f"  {rel['entity']} -[{rel['relationship']}]-> {rel['connected_to']}")
        
        # Query 3: Search by theme
        print("\nQuery: Find research about 'AI diagnosis'")
        results = neo4j.search_by_theme("AI diagnosis")
        for result in results:
            print(f"  Theme: {result['theme']}")
            print(f"  Summary: {result['summary']}\n")
        
        # Visualize the graph
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60 + "\n")
        
        graph_data = neo4j.get_full_graph()
        
        # Static image with NetworkX
        # visualize_with_networkx(graph_data, "lightrag_graph.png")
        
        # Interactive HTML with PyVis
        # visualize_with_pyvis(graph_data, "lightrag_graph_interactive.html")
        
    finally:
        neo4j.close()
        print("\n✅ Done! Connection closed.")


# ============================================================================
# ADDITIONAL NOTES
# ============================================================================

"""
Visualization:

1. **Neo4j Browser** (Built-in, Best Option)
   - Open http://localhost:7474 after starting Neo4j
   - Run Cypher queries to visualize
   - Example: MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 25
   - Interactive, beautiful, and built-in!
"""
