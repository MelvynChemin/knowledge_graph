import json
from neo4j import GraphDatabase
from neo4j_lightrag_storage import load_lightrag_data, Neo4jLightRAG

#Testing to see if the graph merges if two nodes have the same name



neo4j = Neo4jLightRAG(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="yourpassword"  # CHANGE THIS!
    )



s_t_e_1 = [{'name': 'Char1', 'type': 'Person'}, {'name': 'Plc1', 'type': 'Place'}]
s_t_r_1 = [{'source': 'Char1', 'target': 'Plc1', 'relation': 'LIVES_IN'}]
s_k_v_1 = [{'key': 'Char1', 'value': 'A character who lives in Plc1.'}]

s_t_e_2 = [{'name': 'Char2', 'type': 'Person'}, {'name': 'Plc1', 'type': 'Place'}]
s_t_r_2 = [{'source': 'Char2', 'target': 'Plc1', 'relation': 'VISITS'}]
s_k_v_2 = [{'key': 'Char2', 'value': 'A character who visits Plc1.'}]



load_lightrag_data(neo4j, s_t_e_1, s_t_r_1, s_k_v_1)
print('Sucessfully loaded knowledge graph1 data into Neo4j! open http://localhost:7474 to view it.')
load_lightrag_data(neo4j, s_t_e_2, s_t_r_2, s_k_v_2)
print('Sucessfully loaded knowledge graph2 data into Neo4j! open http://localhost:7474 to view it.')