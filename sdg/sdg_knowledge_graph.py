#!/usr/bin/env python3
"""
Convert SDG indicators to knowledge graph using LangChain's LLMGraphTransformer
"""

import json
import os
import asyncio
from dotenv import load_dotenv
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from pyvis.network import Network

load_dotenv()

async def create_sdg_knowledge_graph():
    # Initialize LLM and graph transformer
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    graph_transformer = LLMGraphTransformer(llm=llm)
    
    # Load SDG data
    json_path = os.path.join(os.path.dirname(__file__), 'sdg_indicators.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        sdg_data = json.load(f)
    
    # Convert SDG data to text documents
    documents = []
    
    for goal in sdg_data:
        goal_id = goal['goalId']
        goal_name = goal['goalName']
        
        # Create text representation of each SDG goal and its indicators
        text = f"SDG Goal {goal_id}: {goal_name}\n\n"
        text += f"This is Sustainable Development Goal number {goal_id} of the United Nations.\n"
        text += f"It focuses on: {goal_name}\n\n"
        text += f"Goal {goal_id} has {len(goal['indicators'])} indicators:\n\n"
        
        for indicator in goal['indicators']:
            text += f"Indicator {indicator['code']}: {indicator['description']}\n"
            # Add relationships in the text
            text += f"Indicator {indicator['code']} measures progress toward SDG Goal {goal_id}.\n"
            text += f"This indicator tracks: {indicator['description']}\n\n"
        
        documents.append(Document(page_content=text, metadata={"sdg_goal": goal_id}))
    
    print(f"Created {len(documents)} documents from SDG data")
    
    # Convert to graph documents
    print("Converting to graph documents...")
    all_graph_documents = []
    
    # Process each goal separately to avoid overwhelming the LLM
    for doc in documents:
        print(f"Processing SDG {doc.metadata['sdg_goal']}...")
        try:
            graph_docs = await graph_transformer.aconvert_to_graph_documents([doc])
            all_graph_documents.extend(graph_docs)
        except Exception as e:
            print(f"Error processing SDG {doc.metadata['sdg_goal']}: {e}")
            continue
    
    return all_graph_documents


async def extract_keywords_from_graph(graph_documents, goal_number=None):
    """Extract keywords from the knowledge graph for a specific goal or all goals"""
    
    keyword_clusters = {}
    
    for graph_doc in graph_documents:
        # Get the goal number from metadata
        sdg_goal = graph_doc.source.metadata.get('sdg_goal', 'Unknown')
        
        # Skip if we're looking for a specific goal and this isn't it
        if goal_number and sdg_goal != str(goal_number):
            continue
        
        # Extract unique concepts from nodes
        keywords = set()
        
        for node in graph_doc.nodes:
            # Skip generic nodes
            if node.id.lower() in ['united nations', 'sustainable development goal']:
                continue
            
            # Add node IDs as keywords (they usually contain important concepts)
            if len(node.id) > 3:  # Skip very short nodes
                keywords.add(node.id.lower())
        
        # Extract concepts from relationships
        for rel in graph_doc.relationships:
            # Relationship types often contain important concepts
            if rel.type and len(rel.type) > 3:
                keywords.add(rel.type.lower())
        
        if keywords:
            keyword_clusters[f"Goal {sdg_goal}"] = list(keywords)
    
    return keyword_clusters

def visualize_graph(graph_documents):
    """Visualize the graph following the notebook approach"""
    
    # Create network
    net = Network(height="1200px", width="100%", directed=True,
                      notebook=False, bgcolor="#222222", font_color="white")
    
    # Collect all nodes and relationships
    all_nodes = []
    all_relationships = []
    
    for graph_doc in graph_documents:
        all_nodes.extend(graph_doc.nodes)
        all_relationships.extend(graph_doc.relationships)
    
    # Build lookup for valid nodes
    node_dict = {node.id: node for node in all_nodes}
    
    # Filter out invalid edges and collect valid node IDs
    valid_edges = []
    valid_node_ids = set()
    for rel in all_relationships:
        if rel.source.id in node_dict and rel.target.id in node_dict:
            valid_edges.append(rel)
            valid_node_ids.update([rel.source.id, rel.target.id])

    # Add valid nodes
    for node_id in valid_node_ids:
        node = node_dict[node_id]
        try:
            net.add_node(node.id, label=node.id, title=node.type, group=node.type)
        except:
            continue  # skip if error

    # Add valid edges
    for rel in valid_edges:
        try:
            net.add_edge(rel.source.id, rel.target.id, label=rel.type.lower())
        except:
            continue  # skip if error

    # Configure physics
    net.set_options("""
            {
                "physics": {
                    "forceAtlas2Based": {
                        "gravitationalConstant": -100,
                        "centralGravity": 0.01,
                        "springLength": 200,
                        "springConstant": 0.08
                    },
                    "minVelocity": 0.75,
                    "solver": "forceAtlas2Based"
                }
            }
            """)
        
    output_file = "sdg_knowledge_graph.html"
    net.save_graph(output_file)
    print(f"Graph saved to {os.path.abspath(output_file)}")

    # Try to open in browser
    try:
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(output_file)}")
    except:
        print("Could not open browser automatically")

if __name__ == "__main__":
    # Run the async function
    graph_documents = asyncio.run(create_sdg_knowledge_graph())
    
    print(f"\nTotal graph documents created: {len(graph_documents)}")
    
    # Visualize the graph (exactly like in the notebook)
    visualize_graph(graph_documents)
    
    # Print some nodes and relationships like in the notebook
    if graph_documents:
        print(f"\nNodes from first document: {graph_documents[0].nodes}")
        print(f"\nRelationships from first document: {graph_documents[0].relationships}")