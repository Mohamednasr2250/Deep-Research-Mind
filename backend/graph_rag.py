import json
import re
import mlflow
from typing import List, Dict, Tuple
from langchain_core.documents import Document

try:
    import networkx as nx
except ImportError:
    nx = None


def extract_entities_relations(chunk_text: str, llm) -> List[Tuple[str, str, str]]:
    prompt = f"""Extract key entities and their relationships from this text.
Output ONLY triples in this exact format, one per line:
entity1 | relation | entity2

Text:
{chunk_text[:800]}

Triples:"""

    try:
        response = llm.invoke(prompt).strip()
    except Exception:
        return []

    triples = []
    for line in response.split("\n"):
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 3 and all(parts):
            triples.append(tuple(parts))

    return triples


def extract_from_documents(docs: List[Document], llm) -> List[Tuple[str, str, str, str]]:
    all_triples = []
    for doc in docs:
        source = doc.metadata.get("paper_title", doc.metadata.get("source", "unknown"))
        triples = extract_entities_relations(doc.page_content, llm)
        for (e1, rel, e2) in triples:
            all_triples.append((e1, rel, e2, source))

    with mlflow.start_run(run_name="graph_extraction"):
        mlflow.log_metric("triples_extracted", len(all_triples))
        mlflow.log_metric("chunks_processed", len(docs))

    return all_triples


def build_graph(triples: List[Tuple[str, str, str, str]]):
    if nx is None:
        raise ImportError("networkx not installed — run: pip install networkx python-louvain")

    G = nx.Graph()
    for e1, rel, e2, source in triples:
        G.add_node(e1)
        G.add_node(e2)
        if G.has_edge(e1, e2):
            G[e1][e2]["relations"].append(rel)
            G[e1][e2]["sources"].add(source)
        else:
            G.add_edge(e1, e2, relations=[rel], sources={source})

    return G


def detect_communities(G) -> Dict[int, List[str]]:
    try:
        import community as community_louvain
    except ImportError:
        raise ImportError("python-louvain not installed — run: pip install python-louvain")

    partition = community_louvain.best_partition(G)

    communities: Dict[int, List[str]] = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, []).append(node)

    with mlflow.start_run(run_name="community_detection"):
        mlflow.log_metric("num_communities", len(communities))
        mlflow.log_metric("num_nodes", G.number_of_nodes())

    return communities


def summarize_community(entities: List[str], G, llm) -> str:
    edges_text = []
    subgraph = G.subgraph(entities)
    for u, v, data in subgraph.edges(data=True):
        relations = ", ".join(data.get("relations", []))
        edges_text.append(f"{u} --[{relations}]--> {v}")

    if not edges_text:
        return f"Entities: {', '.join(entities)} (no extracted relationships within this cluster)"

    prompt = f"""These entities and relationships form a cluster/theme extracted from research papers:
{chr(10).join(edges_text[:20])}

Summarize in 2-3 sentences what overall theme or topic this cluster represents:"""

    try:
        return llm.invoke(prompt).strip()
    except Exception:
        return f"Theme involving: {', '.join(entities[:5])}"


def build_community_summaries(G, communities: Dict[int, List[str]], llm) -> Dict[int, dict]:
    summaries = {}
    for comm_id, entities in communities.items():
        if len(entities) < 2:
            continue
        summary = summarize_community(entities, G, llm)
        summaries[comm_id] = {"entities": entities, "summary": summary}
    return summaries


def find_relationship_path(G, entity1: str, entity2: str) -> List[dict]:
    if nx is None or entity1 not in G or entity2 not in G:
        return []

    try:
        path = nx.shortest_path(G, source=entity1, target=entity2)
    except nx.NetworkXNoPath:
        return []

    path_details = []
    for i in range(len(path) - 1):
        edge_data = G[path[i]][path[i + 1]]
        path_details.append({
            "from": path[i],
            "to": path[i + 1],
            "relations": edge_data.get("relations", []),
            "sources": list(edge_data.get("sources", []))
        })

    return path_details


def answer_relationship_query(query: str, G, llm, entity_hint_1: str = None, entity_hint_2: str = None) -> str:
    if not entity_hint_1 or not entity_hint_2:
        extract_prompt = f"""Extract the two main entities/concepts being compared or related in this query.
Output format: entity1 | entity2

Query: {query}"""
        try:
            response = llm.invoke(extract_prompt).strip()
            parts = [p.strip() for p in response.split("|")]
            if len(parts) == 2:
                entity_hint_1, entity_hint_2 = parts
        except Exception:
            return "Could not identify two entities to relate from this query."

    path = find_relationship_path(G, entity_hint_1, entity_hint_2)

    if not path:
        return f"No direct relationship path found between '{entity_hint_1}' and '{entity_hint_2}' in the graph."

    path_text = "\n".join(
        f"{p['from']} --[{', '.join(p['relations'])}]--> {p['to']} (source: {', '.join(p['sources'])})"
        for p in path
    )

    prompt = f"""Based on this relationship path extracted from research papers, answer the question.

Question: {query}
Relationship path:
{path_text}

Answer:"""
    return llm.invoke(prompt).strip()
