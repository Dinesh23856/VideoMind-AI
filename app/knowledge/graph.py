"""Knowledge Graph representation, neighborhood traversal, and prerequisite path querying."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set
from app.domain.models import ConceptNode, KnowledgeEdgeType, RelationshipEdge


class KnowledgeGraph:
    """
    In-memory and queryable Knowledge Graph engine.
    Stores ConceptNodes and RelationshipEdges with fast bidirectional lookup.
    """

    def __init__(self):
        self.nodes: Dict[str, ConceptNode] = {}
        # adjacency: source_id -> list of edges
        self.adj: Dict[str, List[RelationshipEdge]] = {}
        # reverse adjacency: target_id -> list of edges
        self.rev_adj: Dict[str, List[RelationshipEdge]] = {}

    def add_concept(self, concept: ConceptNode) -> None:
        self.nodes[concept.id] = concept
        if concept.id not in self.adj:
            self.adj[concept.id] = []
        if concept.id not in self.rev_adj:
            self.rev_adj[concept.id] = []

    add_node = add_concept

    def add_edge(
        self,
        edge: RelationshipEdge | str,
        target_id: Optional[str] = None,
        relation_type: str = "RELATED_TO",
        confidence: float = 1.0,
    ) -> None:
        if isinstance(edge, str):
            if target_id is None:
                raise ValueError("target_id must be provided when adding an edge by ID strings")
            edge = RelationshipEdge(
                source_id=edge,
                target_id=target_id,
                relation=getattr(KnowledgeEdgeType, relation_type, KnowledgeEdgeType.RELATED_TO),
                confidence=confidence,
            )


        if edge.source_id not in self.adj:
            self.adj[edge.source_id] = []
        self.adj[edge.source_id].append(edge)

        if edge.target_id not in self.rev_adj:
            self.rev_adj[edge.target_id] = []
        self.rev_adj[edge.target_id].append(edge)


    def get_concept_by_name(self, name: str) -> Optional[ConceptNode]:
        target = name.strip().lower()
        for node in self.nodes.values():
            if node.name.lower() == target or any(a.lower() == target for a in node.aliases):
                return node
        return None

    def get_neighbors(self, concept_id: str, max_depth: int = 1) -> Dict[str, Any]:
        """Returns connected sub-graph up to max_depth."""
        visited_nodes: Set[str] = set()
        collected_edges: List[RelationshipEdge] = []

        def dfs(curr_id: str, depth: int):
            if depth > max_depth or curr_id in visited_nodes:
                return
            visited_nodes.add(curr_id)
            for edge in self.adj.get(curr_id, []):
                collected_edges.append(edge)
                dfs(edge.target_id, depth + 1)
            for edge in self.rev_adj.get(curr_id, []):
                collected_edges.append(edge)
                dfs(edge.source_id, depth + 1)

        dfs(concept_id, 0)
        return {
            "nodes": [self.nodes[n_id] for n_id in visited_nodes if n_id in self.nodes],
            "edges": collected_edges,
        }

    def get_prerequisites(self, concept_name: str) -> List[ConceptNode]:
        """Finds all concepts that must be learned before this concept."""
        concept = self.get_concept_by_name(concept_name)
        if not concept:
            return []

        prereqs: List[ConceptNode] = []
        visited: Set[str] = set()

        def collect(curr_id: str):
            for edge in self.adj.get(curr_id, []):
                if edge.relation in (KnowledgeEdgeType.PREREQUISITE_OF, KnowledgeEdgeType.DEPENDS_ON):
                    target_node = self.nodes.get(edge.target_id)
                    if target_node and target_node.id not in visited:
                        visited.add(target_node.id)
                        prereqs.append(target_node)
                        collect(target_node.id)

        collect(concept.id)
        return prereqs

    def export_cytoscape_elements(self) -> List[Dict[str, Any]]:
        """Exports graph in Cytoscape.js format for interactive frontend visualization."""
        elements: List[Dict[str, Any]] = []
        for node_id, node in self.nodes.items():
            elements.append(
                {
                    "data": {
                        "id": node_id,
                        "label": node.name,
                        "definition": node.definition,
                        "importance": node.importance,
                    }
                }
            )
        for src, edges in self.adj.items():
            for edge in edges:
                elements.append(
                    {
                        "data": {
                            "id": f"{edge.source_id}_{edge.target_id}",
                            "source": edge.source_id,
                            "target": edge.target_id,
                            "label": edge.relation.value,
                        }
                    }
                )
        return elements
