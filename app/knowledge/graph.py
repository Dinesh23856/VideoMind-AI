"""
Knowledge Graph 3.0: Provenance-Aware Temporal Knowledge Graph & Evidence Graph.

Features:
- Typed multi-modal nodes (VIDEO, LECTURE, SPEAKER, TOPIC, CONCEPT, ENTITY, CLAIM,
  DEFINITION, EXAMPLE, CODE, FORMULA, QUESTION, ANSWER, SOURCE, TRANSCRIPT_SEGMENT,
  KEYFRAME, OCR_REGION, AUDIO_EVENT)
- Provenance-aware relationships with timestamps, confidence, and source citations
- Bounded multi-hop traversal with cycle prevention and depth/node limits
- Temporal knowledge queries (before, after, between, examples after definition)
- Allen's Interval Algebra integration for multimodal event intervals
- First-class claim & evidence graph integration with contradiction detection
- Graph quality control (dangling edge, orphan node, duplicate edge, and leak checks)
- Multi-tenant workspace isolation across all lookups, traversals, and queries
- Safe graph caching with version invalidation and snapshot import/export
- Cytoscape.js, Mermaid, and Neo4j Cypher exporters
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from app.domain.models import (
    ConceptNode,
    FactClaim,
    FactVerificationStatus,
    KnowledgeEdgeType,
    KnowledgeObjectType,
    RelationshipEdge,
)


class KnowledgeGraph:
    """
    Canonical Knowledge Graph & Evidence Graph engine for VideoMind-AI.
    Provides graph querying, multi-hop traversal, temporal reasoning, claim verification,
    and multi-tenant isolation.
    """

    def __init__(self):
        self.nodes: Dict[str, ConceptNode] = {}
        # adjacency: source_id -> list of edges
        self.adj: Dict[str, List[RelationshipEdge]] = {}
        # reverse adjacency: target_id -> list of edges
        self.rev_adj: Dict[str, List[RelationshipEdge]] = {}
        # In-memory query cache: key -> cached result
        self._query_cache: Dict[str, Any] = {}
        # Graph version counter for cache invalidation
        self.version: int = 1

    def _invalidate_cache(self) -> None:
        self.version += 1
        self._query_cache.clear()

    # ─────────────────────────────────────────────────────────────────────────
    # Node Management
    # ─────────────────────────────────────────────────────────────────────────

    def add_concept(self, concept: ConceptNode) -> None:
        """Adds a ConceptNode to the graph, initializing adjacency lists."""
        self.nodes[concept.id] = concept
        if concept.id not in self.adj:
            self.adj[concept.id] = []
        if concept.id not in self.rev_adj:
            self.rev_adj[concept.id] = []
        self._invalidate_cache()

    add_node = add_concept

    def register_node(
        self,
        node_id: str,
        name: str,
        definition: str = "",
        node_type: KnowledgeObjectType = KnowledgeObjectType.CONCEPT,
        importance: float = 0.8,
        timestamps: Optional[List[float]] = None,
        aliases: Optional[List[str]] = None,
        category: str = "General",
        video_id: Optional[str] = None,
        document_id: Optional[str] = None,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        source_modality: str = "text",
        provenance: Optional[Dict[str, Any]] = None,
        classification: str = "REAL",
        tenant_id: str = "default",
        workspace_id: str = "default",
        user_id: Optional[str] = None,
    ) -> ConceptNode:
        """Helper to create and add a typed node with complete provenance."""
        # Convert start_ms/end_ms to timestamps if timestamps not provided
        ts = timestamps or []
        if not ts and start_ms is not None:
            ts = [round(start_ms / 1000.0, 3)]
            if end_ms is not None:
                ts.append(round(end_ms / 1000.0, 3))

        node = ConceptNode(
            id=node_id,
            name=name,
            definition=definition,
            importance=importance,
            timestamps=ts,
            aliases=aliases or [],
            category=category,
            video_id=video_id,
            node_type=node_type,
            document_id=document_id,
            start_ms=start_ms,
            end_ms=end_ms,
            source_modality=source_modality,
            provenance=provenance or {},
            classification=classification,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            user_id=user_id,
        )
        self.add_concept(node)
        return node

    def get_node(self, node_id: str, tenant_id: Optional[str] = None) -> Optional[ConceptNode]:
        """Retrieves a node by ID, respecting tenant isolation."""
        node = self.nodes.get(node_id)
        if not node:
            return None
        if tenant_id and node.tenant_id not in (tenant_id, "default"):
            return None
        return node

    def get_nodes(
        self,
        tenant_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None,
        node_type: Optional[KnowledgeObjectType] = None,
        video_id: Optional[str] = None,
    ) -> List[ConceptNode]:
        """Filters nodes by tenant, workspace, user, node type, and video."""
        result: List[ConceptNode] = []
        for node in self.nodes.values():
            if tenant_id and node.tenant_id not in (tenant_id, "default"):
                continue
            if workspace_id and node.workspace_id not in (workspace_id, "default"):
                continue
            if user_id and node.user_id and node.user_id != user_id:
                continue
            if node_type and node.node_type != node_type:
                continue
            if video_id and node.video_id != video_id:
                continue
            result.append(node)
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Edge Management
    # ─────────────────────────────────────────────────────────────────────────

    def add_edge(
        self,
        edge: RelationshipEdge | str,
        target_id: Optional[str] = None,
        relation_type: str = "RELATED_TO",
        confidence: float = 1.0,
        description: str = "",
        video_id: Optional[str] = None,
        document_id: Optional[str] = None,
        timestamp_start_ms: Optional[int] = None,
        timestamp_end_ms: Optional[int] = None,
        source_modality: Optional[str] = None,
        classification: str = "REAL",
        tenant_id: str = "default",
        workspace_id: str = "default",
        user_id: Optional[str] = None,
    ) -> RelationshipEdge:
        """Adds a relationship edge with full provenance and tenant metadata."""
        if isinstance(edge, str):
            if target_id is None:
                raise ValueError("target_id must be provided when adding an edge by ID strings")
            rel_enum = getattr(KnowledgeEdgeType, relation_type, None)
            if rel_enum is None:
                rel_enum = KnowledgeEdgeType.RELATED_TO
            edge = RelationshipEdge(
                source_id=edge,
                target_id=target_id,
                relation=rel_enum,
                confidence=confidence,
                description=description,
                video_id=video_id,
                document_id=document_id,
                timestamp_start_ms=timestamp_start_ms,
                timestamp_end_ms=timestamp_end_ms,
                source_modality=source_modality,
                classification=classification,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                user_id=user_id,
            )

        if edge.source_id not in self.adj:
            self.adj[edge.source_id] = []
        self.adj[edge.source_id].append(edge)

        if edge.target_id not in self.rev_adj:
            self.rev_adj[edge.target_id] = []
        self.rev_adj[edge.target_id].append(edge)

        self._invalidate_cache()
        return edge

    add_relationship = add_edge

    def get_edges(
        self,
        tenant_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        video_id: Optional[str] = None,
    ) -> List[RelationshipEdge]:
        """Returns all edges matching tenant/workspace filters."""
        result: List[RelationshipEdge] = []
        seen = set()
        for edges in self.adj.values():
            for e in edges:
                edge_sig = (e.source_id, e.target_id, e.relation.value)
                if edge_sig in seen:
                    continue
                if tenant_id and e.tenant_id not in (tenant_id, "default"):
                    continue
                if workspace_id and e.workspace_id not in (workspace_id, "default"):
                    continue
                if video_id and e.video_id != video_id:
                    continue
                seen.add(edge_sig)
                result.append(e)
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Concept Queries & Lookups
    # ─────────────────────────────────────────────────────────────────────────

    def get_concept_by_name(self, name: str, tenant_id: Optional[str] = None) -> Optional[ConceptNode]:
        """Finds concept node by name or alias, case-insensitively, with tenant checking."""
        target = name.strip().lower()
        for node in self.nodes.values():
            if tenant_id and node.tenant_id not in (tenant_id, "default"):
                continue
            if node.name.lower() == target or any(a.lower() == target for a in node.aliases):
                return node
        return None

    def search_nodes(
        self,
        query: str,
        limit: int = 10,
        tenant_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> List[ConceptNode]:
        """Full-text keyword search across node names, definitions, and aliases."""
        q_lower = query.strip().lower()
        candidates: List[Tuple[ConceptNode, float]] = []

        for node in self.nodes.values():
            if tenant_id and node.tenant_id not in (tenant_id, "default"):
                continue
            if workspace_id and node.workspace_id not in (workspace_id, "default"):
                continue

            score = 0.0
            name_lower = node.name.lower()
            def_lower = node.definition.lower()

            if name_lower == q_lower:
                score += 5.0
            elif q_lower in name_lower:
                score += 3.0
            elif any(q_lower in a.lower() for a in node.aliases):
                score += 2.5
            elif q_lower in def_lower:
                score += 1.0

            if score > 0:
                candidates.append((node, score * node.importance))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in candidates[:limit]]

    # ─────────────────────────────────────────────────────────────────────────
    # Subgraph & Bounded Traversal
    # ─────────────────────────────────────────────────────────────────────────

    def get_neighbors(
        self,
        concept_id: str,
        max_depth: int = 1,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Returns connected sub-graph up to max_depth, respecting tenant isolation."""
        visited_nodes: Set[str] = set()
        collected_edges: List[RelationshipEdge] = []

        def dfs(curr_id: str, depth: int):
            if depth > max_depth or curr_id in visited_nodes:
                return
            node = self.nodes.get(curr_id)
            if not node:
                return
            if tenant_id and node.tenant_id not in (tenant_id, "default"):
                return

            visited_nodes.add(curr_id)
            for edge in self.adj.get(curr_id, []):
                if tenant_id and edge.tenant_id not in (tenant_id, "default"):
                    continue
                collected_edges.append(edge)
                dfs(edge.target_id, depth + 1)
            for edge in self.rev_adj.get(curr_id, []):
                if tenant_id and edge.tenant_id not in (tenant_id, "default"):
                    continue
                collected_edges.append(edge)
                dfs(edge.source_id, depth + 1)

        dfs(concept_id, 0)
        return {
            "nodes": [self.nodes[n_id] for n_id in visited_nodes if n_id in self.nodes],
            "edges": collected_edges,
        }

    def traverse_bounded(
        self,
        start_node_ids: str | List[str],
        relationship_types: Optional[List[str]] = None,
        max_hops: int = 2,
        max_nodes: int = 50,
        direction: str = "both",  # "outgoing", "incoming", "both"
        tenant_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Bounded multi-hop graph traversal with cycle prevention, max_hops, and max_nodes limits.
        Supports single or multiple start node seeds and enforces multi-tenant workspace isolation.
        """
        if isinstance(start_node_ids, str):
            seeds = [start_node_ids]
        else:
            seeds = list(start_node_ids)

        valid_seeds = [
            sid for sid in seeds
            if sid in self.nodes and (not tenant_id or self.nodes[sid].tenant_id in (tenant_id, "default"))
        ]
        if not valid_seeds:
            return {
                "start_node_id": seeds[0] if seeds else "",
                "start_node_ids": seeds,
                "nodes": [],
                "edges": [],
                "paths": [],
                "total_nodes": 0,
                "total_edges": 0,
                "hops_reached": 0,
            }

        rel_filter_set = set(relationship_types) if relationship_types else None
        visited_nodes: Set[str] = set(valid_seeds)
        visited_edges: List[RelationshipEdge] = []
        seen_edge_keys: Set[Tuple[str, str, str]] = set()
        paths: List[List[str]] = [[s] for s in valid_seeds]
        queue: List[Tuple[str, int, List[str]]] = [(s, 0, [s]) for s in valid_seeds]
        max_hop_reached = 0

        while queue and len(visited_nodes) < max_nodes:
            curr_id, depth, curr_path = queue.pop(0)
            max_hop_reached = max(max_hop_reached, depth)

            if depth >= max_hops:
                continue

            candidates: List[Tuple[RelationshipEdge, str]] = []

            if direction in ("outgoing", "both"):
                for edge in self.adj.get(curr_id, []):
                    if rel_filter_set and edge.relation.value not in rel_filter_set:
                        continue
                    if tenant_id and edge.tenant_id not in (tenant_id, "default"):
                        continue
                    candidates.append((edge, edge.target_id))

            if direction in ("incoming", "both"):
                for edge in self.rev_adj.get(curr_id, []):
                    if rel_filter_set and edge.relation.value not in rel_filter_set:
                        continue
                    if tenant_id and edge.tenant_id not in (tenant_id, "default"):
                        continue
                    candidates.append((edge, edge.source_id))

            for edge, next_id in candidates:
                next_node = self.nodes.get(next_id)
                if not next_node:
                    continue
                if tenant_id and next_node.tenant_id not in (tenant_id, "default"):
                    continue
                if workspace_id and next_node.workspace_id not in (workspace_id, "default"):
                    continue

                edge_k = (edge.source_id, edge.target_id, edge.relation.value)
                if edge_k not in seen_edge_keys:
                    seen_edge_keys.add(edge_k)
                    visited_edges.append(edge)

                if next_id not in visited_nodes and len(visited_nodes) < max_nodes:
                    visited_nodes.add(next_id)
                    new_path = curr_path + [next_id]
                    paths.append(new_path)
                    queue.append((next_id, depth + 1, new_path))

        return {
            "start_node_id": seeds[0] if seeds else "",
            "start_node_ids": seeds,
            "nodes": [self.nodes[n_id] for n_id in visited_nodes if n_id in self.nodes],
            "edges": visited_edges,
            "paths": paths,
            "total_nodes": len(visited_nodes),
            "total_edges": len(visited_edges),
            "hops_reached": max_hop_reached,
        }

    def traverse(
        self,
        start_node_id: str,
        relationship_types: Optional[List[str]] = None,
        max_hops: int = 2,
        max_nodes: int = 50,
        direction: str = "both",
        tenant_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Bounded single-seed traversal delegating to traverse_bounded."""
        return self.traverse_bounded(
            start_node_ids=start_node_id,
            relationship_types=relationship_types,
            max_hops=max_hops,
            max_nodes=max_nodes,
            direction=direction,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            user_id=user_id,
        )


    # ─────────────────────────────────────────────────────────────────────────
    # Prerequisites & Curriculum
    # ─────────────────────────────────────────────────────────────────────────

    def get_prerequisites(self, concept_name: str, tenant_id: Optional[str] = None) -> List[ConceptNode]:
        """Finds all concepts that must be learned before this concept."""
        concept = self.get_concept_by_name(concept_name, tenant_id=tenant_id)
        if not concept:
            return []

        prereqs: List[ConceptNode] = []
        visited: Set[str] = set()

        def collect(curr_id: str):
            for edge in self.adj.get(curr_id, []):
                if edge.relation == KnowledgeEdgeType.DEPENDS_ON:
                    target_node = self.nodes.get(edge.target_id)
                    if target_node and target_node.id not in visited and target_node.id != concept.id:
                        if tenant_id and target_node.tenant_id not in (tenant_id, "default"):
                            continue
                        visited.add(target_node.id)
                        prereqs.append(target_node)
                        collect(target_node.id)

            for edge in self.rev_adj.get(curr_id, []):
                if edge.relation == KnowledgeEdgeType.PREREQUISITE_OF:
                    src_node = self.nodes.get(edge.source_id)
                    if src_node and src_node.id not in visited and src_node.id != concept.id:
                        if tenant_id and src_node.tenant_id not in (tenant_id, "default"):
                            continue
                        visited.add(src_node.id)
                        prereqs.append(src_node)
                        collect(src_node.id)

        collect(concept.id)
        return prereqs

    def get_topological_learning_path(self, tenant_id: Optional[str] = None) -> List[ConceptNode]:
        """Computes a topologically ordered learning curriculum based on prerequisite dependencies."""
        relevant_nodes = {
            nid: node
            for nid, node in self.nodes.items()
            if not tenant_id or node.tenant_id in (tenant_id, "default")
        }
        in_degree: Dict[str, int] = {node_id: 0 for node_id in relevant_nodes}

        # Calculate prerequisite dependencies
        for src_id, edges in self.adj.items():
            if src_id not in relevant_nodes:
                continue
            for edge in edges:
                if edge.target_id in relevant_nodes and edge.relation in (
                    KnowledgeEdgeType.PREREQUISITE_OF,
                    KnowledgeEdgeType.DEPENDS_ON,
                ):
                    in_degree[edge.target_id] = in_degree.get(edge.target_id, 0) + 1

        queue = [n_id for n_id, deg in in_degree.items() if deg == 0]
        sorted_nodes: List[ConceptNode] = []

        while queue:
            curr_id = queue.pop(0)
            if curr_id in relevant_nodes:
                sorted_nodes.append(relevant_nodes[curr_id])
            for edge in self.adj.get(curr_id, []):
                if edge.target_id in relevant_nodes and edge.relation in (
                    KnowledgeEdgeType.PREREQUISITE_OF,
                    KnowledgeEdgeType.DEPENDS_ON,
                ):
                    in_degree[edge.target_id] = max(0, in_degree[edge.target_id] - 1)
                    if in_degree[edge.target_id] == 0:
                        queue.append(edge.target_id)

        # Include any remaining nodes with cycles or unattached
        for node_id, node in relevant_nodes.items():
            if node not in sorted_nodes:
                sorted_nodes.append(node)

        return sorted_nodes

    # ─────────────────────────────────────────────────────────────────────────
    # Temporal Knowledge Reasoning
    # ─────────────────────────────────────────────────────────────────────────

    def _get_node_time_range(self, node: ConceptNode) -> Tuple[Optional[float], Optional[float]]:
        """Returns (start_sec, end_sec) for a node if available."""
        if node.start_ms is not None:
            s = node.start_ms / 1000.0
            e = (node.end_ms / 1000.0) if node.end_ms is not None else s
            return s, e
        if node.timestamps:
            return min(node.timestamps), max(node.timestamps)
        return None, None

    def what_was_discussed_before(
        self,
        target_node_id_or_time: str | float,
        delta_sec: float = 120.0,
        video_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[ConceptNode]:
        """Finds concepts discussed within delta_sec immediately before target time or concept start."""
        if isinstance(target_node_id_or_time, str):
            node = self.get_node(target_node_id_or_time, tenant_id=tenant_id)
            if not node:
                return []
            start_t, _ = self._get_node_time_range(node)
            if start_t is None:
                return []
            target_t = start_t
            if not video_id:
                video_id = node.video_id
        else:
            target_t = float(target_node_id_or_time)

        candidates: List[Tuple[ConceptNode, float]] = []
        for n in self.nodes.values():
            if tenant_id and n.tenant_id not in (tenant_id, "default"):
                continue
            if video_id and n.video_id != video_id:
                continue
            ns, ne = self._get_node_time_range(n)
            if ns is not None and ne is not None:
                if ne <= target_t and (target_t - ne) <= delta_sec:
                    candidates.append((n, target_t - ne))

        candidates.sort(key=lambda x: x[1])  # Closest prior first
        return [c[0] for c in candidates[:limit]]

    def what_was_discussed_after(
        self,
        target_node_id_or_time: str | float,
        delta_sec: float = 120.0,
        video_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[ConceptNode]:
        """Finds concepts discussed within delta_sec immediately after target time or concept end."""
        if isinstance(target_node_id_or_time, str):
            node = self.get_node(target_node_id_or_time, tenant_id=tenant_id)
            if not node:
                return []
            _, end_t = self._get_node_time_range(node)
            if end_t is None:
                return []
            target_t = end_t
            if not video_id:
                video_id = node.video_id
        else:
            target_t = float(target_node_id_or_time)

        candidates: List[Tuple[ConceptNode, float]] = []
        for n in self.nodes.values():
            if tenant_id and n.tenant_id not in (tenant_id, "default"):
                continue
            if video_id and n.video_id != video_id:
                continue
            ns, ne = self._get_node_time_range(n)
            if ns is not None:
                if ns >= target_t and (ns - target_t) <= delta_sec:
                    candidates.append((n, ns - target_t))

        candidates.sort(key=lambda x: x[1])  # Closest following first
        return [c[0] for c in candidates[:limit]]

    def concepts_between(
        self,
        start_time: float,
        end_time: float,
        video_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> List[ConceptNode]:
        """Returns all concepts that occur within or overlap the [start_time, end_time] interval."""
        results: List[ConceptNode] = []
        for n in self.nodes.values():
            if tenant_id and n.tenant_id not in (tenant_id, "default"):
                continue
            if video_id and n.video_id != video_id:
                continue
            ns, ne = self._get_node_time_range(n)
            if ns is not None and ne is not None:
                if max(ns, start_time) <= min(ne, end_time):
                    results.append(n)
        results.sort(key=lambda x: self._get_node_time_range(x)[0] or 0.0)
        return results

    def examples_after_definition(
        self,
        concept_name_or_id: str,
        video_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> List[ConceptNode]:
        """Finds EXAMPLE nodes occurring after the definition or first introduction of a concept."""
        concept = self.get_node(concept_name_or_id, tenant_id=tenant_id) or self.get_concept_by_name(concept_name_or_id, tenant_id=tenant_id)
        if not concept:
            return []

        c_start, c_end = self._get_node_time_range(concept)
        threshold_t = c_end if c_end is not None else (c_start if c_start is not None else 0.0)

        # 1. First look for explicit EXAMPLE_OF edges
        explicit_examples: List[ConceptNode] = []
        for edge in self.rev_adj.get(concept.id, []):
            if edge.relation == KnowledgeEdgeType.EXAMPLE_OF:
                src_node = self.nodes.get(edge.source_id)
                if src_node:
                    explicit_examples.append(src_node)
        if explicit_examples:
            return explicit_examples

        # 2. Look for EXAMPLE nodes occurring temporally after definition
        results: List[ConceptNode] = []
        for n in self.nodes.values():
            if tenant_id and n.tenant_id not in (tenant_id, "default"):
                continue
            if n.node_type in (KnowledgeObjectType.EXAMPLE, KnowledgeObjectType.CODE_SNIPPET, KnowledgeObjectType.CODE):
                ns, _ = self._get_node_time_range(n)
                if ns is not None and ns >= threshold_t:
                    results.append(n)

        results.sort(key=lambda x: self._get_node_time_range(x)[0] or 0.0)
        return results

    def find_first_introduction(
        self,
        concept_name: str,
        video_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> Optional[ConceptNode]:
        """Finds the earliest occurrence or definition of a concept."""
        concept = self.get_concept_by_name(concept_name, tenant_id=tenant_id)
        return concept

    def temporal_relation(self, node_a_id: str, node_b_id: str) -> Optional[str]:
        """Computes Allen's interval algebra relation between two nodes."""
        from app.reasoning.temporal import compute_temporal_relation

        node_a = self.nodes.get(node_a_id)
        node_b = self.nodes.get(node_b_id)
        if not node_a or not node_b:
            return None
        sa, ea = self._get_node_time_range(node_a)
        sb, eb = self._get_node_time_range(node_b)
        if sa is None or ea is None or sb is None or eb is None:
            return None
        rel = compute_temporal_relation(sa, ea, sb, eb)
        return rel.value

    def find_interval_relations(
        self,
        reference_node_id: str,
        target_relation: str,
        tenant_id: Optional[str] = None,
        video_id: Optional[str] = None,
    ) -> List[ConceptNode]:
        """
        Finds all nodes having a specific Allen's interval relation to the reference node.
        Example relations: BEFORE, AFTER, OVERLAPS, DURING, CONTAINS, MEETS, etc.
        """
        ref_node = self.get_node(reference_node_id, tenant_id=tenant_id)
        if not ref_node:
            return []

        target_rel = target_relation.strip().upper()
        matching_nodes: List[ConceptNode] = []

        for node_id, node in self.nodes.items():
            if node_id == reference_node_id:
                continue
            if tenant_id and node.tenant_id not in (tenant_id, "default"):
                continue
            if video_id and node.video_id != video_id:
                continue

            rel = self.temporal_relation(ref_node.id, node.id)
            if rel == target_rel:
                matching_nodes.append(node)

        return matching_nodes

    # ─────────────────────────────────────────────────────────────────────────
    # Claim & Evidence Graph Integration
    # ─────────────────────────────────────────────────────────────────────────

    def add_claim(
        self,
        claim_id: str,
        claim_text: str,
        confidence: float = 0.9,
        source_id: Optional[str] = None,
        timestamps: Optional[List[float]] = None,
        tenant_id: str = "default",
        workspace_id: str = "default",
        user_id: Optional[str] = None,
    ) -> ConceptNode:
        """Registers a factual claim node into the graph."""
        return self.register_node(
            node_id=claim_id,
            name=claim_text[:60] + "..." if len(claim_text) > 60 else claim_text,
            definition=claim_text,
            node_type=KnowledgeObjectType.CLAIM,
            importance=confidence,
            timestamps=timestamps or [],
            video_id=source_id,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            user_id=user_id,
        )

    def link_evidence(
        self,
        claim_id: str,
        evidence_id: str,
        relation_type: str = "SUPPORTS",
        confidence: float = 1.0,
        provenance: str = "",
        tenant_id: str = "default",
        workspace_id: str = "default",
    ) -> RelationshipEdge:
        """Links an evidence node to a claim (e.g. SUPPORTS or CONTRADICTS)."""
        return self.add_edge(
            edge=evidence_id,
            target_id=claim_id,
            relation_type=relation_type,
            confidence=confidence,
            description=provenance,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )

    def get_claim_evidence(self, claim_id: str, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Calculates verification status, supporting evidence, and contradictions for a claim."""
        claim_node = self.get_node(claim_id, tenant_id=tenant_id)
        if not claim_node:
            return {"claim_id": claim_id, "status": "NOT_AVAILABLE", "error": "Claim not found"}

        supports: List[Dict[str, Any]] = []
        contradictions: List[Dict[str, Any]] = []

        # Find edges targeting this claim
        for edge in self.rev_adj.get(claim_id, []):
            if tenant_id and edge.tenant_id not in (tenant_id, "default"):
                continue
            ev_node = self.nodes.get(edge.source_id)
            ev_info = {
                "evidence_id": edge.source_id,
                "content": ev_node.definition if ev_node else "",
                "relation": edge.relation.value,
                "confidence": edge.confidence,
                "provenance": edge.description,
                "timestamps": ev_node.timestamps if ev_node else [],
                "source_id": ev_node.video_id if ev_node else None,
            }
            if edge.relation in (KnowledgeEdgeType.SUPPORTS, KnowledgeEdgeType.DERIVED_FROM):
                supports.append(ev_info)
            elif edge.relation in (KnowledgeEdgeType.CONTRADICTS, KnowledgeEdgeType.CONTRASTS_WITH):
                contradictions.append(ev_info)

        # Also check edges originating from this claim
        for edge in self.adj.get(claim_id, []):
            if tenant_id and edge.tenant_id not in (tenant_id, "default"):
                continue
            ev_node = self.nodes.get(edge.target_id)
            ev_info = {
                "evidence_id": edge.target_id,
                "content": ev_node.definition if ev_node else "",
                "relation": edge.relation.value,
                "confidence": edge.confidence,
                "provenance": edge.description,
                "timestamps": ev_node.timestamps if ev_node else [],
                "source_id": ev_node.video_id if ev_node else None,
            }
            if edge.relation in (KnowledgeEdgeType.SUPPORTS, KnowledgeEdgeType.DERIVED_FROM):
                supports.append(ev_info)
            elif edge.relation in (KnowledgeEdgeType.CONTRADICTS, KnowledgeEdgeType.CONTRASTS_WITH):
                contradictions.append(ev_info)

        support_score = sum(s["confidence"] for s in supports)
        contradiction_score = sum(c["confidence"] for c in contradictions)

        if contradiction_score > 0 and contradiction_score >= support_score:
            status = "CONTRADICTED"
        elif support_score >= 1.0:
            status = "VERIFIED"
        elif support_score > 0.0:
            status = "PARTIALLY_SUPPORTED"
        else:
            status = "UNVERIFIED"

        return {
            "claim_id": claim_id,
            "claim_text": claim_node.definition or claim_node.name,
            "status": status,
            "support_score": round(support_score, 3),
            "contradiction_score": round(contradiction_score, 3),
            "supports": supports,
            "contradictions": contradictions,
            "classification": "REAL" if (supports or contradictions) else "NOT_AVAILABLE",
        }

    def detect_contradictions(
        self,
        claim_id_or_concept: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Scans graph for contradicting relationship edges.
        Exposes both sides with their citations truthfully without fabricating resolution.
        """
        contradictions: List[Dict[str, Any]] = []
        checked_pairs: Set[Tuple[str, str]] = set()

        edges_to_check = self.get_edges(tenant_id=tenant_id)
        for edge in edges_to_check:
            if edge.relation in (KnowledgeEdgeType.CONTRADICTS, KnowledgeEdgeType.CONTRASTS_WITH):
                pair = tuple(sorted([edge.source_id, edge.target_id]))
                if pair in checked_pairs:
                    continue
                checked_pairs.add(pair)

                src = self.nodes.get(edge.source_id)
                tgt = self.nodes.get(edge.target_id)
                if not src or not tgt:
                    continue

                if claim_id_or_concept and claim_id_or_concept not in (src.id, tgt.id, src.name, tgt.name):
                    continue

                contradictions.append(
                    {
                        "source_id": edge.source_id,
                        "source_name": src.name,
                        "source_claim": src.definition,
                        "source_citation": src.video_id or "unknown",
                        "target_id": edge.target_id,
                        "target_name": tgt.name,
                        "target_claim": tgt.definition,
                        "target_citation": tgt.video_id or "unknown",
                        "relation": edge.relation.value,
                        "confidence": edge.confidence,
                        "description": edge.description,
                        "status": "CONTRADICTED",
                        "epistemic_verdict": "Unresolved contradiction between sources; both citations preserved.",
                        "classification": "HEURISTIC",
                    }
                )

        # Cross-claim heuristic contradiction detection for CLAIM nodes
        from app.knowledge.verification import FactVerificationEngine
        verifier = FactVerificationEngine()
        claim_nodes = [
            n for n in self.nodes.values()
            if n.node_type == KnowledgeObjectType.CLAIM
            and (not tenant_id or n.tenant_id in (tenant_id, "default"))
        ]

        for i, ca in enumerate(claim_nodes):
            for cb in claim_nodes[i + 1 :]:
                pair = tuple(sorted([ca.id, cb.id]))
                if pair in checked_pairs:
                    continue
                if claim_id_or_concept and claim_id_or_concept not in (ca.id, cb.id, ca.name, cb.name):
                    continue

                from app.domain.models import FactClaim
                dummy_a = FactClaim(id=ca.id, claim=ca.definition or ca.name, evidence="", video_id=ca.video_id, source_timestamps=ca.timestamps)
                dummy_b = FactClaim(id=cb.id, claim=cb.definition or cb.name, evidence="", video_id=cb.video_id, source_timestamps=cb.timestamps)

                contra = verifier.detect_claim_contradiction(dummy_a, dummy_b)
                if contra:
                    checked_pairs.add(pair)
                    contradictions.append({
                        "source_id": ca.id,
                        "source_name": ca.name,
                        "source_claim": ca.definition or ca.name,
                        "source_citation": ca.video_id or "unknown",
                        "target_id": cb.id,
                        "target_name": cb.name,
                        "target_claim": cb.definition or cb.name,
                        "target_citation": cb.video_id or "unknown",
                        "relation": KnowledgeEdgeType.CONTRADICTS.value,
                        "confidence": contra.get("confidence", 0.90),
                        "description": contra.get("epistemic_verdict", "Heuristic contradiction detected."),
                        "status": "CONTRADICTED",
                        "epistemic_verdict": contra.get("epistemic_verdict", "Unresolved contradiction; both citations preserved."),
                        "classification": "HEURISTIC",
                    })

        return contradictions

    def resolve_entities(
        self,
        tenant_id: str = "default",
        workspace_id: str = "default",
    ) -> Dict[str, Any]:
        """
        Runs EntityResolver across all nodes within the specified tenant/workspace.
        Merges duplicate and aliased concept nodes, rewiring existing edges
        from duplicates to canonical nodes to maintain graph topology.
        """
        from app.knowledge.resolver import EntityResolver

        resolver = EntityResolver()
        nodes_to_resolve = [
            n for n in self.nodes.values()
            if n.tenant_id in (tenant_id, "default") and n.workspace_id in (workspace_id, "default")
        ]
        resolved = resolver.resolve_entities(nodes_to_resolve, tenant_id=tenant_id, workspace_id=workspace_id)

        canonical_map: Dict[str, str] = {}
        for n in nodes_to_resolve:
            if getattr(n, "canonical_id", None):
                canonical_map[n.id] = n.canonical_id

        rewired_edges_count = 0
        pruned_nodes_count = 0

        for dup_id, can_id in canonical_map.items():
            if dup_id == can_id:
                continue

            # Rewire outgoing edges from dup_id
            if dup_id in self.adj:
                for edge in self.adj[dup_id]:
                    edge.source_id = can_id
                    if can_id not in self.adj:
                        self.adj[can_id] = []
                    self.adj[can_id].append(edge)
                    rewired_edges_count += 1
                del self.adj[dup_id]

            # Rewire incoming edges to dup_id
            if dup_id in self.rev_adj:
                for edge in self.rev_adj[dup_id]:
                    edge.target_id = can_id
                    if can_id not in self.rev_adj:
                        self.rev_adj[can_id] = []
                    self.rev_adj[can_id].append(edge)
                    rewired_edges_count += 1
                del self.rev_adj[dup_id]

            # Remove duplicate node from self.nodes
            if dup_id in self.nodes:
                del self.nodes[dup_id]
                pruned_nodes_count += 1

        self._invalidate_cache()
        return {
            "canonical_nodes_count": len(resolved),
            "pruned_nodes_count": pruned_nodes_count,
            "rewired_edges_count": rewired_edges_count,
            "tenant_id": tenant_id,
            "workspace_id": workspace_id,
        }

    def integrate_multimodal_knowledge(
        self,
        extraction_data: Dict[str, Any],
        tenant_id: str = "default",
        workspace_id: str = "default",
        user_id: Optional[str] = None,
        video_id: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        Integrates multimodal extraction results into canonical KnowledgeGraph.
        Adds concepts, entities, OCR nodes, scene nodes, audio nodes, document nodes,
        code snippets, formulas, claims, and relationship edges.
        """
        import re
        added_nodes = 0
        added_edges = 0
        added_claims = 0

        for c in extraction_data.get("concepts", []):
            if c.id not in self.nodes:
                self.add_concept(c)
                added_nodes += 1

        for ent in extraction_data.get("entities", []):
            if ent.id not in self.nodes:
                self.add_concept(ent)
                added_nodes += 1

        for o_node in extraction_data.get("ocr_nodes", []):
            if o_node.id not in self.nodes:
                self.add_concept(o_node)
                added_nodes += 1

        for s_node in extraction_data.get("scene_nodes", []):
            if s_node.id not in self.nodes:
                self.add_concept(s_node)
                added_nodes += 1

        for a_node in extraction_data.get("audio_nodes", []):
            if a_node.id not in self.nodes:
                self.add_concept(a_node)
                added_nodes += 1

        for d_node in extraction_data.get("doc_nodes", []):
            if d_node.id not in self.nodes:
                self.add_concept(d_node)
                added_nodes += 1

        for cs in extraction_data.get("code_snippets", []):
            if cs.id not in self.nodes:
                self.register_node(
                    node_id=cs.id,
                    name=f"Code: {cs.language}",
                    definition=cs.code[:150],
                    node_type=KnowledgeObjectType.CODE,
                    importance=0.75,
                    video_id=video_id,
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    user_id=user_id,
                    source_modality="code",
                    classification="REAL",
                    provenance={"language": cs.language, "imports": cs.imports, "functions": cs.functions},
                )
                added_nodes += 1

        for md in extraction_data.get("math_derivations", []):
            if md.id not in self.nodes:
                self.register_node(
                    node_id=md.id,
                    name=f"Formula: {md.latex[:30]}",
                    definition=md.latex,
                    node_type=KnowledgeObjectType.FORMULA,
                    importance=0.80,
                    video_id=video_id,
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    user_id=user_id,
                    source_modality="formula",
                    classification="REAL",
                    provenance={"latex": md.latex, "variables": md.variables},
                )
                added_nodes += 1

        for clm in extraction_data.get("claims", []):
            clm_node = self.add_claim(
                claim_id=clm.id,
                claim_text=clm.claim,
                confidence=clm.confidence,
                source_id=clm.video_id or video_id,
                timestamps=clm.source_timestamps,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                user_id=user_id,
            )
            added_claims += 1

            for nid, node in list(self.nodes.items()):
                if node.node_type == KnowledgeObjectType.CLAIM or nid == clm.id:
                    continue
                if node.name.lower() in clm.claim.lower():
                    self.link_evidence(
                        claim_id=clm.id,
                        evidence_id=node.id,
                        relation_type="SUPPORTS" if clm.status != FactVerificationStatus.CONTRADICTED else "CONTRADICTS",
                        confidence=clm.confidence,
                        provenance=clm.evidence,
                        tenant_id=tenant_id,
                        workspace_id=workspace_id,
                    )
                    added_edges += 1

        for edge in extraction_data.get("edges", []):
            self.add_edge(edge)
            added_edges += 1

        resolution_stats = self.resolve_entities(tenant_id=tenant_id, workspace_id=workspace_id)

        return {
            "nodes_added": added_nodes,
            "edges_added": added_edges,
            "claims_added": added_claims,
            "canonical_nodes": resolution_stats["canonical_nodes_count"],
            "pruned_nodes": resolution_stats["pruned_nodes_count"],
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Graph Quality Control & Validation
    # ─────────────────────────────────────────────────────────────────────────

    def validate_graph(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Validates graph integrity:
        - Dangling edges (source or target node missing from graph)
        - Orphan nodes (no incoming or outgoing edges)
        - Duplicate edges (duplicate source, target, relation)
        - Cross-tenant edge leaks (edge connecting nodes belonging to different non-default tenants)
        """
        dangling_edges: List[Dict[str, Any]] = []
        duplicate_edges: List[Dict[str, Any]] = []
        cross_tenant_leaks: List[Dict[str, Any]] = []
        seen_edges: Set[Tuple[str, str, str]] = set()

        relevant_nodes = {
            nid: n for nid, n in self.nodes.items()
            if not tenant_id or n.tenant_id in (tenant_id, "default")
        }

        connected_nodes: Set[str] = set()

        for src_id, edges in self.adj.items():
            for e in edges:
                if tenant_id and e.tenant_id not in (tenant_id, "default"):
                    continue

                edge_key = (e.source_id, e.target_id, e.relation.value)
                if edge_key in seen_edges:
                    duplicate_edges.append(
                        {"source": e.source_id, "target": e.target_id, "relation": e.relation.value}
                    )
                else:
                    seen_edges.add(edge_key)

                src_exists = e.source_id in self.nodes
                tgt_exists = e.target_id in self.nodes

                if not src_exists or not tgt_exists:
                    dangling_edges.append(
                        {
                            "source_id": e.source_id,
                            "target_id": e.target_id,
                            "source_missing": not src_exists,
                            "target_missing": not tgt_exists,
                        }
                    )
                else:
                    connected_nodes.add(e.source_id)
                    connected_nodes.add(e.target_id)
                    src_node = self.nodes[e.source_id]
                    tgt_node = self.nodes[e.target_id]
                    if (
                        src_node.tenant_id != "default"
                        and tgt_node.tenant_id != "default"
                        and src_node.tenant_id != tgt_node.tenant_id
                    ):
                        cross_tenant_leaks.append(
                            {
                                "source_id": e.source_id,
                                "source_tenant": src_node.tenant_id,
                                "target_id": e.target_id,
                                "target_tenant": tgt_node.tenant_id,
                            }
                        )

        orphan_nodes = [nid for nid in relevant_nodes if nid not in connected_nodes]

        is_valid = len(dangling_edges) == 0 and len(cross_tenant_leaks) == 0

        return {
            "is_valid": is_valid,
            "total_nodes": len(relevant_nodes),
            "total_edges": len(seen_edges),
            "dangling_edge_count": len(dangling_edges),
            "orphan_node_count": len(orphan_nodes),
            "duplicate_edge_count": len(duplicate_edges),
            "cross_tenant_leak_count": len(cross_tenant_leaks),
            "dangling_edges": dangling_edges[:10],
            "orphan_nodes": orphan_nodes[:20],
            "duplicate_edges": duplicate_edges[:10],
            "cross_tenant_leaks": cross_tenant_leaks,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Snapshot Import / Export & Safe Caching
    # ─────────────────────────────────────────────────────────────────────────

    def export_snapshot(
        self,
        tenant_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Exports graph state to a JSON-serializable snapshot dict."""
        nodes_list = [
            n.model_dump() for n in self.get_nodes(tenant_id=tenant_id, workspace_id=workspace_id)
        ]
        edges_list = [
            e.model_dump() for e in self.get_edges(tenant_id=tenant_id, workspace_id=workspace_id)
        ]
        return {
            "version": self.version,
            "tenant_id": tenant_id or "default",
            "workspace_id": workspace_id or "default",
            "node_count": len(nodes_list),
            "edge_count": len(edges_list),
            "nodes": nodes_list,
            "edges": edges_list,
        }

    def import_snapshot(self, snapshot: Dict[str, Any], overwrite: bool = False) -> None:
        """Imports graph state from a snapshot dict."""
        if overwrite:
            self.nodes.clear()
            self.adj.clear()
            self.rev_adj.clear()

        for n_data in snapshot.get("nodes", []):
            node = ConceptNode(**n_data)
            self.add_concept(node)

        for e_data in snapshot.get("edges", []):
            edge = RelationshipEdge(**e_data)
            self.add_edge(edge)

        self._invalidate_cache()

    # ─────────────────────────────────────────────────────────────────────────
    # Visual Exporters
    # ─────────────────────────────────────────────────────────────────────────

    def export_cytoscape_elements(
        self,
        tenant_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Exports graph in Cytoscape.js format for interactive UI visualization."""
        elements: List[Dict[str, Any]] = []
        nodes = self.get_nodes(tenant_id=tenant_id, workspace_id=workspace_id)
        edges = self.get_edges(tenant_id=tenant_id, workspace_id=workspace_id)

        for node in nodes:
            elements.append(
                {
                    "data": {
                        "id": node.id,
                        "label": node.name,
                        "definition": node.definition,
                        "importance": node.importance,
                        "node_type": node.node_type.value,
                        "video_id": node.video_id,
                        "timestamps": node.timestamps,
                    }
                }
            )

        for edge in edges:
            elements.append(
                {
                    "data": {
                        "id": f"{edge.source_id}_{edge.target_id}_{edge.relation.value}",
                        "source": edge.source_id,
                        "target": edge.target_id,
                        "label": edge.relation.value,
                        "confidence": edge.confidence,
                        "description": edge.description,
                    }
                }
            )
        return elements

    def export_mermaid_diagram(
        self,
        tenant_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> str:
        """Exports graph as a GitHub-compatible Mermaid flowchart."""
        lines = ["graph TD"]
        nodes = self.get_nodes(tenant_id=tenant_id, workspace_id=workspace_id)
        edges = self.get_edges(tenant_id=tenant_id, workspace_id=workspace_id)

        for node in nodes:
            clean_name = node.name.replace('"', "'")
            lines.append(f'    {node.id}["{clean_name}"]')

        for edge in edges:
            lines.append(f"    {edge.source_id} -->|{edge.relation.value}| {edge.target_id}")

        return "\n".join(lines)

    def export_cypher_statements(
        self,
        tenant_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> List[str]:
        """Exports graph as Neo4j Cypher CREATE statements."""
        statements: List[str] = []
        nodes = self.get_nodes(tenant_id=tenant_id, workspace_id=workspace_id)
        edges = self.get_edges(tenant_id=tenant_id, workspace_id=workspace_id)

        for node in nodes:
            escaped_def = node.definition.replace("'", "\\'")
            escaped_name = node.name.replace("'", "\\'")
            statements.append(
                f"MERGE (c:Concept {{id: '{node.id}'}}) "
                f"ON CREATE SET c.name = '{escaped_name}', c.definition = '{escaped_def}', c.importance = {node.importance}, c.node_type = '{node.node_type.value}';"
            )

        for edge in edges:
            statements.append(
                f"MATCH (s:Concept {{id: '{edge.source_id}'}}), (t:Concept {{id: '{edge.target_id}'}}) "
                f"MERGE (s)-[:{edge.relation.value} {{confidence: {edge.confidence}}}]->(t);"
            )

        return statements

    def integrate_visual_intelligence(
        self,
        scenes: List[Any],
        frames: List[Any],
        events: Optional[List[Any]] = None,
        video_id: str = "default_video",
        video_title: str = "Video",
        tenant_id: str = "default",
        workspace_id: str = "default",
        user_id: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        Integrates visual intelligence artifacts into the knowledge graph:
        - SCENE nodes for temporal visual scenes
        - KEYFRAME nodes for salient visual frames
        - CODE / FORMULA / OCR_REGION typed nodes
        - EVENT nodes for detected pedagogical & visual events
        - Establishes provenance-rich typed edges connecting VIDEO, SCENES, KEYFRAMES, and EVENTS.
        """
        # 1. Register Source Video Node
        video_node_id = f"video_{video_id}"
        if video_node_id not in self.nodes:
            self.register_node(
                node_id=video_node_id,
                name=video_title,
                definition=f"Source video: {video_title}",
                node_type=KnowledgeObjectType.VIDEO,
                video_id=video_id,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                user_id=user_id,
                classification="REAL",
            )

        # 2. Register SCENE Nodes & Edges
        for scn in scenes:
            s_id = getattr(scn, "scene_id", f"scn_{video_id}")
            s_node_id = f"scn_node_{video_id}_{s_id}"
            start_ms = getattr(scn, "start_ms", int(getattr(scn, "start_time", 0.0) * 1000))
            end_ms = getattr(scn, "end_ms", int(getattr(scn, "end_time", 0.0) * 1000))

            if s_node_id not in self.nodes:
                self.register_node(
                    node_id=s_node_id,
                    name=f"Scene {s_id}",
                    definition=f"Visual scene spanning {start_ms/1000.0:.1f}s - {end_ms/1000.0:.1f}s",
                    node_type=KnowledgeObjectType.SCENE,
                    video_id=video_id,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    user_id=user_id,
                    classification=getattr(scn, "classification", "REAL"),
                )
                self.add_edge(
                    edge=video_node_id,
                    target_id=s_node_id,
                    relation_type="CONTAINS_SCENE",
                    video_id=video_id,
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    user_id=user_id,
                    classification="REAL",
                )

        # 3. Register KEYFRAME Nodes & Content Artifacts
        for frame in frames:
            f_id = getattr(frame, "frame_id", f"frm_{video_id}")
            f_node_id = f"kf_node_{video_id}_{f_id}"
            f_ts = getattr(frame, "timestamp", 0.0)
            f_type = getattr(frame, "frame_type", "slide")
            ocr_text = getattr(frame, "ocr_text", "")
            f_desc = getattr(frame, "description", "")
            start_ms = int(f_ts * 1000)

            if f_node_id not in self.nodes:
                self.register_node(
                    node_id=f_node_id,
                    name=f"Keyframe {f_id} ({f_type})",
                    definition=f_desc or f"Visual keyframe at {f_ts:.1f}s",
                    node_type=KnowledgeObjectType.KEYFRAME,
                    video_id=video_id,
                    start_ms=start_ms,
                    end_ms=start_ms + 3000,
                    source_modality="video",
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    user_id=user_id,
                    classification="REAL",
                )

            # Link frame to its scene
            scn_id = getattr(frame, "scene_id", None)
            if scn_id:
                s_node_id = f"scn_node_{video_id}_{scn_id}"
                if s_node_id in self.nodes:
                    self.add_edge(
                        edge=s_node_id,
                        target_id=f_node_id,
                        relation_type="VISUALLY_SHOWN_IN",
                        video_id=video_id,
                        tenant_id=tenant_id,
                        workspace_id=workspace_id,
                        user_id=user_id,
                        classification="REAL",
                    )

            # If OCR text is present, register OCR_REGION node and link
            if ocr_text:
                ocr_node_id = f"ocr_node_{video_id}_{f_id}"
                if ocr_node_id not in self.nodes:
                    self.register_node(
                        node_id=ocr_node_id,
                        name=f"OCR {f_id}",
                        definition=ocr_text[:300],
                        node_type=KnowledgeObjectType.OCR_REGION,
                        video_id=video_id,
                        start_ms=start_ms,
                        end_ms=start_ms + 3000,
                        source_modality="ocr",
                        tenant_id=tenant_id,
                        workspace_id=workspace_id,
                        user_id=user_id,
                        classification="REAL",
                    )
                    self.add_edge(
                        edge=f_node_id,
                        target_id=ocr_node_id,
                        relation_type="HAS_OCR",
                        video_id=video_id,
                        tenant_id=tenant_id,
                        workspace_id=workspace_id,
                        user_id=user_id,
                        classification="REAL",
                    )

            # If frame depicts code, register CODE node
            if f_type == "code":
                code_node_id = f"code_node_{video_id}_{f_id}"
                if code_node_id not in self.nodes:
                    self.register_node(
                        node_id=code_node_id,
                        name=f"Code Snippet {f_id}",
                        definition=ocr_text[:300] or "Source code on screen",
                        node_type=KnowledgeObjectType.CODE,
                        video_id=video_id,
                        start_ms=start_ms,
                        end_ms=start_ms + 3000,
                        source_modality="video",
                        tenant_id=tenant_id,
                        workspace_id=workspace_id,
                        user_id=user_id,
                        classification="HEURISTIC",
                    )
                    self.add_edge(
                        edge=f_node_id,
                        target_id=code_node_id,
                        relation_type="DEPICTS",
                        video_id=video_id,
                        tenant_id=tenant_id,
                        workspace_id=workspace_id,
                        user_id=user_id,
                        classification="HEURISTIC",
                    )

            # If frame depicts formula, register FORMULA node
            elif f_type == "formula":
                form_node_id = f"form_node_{video_id}_{f_id}"
                if form_node_id not in self.nodes:
                    self.register_node(
                        node_id=form_node_id,
                        name=f"Formula {f_id}",
                        definition=ocr_text[:300] or "Mathematical derivation",
                        node_type=KnowledgeObjectType.FORMULA,
                        video_id=video_id,
                        start_ms=start_ms,
                        end_ms=start_ms + 3000,
                        source_modality="video",
                        tenant_id=tenant_id,
                        workspace_id=workspace_id,
                        user_id=user_id,
                        classification="HEURISTIC",
                    )
                    self.add_edge(
                        edge=f_node_id,
                        target_id=form_node_id,
                        relation_type="DEPICTS",
                        video_id=video_id,
                        tenant_id=tenant_id,
                        workspace_id=workspace_id,
                        user_id=user_id,
                        classification="HEURISTIC",
                    )

        # 4. Register EVENT Nodes
        for evt in (events or []):
            evt_id = getattr(evt, "event_id", f"evt_{video_id}")
            evt_type = getattr(evt, "event_type", "event")
            e_node_id = f"evt_node_{video_id}_{evt_id}"
            start_ms = getattr(evt, "start_ms", int(getattr(evt, "start_time", 0.0) * 1000))
            end_ms = getattr(evt, "end_ms", int(getattr(evt, "end_time", 0.0) * 1000))

            if e_node_id not in self.nodes:
                self.register_node(
                    node_id=e_node_id,
                    name=f"Event: {evt_type}",
                    definition=getattr(evt, "description", ""),
                    node_type=KnowledgeObjectType.EVENT,
                    video_id=video_id,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    user_id=user_id,
                    classification="HEURISTIC",
                )

            # Link event to keyframe if present
            frame_id = getattr(evt, "frame_id", None)
            if frame_id:
                kf_node_id = f"kf_node_{video_id}_{frame_id}"
                if kf_node_id in self.nodes:
                    self.add_edge(
                        edge=e_node_id,
                        target_id=kf_node_id,
                        relation_type="REFERENCES",
                        video_id=video_id,
                        tenant_id=tenant_id,
                        workspace_id=workspace_id,
                        user_id=user_id,
                        classification="HEURISTIC",
                    )

        return {
            "scenes": len(scenes),
            "keyframes": len(frames),
            "events": len(events or []),
        }
