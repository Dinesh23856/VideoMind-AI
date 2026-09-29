"""
Knowledge Reasoning Engine: Higher-order reasoning over graph, transcripts, and evidence.
Supports concept comparison, prerequisite gap analysis, and study plan synthesis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.domain.models import ConceptNode, KnowledgeEdgeType
from app.knowledge.graph import KnowledgeGraph
from app.retrieval.hybrid_search import HybridSearchEngine


class ConceptComparisonResult(BaseModel):
    concept_a: str
    concept_b: str
    direct_relationship: Optional[str] = None
    common_neighbors: List[str] = Field(default_factory=list)
    distinctions: List[str] = Field(default_factory=list)
    shared_attributes: List[str] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list)
    summary: str = ""


class PrerequisiteReadinessResult(BaseModel):
    target_concept: str
    is_ready: bool
    readiness_percentage: float
    mastered_prerequisites: List[str] = Field(default_factory=list)
    missing_prerequisites: List[str] = Field(default_factory=list)
    recommended_next_step: Optional[str] = None
    explanation: str = ""


class ReasoningEngine:
    """
    Reasoning layer executing structured multi-step inferences over the Knowledge Graph
    and multimodal transcript corpus.
    """

    def __init__(self, graph: Optional[KnowledgeGraph] = None, search_engine: Optional[HybridSearchEngine] = None):
        self.graph = graph or KnowledgeGraph()
        self.search_engine = search_engine or HybridSearchEngine()

    def compare_concepts(self, concept_a_name: str, concept_b_name: str) -> ConceptComparisonResult:
        """Analyzes commonalities, direct connections, and differences between two concepts."""
        node_a = self.graph.get_concept_by_name(concept_a_name)
        node_b = self.graph.get_concept_by_name(concept_b_name)

        name_a = node_a.name if node_a else concept_a_name
        name_b = node_b.name if node_b else concept_b_name

        direct_rel = None
        common_neighbors = []

        if node_a and node_b:
            # Check direct edge from A to B or B to A
            for edge in self.graph.adj.get(node_a.id, []):
                if edge.target_id == node_b.id:
                    direct_rel = f"{name_a} {edge.relation.value} {name_b}"
                    break
            if not direct_rel:
                for edge in self.graph.adj.get(node_b.id, []):
                    if edge.target_id == node_a.id:
                        direct_rel = f"{name_b} {edge.relation.value} {name_a}"
                        break

            # Find common neighbors
            neighbors_a = {e.target_id for e in self.graph.adj.get(node_a.id, [])}.union(
                {e.source_id for e in self.graph.rev_adj.get(node_a.id, [])}
            )
            neighbors_b = {e.target_id for e in self.graph.adj.get(node_b.id, [])}.union(
                {e.source_id for e in self.graph.rev_adj.get(node_b.id, [])}
            )
            shared_ids = neighbors_a.intersection(neighbors_b)
            common_neighbors = [self.graph.nodes[n_id].name for n_id in shared_ids if n_id in self.graph.nodes]

        # Retrieve evidence snippets from search engine
        hits = self.search_engine.search(f"{name_a} {name_b}", k=3)
        citations = [f"{h.timestamp_formatted} {h.snippet}" for h in hits]

        def_a = node_a.definition if node_a else "No formal definition indexed."
        def_b = node_b.definition if node_b else "No formal definition indexed."

        distinctions = [
            f"{name_a}: {def_a}",
            f"{name_b}: {def_b}",
        ]

        summary = f"Comparison between {name_a} and {name_b}. "
        if direct_rel:
            summary += f"Direct relationship: {direct_rel}. "
        if common_neighbors:
            summary += f"Shared conceptual neighbors: {', '.join(common_neighbors)}. "

        return ConceptComparisonResult(
            concept_a=name_a,
            concept_b=name_b,
            direct_relationship=direct_rel,
            common_neighbors=common_neighbors,
            distinctions=distinctions,
            shared_attributes=[f"Both are core topics in lecture domain with {len(common_neighbors)} shared links."],
            citations=citations,
            summary=summary.strip(),
        )

    def analyze_prerequisite_readiness(
        self,
        target_concept_name: str,
        mastered_concepts: List[str],
    ) -> PrerequisiteReadinessResult:
        """Calculates learner readiness for a target concept based on prerequisite satisfaction."""
        all_prereqs = self.graph.get_prerequisites(target_concept_name)
        if not all_prereqs:
            return PrerequisiteReadinessResult(
                target_concept=target_concept_name,
                is_ready=True,
                readiness_percentage=100.0,
                mastered_prerequisites=[],
                missing_prerequisites=[],
                recommended_next_step=None,
                explanation=f"'{target_concept_name}' has no prerequisite dependencies in the knowledge graph; you may begin immediately.",
            )

        mastered_lower = {m.strip().lower() for m in mastered_concepts}
        mastered_list: List[str] = []
        missing_list: List[str] = []

        for p in all_prereqs:
            if p.name.lower() in mastered_lower:
                mastered_list.append(p.name)
            else:
                missing_list.append(p.name)

        readiness = round((len(mastered_list) / len(all_prereqs)) * 100.0, 1)
        is_ready = len(missing_list) == 0
        recommended = missing_list[0] if missing_list else None

        explanation = (
            f"You have satisfied {len(mastered_list)} of {len(all_prereqs)} prerequisites ({readiness}%). "
        )
        if missing_list:
            explanation += f"We recommend learning '{recommended}' before proceeding to '{target_concept_name}'."
        else:
            explanation += f"You are fully prepared to study '{target_concept_name}'!"

        return PrerequisiteReadinessResult(
            target_concept=target_concept_name,
            is_ready=is_ready,
            readiness_percentage=readiness,
            mastered_prerequisites=mastered_list,
            missing_prerequisites=missing_list,
            recommended_next_step=recommended,
            explanation=explanation,
        )

    def synthesize_study_plan(
        self,
        target_topics: List[str] | str,
        hours_available: float = 5.0,
        mastered_concepts: Optional[List[str]] = None,
        available_time_minutes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Synthesizes a structured study curriculum ordered by prerequisite dependency."""
        if isinstance(target_topics, str):
            topics_list = [target_topics]
            primary_target = target_topics
        else:
            topics_list = target_topics
            primary_target = topics_list[0] if topics_list else ""

        if available_time_minutes is not None:
            hours_available = round(available_time_minutes / 60.0, 2)

        ordered_nodes = self.graph.get_topological_learning_path()
        target_lower = {t.strip().lower() for t in topics_list}
        mastered_lower = {m.strip().lower() for m in (mastered_concepts or [])}

        relevant_nodes = [n for n in ordered_nodes if n.name.lower() in target_lower or any(t in n.name.lower() for t in target_lower)]
        if not relevant_nodes:
            relevant_nodes = ordered_nodes[:5]

        # Filter out already mastered concepts unless all would be filtered
        unmastered_nodes = [n for n in relevant_nodes if n.name.lower() not in mastered_lower]
        active_nodes = unmastered_nodes if unmastered_nodes else relevant_nodes

        time_per_topic = round(hours_available / max(len(active_nodes), 1), 2)
        schedule = []

        for idx, node in enumerate(active_nodes):
            schedule.append({
                "step": idx + 1,
                "topic": node.name,
                "definition": node.definition,
                "allocated_hours": time_per_topic,
                "timestamps": node.timestamps[:3] if node.timestamps else [0.0],
                "recommended_action": f"Review lecture interval at {node.timestamps[0] if node.timestamps else 0}s and complete practice questions.",
            })

        return {
            "target_concept": primary_target,
            "target_topics": topics_list,
            "total_hours": hours_available,
            "topics_count": len(active_nodes),
            "learning_steps": schedule,
            "schedule": schedule,
        }
