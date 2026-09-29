"""Counterfactual Reasoning and Prerequisite Simulation over Knowledge Graphs."""

from __future__ import annotations

from typing import Any, Dict, List, Set
from pydantic import BaseModel, Field

from app.domain.models import ConceptNode, KnowledgeEdgeType, RelationshipEdge


class CounterfactualImpact(BaseModel):
    removed_concept_id: str
    unlocked_concepts_lost: List[str] = Field(default_factory=list)
    impact_score: float = 0.0  # ratio of total downstream nodes affected
    alternative_prerequisite_paths: List[str] = Field(default_factory=list)
    criticality: str = "MODERATE"  # LOW, MODERATE, HIGH, CRITICAL


class CounterfactualSimulator:
    """Performs graph surgery and reachability simulation to evaluate concept criticality."""

    def __init__(self, concepts: List[ConceptNode], edges: List[RelationshipEdge]) -> None:
        self.concepts = {c.id: c for c in concepts}
        # Build DAG: prerequisite -> downstream
        self.downstream: Dict[str, List[str]] = {}
        self.prerequisites: Dict[str, List[str]] = {}

        for edge in edges:
            if edge.relation == KnowledgeEdgeType.PREREQUISITE_OF:
                self.downstream.setdefault(edge.source_id, []).append(edge.target_id)
                self.prerequisites.setdefault(edge.target_id, []).append(edge.source_id)

    def simulate_omission(self, concept_id: str) -> CounterfactualImpact:
        if concept_id not in self.concepts:
            raise KeyError(f"Concept {concept_id} not present in simulation model.")

        # Breadth-first reachability of downstream lost concepts
        lost: Set[str] = set()
        queue = list(self.downstream.get(concept_id, []))

        while queue:
            curr = queue.pop(0)
            if curr not in lost:
                # Check if this node has alternative independent prerequisites
                all_prereqs = self.prerequisites.get(curr, [])
                remaining_prereqs = [p for p in all_prereqs if p != concept_id and p not in lost]
                if not remaining_prereqs:
                    # Lost completely
                    lost.add(curr)
                    queue.extend(self.downstream.get(curr, []))

        total_nodes = max(1, len(self.concepts) - 1)
        ratio = len(lost) / total_nodes

        if ratio > 0.5:
            crit = "CRITICAL"
        elif ratio > 0.25:
            crit = "HIGH"
        elif ratio > 0.0:
            crit = "MODERATE"
        else:
            crit = "LOW"

        return CounterfactualImpact(
            removed_concept_id=concept_id,
            unlocked_concepts_lost=[self.concepts[cid].name for cid in lost if cid in self.concepts],
            impact_score=round(ratio, 3),
            criticality=crit,
        )
