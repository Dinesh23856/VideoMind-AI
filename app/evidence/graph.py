"""Independent Multi-Modal Evidence Graph subsystem."""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field


class EvidenceNodeType(str, Enum):
    CLAIM = "CLAIM"
    TRANSCRIPT_SPAN = "TRANSCRIPT_SPAN"
    VIDEO_FRAME = "VIDEO_FRAME"
    OCR_TEXT = "OCR_TEXT"
    CODE_SNIPPET = "CODE_SNIPPET"
    EQUATION = "EQUATION"
    DOCUMENT_PAGE = "DOCUMENT_PAGE"


class EvidenceEdgeType(str, Enum):
    SUPPORTED_BY = "SUPPORTED_BY"
    CONTRADICTED_BY = "CONTRADICTED_BY"
    DERIVED_FROM = "DERIVED_FROM"
    IMAGE_REGION_OF = "IMAGE_REGION_OF"
    EXEMPLIFIES = "EXEMPLIFIES"
    CORROBORATES = "CORROBORATES"


class EvidenceNode(BaseModel):
    node_id: str = Field(default_factory=lambda: f"evn_{uuid.uuid4().hex[:12]}")
    node_type: EvidenceNodeType
    content: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_id: Optional[str] = None
    timestamp: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvidenceEdge(BaseModel):
    source_id: str  # e.g., the claim
    target_id: str  # e.g., the supporting snippet
    relation: EvidenceEdgeType
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    provenance: str = ""


class EvidenceGraph:
    """Directed multi-modal evidence graph modeling corroboration, support, and contradictions."""

    def __init__(self) -> None:
        self._nodes: Dict[str, EvidenceNode] = {}
        self._edges: List[EvidenceEdge] = []
        self._outgoing: Dict[str, List[EvidenceEdge]] = {}
        self._incoming: Dict[str, List[EvidenceEdge]] = {}

    def add_node(self, node: EvidenceNode) -> str:
        self._nodes[node.node_id] = node
        return node.node_id

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: EvidenceEdgeType,
        weight: float = 1.0,
        provenance: str = "",
    ) -> EvidenceEdge:
        if source_id not in self._nodes or target_id not in self._nodes:
            raise KeyError(f"Both nodes must exist in EvidenceGraph: {source_id} -> {target_id}")

        edge = EvidenceEdge(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            weight=weight,
            provenance=provenance,
        )
        self._edges.append(edge)
        self._outgoing.setdefault(source_id, []).append(edge)
        self._incoming.setdefault(target_id, []).append(edge)
        return edge

    def get_node(self, node_id: str) -> Optional[EvidenceNode]:
        return self._nodes.get(node_id)

    def get_supporting_evidence(self, claim_id: str) -> List[Tuple[EvidenceNode, float]]:
        """Returns all evidence nodes directly or corroboratively supporting this claim."""
        results: List[Tuple[EvidenceNode, float]] = []
        edges = self._outgoing.get(claim_id, [])
        for e in edges:
            if e.relation in (EvidenceEdgeType.SUPPORTED_BY, EvidenceEdgeType.CORROBORATES):
                node = self._nodes.get(e.target_id)
                if node:
                    results.append((node, e.weight * node.confidence))
        return results

    def get_contradicting_evidence(self, claim_id: str) -> List[Tuple[EvidenceNode, float]]:
        """Returns all evidence nodes directly contradicting this claim."""
        results: List[Tuple[EvidenceNode, float]] = []
        edges = self._outgoing.get(claim_id, [])
        for e in edges:
            if e.relation == EvidenceEdgeType.CONTRADICTED_BY:
                node = self._nodes.get(e.target_id)
                if node:
                    results.append((node, e.weight * node.confidence))
        return results

    def compute_claim_verification(self, claim_id: str) -> Dict[str, Any]:
        """Calculates net support score and verification status for a claim."""
        claim_node = self.get_node(claim_id)
        if not claim_node:
            raise KeyError(f"Claim node not found: {claim_id}")

        supports = self.get_supporting_evidence(claim_id)
        contradictions = self.get_contradicting_evidence(claim_id)

        support_score = sum(score for _, score in supports)
        contradiction_score = sum(score for _, score in contradictions)

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
            "claim_text": claim_node.content,
            "status": status,
            "support_score": round(support_score, 3),
            "contradiction_score": round(contradiction_score, 3),
            "support_count": len(supports),
            "contradiction_count": len(contradictions),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [n.model_dump() for n in self._nodes.values()],
            "edges": [e.model_dump() for e in self._edges],
        }

    def __len__(self) -> int:
        return len(self._nodes)
