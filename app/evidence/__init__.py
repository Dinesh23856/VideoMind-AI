"""Evidence Engine: Evidence Graph, Claim Resolution, and Source Trust."""

from app.evidence.graph import (
    EvidenceEdgeType,
    EvidenceGraph,
    EvidenceNode,
    EvidenceNodeType,
)
from app.evidence.resolver import ClaimResolutionEngine, ResolvedClaim
from app.evidence.trust import SourceMetadata, SourceTrustModel

__all__ = [
    "EvidenceNodeType",
    "EvidenceEdgeType",
    "EvidenceNode",
    "EvidenceGraph",
    "ClaimResolutionEngine",
    "ResolvedClaim",
    "SourceMetadata",
    "SourceTrustModel",
]
