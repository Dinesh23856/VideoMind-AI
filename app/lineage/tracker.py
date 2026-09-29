"""Fine-Grained Multimodal Data Lineage and Provenance Graph."""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field


class LineageNodeType(str, Enum):
    MEDIA_FILE = "MEDIA_FILE"
    AUDIO_STREAM = "AUDIO_STREAM"
    TRANSCRIPT_SEGMENT = "TRANSCRIPT_SEGMENT"
    EXTRACTED_FRAME = "EXTRACTED_FRAME"
    OCR_RESULT = "OCR_RESULT"
    KNOWLEDGE_CONCEPT = "KNOWLEDGE_CONCEPT"
    SYNTHESIZED_RESPONSE = "SYNTHESIZED_RESPONSE"


class LineageNode(BaseModel):
    node_id: str = Field(default_factory=lambda: f"lin_{uuid.uuid4().hex[:10]}")
    node_type: LineageNodeType
    name: str
    content_hash: str
    created_at: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LineageEdge(BaseModel):
    parent_id: str
    child_id: str
    transform_name: str
    model_version: str = "v1"
    parameters: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


class LineageTracker:
    """Tracks end-to-end data provenance from raw media to synthesized AI responses."""

    def __init__(self) -> None:
        self._nodes: Dict[str, LineageNode] = {}
        self._edges: List[LineageEdge] = []
        self._parents: Dict[str, List[str]] = {}  # child -> parents
        self._children: Dict[str, List[str]] = {}  # parent -> children

    def register_node(self, node: LineageNode) -> str:
        self._nodes[node.node_id] = node
        return node.node_id

    def record_derivation(
        self,
        parent_id: str,
        child_id: str,
        transform_name: str,
        model_version: str = "1.0",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> LineageEdge:
        if parent_id not in self._nodes or child_id not in self._nodes:
            raise KeyError(f"Both nodes must exist in LineageTracker: {parent_id} -> {child_id}")

        edge = LineageEdge(
            parent_id=parent_id,
            child_id=child_id,
            transform_name=transform_name,
            model_version=model_version,
            parameters=parameters or {},
        )
        self._edges.append(edge)
        self._parents.setdefault(child_id, []).append(parent_id)
        self._children.setdefault(parent_id, []).append(child_id)
        return edge

    def trace_upstream(self, node_id: str) -> List[LineageNode]:
        """Traces the complete recursive ancestry (roots) that produced this node."""
        visited: Set[str] = set()
        queue = [node_id]
        ancestry: List[LineageNode] = []

        while queue:
            curr = queue.pop(0)
            for parent_id in self._parents.get(curr, []):
                if parent_id not in visited:
                    visited.add(parent_id)
                    ancestry.append(self._nodes[parent_id])
                    queue.append(parent_id)

        return ancestry

    def trace_downstream(self, node_id: str) -> List[LineageNode]:
        """Traces all derivative nodes that would be impacted if this node is modified."""
        visited: Set[str] = set()
        queue = [node_id]
        descendants: List[LineageNode] = []

        while queue:
            curr = queue.pop(0)
            for child_id in self._children.get(curr, []):
                if child_id not in visited:
                    visited.add(child_id)
                    descendants.append(self._nodes[child_id])
                    queue.append(child_id)

        return descendants

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [n.model_dump() for n in self._nodes.values()],
            "edges": [e.model_dump() for e in self._edges],
        }
