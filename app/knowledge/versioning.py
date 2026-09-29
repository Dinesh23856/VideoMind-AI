"""Git-like Knowledge Version Control and Diff Engine."""

from __future__ import annotations

import copy
import hashlib
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field

from app.domain.models import ConceptNode


class KnowledgeRevision(BaseModel):
    revision_id: str = Field(default_factory=lambda: f"rev_{uuid.uuid4().hex[:10]}")
    parent_id: Optional[str] = None
    branch: str = "main"
    message: str
    author: str = "system"
    timestamp: str = Field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    snapshot_hash: str
    concepts: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    edges: List[Dict[str, Any]] = Field(default_factory=list)


class KnowledgeDiff(BaseModel):
    from_revision: str
    to_revision: str
    added_concepts: List[str] = Field(default_factory=list)
    removed_concepts: List[str] = Field(default_factory=list)
    modified_concepts: List[Dict[str, Any]] = Field(default_factory=list)
    added_edges: int = 0
    removed_edges: int = 0
    summary: str = ""


class KnowledgeVersionControl:
    """Manages branches, commits, and diffs for knowledge representations."""

    def __init__(self) -> None:
        self._revisions: Dict[str, KnowledgeRevision] = {}
        self._branches: Dict[str, str] = {}  # branch_name -> head_revision_id

    @staticmethod
    def _compute_hash(concepts: Dict[str, Any], edges: List[Any]) -> str:
        serialized = json.dumps({"c": concepts, "e": edges}, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def commit(
        self,
        concepts: List[ConceptNode],
        edges: Optional[List[Dict[str, Any]]] = None,
        branch: str = "main",
        message: str = "Knowledge update",
        author: str = "system",
    ) -> KnowledgeRevision:
        parent_id = self._branches.get(branch)
        concepts_dict = {c.id: c.model_dump() for c in concepts}
        edges_list = copy.deepcopy(edges or [])
        snapshot_hash = self._compute_hash(concepts_dict, edges_list)

        rev = KnowledgeRevision(
            parent_id=parent_id,
            branch=branch,
            message=message,
            author=author,
            snapshot_hash=snapshot_hash,
            concepts=concepts_dict,
            edges=edges_list,
        )
        self._revisions[rev.revision_id] = rev
        self._branches[branch] = rev.revision_id
        return rev

    def get_revision(self, revision_id: str) -> Optional[KnowledgeRevision]:
        return self._revisions.get(revision_id)

    def get_head(self, branch: str = "main") -> Optional[KnowledgeRevision]:
        head_id = self._branches.get(branch)
        return self._revisions.get(head_id) if head_id else None

    def diff(self, from_rev_id: str, to_rev_id: str) -> KnowledgeDiff:
        from_rev = self._revisions.get(from_rev_id)
        to_rev = self._revisions.get(to_rev_id)
        if not from_rev or not to_rev:
            raise KeyError(f"Revision {from_rev_id} or {to_rev_id} not found")

        old_c_keys = set(from_rev.concepts.keys())
        new_c_keys = set(to_rev.concepts.keys())

        added = list(new_c_keys - old_c_keys)
        removed = list(old_c_keys - new_c_keys)
        modified = []

        common = old_c_keys & new_c_keys
        for cid in common:
            old_c = from_rev.concepts[cid]
            new_c = to_rev.concepts[cid]
            if old_c != new_c:
                modified.append({
                    "concept_id": cid,
                    "name": new_c.get("name"),
                    "old_definition": old_c.get("definition"),
                    "new_definition": new_c.get("definition"),
                    "old_importance": old_c.get("importance"),
                    "new_importance": new_c.get("importance"),
                })

        added_edges = max(0, len(to_rev.edges) - len(from_rev.edges))
        removed_edges = max(0, len(from_rev.edges) - len(to_rev.edges))

        summary = (
            f"+{len(added)} concepts, -{len(removed)} concepts, "
            f"~{len(modified)} modified concepts; "
            f"{len(to_rev.edges)} total edges."
        )

        return KnowledgeDiff(
            from_revision=from_rev_id,
            to_revision=to_rev_id,
            added_concepts=added,
            removed_concepts=removed,
            modified_concepts=modified,
            added_edges=added_edges,
            removed_edges=removed_edges,
            summary=summary,
        )

    def branch(self, new_branch: str, from_branch: str = "main") -> str:
        head = self.get_head(from_branch)
        if not head:
            raise ValueError(f"Source branch '{from_branch}' does not exist or has no revisions")
        self._branches[new_branch] = head.revision_id
        return head.revision_id
