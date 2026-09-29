"""
Knowledge Object Store: First-class typed knowledge entities with versioning,
provenance, confidence scores, and bidirectional relationships.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional
from app.domain.models import KnowledgeObject, KnowledgeObjectType, RelationshipEdge


class KnowledgeObjectStore:
    """
    Manages typed first-class knowledge objects:
    Concept, Definition, Fact, Claim, Equation, Procedure, Example,
    CodeSnippet, Question, Answer, Person, Tool, Technology, Topic, Chapter, Visual, Evidence.
    """

    def __init__(self):
        self._objects: Dict[str, KnowledgeObject] = {}
        # index by type
        self._type_index: Dict[KnowledgeObjectType, List[str]] = {}
        # index by source/video
        self._source_index: Dict[str, List[str]] = {}

    def put_object(self, obj: KnowledgeObject) -> KnowledgeObject:
        """Stores or updates a knowledge object, updating version if content changed."""
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        existing = self._objects.get(obj.id)
        if existing:
            # Check content difference
            if existing.content != obj.content or existing.title != obj.title:
                obj.version = existing.version + 1
                obj.created_at = existing.created_at
                obj.updated_at = now
            else:
                obj.version = existing.version
                obj.created_at = existing.created_at
                obj.updated_at = existing.updated_at
        else:
            obj.created_at = obj.created_at or now
            obj.updated_at = now

        self._objects[obj.id] = obj

        # Update type index
        if obj.object_type not in self._type_index:
            self._type_index[obj.object_type] = []
        if obj.id not in self._type_index[obj.object_type]:
            self._type_index[obj.object_type].append(obj.id)

        # Update source index
        if obj.source_id:
            if obj.source_id not in self._source_index:
                self._source_index[obj.source_id] = []
            if obj.id not in self._source_index[obj.source_id]:
                self._source_index[obj.source_id].append(obj.id)

        return obj

    def get_object(self, obj_id: str) -> Optional[KnowledgeObject]:
        return self._objects.get(obj_id)

    def list_by_type(self, object_type: KnowledgeObjectType) -> List[KnowledgeObject]:
        ids = self._type_index.get(object_type, [])
        return [self._objects[i] for i in ids if i in self._objects]

    def list_by_source(self, source_id: str) -> List[KnowledgeObject]:
        ids = self._source_index.get(source_id, [])
        return [self._objects[i] for i in ids if i in self._objects]

    def search_by_text(self, term: str) -> List[KnowledgeObject]:
        t = term.lower().strip()
        matches = []
        for obj in self._objects.values():
            if t in obj.title.lower() or t in obj.content.lower():
                matches.append(obj)
        return matches

    def compute_content_hash(self, obj: KnowledgeObject) -> str:
        payload = f"{obj.id}:{obj.object_type.value}:{obj.title}:{obj.content}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
