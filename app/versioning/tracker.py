"""
Source & Artifact Versioning 2.0: Tracks content hashes, revision history,
and audit trails for transcripts, notes, and compiled artifacts.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from typing import Dict, List, Optional
from app.domain.models import VersionedArtifactRecord


class SourceVersionTracker:
    """
    Manages immutable revision history for artifacts and transcripts.
    Guarantees that re-processing or edits preserve complete version lineages with cryptographic hashes.
    """

    def __init__(self):
        # (job_id, artifact_type) -> List[VersionedArtifactRecord]
        self._history: Dict[tuple[str, str], List[VersionedArtifactRecord]] = {}

    def record_artifact_version(
        self,
        job_id: str,
        artifact_type: str,
        content: str | bytes,
        creator_id: Optional[str] = None,
        model_version: str = "gpt-4o",
    ) -> VersionedArtifactRecord:
        """Computes content hash and registers a new immutable version record."""
        raw_bytes = content.encode("utf-8") if isinstance(content, str) else content
        content_hash = hashlib.sha256(raw_bytes).hexdigest()

        key = (job_id, artifact_type)
        current_versions = self._history.get(key, [])

        # If latest version has identical hash, return existing version
        if current_versions and current_versions[-1].content_hash == content_hash:
            return current_versions[-1]

        next_version = len(current_versions) + 1
        record = VersionedArtifactRecord(
            artifact_id=f"art_{uuid.uuid4().hex[:8]}",
            job_id=job_id,
            artifact_type=artifact_type,
            version=next_version,
            source_version=1,
            model_version=model_version,
            content_hash=content_hash,
            creator_id=creator_id,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

        if key not in self._history:
            self._history[key] = []
        self._history[key].append(record)

        return record

    def get_latest_version(self, job_id: str, artifact_type: str) -> Optional[VersionedArtifactRecord]:
        history = self._history.get((job_id, artifact_type), [])
        return history[-1] if history else None

    def get_version_history(self, job_id: str, artifact_type: str) -> List[VersionedArtifactRecord]:
        return self._history.get((job_id, artifact_type), [])
