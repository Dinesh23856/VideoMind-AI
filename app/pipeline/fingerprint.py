"""Content-Addressable Fingerprinting for pipeline artifacts and media inputs."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PipelineArtifactFingerprint(BaseModel):
    artifact_type: str  # raw_media, audio, transcript, frames, knowledge, embeddings
    sha256: str
    size_bytes: int = 0
    parameters_hash: str = ""
    combined_hash: str = ""


class ContentFingerprintEngine:
    """Computes deterministic cryptographic fingerprints for video, audio, and intermediate AI outputs."""

    @staticmethod
    def hash_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def hash_file(file_path: str, chunk_size: int = 65536) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found for fingerprinting: {file_path}")
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def hash_dict(obj: Dict[str, Any]) -> str:
        serialized = json.dumps(obj, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @classmethod
    def fingerprint_artifact(
        cls,
        artifact_type: str,
        content: Any,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> PipelineArtifactFingerprint:
        params_hash = cls.hash_dict(parameters or {})
        
        if isinstance(content, bytes):
            data_hash = cls.hash_bytes(content)
            size = len(content)
        elif isinstance(content, str):
            if os.path.isfile(content):
                data_hash = cls.hash_file(content)
                size = os.path.getsize(content)
            else:
                encoded = content.encode("utf-8")
                data_hash = cls.hash_bytes(encoded)
                size = len(encoded)
        elif isinstance(content, (dict, list)):
            encoded = json.dumps(content, sort_keys=True).encode("utf-8")
            data_hash = cls.hash_bytes(encoded)
            size = len(encoded)
        else:
            encoded = str(content).encode("utf-8")
            data_hash = cls.hash_bytes(encoded)
            size = len(encoded)

        combined = hashlib.sha256(f"{data_hash}:{params_hash}".encode("utf-8")).hexdigest()

        return PipelineArtifactFingerprint(
            artifact_type=artifact_type,
            sha256=data_hash,
            size_bytes=size,
            parameters_hash=params_hash,
            combined_hash=combined,
        )
