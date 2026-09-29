"""Dependency-Aware Invalidation Graph for incremental video knowledge pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Set


class PipelineStage(str, Enum):
    MEDIA_INGEST = "MEDIA_INGEST"
    AUDIO_EXTRACT = "AUDIO_EXTRACT"
    TRANSCRIPTION = "TRANSCRIPTION"
    DIARIZATION = "DIARIZATION"
    VISION_SAMPLING = "VISION_SAMPLING"
    OCR = "OCR"
    KNOWLEDGE_EXTRACTION = "KNOWLEDGE_EXTRACTION"
    EMBEDDING_GENERATION = "EMBEDDING_GENERATION"
    RAG_INDEXING = "RAG_INDEXING"


class DependencyInvalidationGraph:
    """Manages stage dependencies and calculates downstream re-execution cascades."""

    # Stage -> list of direct downstream dependent stages
    DEFAULT_DEPENDENCIES: Dict[PipelineStage, List[PipelineStage]] = {
        PipelineStage.MEDIA_INGEST: [PipelineStage.AUDIO_EXTRACT, PipelineStage.VISION_SAMPLING],
        PipelineStage.AUDIO_EXTRACT: [PipelineStage.TRANSCRIPTION],
        PipelineStage.TRANSCRIPTION: [PipelineStage.DIARIZATION, PipelineStage.KNOWLEDGE_EXTRACTION],
        PipelineStage.VISION_SAMPLING: [PipelineStage.OCR],
        PipelineStage.OCR: [PipelineStage.KNOWLEDGE_EXTRACTION],
        PipelineStage.DIARIZATION: [PipelineStage.KNOWLEDGE_EXTRACTION],
        PipelineStage.KNOWLEDGE_EXTRACTION: [PipelineStage.EMBEDDING_GENERATION],
        PipelineStage.EMBEDDING_GENERATION: [PipelineStage.RAG_INDEXING],
        PipelineStage.RAG_INDEXING: [],
    }

    def __init__(self, dependencies: Dict[PipelineStage, List[PipelineStage]] | None = None) -> None:
        self.dependencies = dependencies or self.DEFAULT_DEPENDENCIES

    def compute_invalidated_stages(self, modified_stages: List[PipelineStage]) -> List[PipelineStage]:
        """Calculates the minimal topological set of stages that must re-run."""
        invalidated: Set[PipelineStage] = set()
        queue = list(modified_stages)

        while queue:
            current = queue.pop(0)
            if current not in invalidated:
                invalidated.add(current)
                for downstream in self.dependencies.get(current, []):
                    if downstream not in invalidated:
                        queue.append(downstream)

        # Return in pipeline canonical order
        order = list(PipelineStage)
        return [stage for stage in order if stage in invalidated]

    def can_skip_stage(self, stage: PipelineStage, invalidated_stages: List[PipelineStage]) -> bool:
        """Determines whether a cached stage output can be reused safely."""
        return stage not in invalidated_stages
