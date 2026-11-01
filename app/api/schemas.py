"""Pydantic request and response schemas for VideoMind-AI API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.domain.models import TutorMode


# ─────────────────────────────────────────────────────────────────────────────
# Ingestion & Jobs Schemas
# ─────────────────────────────────────────────────────────────────────────────

class UrlJobRequest(BaseModel):
    url: str
    title: Optional[str] = None


class JobAccepted(BaseModel):
    job_id: str
    status: str = "ACCEPTED"
    progress_url: str


class VideoChatRequest(BaseModel):
    video_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    history: Optional[List[Dict[str, str]]] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Authentication Schemas
# ─────────────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str
    password: str
    full_name: Optional[str] = None
    role: str = "USER"


class LoginRequest(BaseModel):
    email: str
    password: str


# ─────────────────────────────────────────────────────────────────────────────
# Bookmarks Schemas
# ─────────────────────────────────────────────────────────────────────────────

class BookmarkRequest(BaseModel):
    video_id: str
    timestamp: float
    title: str
    note_text: Optional[str] = None
    category: str = "Note"


# ─────────────────────────────────────────────────────────────────────────────
# Courses & Learning Schemas
# ─────────────────────────────────────────────────────────────────────────────

class TutorChatRequest(BaseModel):
    message: str
    mode: TutorMode = TutorMode.INTERMEDIATE
    video_id: Optional[str] = None
    hints_count: int = 0
    solution_attempt: Optional[str] = None


class CourseCreateRequest(BaseModel):
    title: str
    description: str = ""
    video_ids: List[str] = []


class LearningPathRequest(BaseModel):
    mastered_concepts: List[str] = []
    weak_concepts: List[str] = []
    video_id: Optional[str] = None


class CalibrateMasteryRequest(BaseModel):
    concept_id: str = Field(..., min_length=1)
    concept_name: str = Field(..., min_length=1)
    is_correct: bool
    score: Optional[float] = 1.0


class ScheduleReviewRequest(BaseModel):
    concept_id: str = Field(..., min_length=1)
    question_text: str = Field(..., min_length=1)
    question_id: Optional[str] = None
    review_type: Optional[str] = "concept"


class GradeReviewRequest(BaseModel):
    quality: int = Field(..., ge=0, le=5)


class GenerateStudyPackRequest(BaseModel):
    video_id: str = Field(..., min_length=1)


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge Graph Schemas
# ─────────────────────────────────────────────────────────────────────────────

class TraverseRequest(BaseModel):
    start_node_id: str
    relationship_types: Optional[List[str]] = None
    max_hops: int = 2
    max_nodes: int = 50
    direction: str = "both"
    tenant_id: str = "default"
    workspace_id: str = "default"
    user_id: Optional[str] = None


class NodeCreateRequest(BaseModel):
    id: str
    name: str
    definition: str = ""
    node_type: str = "CONCEPT"
    importance: float = 0.8
    timestamps: List[float] = []
    aliases: List[str] = []
    category: str = "General"
    video_id: Optional[str] = None
    document_id: Optional[str] = None
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    source_modality: str = "text"
    classification: str = "REAL"
    tenant_id: str = "default"
    workspace_id: str = "default"
    user_id: Optional[str] = None


class EdgeCreateRequest(BaseModel):
    source_id: str
    target_id: str
    relation: str = "RELATED_TO"
    confidence: float = 1.0
    description: str = ""
    video_id: Optional[str] = None
    document_id: Optional[str] = None
    timestamp_start_ms: Optional[int] = None
    timestamp_end_ms: Optional[int] = None
    source_modality: Optional[str] = None
    classification: str = "REAL"
    tenant_id: str = "default"
    workspace_id: str = "default"
    user_id: Optional[str] = None


class SnapshotImportRequest(BaseModel):
    snapshot: Dict[str, Any]
    overwrite: bool = False


class KnowledgeDiffRequest(BaseModel):
    from_revision: str
    to_revision: str


# ─────────────────────────────────────────────────────────────────────────────
# Collections & Reasoning Schemas
# ─────────────────────────────────────────────────────────────────────────────

class CreateCollectionRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = ""
    video_ids: Optional[List[str]] = Field(default_factory=list)


class AddVideoToCollectionRequest(BaseModel):
    video_id: str


class CollectionSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: Optional[int] = 5


class CompareConceptsRequest(BaseModel):
    concept_a: str = Field(..., min_length=1)
    concept_b: str = Field(..., min_length=1)


class PrerequisiteCheckRequest(BaseModel):
    target_concept: str = Field(..., min_length=1)
    mastered_concepts: List[str] = Field(default_factory=list)


class StudyPlanRequest(BaseModel):
    target_concept: str = Field(..., min_length=1)
    mastered_concepts: List[str] = Field(default_factory=list)
    available_time_minutes: Optional[int] = 60


# ─────────────────────────────────────────────────────────────────────────────
# Code, Math & Admin Schemas
# ─────────────────────────────────────────────────────────────────────────────

class AnalyzeCodeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    timestamp: Optional[float] = 0.0


class AnalyzeMathRequest(BaseModel):
    text: str = Field(..., min_length=1)
    timestamp: Optional[float] = 0.0


class AdjustQuotaRequest(BaseModel):
    user_id: str
    max_video_duration_minutes: Optional[int] = None
    max_storage_bytes: Optional[int] = None
    max_monthly_jobs: Optional[int] = None
    max_monthly_queries: Optional[int] = None


# ─────────────────────────────────────────────────────────────────────────────
# Next-Gen Subsystems Schemas
# ─────────────────────────────────────────────────────────────────────────────

class EvidenceVerifyRequest(BaseModel):
    claim_text: str
    supporting_snippets: List[str] = Field(default_factory=list)
    contradicting_snippets: List[str] = Field(default_factory=list)


class AgentResearchRequest(BaseModel):
    topic: str
    passages: List[Dict[str, Any]] = Field(default_factory=list)


class AgentProjectRequest(BaseModel):
    topic: str
    concept_names: List[str] = Field(default_factory=list)


class DocumentIngestRequest(BaseModel):
    doc_id: str
    doc_title: str
    content: str
    media_type: str = "markdown"


class CrossSearchRequest(BaseModel):
    query: str
    top_k: int = 8


class PrivacyRedactRequest(BaseModel):
    text: str


# ─────────────────────────────────────────────────────────────────────────────
# AI Gateway Schemas
# ─────────────────────────────────────────────────────────────────────────────

class StructuredAIRequest(BaseModel):
    prompt: str
    schema_definition: Optional[Dict[str, Any]] = None
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    max_retries: int = 2


class VisionAIRequest(BaseModel):
    image_base64: str
    prompt: str
    system_prompt: Optional[str] = None
    model: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Vector Hybrid RAG Schemas
# ─────────────────────────────────────────────────────────────────────────────

class RAGQueryRequest(BaseModel):
    query: str
    tenant_id: str = "default"
    workspace_id: str = "default"
    video_id: Optional[str] = None
    strategy: str = "hybrid"
    top_k: int = 5


class RAGSearchRequest(BaseModel):
    query: str
    tenant_id: str = "default"
    workspace_id: str = "default"
    video_id: Optional[str] = None
    modality: Optional[str] = None
    strategy: str = "hybrid"
    top_k: int = 8


class RAGIndexRequest(BaseModel):
    documents: List[Dict[str, Any]]
    tenant_id: str = "default"
    workspace_id: str = "default"


class RAGDeleteRequest(BaseModel):
    ids: Optional[List[str]] = None
    filter: Optional[Dict[str, Any]] = None
    tenant_id: Optional[str] = None
