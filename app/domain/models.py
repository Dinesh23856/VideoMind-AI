"""Core domain data models and schemas for VideoMind-AI."""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class PipelineState(str, Enum):
    CREATED = "CREATED"
    QUEUED = "QUEUED"
    VALIDATING = "VALIDATING"
    DOWNLOADING = "DOWNLOADING"
    INGESTING = "INGESTING"
    EXTRACTING_AUDIO = "EXTRACTING_AUDIO"
    TRANSCRIBING = "TRANSCRIBING"
    ANALYZING_AUDIO = "ANALYZING_AUDIO"
    DIARIZING = "DIARIZING"
    ANALYZING_VIDEO = "ANALYZING_VIDEO"
    OCR_PROCESSING = "OCR_PROCESSING"
    SEGMENTING = "SEGMENTING"
    STRUCTURING = "STRUCTURING"
    EXTRACTING_KNOWLEDGE = "EXTRACTING_KNOWLEDGE"
    BUILDING_KNOWLEDGE = "BUILDING_KNOWLEDGE"
    INDEXING = "INDEXING"
    GENERATING_ARTIFACTS = "GENERATING_ARTIFACTS"
    FINALIZING = "FINALIZING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class SpeakerRole(str, Enum):
    HOST = "Host"
    GUEST = "Guest"
    PROFESSOR = "Professor"
    STUDENT = "Student"
    SPEAKER_1 = "Speaker 1"
    SPEAKER_2 = "Speaker 2"
    SPEAKER_3 = "Speaker 3"
    INTERVIEWER = "Interviewer"
    INTERVIEWEE = "Interviewee"
    UNKNOWN = "Unknown"


class FactVerificationStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    VERIFIED = "VERIFIED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNVERIFIED = "UNVERIFIED"
    CONFLICTING = "CONFLICTING"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class KnowledgeEdgeType(str, Enum):
    IS_A = "IS_A"
    PART_OF = "PART_OF"
    USES = "USES"
    DEPENDS_ON = "DEPENDS_ON"
    RELATED_TO = "RELATED_TO"
    EXAMPLE_OF = "EXAMPLE_OF"
    IMPLEMENTS = "IMPLEMENTS"
    CONTRADICTS = "CONTRADICTS"
    SUPPORTS = "SUPPORTS"
    PREREQUISITE_OF = "PREREQUISITE_OF"
    MENTIONED_IN = "MENTIONED_IN"
    EXPLAINED_BY = "EXPLAINED_BY"
    VISUALLY_SHOWN_IN = "VISUALLY_SHOWN_IN"
    USED_BY = "USED_BY"
    CONTRASTS_WITH = "CONTRASTS_WITH"
    IMPLEMENTED_BY = "IMPLEMENTED_BY"
    EXPLAINS = "EXPLAINS"
    # Extended graph relationship types
    MENTIONS = "MENTIONS"
    DEFINES = "DEFINES"
    FOLLOWS = "FOLLOWS"
    DERIVED_FROM = "DERIVED_FROM"
    OCCURS_IN = "OCCURS_IN"
    SPOKEN_BY = "SPOKEN_BY"
    APPEARS_IN = "APPEARS_IN"
    REFERENCES = "REFERENCES"
    ANSWERS = "ANSWERS"
    ASKS = "ASKS"
    SIMILAR_TO = "SIMILAR_TO"
    PRECEDES = "PRECEDES"
    CAUSES = "CAUSES"
    CONTAINS_SCENE = "CONTAINS_SCENE"
    DEPICTS = "DEPICTS"
    HAS_OCR = "HAS_OCR"


class KnowledgeObjectType(str, Enum):
    CONCEPT = "CONCEPT"
    DEFINITION = "DEFINITION"
    FACT = "FACT"
    CLAIM = "CLAIM"
    EQUATION = "EQUATION"
    PROCEDURE = "PROCEDURE"
    EXAMPLE = "EXAMPLE"
    CODE_SNIPPET = "CODE_SNIPPET"
    QUESTION = "QUESTION"
    ANSWER = "ANSWER"
    PERSON = "PERSON"
    TOOL = "TOOL"
    TECHNOLOGY = "TECHNOLOGY"
    TOPIC = "TOPIC"
    CHAPTER = "CHAPTER"
    VISUAL = "VISUAL"
    TRANSCRIPT_SEGMENT = "TRANSCRIPT_SEGMENT"
    SOURCE = "SOURCE"
    EVIDENCE = "EVIDENCE"
    # Multimodal knowledge node types
    VIDEO = "VIDEO"
    DOCUMENT = "DOCUMENT"
    LECTURE = "LECTURE"
    SPEAKER = "SPEAKER"
    ENTITY = "ENTITY"
    CODE = "CODE"
    FORMULA = "FORMULA"
    KEYFRAME = "KEYFRAME"
    OCR_REGION = "OCR_REGION"
    AUDIO_EVENT = "AUDIO_EVENT"
    SCENE = "SCENE"
    EVENT = "EVENT"
    TEMPORAL = "TEMPORAL"


class LearnerMasteryState(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    LEARNING = "LEARNING"
    PRACTICING = "PRACTICING"
    MASTERED = "MASTERED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


class CollaborationPermission(str, Enum):
    VIEW = "VIEW"
    COMMENT = "COMMENT"
    EDIT = "EDIT"
    MANAGE = "MANAGE"


class TutorMode(str, Enum):
    BEGINNER = "Beginner"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"
    EXPERT = "Expert"
    INTERVIEW = "Interview"
    EXAM = "Exam"
    RESEARCH = "Research"
    DEVELOPER = "Developer"
    CODING = "Developer"
    PROJECT = "Developer"


class UserRole(str, Enum):
    USER = "USER"
    TEACHER = "TEACHER"
    STUDENT = "STUDENT"
    RESEARCHER = "RESEARCHER"
    ADMIN = "ADMIN"


# ─────────────────────────────────────────────────────────────────────────────
# Audio & Multimodal Entities
# ─────────────────────────────────────────────────────────────────────────────

class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float
    probability: float = 1.0


class VisualFrameContext(BaseModel):
    frame_id: str
    timestamp: float
    frame_type: str = "slide"  # slide, diagram, code, terminal, chart, equation, speaker, formula, whiteboard, document
    description: str = ""
    ocr_text: str = ""
    image_path: Optional[str] = None
    confidence: float = 0.95
    bounding_boxes: List[Dict[str, Any]] = Field(default_factory=list)
    perceptual_hash: Optional[str] = None
    scene_id: Optional[str] = None
    selection_reason: Optional[str] = None
    classification: str = "HEURISTIC"


class AudioEvent(BaseModel):
    event_id: str
    start_time: float
    end_time: float
    event_type: str  # speech, silence, question_inflection, emphasis, applause, music, noise
    confidence: float = 0.9
    speaker: Optional[str] = None


class TranscriptSegment(BaseModel):
    id: str
    start: float
    end: float
    speaker: str = "Speaker 1"
    speaker_role: SpeakerRole = SpeakerRole.SPEAKER_1
    text: str
    language: str = "en"
    translated_text: Optional[str] = None
    confidence: float = 0.95
    topics: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    concepts: List[str] = Field(default_factory=list)
    visual_refs: List[str] = Field(default_factory=list)
    code_refs: List[str] = Field(default_factory=list)
    math_refs: List[str] = Field(default_factory=list)
    importance: float = 0.5
    words: List[WordTimestamp] = Field(default_factory=list)


class MultimodalTimelinePoint(BaseModel):
    timestamp: float
    audio_event: Optional[AudioEvent] = None
    speaker: Optional[str] = None
    transcript_segment_id: Optional[str] = None
    transcript_text: str = ""
    visual_frame_id: Optional[str] = None
    frame_type: Optional[str] = None
    ocr_text: str = ""
    concept_tags: List[str] = Field(default_factory=list)
    scene_id: Optional[str] = None
    multimodal_event_id: Optional[str] = None
    multimodal_event_type: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge & Semantic Entities (First-Class Knowledge Objects)
# ─────────────────────────────────────────────────────────────────────────────

class ConceptNode(BaseModel):
    id: str
    name: str
    definition: str
    importance: float = 0.8
    timestamps: List[float] = Field(default_factory=list)
    aliases: List[str] = Field(default_factory=list)
    category: str = "General"
    video_id: Optional[str] = None
    node_type: KnowledgeObjectType = KnowledgeObjectType.CONCEPT
    document_id: Optional[str] = None
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    source_modality: Optional[str] = "text"
    provenance: Dict[str, Any] = Field(default_factory=dict)
    classification: str = "REAL"
    tenant_id: str = "default"
    workspace_id: str = "default"
    user_id: Optional[str] = None
    canonical_id: Optional[str] = None
    mention_count: int = 1
    evidence_ids: List[str] = Field(default_factory=list)
    source_id: Optional[str] = None


class RelationshipEdge(BaseModel):
    source_id: str
    target_id: str
    relation: KnowledgeEdgeType
    confidence: float = 1.0
    description: str = ""
    source_id_ref: Optional[str] = None
    video_id: Optional[str] = None
    document_id: Optional[str] = None
    timestamp_start_ms: Optional[int] = None
    timestamp_end_ms: Optional[int] = None
    source_modality: Optional[str] = None
    created_at: str = Field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    created_by: str = "system"
    classification: str = "REAL"
    tenant_id: str = "default"
    workspace_id: str = "default"
    user_id: Optional[str] = None


class FactClaim(BaseModel):
    id: str
    claim: str
    evidence: str
    confidence: float = 0.9
    source_timestamps: List[float] = Field(default_factory=list)
    status: FactVerificationStatus = FactVerificationStatus.SUPPORTED
    external_source: Optional[str] = None
    contradictions: List[str] = Field(default_factory=list)
    tenant_id: str = "default"
    workspace_id: str = "default"
    user_id: Optional[str] = None
    classification: str = "REAL"
    source_ids: List[str] = Field(default_factory=list)
    source_id: Optional[str] = None
    video_id: Optional[str] = None
    document_id: Optional[str] = None
    modality: Optional[str] = "transcript"
    evidence_id: Optional[str] = None
    evidence_ref: Optional[str] = None
    extraction_method: str = "HEURISTIC"
    epistemic_classification: str = "HEURISTIC"
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None


class KnowledgeObject(BaseModel):
    id: str
    object_type: KnowledgeObjectType
    title: str = ""
    content: str
    confidence: float = 1.0
    provenance: Dict[str, Any] = Field(default_factory=dict)
    timestamps: List[float] = Field(default_factory=list)
    source_id: Optional[str] = None
    relationships: List[RelationshipEdge] = Field(default_factory=list)
    version: int = 1
    created_at: str = Field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    updated_at: str = Field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


class CodeSnippet(BaseModel):
    id: str
    language: str = "python"
    code: str
    purpose: str = ""
    explanation: str = ""
    complexity: str = "O(n)"
    runnable: bool = False
    imports: List[str] = Field(default_factory=list)
    functions: List[str] = Field(default_factory=list)
    inputs: List[str] = Field(default_factory=list)
    outputs: List[str] = Field(default_factory=list)
    potential_issues: List[str] = Field(default_factory=list)
    refactoring_suggestions: List[str] = Field(default_factory=list)
    timestamps: List[float] = Field(default_factory=list)


class MathDerivation(BaseModel):
    id: str
    latex: str
    variables: Dict[str, str] = Field(default_factory=dict)
    assumptions: List[str] = Field(default_factory=list)
    derivation_steps: List[str] = Field(default_factory=list)
    intuition: str = ""
    worked_example: Optional[str] = None
    timestamps: List[float] = Field(default_factory=list)


class ChapterItem(BaseModel):
    id: str
    title: str
    start_time: float
    end_time: float
    summary: str
    learning_objectives: List[str] = Field(default_factory=list)
    key_concepts: List[str] = Field(default_factory=list)
    difficulty: str = "Intermediate"
    prerequisites: List[str] = Field(default_factory=list)
    importance: float = 0.8
    visual_summary: Optional[str] = None


class QuizQuestion(BaseModel):
    id: str
    type: str = "mcq"  # mcq, true_false, fill_blank, code, short_answer
    question: str
    options: List[str] = Field(default_factory=list)
    answer: str
    explanation: str
    hint: str = ""
    difficulty: str = "Medium"  # Easy, Medium, Hard, Very Hard
    source_timestamps: List[float] = Field(default_factory=list)


class AnkiCardModel(BaseModel):
    id: str
    front: str
    back: str
    card_type: str = "basic"  # basic, cloze, code, formula
    tags: List[str] = Field(default_factory=list)
    difficulty: str = "Medium"
    cloze_text: Optional[str] = None
    code_snippet: Optional[str] = None
    latex_math: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Course & Learning Entities (Learner Model & Spaced Repetition)
# ─────────────────────────────────────────────────────────────────────────────

class CourseLesson(BaseModel):
    id: str
    title: str
    video_id: str
    duration_seconds: float
    summary: str
    concepts: List[str] = Field(default_factory=list)
    prerequisites: List[str] = Field(default_factory=list)
    quizzes: List[QuizQuestion] = Field(default_factory=list)
    coding_tasks: List[CodeSnippet] = Field(default_factory=list)
    flashcards: List[AnkiCardModel] = Field(default_factory=list)


class CourseModule(BaseModel):
    id: str
    title: str
    description: str
    lessons: List[CourseLesson] = Field(default_factory=list)


class Course(BaseModel):
    id: str
    title: str
    description: str
    difficulty: str = "Intermediate"
    estimated_duration_hours: float = 1.0
    prerequisites: List[str] = Field(default_factory=list)
    learning_outcomes: List[str] = Field(default_factory=list)
    modules: List[CourseModule] = Field(default_factory=list)


class ConceptMasteryRecord(BaseModel):
    concept_id: str
    concept_name: str
    mastery_state: LearnerMasteryState = LearnerMasteryState.NOT_STARTED
    confidence: float = 0.0
    recent_performance: float = 0.0
    attempt_count: int = 0
    correct_count: int = 0
    last_seen: Optional[float] = None
    next_review: Optional[float] = None
    ease_factor: float = 2.5
    interval_days: int = 0
    prerequisites: List[str] = Field(default_factory=list)


class SpacedReviewItem(BaseModel):
    item_id: str
    user_id: str
    concept_id: str
    question_id: Optional[str] = None
    question_text: str
    due_timestamp: float
    interval_days: int = 1
    repetition_number: int = 0
    ease_factor: float = 2.5
    review_type: str = "concept"  # concept, flashcard, quiz


class LearnerProfile(BaseModel):
    user_id: str
    watched_videos: List[str] = Field(default_factory=list)
    mastered_concepts: List[str] = Field(default_factory=list)
    weak_concepts: List[str] = Field(default_factory=list)
    concept_mastery: Dict[str, ConceptMasteryRecord] = Field(default_factory=dict)
    quiz_scores: Dict[str, float] = Field(default_factory=dict)
    total_study_seconds: float = 0.0
    preferences: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Collaboration, Quotas, Collections & Governance
# ─────────────────────────────────────────────────────────────────────────────

class MultiVideoCollection(BaseModel):
    id: str
    user_id: str
    title: str
    description: str = ""
    video_ids: List[str] = Field(default_factory=list)
    shared_concept_count: int = 0
    created_at: str = Field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    updated_at: str = Field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


class SmartBookmark(BaseModel):
    id: str
    user_id: str
    video_id: str
    timestamp: float
    title: str = ""
    text_snippet: str = ""
    visual_frame_id: Optional[str] = None
    concept_tags: List[str] = Field(default_factory=list)
    personal_note: str = ""
    ai_summary: Optional[str] = None
    created_at: str = Field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


class SharedResourceRecord(BaseModel):
    resource_id: str
    resource_type: str  # video, course, collection, bookmark
    owner_id: str
    grantee_id: str
    permission: CollaborationPermission = CollaborationPermission.VIEW
    created_at: str = Field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


class UserQuota(BaseModel):
    user_id: str
    role: str = "STUDENT"
    max_video_duration_minutes: int = 120
    max_storage_bytes: int = 5368709120  # 5 GB
    max_monthly_jobs: int = 50
    max_monthly_queries: int = 500
    current_storage_bytes: int = 0
    current_month_jobs: int = 0
    current_month_queries: int = 0


class AuditLogEntry(BaseModel):
    id: str
    timestamp: str = Field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    user_id: str
    event_type: str  # auth, access, quota, modification, deletion
    resource_id: Optional[str] = None
    action: str
    status: str = "SUCCESS"
    ip_address: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class VersionedArtifactRecord(BaseModel):
    artifact_id: str
    job_id: str
    artifact_type: str  # markdown, docx, pdf, anki, srt, vtt, json
    version: int = 1
    source_version: int = 1
    model_version: str = "gpt-4o"
    content_hash: str = ""
    creator_id: Optional[str] = None
    created_at: str = Field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


# ─────────────────────────────────────────────────────────────────────────────
# Observability & Cost Tracking
# ─────────────────────────────────────────────────────────────────────────────

class ProcessingCostRecord(BaseModel):
    asr_cost_usd: float = 0.0
    llm_tokens_in: int = 0
    llm_tokens_out: int = 0
    llm_cost_usd: float = 0.0
    vision_cost_usd: float = 0.0
    storage_bytes: int = 0
    total_cost_usd: float = 0.0
    duration_seconds: float = 0.0
