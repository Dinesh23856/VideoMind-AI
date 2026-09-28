"""Core domain data models and schemas for VideoMind-AI."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class PipelineState(str, Enum):
    CREATED = "CREATED"
    QUEUED = "QUEUED"
    DOWNLOADING = "DOWNLOADING"
    INGESTING = "INGESTING"
    EXTRACTING_AUDIO = "EXTRACTING_AUDIO"
    TRANSCRIBING = "TRANSCRIBING"
    DIARIZING = "DIARIZING"
    ANALYZING_VIDEO = "ANALYZING_VIDEO"
    OCR_PROCESSING = "OCR_PROCESSING"
    SEGMENTING = "SEGMENTING"
    EXTRACTING_KNOWLEDGE = "EXTRACTING_KNOWLEDGE"
    INDEXING = "INDEXING"
    GENERATING_ARTIFACTS = "GENERATING_ARTIFACTS"
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
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNVERIFIED = "UNVERIFIED"
    CONFLICTING = "CONFLICTING"


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


class TutorMode(str, Enum):
    BEGINNER = "Beginner"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"
    EXPERT = "Expert"
    INTERVIEW = "Interview"
    EXAM = "Exam"
    RESEARCH = "Research"
    DEVELOPER = "Developer"


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
    frame_type: str = "slide"  # slide, diagram, code, terminal, chart, equation, speaker
    description: str = ""
    ocr_text: str = ""
    image_path: Optional[str] = None
    confidence: float = 0.95


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


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge & Semantic Entities
# ─────────────────────────────────────────────────────────────────────────────

class ConceptNode(BaseModel):
    id: str
    name: str
    definition: str
    importance: float = 0.8
    timestamps: List[float] = Field(default_factory=list)
    aliases: List[str] = Field(default_factory=list)
    category: str = "General"


class RelationshipEdge(BaseModel):
    source_id: str
    target_id: str
    relation: KnowledgeEdgeType
    confidence: float = 1.0
    description: str = ""


class FactClaim(BaseModel):
    id: str
    claim: str
    evidence: str
    confidence: float = 0.9
    source_timestamps: List[float] = Field(default_factory=list)
    status: FactVerificationStatus = FactVerificationStatus.SUPPORTED
    external_source: Optional[str] = None


class CodeSnippet(BaseModel):
    id: str
    language: str = "python"
    code: str
    explanation: str = ""
    complexity: str = "O(n)"
    runnable: bool = False
    imports: List[str] = Field(default_factory=list)
    functions: List[str] = Field(default_factory=list)
    timestamps: List[float] = Field(default_factory=list)


class MathDerivation(BaseModel):
    id: str
    latex: str
    variables: Dict[str, str] = Field(default_factory=dict)
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
# Course & Learning Entities
# ─────────────────────────────────────────────────────────────────────────────

class CourseLesson(BaseModel):
    id: str
    title: str
    video_id: str
    duration_seconds: float
    summary: str
    concepts: List[str] = Field(default_factory=list)
    prerequisites: List[str] = Field(default_factory=list)


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


class LearnerProfile(BaseModel):
    user_id: str
    watched_videos: List[str] = Field(default_factory=list)
    mastered_concepts: List[str] = Field(default_factory=list)
    weak_concepts: List[str] = Field(default_factory=list)
    quiz_scores: Dict[str, float] = Field(default_factory=dict)
    total_study_seconds: float = 0.0


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
