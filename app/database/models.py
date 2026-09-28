"""SQLAlchemy ORM models for VideoMind-AI relational persistence."""

from __future__ import annotations

import time
from sqlalchemy import (
    Boolean,
    Column,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from .session import Base


class User(Base):
    __tablename__ = "users"

    id = Column(String(64), primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    role = Column(String(32), default="USER", nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(Float, default=time.time, nullable=False)

    videos = relationship("VideoRecord", back_populates="owner")
    bookmarks = relationship("BookmarkRecord", back_populates="user")


class VideoRecord(Base):
    __tablename__ = "videos"

    id = Column(String(64), primary_key=True, index=True)
    user_id = Column(String(64), ForeignKey("users.id"), nullable=True, index=True)
    title = Column(String(512), nullable=False)
    source_type = Column(String(32), default="url", nullable=False)  # url, file
    source_url = Column(Text, nullable=True)
    file_path = Column(Text, nullable=True)
    duration_seconds = Column(Float, default=0.0)
    language = Column(String(16), default="en")
    created_at = Column(Float, default=time.time, nullable=False)

    owner = relationship("User", back_populates="videos")
    jobs = relationship("JobRecord", back_populates="video", cascade="all, delete-orphan")
    transcripts = relationship("TranscriptSegmentRecord", back_populates="video", cascade="all, delete-orphan")
    entities = relationship("KnowledgeEntityRecord", back_populates="video", cascade="all, delete-orphan")


class JobRecord(Base):
    __tablename__ = "jobs"

    id = Column(String(64), primary_key=True, index=True)
    video_id = Column(String(64), ForeignKey("videos.id"), nullable=True, index=True)
    status = Column(String(32), default="CREATED", index=True, nullable=False)
    current_stage = Column(String(64), default="CREATED", nullable=False)
    progress = Column(Float, default=0.0)
    message = Column(Text, default="Job created.")
    error = Column(Text, nullable=True)
    cost_usd = Column(Float, default=0.0)
    created_at = Column(Float, default=time.time, nullable=False)
    updated_at = Column(Float, default=time.time, onupdate=time.time, nullable=False)

    video = relationship("VideoRecord", back_populates="jobs")
    checkpoints = relationship("StageCheckpointRecord", back_populates="job", cascade="all, delete-orphan")


class StageCheckpointRecord(Base):
    __tablename__ = "stage_checkpoints"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(64), ForeignKey("jobs.id"), nullable=False, index=True)
    stage = Column(String(64), nullable=False)
    started_at = Column(Float, nullable=False)
    completed_at = Column(Float, nullable=True)
    data_json = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)

    job = relationship("JobRecord", back_populates="checkpoints")


class TranscriptSegmentRecord(Base):
    __tablename__ = "transcript_segments"

    id = Column(String(64), primary_key=True, index=True)
    video_id = Column(String(64), ForeignKey("videos.id"), nullable=False, index=True)
    start_time = Column(Float, nullable=False, index=True)
    end_time = Column(Float, nullable=False)
    speaker = Column(String(64), default="Speaker 1")
    speaker_role = Column(String(32), default="Speaker 1")
    text = Column(Text, nullable=False)
    language = Column(String(16), default="en")
    translated_text = Column(Text, nullable=True)
    confidence = Column(Float, default=1.0)
    importance = Column(Float, default=0.5)
    data_json = Column(Text, nullable=True)

    video = relationship("VideoRecord", back_populates="transcripts")


class KnowledgeEntityRecord(Base):
    __tablename__ = "knowledge_entities"

    id = Column(String(64), primary_key=True, index=True)
    video_id = Column(String(64), ForeignKey("videos.id"), nullable=False, index=True)
    entity_type = Column(String(32), index=True, nullable=False)  # Concept, Claim, Code, Formula, Chapter
    name = Column(String(255), index=True, nullable=False)
    definition = Column(Text, nullable=True)
    importance = Column(Float, default=0.8)
    data_json = Column(Text, nullable=True)
    created_at = Column(Float, default=time.time, nullable=False)

    video = relationship("VideoRecord", back_populates="entities")


class KnowledgeEdgeRecord(Base):
    __tablename__ = "knowledge_edges"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(String(64), nullable=False, index=True)
    target_id = Column(String(64), nullable=False, index=True)
    relation = Column(String(64), nullable=False)
    confidence = Column(Float, default=1.0)
    description = Column(Text, nullable=True)

    __table_args__ = (
        Index("idx_edge_source_target", "source_id", "target_id"),
    )


class BookmarkRecord(Base):
    __tablename__ = "bookmarks"

    id = Column(String(64), primary_key=True, index=True)
    user_id = Column(String(64), ForeignKey("users.id"), nullable=False, index=True)
    video_id = Column(String(64), nullable=False, index=True)
    timestamp = Column(Float, nullable=False)
    title = Column(String(255), nullable=False)
    note_text = Column(Text, nullable=True)
    category = Column(String(32), default="Note")  # Bookmark, Highlight, Note, Question, Important
    created_at = Column(Float, default=time.time, nullable=False)

    user = relationship("User", back_populates="bookmarks")


class QuizRecord(Base):
    __tablename__ = "quizzes"

    id = Column(String(64), primary_key=True, index=True)
    video_id = Column(String(64), nullable=False, index=True)
    question = Column(Text, nullable=False)
    question_type = Column(String(32), default="mcq")
    options_json = Column(Text, nullable=True)
    answer = Column(Text, nullable=False)
    explanation = Column(Text, nullable=True)
    hint = Column(Text, nullable=True)
    difficulty = Column(String(32), default="Medium")
    created_at = Column(Float, default=time.time, nullable=False)


class CourseRecord(Base):
    __tablename__ = "courses"

    id = Column(String(64), primary_key=True, index=True)
    title = Column(String(512), nullable=False)
    description = Column(Text, nullable=True)
    difficulty = Column(String(32), default="Intermediate")
    estimated_duration_hours = Column(Float, default=1.0)
    structure_json = Column(Text, nullable=False)
    created_at = Column(Float, default=time.time, nullable=False)


class AuditLogRecord(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64), nullable=True, index=True)
    action = Column(String(64), nullable=False, index=True)
    resource = Column(String(255), nullable=True)
    ip_address = Column(String(64), nullable=True)
    status = Column(String(32), default="SUCCESS")
    timestamp = Column(Float, default=time.time, nullable=False, index=True)
    details_json = Column(Text, nullable=True)
