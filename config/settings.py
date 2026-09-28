from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from typing import Any, List, Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    app_env: str = "development"
    debug: bool = True
    secret_key: str = "change-me"

    # Storage
    scratch_dir: str = Field(
        default_factory=lambda: os.getenv("SCRATCH_DIR") or os.path.join(tempfile.gettempdir(), "media_proc")
    )
    s3_bucket: Optional[str] = None
    s3_endpoint_url: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    presigned_url_expiry_seconds: int = 86400

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # ASR
    whisper_model_size: str = "large-v3"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_batch_size: int = 8
    asr_provider: Literal["faster-whisper", "groq", "deepgram", "offline"] = "faster-whisper"
    groq_api_key: Optional[str] = None
    deepgram_api_key: Optional[str] = None

    # LLM
    llm_provider: Literal["openai", "anthropic", "google", "groq", "offline"] = "openai"
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    llm_model: str = "gpt-4o"
    llm_max_tokens: int = 8192

    # Notion
    notion_api_key: Optional[str] = None
    notion_parent_page_id: Optional[str] = None

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_base_url: Optional[str] = None
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])

    # Database & Storage
    database_url: str = Field(
        default_factory=lambda: os.getenv("DATABASE_URL") or "sqlite:///./videomind.db"
    )
    jwt_secret_key: str = "videomind-jwt-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 1440
    max_upload_size_bytes: int = 1073741824  # 1 GB

    # Intelligence Pipeline Feature Flags
    enable_vision_pipeline: bool = True
    enable_diarization: bool = True
    enable_fact_verification: bool = True
    vector_store_type: str = "inmemory"

    @field_validator("asr_provider", mode="before")
    @classmethod
    def normalize_asr_provider(cls, v: Any) -> str:
        if isinstance(v, str):
            return v.strip().lower()
        return v

    @field_validator("llm_provider", mode="before")
    @classmethod
    def normalize_llm_provider(cls, v: Any) -> str:
        if isinstance(v, str):
            return v.strip().lower()
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
