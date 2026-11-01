from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from typing import Any, List, Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=None if os.getenv("APP_ENV") == "test" else ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    app_env: str = Field(default_factory=lambda: os.getenv("APP_ENV", "development"))
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
    asr_provider: Literal["faster-whisper", "groq", "deepgram", "offline"] = Field(
        default_factory=lambda: "offline" if os.getenv("APP_ENV") == "test" else "faster-whisper"
    )
    groq_api_key: Optional[str] = None
    deepgram_api_key: Optional[str] = None

    # LLM & Local-First Frontier AI
    llm_provider: Literal["openai", "anthropic", "google", "groq", "ollama", "local", "llama_cpp", "transformers", "grok", "offline"] = Field(
        default_factory=lambda: "offline" if os.getenv("APP_ENV") == "test" else "openai"
    )
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    grok_api_key: Optional[str] = None
    xai_api_key: Optional[str] = None
    llm_model: str = "gpt-4o"
    llm_max_tokens: int = 8192
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    llama_cpp_base_url: str = "http://localhost:8080"
    transformers_server_url: Optional[str] = None
    local_llm_url: Optional[str] = None
    zero_cost_mode: bool = True

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

    @field_validator("jwt_secret_key")
    @classmethod
    def validate_jwt_secret(cls, v: str, info) -> str:
        unsafe_defaults = {
            "videomind-jwt-secret-key-change-in-production",
            "videomind-secret-key-change-in-production",
            "change-me",
            "secret",
            "jwt-secret",
            "default",
            "password",
            "123456",
        }
        # In production environments, refuse to start with a known unsafe/default secret
        env = (info.data.get("app_env") if info.data else None) or os.getenv("APP_ENV", "development")
        env = str(env).strip().lower()
        if env in ("production", "prod"):
            if not v or v in unsafe_defaults or len(v) < 32:
                raise ValueError(
                    "CRITICAL SECURITY CONFIGURATION ERROR: In production mode ('production'), "
                    "'jwt_secret_key' must be configured with a cryptographically strong secret "
                    "of at least 32 characters. Default/unsafe secrets are strictly blocked."
                )
        return v

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: str, info) -> str:
        env = (info.data.get("app_env") if info.data else None) or os.getenv("APP_ENV", "development")
        env = str(env).strip().lower()
        if env in ("production", "prod"):
            if not v or v in ("change-me", "default", "secret") or len(v) < 16:
                raise ValueError(
                    "CRITICAL SECURITY CONFIGURATION ERROR: In production mode ('production'), "
                    "'secret_key' must be configured with a secure key (at least 16 characters)."
                )
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
