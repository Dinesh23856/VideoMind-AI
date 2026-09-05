from functools import lru_cache
from typing import List, Literal, Optional

from pydantic import Field
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
    scratch_dir: str = "/tmp/media_proc"
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
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"
    whisper_batch_size: int = 16
    asr_provider: Literal["faster-whisper", "groq", "deepgram"] = "faster-whisper"
    groq_api_key: Optional[str] = None
    deepgram_api_key: Optional[str] = None

    # LLM
    llm_provider: Literal["openai", "anthropic", "google"] = "openai"
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
    cors_origins: List[str] = Field(default_factory=lambda: ["http://localhost:3000"])


@lru_cache
def get_settings() -> Settings:
    return Settings()
