"""Centralized error taxonomy and standardized structured exception handling."""

from __future__ import annotations

from typing import Any, Dict, Optional


class VideoMindException(Exception):
    """Base exception for all domain and operational errors in VideoMind-AI."""

    error_code: str = "INTERNAL_ERROR"
    status_code: int = 500
    retryable: bool = False

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        status_code: Optional[int] = None,
        retryable: Optional[bool] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        if error_code is not None:
            self.error_code = error_code
        if status_code is not None:
            self.status_code = status_code
        if retryable is not None:
            self.retryable = retryable
        self.details = details or {}

    def to_dict(self, request_id: Optional[str] = None, job_id: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "error_code": self.error_code,
            "message": self.message,
            "retryable": self.retryable,
        }
        if request_id:
            payload["request_id"] = request_id
        if job_id:
            payload["job_id"] = job_id
        if self.details:
            payload["details"] = self.details
        return payload


class ValidationError(VideoMindException):
    error_code = "VALIDATION_ERROR"
    status_code = 422
    retryable = False


class AuthenticationError(VideoMindException):
    error_code = "AUTHENTICATION_ERROR"
    status_code = 401
    retryable = False


class AuthorizationError(VideoMindException):
    error_code = "AUTHORIZATION_ERROR"
    status_code = 403
    retryable = False


class ResourceNotFoundError(VideoMindException):
    error_code = "RESOURCE_NOT_FOUND"
    status_code = 404
    retryable = False


class RateLimitExceededError(VideoMindException):
    error_code = "RATE_LIMITED"
    status_code = 429
    retryable = True


class ProviderTimeoutError(VideoMindException):
    error_code = "PROVIDER_TIMEOUT"
    status_code = 504
    retryable = True


class ProviderUnavailableError(VideoMindException):
    error_code = "PROVIDER_UNAVAILABLE"
    status_code = 503
    retryable = True


class MediaProcessingError(VideoMindException):
    error_code = "MEDIA_PROCESSING_ERROR"
    status_code = 500
    retryable = False


class VideoProcessingError(MediaProcessingError):
    """Alias for media processing failures during video ingestion."""
    pass


class StorageError(VideoMindException):
    error_code = "STORAGE_ERROR"
    status_code = 500
    retryable = True


class SSRFSecurityError(VideoMindException):
    error_code = "SSRF_ATTEMPT_BLOCKED"
    status_code = 400
    retryable = False


class PromptInjectionDetectedError(VideoMindException):
    error_code = "PROMPT_INJECTION_DETECTED"
    status_code = 400
    retryable = False


class PipelineExecutionError(VideoMindException):
    error_code = "PIPELINE_EXECUTION_ERROR"
    status_code = 500
    retryable = True
