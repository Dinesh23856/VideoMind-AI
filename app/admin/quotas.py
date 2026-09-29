"""
Resource & Quota Management 2.0: Per-user, per-role, and per-tenant limits
for video duration, storage bytes, pipeline jobs, and search queries.
"""

from __future__ import annotations

from typing import Dict, Optional
from app.core.errors import VideoProcessingError
from app.domain.models import UserQuota


class QuotaExceededError(VideoProcessingError):
    """Raised when a user attempts an action exceeding their tier quota."""
    pass


class QuotaManager:
    """
    Tracks and enforces resource quotas across user roles:
    STUDENT, TEACHER, RESEARCHER, ADMIN.
    """

    ROLE_DEFAULT_LIMITS = {
        "STUDENT": {
            "max_video_duration_minutes": 120,
            "max_storage_bytes": 5 * 1024 * 1024 * 1024,  # 5 GB
            "max_monthly_jobs": 50,
            "max_monthly_queries": 500,
        },
        "TEACHER": {
            "max_video_duration_minutes": 360,
            "max_storage_bytes": 25 * 1024 * 1024 * 1024,  # 25 GB
            "max_monthly_jobs": 200,
            "max_monthly_queries": 2000,
        },
        "RESEARCHER": {
            "max_video_duration_minutes": 720,
            "max_storage_bytes": 100 * 1024 * 1024 * 1024,  # 100 GB
            "max_monthly_jobs": 500,
            "max_monthly_queries": 5000,
        },
        "ADMIN": {
            "max_video_duration_minutes": 999999,
            "max_storage_bytes": 1024 * 1024 * 1024 * 1024,  # 1 TB
            "max_monthly_jobs": 999999,
            "max_monthly_queries": 999999,
        },
    }

    def __init__(self):
        # user_id -> UserQuota
        self._quotas: Dict[str, UserQuota] = {}

    def get_or_create_quota(self, user_id: str, role: str = "STUDENT") -> UserQuota:
        if user_id not in self._quotas:
            limits = self.ROLE_DEFAULT_LIMITS.get(role.upper(), self.ROLE_DEFAULT_LIMITS["STUDENT"])
            self._quotas[user_id] = UserQuota(
                user_id=user_id,
                role=role.upper(),
                max_video_duration_minutes=limits["max_video_duration_minutes"],
                max_storage_bytes=limits["max_storage_bytes"],
                max_monthly_jobs=limits["max_monthly_jobs"],
                max_monthly_queries=limits["max_monthly_queries"],
            )
        return self._quotas[user_id]

    def check_and_increment_job(self, user_id: str, video_duration_minutes: float = 0.0, role: str = "STUDENT") -> None:
        """Enforces limits before initiating a new pipeline job."""
        quota = self.get_or_create_quota(user_id, role)
        if quota.role == "ADMIN":
            quota.current_month_jobs += 1
            return

        if quota.current_month_jobs >= quota.max_monthly_jobs:
            raise QuotaExceededError(
                f"Monthly job limit reached ({quota.current_month_jobs}/{quota.max_monthly_jobs}). Please upgrade your tier."
            )

        if video_duration_minutes > quota.max_video_duration_minutes:
            raise QuotaExceededError(
                f"Video duration ({video_duration_minutes:.1f}m) exceeds your tier maximum of {quota.max_video_duration_minutes}m."
            )

        quota.current_month_jobs += 1

    def check_and_increment_query(self, user_id: str, role: str = "STUDENT") -> None:
        """Enforces query rate / monthly query limits."""
        quota = self.get_or_create_quota(user_id, role)
        if quota.role == "ADMIN":
            quota.current_month_queries += 1
            return

        if quota.current_month_queries >= quota.max_monthly_queries:
            raise QuotaExceededError(
                f"Monthly query limit reached ({quota.current_month_queries}/{quota.max_monthly_queries})."
            )

        quota.current_month_queries += 1

    def update_storage_usage(self, user_id: str, bytes_delta: int) -> None:
        quota = self.get_or_create_quota(user_id)
        quota.current_storage_bytes = max(0, quota.current_storage_bytes + bytes_delta)
        if quota.role != "ADMIN" and quota.current_storage_bytes > quota.max_storage_bytes:
            raise QuotaExceededError(
                f"Storage limit exceeded ({quota.current_storage_bytes / 1e9:.2f} GB / {quota.max_storage_bytes / 1e9:.2f} GB)."
            )
