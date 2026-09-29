"""
Admin Console Services: Operational health, user analytics, queue depths,
storage utilization, and security event auditing.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from app.domain.models import AuditLogEntry, PipelineState


class AdminConsoleService:
    """
    Backing service for the Admin Operations Center.
    Requires ADMIN role to access.
    """

    def __init__(self):
        self.audit_log: List[AuditLogEntry] = []

    def record_audit_event(
        self,
        user_id: str,
        event_type: str,
        action: str,
        resource_id: Optional[str] = None,
        status: str = "SUCCESS",
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditLogEntry:
        entry = AuditLogEntry(
            id=f"audit_{int(time.time() * 1000)}",
            user_id=user_id,
            event_type=event_type,
            resource_id=resource_id,
            action=action,
            status=status,
            details=details or {},
        )
        self.audit_log.append(entry)
        if len(self.audit_log) > 1000:
            self.audit_log.pop(0)
        return entry

    def get_system_telemetry(self, db_session: Any) -> Dict[str, Any]:
        """Gathers system-wide operations overview from the database."""
        from app.database.models import User, VideoRecord, JobRecord, CourseRecord

        total_users = db_session.query(User).count()
        total_videos = db_session.query(VideoRecord).count()
        total_courses = db_session.query(CourseRecord).count()

        completed_jobs = db_session.query(JobRecord).filter(JobRecord.status == "COMPLETED").count()
        failed_jobs = db_session.query(JobRecord).filter(JobRecord.status == "FAILED").count()
        active_jobs = db_session.query(JobRecord).filter(
            JobRecord.status.in_(["QUEUED", "DOWNLOADING", "TRANSCRIBING", "INDEXING"])
        ).count()

        return {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "users": {
                "total": total_users,
            },
            "media": {
                "total_videos": total_videos,
                "total_courses": total_courses,
            },
            "jobs": {
                "active": active_jobs,
                "completed": completed_jobs,
                "failed": failed_jobs,
                "total": completed_jobs + failed_jobs + active_jobs,
            },
            "recent_audit_events": [e.model_dump() for e in self.audit_log[-10:]],
            "system_health": "HEALTHY" if failed_jobs < 10 else "ATTENTION_REQUIRED",
        }
