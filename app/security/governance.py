"""
Data Governance & Privacy 2.0: Data retention lifecycle, user data export,
and GDPR Right-to-Erasure deletion workflows.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from app.memory.personal_memory import PersonalMemoryManager


class DataGovernanceManager:
    """
    Manages enterprise privacy compliance, data retention, export, and user erasure.
    """

    def __init__(self, memory_manager: Optional[PersonalMemoryManager] = None):
        self.memory_manager = memory_manager or PersonalMemoryManager()

    def export_user_data(self, user_id: str, db_session: Any) -> Dict[str, Any]:
        """Gathers all tenant data across videos, courses, bookmarks, and memory for user export."""
        from app.database.models import User, VideoRecord, CourseRecord, BookmarkRecord

        user = db_session.query(User).filter(User.id == user_id).first()
        videos = db_session.query(VideoRecord).filter(VideoRecord.user_id == user_id).all()
        courses = db_session.query(CourseRecord).filter(CourseRecord.user_id == user_id).all()
        bookmarks = db_session.query(BookmarkRecord).filter(BookmarkRecord.user_id == user_id).all()
        memory = self.memory_manager.export_memory(user_id)

        return {
            "export_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "user": {
                "id": user.id if user else user_id,
                "email": user.email if user else "",
                "role": user.role if user else "STUDENT",
            },
            "videos_count": len(videos),
            "videos": [
                {"id": v.id, "title": v.title, "url": v.url, "created_at": str(v.created_at)}
                for v in videos
            ],
            "courses_count": len(courses),
            "courses": [
                {"id": c.id, "title": c.title, "created_at": str(c.created_at)}
                for c in courses
            ],
            "bookmarks": [
                {"id": b.id, "video_id": b.video_id, "timestamp": b.timestamp, "note": getattr(b, "note_text", "")}
                for b in bookmarks
            ],
            "learning_memory": memory,
        }

    def delete_user_data(self, user_id: str, db_session: Any) -> Dict[str, Any]:
        """
        Executes GDPR Right-to-Erasure: permanently purges user profile,
        videos, courses, bookmarks, and learning memory.
        """
        from app.database.models import User, VideoRecord, CourseRecord, BookmarkRecord, JobRecord

        # 1. Delete associated bookmarks
        db_session.query(BookmarkRecord).filter(BookmarkRecord.user_id == user_id).delete()

        # 2. Delete courses
        db_session.query(CourseRecord).filter(CourseRecord.user_id == user_id).delete()

        # 3. Delete jobs and videos
        db_session.query(JobRecord).filter(JobRecord.user_id == user_id).delete()
        db_session.query(VideoRecord).filter(VideoRecord.user_id == user_id).delete()

        # 4. Delete user account
        db_session.query(User).filter(User.id == user_id).delete()
        db_session.commit()

        # 5. Purge memory
        self.memory_manager.delete_memory(user_id)

        return {
            "user_id": user_id,
            "status": "PURGED",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "message": "All personal records, videos, courses, bookmarks, and memories permanently erased.",
        }
