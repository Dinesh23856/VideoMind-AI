"""AI Tutor, Course Generation, Mastery Tracking, and Spaced Repetition router."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api import deps
from app.api.schemas import (
    CalibrateMasteryRequest,
    CourseCreateRequest,
    GenerateStudyPackRequest,
    GradeReviewRequest,
    LearningPathRequest,
    ScheduleReviewRequest,
    TutorChatRequest,
)
from app.database.models import (
    CourseRecord,
    KnowledgeEntityRecord,
    TranscriptSegmentRecord,
    VideoRecord,
)
from app.database.session import SessionLocal
from app.domain.models import ConceptNode, LearnerProfile, TranscriptSegment, UserRole
from app.learning.course import CourseGenerator, LearningPathEngine
from app.security.auth import (
    get_current_user,
    get_optional_current_user,
    verify_resource_ownership,
)

router = APIRouter(tags=["AI Tutor & Learning"])


@router.post("/api/v1/tutor/chat")
async def tutor_chat(
    body: TutorChatRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """Interactive Socratic AI Tutor with 8 pedagogical modes and grounded citations."""
    if body.video_id:
        deps._get_and_check_video_access(body.video_id, current_user)

    res = deps.global_tutor_engine.interact(
        user_message=body.message,
        mode=body.mode,
        video_id=body.video_id,
        context_hints_given=body.hints_count,
        student_solution=body.solution_attempt,
    )
    return res


@router.post("/api/v1/courses", status_code=status.HTTP_201_CREATED)
async def create_course(
    body: CourseCreateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Compiles multiple video lectures into a cohesive, structured course."""
    with SessionLocal() as db:
        videos_meta = []
        for vid_id in body.video_ids:
            v = db.query(VideoRecord).filter(VideoRecord.id == vid_id).first()
            if not v:
                raise HTTPException(status_code=404, detail=f"Video '{vid_id}' not found")
            verify_resource_ownership(v.user_id, current_user, "video")
            videos_meta.append({
                "id": v.id,
                "title": v.title,
                "duration": v.duration_seconds,
                "concepts": [],
                "summary": f"Lecture: {v.title}",
            })

        course_obj = CourseGenerator().compile_course(
            title=body.title,
            description=body.description,
            videos=videos_meta,
        )

        course_rec = CourseRecord(
            id=course_obj.id,
            user_id=current_user["id"],
            title=course_obj.title,
            description=course_obj.description,
            difficulty=course_obj.difficulty,
            estimated_duration_hours=course_obj.estimated_duration_hours,
            structure_json=json.dumps(course_obj.model_dump(), default=str),
        )
        db.add(course_rec)
        db.commit()
        return course_obj.model_dump()


@router.get("/api/v1/courses")
async def list_courses(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """List courses owned by current user (or all courses for ADMIN)."""
    with SessionLocal() as db:
        query = db.query(CourseRecord)
        if current_user.get("role") != UserRole.ADMIN.value:
            query = query.filter(CourseRecord.user_id == current_user["id"])
        total = query.count()
        courses = query.order_by(CourseRecord.created_at.desc()).offset(offset).limit(limit).all()
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "courses": [
                {
                    "id": c.id,
                    "title": c.title,
                    "description": c.description,
                    "difficulty": c.difficulty,
                    "estimated_duration_hours": c.estimated_duration_hours,
                    "created_at": c.created_at,
                }
                for c in courses
            ],
        }


@router.get("/api/v1/courses/{course_id}")
async def get_course(
    course_id: str,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """Retrieve full course details with ownership protection."""
    with SessionLocal() as db:
        course = db.query(CourseRecord).filter(CourseRecord.id == course_id).first()
        if not course:
            raise HTTPException(status_code=404, detail="Course not found")
        if course.user_id:
            if not current_user:
                raise HTTPException(status_code=401, detail="Authentication required")
            verify_resource_ownership(course.user_id, current_user, "course")

        return {
            "id": course.id,
            "title": course.title,
            "description": course.description,
            "difficulty": course.difficulty,
            "estimated_duration_hours": course.estimated_duration_hours,
            "created_at": course.created_at,
            "structure": json.loads(course.structure_json),
        }


@router.delete("/api/v1/courses/{course_id}")
async def delete_course(
    course_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Deletes a course owned by current user."""
    with SessionLocal() as db:
        course = db.query(CourseRecord).filter(CourseRecord.id == course_id).first()
        if not course:
            raise HTTPException(status_code=404, detail="Course not found")
        verify_resource_ownership(course.user_id, current_user, "course")
        db.delete(course)
        db.commit()
        return {"status": "deleted", "course_id": course_id}


@router.post("/api/v1/learning/path")
async def generate_learning_path(
    body: LearningPathRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """Computes personalized, adaptive learning paths based on mastery and weakness."""
    profile = LearnerProfile(
        user_id=current_user["id"] if current_user else "guest",
        mastered_concepts=body.mastered_concepts,
        weak_concepts=body.weak_concepts,
    )
    concepts = []
    if body.video_id:
        deps._get_and_check_video_access(body.video_id, current_user)
        with SessionLocal() as db:
            entities = db.query(KnowledgeEntityRecord).filter(KnowledgeEntityRecord.video_id == body.video_id).all()
            for e in entities:
                concepts.append(ConceptNode(id=e.id, name=e.name, definition=e.definition or "", importance=e.importance))

    path = LearningPathEngine().generate_personalized_path(profile, concepts)
    return {"user_id": profile.user_id, "path": path}


@router.get("/api/v1/learning/mastery/{concept_id}")
async def get_concept_mastery(
    concept_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    rec = deps.global_mastery_tracker.get_or_create_record(current_user["id"], concept_id, concept_id)
    return rec.model_dump()


@router.post("/api/v1/learning/mastery/calibrate")
async def record_mastery_attempt(
    req: CalibrateMasteryRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    rec = deps.global_mastery_tracker.record_attempt(
        current_user["id"],
        req.concept_id,
        req.concept_name,
        req.is_correct,
        req.score or 1.0,
    )
    return rec.model_dump()


@router.post("/api/v1/learning/reviews/schedule")
async def schedule_review(
    req: ScheduleReviewRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    item = deps.global_spaced_scheduler.schedule_initial_review(
        user_id=current_user["id"],
        concept_id=req.concept_id,
        question_text=req.question_text,
        question_id=req.question_id,
        review_type=req.review_type or "concept",
    )
    return item.model_dump()


@router.get("/api/v1/learning/reviews/due")
async def get_due_reviews(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    items = deps.global_spaced_scheduler.get_due_reviews(current_user["id"])
    return {"due_reviews": [item.model_dump() for item in items]}


@router.post("/api/v1/learning/reviews/{item_id}/grade")
async def grade_review_item(
    item_id: str,
    req: GradeReviewRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    updated = deps.global_spaced_scheduler.process_review_result(item_id, req.quality)
    return updated.model_dump()


@router.post("/api/v1/study-materials/generate")
async def generate_study_materials(
    req: GenerateStudyPackRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    video = deps._get_and_check_video_access(req.video_id, current_user)
    with SessionLocal() as db:
        seg_records = (
            db.query(TranscriptSegmentRecord)
            .filter(TranscriptSegmentRecord.video_id == req.video_id)
            .order_by(TranscriptSegmentRecord.start_time.asc())
            .all()
        )
        entity_records = (
            db.query(KnowledgeEntityRecord)
            .filter(KnowledgeEntityRecord.video_id == req.video_id)
            .all()
        )
    segments = [
        TranscriptSegment(
            start_time=s.start_time,
            end_time=s.end_time,
            text=s.text,
            speaker=s.speaker,
            speaker_role=s.speaker_role,
            importance=s.importance,
        )
        for s in seg_records
    ]
    concepts = [
        ConceptNode(
            id=e.id,
            name=e.name,
            definition=e.definition or "",
            category=e.entity_type,
            importance=e.importance,
        )
        for e in entity_records
    ]
    pack = deps.global_study_material_gen.generate_pack(
        video_id=video.id,
        video_title=video.title,
        segments=segments,
        concepts=concepts,
    )
    return pack.model_dump()
