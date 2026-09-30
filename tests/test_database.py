import uuid
from app.database.models import JobRecord, StageCheckpointRecord, User, VideoRecord
from app.database.session import SessionLocal, init_db



def test_database_crud():
    init_db()
    with SessionLocal() as db:
        user_id = f"test_usr_{uuid.uuid4().hex[:8]}"
        user = User(
            id=user_id,
            email=f"{user_id}@videomind.ai",
            hashed_password="hashed_pw_secret",
            role="TEACHER",
        )
        db.add(user)
        db.commit()

        video_id = f"vid_{uuid.uuid4().hex[:8]}"
        video = VideoRecord(
            id=video_id,
            user_id=user.id,
            title="Transformer Architecture Deep Dive",
            source_type="url",
            source_url="https://youtube.com/watch?v=test",
            duration_seconds=3600.0,
        )
        db.add(video)

        job_id = f"job_{uuid.uuid4().hex[:8]}"
        job = JobRecord(
            id=job_id,
            video_id=video.id,
            status="COMPLETED",
            current_stage="COMPLETED",
            progress=100.0,
        )
        db.add(job)
        db.commit()

        # Checkpoint record
        cp = StageCheckpointRecord(
            job_id=job.id,
            stage="TRANSCRIBING",
            started_at=100.0,
            completed_at=115.0,
            data_json='{"segments_count": 42}',
        )
        db.add(cp)
        db.commit()

        # Query back
        retrieved_job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
        assert retrieved_job is not None
        assert retrieved_job.status == "COMPLETED"
        assert retrieved_job.video.title == "Transformer Architecture Deep Dive"
        assert len(retrieved_job.checkpoints) == 1
        assert retrieved_job.checkpoints[0].stage == "TRANSCRIBING"
        assert retrieved_job.checkpoints[0].data_json == '{"segments_count": 42}'

