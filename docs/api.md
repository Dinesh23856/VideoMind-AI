# VideoMind-AI API Reference

VideoMind-AI exposes a production REST API built with FastAPI, Server-Sent Events (SSE), and OpenAPI schemas available interactively at `/docs`.

---

## 1. Authentication & User Management

### `POST /api/v1/auth/register`
Creates a new user account with hashed password and role assignment.

**Request Body:**
```json
{
  "email": "student@university.edu",
  "password": "SecurePassword123!",
  "role": "STUDENT"
}
```

### `POST /api/v1/auth/login`
Authenticates user credentials and returns a signed HS256 JWT access token.

**Response (200 OK):**
```json
{
  "access_token": "eyJhbGciOi...",
  "token_type": "bearer",
  "role": "STUDENT"
}
```

### `GET /api/v1/auth/me`
Retrieves the profile and role metadata of the authenticated user.

---

## 2. Video & Ingestion Endpoints

### `POST /api/v1/jobs/url`
Submits a public video URL (YouTube, Vimeo, etc.) for processing. Protected by SSRF boundary checks.

**Request Body:**
```json
{
  "url": "https://www.youtube.com/watch?v=kCc8FmEb1nY",
  "title": "Karpathy: Intro to Large Language Models"
}
```

**Response (202 Accepted):**
```json
{
  "job_id": "97f0a991-8a90-449e-b2d3-2f8dfa1f9a2e",
  "status": "ACCEPTED",
  "progress_url": "/api/v1/jobs/97f0a991-8a90-449e-b2d3-2f8dfa1f9a2e/progress"
}
```

### `POST /api/v1/jobs/upload`
Uploads a local media file ($\le$ 1 GB) with filename sanitization and MIME inspection.

### `GET /api/v1/jobs/{job_id}/progress`
Server-Sent Events (SSE) stream providing real-time stage updates, percentages, and status changes.

### `POST /api/v1/jobs/{job_id}/cancel`
Cancels an in-flight background pipeline execution (enforcing tenant isolation).

### `GET /api/v1/jobs/{job_id}/artifacts/{name}`
Download educational artifacts:
- `markdown`: `notes.md`
- `docx`: Native OpenXML Word document `notes.docx`
- `pdf`: Typst compiled `notes.pdf`
- `anki`: Anki flashcard deck `deck.apkg`
- `srt`: Subtitles file `subtitles.srt`
- `vtt`: WebVTT subtitles `subtitles.vtt`
- `json`: Structured Knowledge AST `knowledge.json`

### `GET /api/v1/videos`
Lists all processed video libraries belonging to the authenticated tenant.

### `GET /api/v1/videos/{video_id}`
Returns video metadata, duration, speaker breakdown, and keyframe manifests.

### `DELETE /api/v1/videos/{video_id}`
Deletes a video and its indexed vectors (owner or admin only).

### `GET /api/v1/videos/{video_id}/transcript`
Returns timestamped, diarized transcript segments with speaker labels.

---

## 3. Knowledge, Search & RAG Endpoints

### `GET /api/v1/search?q={query}&k=5`
Executes hybrid lexical (BM25) and sparse/dense retrieval across indexed video chunks with citation timestamps.

### `POST /api/v1/tutor/chat`
Engage with the Socratic AI Tutor across 8 pedagogical modes:
- `Beginner`, `Intermediate`, `Advanced`, `Expert`
- `Interview`, `Exam`, `Research`, `Developer`

**Request Body:**
```json
{
  "message": "Why does self-attention have quadratic complexity?",
  "mode": "Developer",
  "video_id": "optional-job-id"
}
```

### `GET /api/v1/knowledge/graph/{job_id}`
Returns Cytoscape.js formatted nodes and edges for interactive visualization.

---

## 4. Courses & Adaptive Learning

### `POST /api/v1/courses`
Synthesizes an structured multi-unit course curriculum from a set of processed video IDs.

### `GET /api/v1/courses`
Lists all courses synthesized by the user.

### `GET /api/v1/courses/{course_id}`
Retrieves syllabus modules, learning objectives, and quiz questions.

### `POST /api/v1/learning/path`
Computes an adaptive prerequisite learning path based on learner goal concepts.

---

## 5. Observability & Health

- `GET /health`: Comprehensive service health, database state, and feature flags
- `GET /live`: Kubernetes / Docker liveness probe
- `GET /ready`: Pod readiness probe (checks DB connectivity and Redis status)
- `GET /metrics`: Prometheus telemetry metrics (request rates, latencies, errors)
