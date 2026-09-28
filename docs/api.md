# VideoMind-AI API Reference

VideoMind-AI exposes a production REST API built with FastAPI, Server-Sent Events (SSE), and OpenAPI schemas available interactively at `/docs`.

---

## 1. Pipeline & Ingestion Endpoints

### `POST /api/v1/jobs/url`
Submits a public video URL (YouTube, Vimeo, etc.) for processing. Includes SSRF protection against private IP targets.

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

---

### `POST /api/v1/jobs/upload`
Uploads a local media file ($\le$ 1 GB).

**Form Data:**
- `file`: Media file (MP4, MKV, MOV, WEBM, MP3, WAV)
- `title` *(optional)*: Lecture title override

---

### `GET /api/v1/jobs/{job_id}/progress`
Server-Sent Events (SSE) stream providing real-time stage updates, percentages, and status changes.

---

### `GET /api/v1/jobs/{job_id}/artifacts/{name}`
Download educational artifacts:
- `markdown`: `notes.md`
- `pdf`: Typst compiled `notes.pdf`
- `anki`: Anki flashcard deck `deck.apkg`
- `srt`: Subtitles file `subtitles.srt`
- `vtt`: WebVTT subtitles `subtitles.vtt`
- `json`: Structured Knowledge AST `knowledge.json`
- `docx`: Word compatible document `notes.docx`

---

## 2. Knowledge & AI Endpoints

### `GET /api/v1/search?q={query}&k=5`
Executes hybrid search across all processed lectures with timestamp citations.

---

### `POST /api/v1/tutor/chat`
Engage with the Socratic AI Tutor.

**Request Body:**
```json
{
  "message": "Why does self-attention have quadratic complexity?",
  "mode": "Developer",
  "video_id": "optional-job-id"
}
```

---

### `GET /api/v1/knowledge/graph/{job_id}`
Returns Cytoscape.js formatted nodes and edges for interactive visualization.

---

## 3. Observability & Health

- `GET /health`: Liveness & metadata
- `GET /live`: Pod liveness probe
- `GET /ready`: Pod readiness probe (checks Redis & DB)
- `GET /metrics`: Prometheus telemetry metrics
