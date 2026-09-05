# Video-to-Knowledge Pipeline

Production-grade architecture that turns dense Data Science / Generative AI lecture videos into structured educational assets:

- Timestamped Markdown notes (LaTeX math, PEP-8 code, hierarchical structure)
- Anki flashcard decks (`.apkg`)
- Publication-ready PDFs (Typst)
- Optional Notion workspace pages
- Real-time progress via Server-Sent Events

Built from the technical blueprint in *Engineering Architecture for Technical Video-to-Knowledge Pipelines*.

## Architecture Overview

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  Ingestion  │────▶│  ASR Engine  │────▶│ LLM Distillation│────▶│ Artifact Compile │
│ yt-dlp /    │     │ faster-whisper│     │ Frontier models │     │ Anki / PDF /     │
│ ffmpeg      │     │ + Silero VAD │     │ (full context)  │     │ Notion           │
└─────────────┘     └──────────────┘     └─────────────────┘     └──────────────────┘
        ▲                                                                      │
        │                    FastAPI + Celery + Redis + SSE                    │
        └──────────────────────────────────────────────────────────────────────┘
```

### Key design decisions (from the reference architecture)

| Stage | Choice | Why |
|-------|--------|-----|
| Ingestion | Prefer human/auto captions via `yt-dlp`; fall back to `ffmpeg` 16 kHz mono WAV | Zero ASR cost when captions exist; demux avoids video decode |
| ASR | `faster-whisper` + Silero VAD + `condition_on_previous_text=False` | 40–70× real-time, no autoregressive hallucination loops |
| Structuring | Single-pass long-context LLM (not Map-Reduce) | Preserves multi-minute mathematical derivations |
| PDF | Typst | Millisecond compile, native math, tiny binary vs Chromium/TeXLive |
| Orchestration | FastAPI (202) + Celery + Redis Pub/Sub + SSE | Non-blocking, live progress, no client polling |

## Deploy to Render (recommended)

Full guide: **[DEPLOY_RENDER.md](./DEPLOY_RENDER.md)**

1. Push this repo to GitHub  
2. Render Dashboard → **New → Blueprint** → select repo  
3. Set secrets on **both** `vtk-api` and `vtk-worker`:
   - `OPENAI_API_KEY`
   - `GROQ_API_KEY`  
4. Open the service URL → web UI is at `/`

On Render, ASR defaults to **Groq** (no GPU needed).

---

## Quick Start (local)

### 1. Prerequisites

- Python 3.11+
- `ffmpeg` on `$PATH`
- Redis
- (Optional) NVIDIA GPU + CUDA for local Whisper
- API keys for LLM (OpenAI / Anthropic / Google) and optionally Groq/Deepgram

### 2. Install

```bash
git clone <this-repo>
cd video-to-knowledge-pipeline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your keys
```

### 3. Run with Docker Compose (recommended)

```bash
docker compose up --build
```

- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Redis: localhost:6379

### 4. Local development (without Docker)

```bash
# Terminal 1 – Redis
redis-server

# Terminal 2 – Celery worker
celery -A app.orchestration.celery_app.celery_app worker -l info --concurrency=1

# Terminal 3 – API
uvicorn app.api.main:app --reload --port 8000
```

### 5. CLI (synchronous, great for debugging)

```bash
python scripts/cli.py process "https://www.youtube.com/watch?v=..." --title "Attention Is All You Need"
# or local file
python scripts/cli.py process ./lecture.mp4 --output ./out
```

## API Usage

### Submit a YouTube URL

```bash
curl -X POST http://localhost:8000/api/v1/jobs/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=...", "title": "Transformers Lecture"}'
```

Response (202):

```json
{
  "job_id": "a1b2c3d4-...",
  "status": "ACCEPTED",
  "progress_url": "/api/v1/jobs/a1b2c3d4-.../progress"
}
```

### Stream progress (SSE)

```javascript
const es = new EventSource("http://localhost:8000/api/v1/jobs/<job_id>/progress");
es.addEventListener("update", (e) => {
  const data = JSON.parse(e.data);
  console.log(data.status, data.progress, data.message);
  if (data.status === "COMPLETED" || data.status === "FAILED") es.close();
});
```

### Upload a local video (≤ 1 GB)

```bash
curl -X POST http://localhost:8000/api/v1/jobs/upload \
  -F "file=@lecture.mp4" \
  -F "title=My Lecture"
```

### Download artifacts

```
GET /api/v1/jobs/{job_id}/artifacts/markdown
GET /api/v1/jobs/{job_id}/artifacts/transcript
GET /api/v1/jobs/{job_id}/artifacts/pdf
GET /api/v1/jobs/{job_id}/artifacts/anki
```

## Configuration

All settings are environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `ASR_PROVIDER` | `faster-whisper` | `faster-whisper` \| `groq` \| `deepgram` |
| `WHISPER_MODEL_SIZE` | `large-v3` | Model size for local ASR |
| `WHISPER_DEVICE` | `cuda` | `cuda` or `cpu` |
| `LLM_PROVIDER` | `openai` | `openai` \| `anthropic` \| `google` |
| `LLM_MODEL` | `gpt-4o` | Model name |
| `NOTION_API_KEY` | – | Optional Notion integration |
| `NOTION_PARENT_PAGE_ID` | – | Parent page for new notes |

## Project Layout

```
video-to-knowledge-pipeline/
├── app/
│   ├── api/              # FastAPI gateway + SSE
│   ├── ingestion/        # yt-dlp captions + ffmpeg demux
│   ├── transcription/    # faster-whisper + VAD
│   ├── structuring/      # LLM prompts + clients
│   ├── artifacts/        # Anki, Notion, Typst PDF
│   └── orchestration/    # Celery tasks + Redis pub/sub
├── config/settings.py
├── scripts/cli.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

## Economic Notes (from the reference)

- Prefer YouTube captions whenever available → $0 ASR cost.
- Under ~2 000 audio-hours/month, serverless ASR (Groq Whisper Turbo ≈ $0.04/h) is usually cheaper than a dedicated GPU instance.
- Frontier LLMs (Claude 3.5 Sonnet / GPT-4o) cost roughly $0.14–0.18 per two-hour lecture; distilled models are cheaper but lose mathematical fidelity.

## License

MIT – adapt freely for educational and production use.
