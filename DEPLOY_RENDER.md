# Deploy VideoMind-AI to Render (Step-by-Step)

This project is fully configured for Render via Infrastructure-as-Code Blueprint (`render.yaml`).

## Architecture on Render

```text
User / Browser
      │
      ▼
┌─────────────────────────────────┐
│  vtk-api (FastAPI Web Service)  │
│  - Serves Web UI & Docs         │
│  - Enqueues Jobs to Redis       │
│  - Streams SSE Progress         │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│  vtk-redis (Managed Redis)      │
│  - Task Queue Broker            │
│  - Real-time Pub/Sub Progress   │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│  vtk-worker (Celery Worker)     │
│  - Captions / Audio Demux       │
│  - Speech-to-Text (Groq/Deepgram│
│  - LLM Synthesis                │
│  - Anki / Typst PDF Generation  │
└─────────────────────────────────┘
```

**ASR on Render uses Groq or Deepgram** (serverless Whisper) — no expensive GPU instance needed.

---

## 1. Prerequisites & API Keys

You need at least:
1. **OpenAI API Key** (or Anthropic / Google Gemini) for lecture distillation:
   https://platform.openai.com/api-keys
2. **Groq API Key** (or Deepgram) for blazing-fast serverless speech-to-text:
   https://console.groq.com/keys

Optional:
- `DEEPGRAM_API_KEY`: For audio files > 25MB without compression
- `NOTION_API_KEY` & `NOTION_PARENT_PAGE_ID`: For direct export into your Notion workspace

---

## 2. Push Code to GitHub

```bash
cd VideoMind-AI
git init
git add .
git commit -m "feat: complete VideoMind-AI pipeline"
git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
git branch -M main
git push -u origin main
```

---

## 3. Launch Render Blueprint

1. Go to your [Render Dashboard](https://dashboard.render.com).
2. Click **New +** → **Blueprint**.
3. Connect your GitHub repository.
4. Render automatically reads `render.yaml` and provisions all 3 services:
   - `vtk-api`: FastAPI Web Service
   - `vtk-worker`: Celery Background Worker
   - `vtk-redis`: Redis Instance
5. Click **Apply**.

---

## 4. Set Secret Environment Variables

In the Render dashboard, set the following environment variables on **both** `vtk-api` and `vtk-worker`:

| Key | Value | Description |
|:---|:---|:---|
| `OPENAI_API_KEY` | `sk-...` | (Or `GOOGLE_API_KEY` / `ANTHROPIC_API_KEY`) |
| `GROQ_API_KEY` | `gsk-...` | (Or `DEEPGRAM_API_KEY`) |
| `LLM_PROVIDER` | `openai` | `openai`, `google`, or `anthropic` |
| `ASR_PROVIDER` | `groq` | `groq`, `deepgram`, or `faster-whisper` |

*(Note: Redis URLs are wired automatically by Render via the Blueprint property link).*

---

## 5. Verify Health & Use the App

Once deployed:
- Web Application: `https://vtk-api.onrender.com/`
- Health Probe: `https://vtk-api.onrender.com/health`
- Swagger OpenAPI Docs: `https://vtk-api.onrender.com/docs`

1. Paste any YouTube lecture URL (or upload a local video/audio file).
2. Click **Start processing**.
3. Watch the real-time Server-Sent Events progress bar.
4. Download your generated timestamped Markdown, Anki `.apkg` deck, and Typst PDF!