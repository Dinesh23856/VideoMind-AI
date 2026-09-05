# Deploy to Render (step-by-step)

This project is ready for Render via Blueprint (`render.yaml`).

## What gets deployed

| Service | Type | Role |
|---------|------|------|
| `vtk-api` | Web | FastAPI + web UI |
| `vtk-worker` | Background Worker | Celery pipeline jobs |
| `vtk-redis` | Redis | Queue + progress pub/sub |

**ASR on Render uses Groq** (serverless Whisper) — no GPU required.

---

## 1. Get API keys

You need at least:

1. **OpenAI** (or Anthropic / Google) → for structured notes  
   https://platform.openai.com/api-keys

2. **Groq** → for speech-to-text (cheap & fast)  
   https://console.groq.com/keys

Optional: Deepgram, Notion.

---

## 2. Push code to GitHub

```bash
cd video-to-knowledge-pipeline
git init
git add .
git commit -m "Video-to-Knowledge pipeline"
# Create a new empty repo on GitHub, then:
git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
git branch -M main
git push -u origin main
```

---

## 3. Create Blueprint on Render

1. Go to [https://dashboard.render.com](https://dashboard.render.com)
2. **New** → **Blueprint**
3. Connect the GitHub repo
4. Render reads `render.yaml` and shows 3 services
5. Click **Apply**

---

## 4. Set secret environment variables

In the Render dashboard, for **both** `vtk-api` and `vtk-worker`:

| Key | Value |
|-----|--------|
| `OPENAI_API_KEY` | `sk-...` |
| `GROQ_API_KEY` | your Groq key |

(Optional) `ANTHROPIC_API_KEY`, `DEEPGRAM_API_KEY`, `NOTION_API_KEY`, `NOTION_PARENT_PAGE_ID`

`sync: false` in `render.yaml` means you must enter these in the UI (they are not stored in git).

Redis URLs are linked automatically from `vtk-redis`.

---

## 5. Wait for deploy

- First build takes several minutes (Docker + ffmpeg + Python deps)
- Open the `vtk-api` URL (e.g. `https://vtk-api.onrender.com`)
- You should see the web UI
- Health: `https://vtk-api.onrender.com/health`

---

## 6. Use the app

1. Paste a YouTube lecture URL (or upload a file)
2. Click **Start processing**
3. Watch live progress
4. Download Markdown / PDF / Anki when complete

API docs: `https://YOUR-SERVICE.onrender.com/docs`

---

## Cost tips (Render)

| Plan | Notes |
|------|--------|
| Free Redis | Works for light use; may sleep |
| Starter web + worker | Recommended minimum for real jobs |
| ASR | Groq Whisper Turbo ≈ **$0.04 / audio hour** |
| LLM | GPT-4o ≈ **$0.14–0.18** per 2-hour lecture |

Free web instances sleep after inactivity — first request after sleep is slow. Upgrade to paid for always-on.

---

## Switch ASR to Deepgram

In both services’ Environment:

```
ASR_PROVIDER=deepgram
DEEPGRAM_API_KEY=...
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Worker not processing | Check worker logs; confirm `CELERY_BROKER_URL` matches Redis |
| Groq 413 / payload too large | Pipeline auto-compresses audio; long videos may need Deepgram |
| LLM errors | Verify `OPENAI_API_KEY` is set on **worker** as well as API |
| Out of disk | Ephemeral `/tmp` is cleaned per job; avoid huge concurrent uploads on free tier |

---

## Local test of production image

```bash
docker build -t vtk .
docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  -e GROQ_API_KEY=gsk-... \
  -e ASR_PROVIDER=groq \
  -e REDIS_URL=redis://host.docker.internal:6379/0 \
  vtk
```
