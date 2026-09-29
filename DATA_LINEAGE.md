# VideoMind-AI: Data Lineage & Provenance Specification

## 1. Overview
In enterprise educational and research contexts, black-box AI responses are unacceptable. VideoMind-AI implements complete, fine-grained **Data Lineage Tracking** (`app/lineage/tracker.py`). Every generated concept, summary, quiz question, or RAG citation maintains an immutable provenance trail tracing back to raw media frames, audio bytes, and model transformation parameters.

---

## 2. Lineage Graph Architecture

```
  [Raw Video/Audio File] (MEDIA_FILE)
            │
            ▼ (ffmpeg_extract)
    [Audio Stream WAV] (AUDIO_STREAM)
            │
            ▼ (whisper_transcription)
  [Transcript Segment] (TRANSCRIPT_SEGMENT) ──┐
                                             ├──► [Concept Node] (KNOWLEDGE_CONCEPT)
    [Visual Keyframe] (EXTRACTED_FRAME)     ──┘           │
            │                                             ▼ (rag_synthesis)
            ▼ (tesseract_ocr)                  [Synthesized Response] (SYNTHESIZED_RESPONSE)
      [OCR Text Span] (OCR_RESULT)
```

### 2.1 Node Hierarchy
- `MEDIA_FILE`: Raw video/audio files with SHA-256 binary hash.
- `AUDIO_STREAM`: Extracted 16kHz mono audio channels.
- `TRANSCRIPT_SEGMENT`: Spoken phrases with speaker diarization tags.
- `EXTRACTED_FRAME`: Visual keyframe imagery with timestamps.
- `OCR_RESULT`: Text extracted from whiteboard/slide regions.
- `KNOWLEDGE_CONCEPT`: Extracted structured epistemic entities.
- `SYNTHESIZED_RESPONSE`: Final AI-generated answers or learning artifacts.

### 2.2 Transform Edges
Edges record:
- `parent_id` $\to$ `child_id`
- `transform_name` (e.g. `whisper_transcription`, `dense_vector_embedding`, `llm_concept_extraction`)
- `model_version`
- `parameters` (e.g. temperature, frame sample rate, embedding dimension)
- `timestamp`

---

## 3. Bidirectional Provenance Queries

### 3.1 Upstream Root Cause Tracing (`trace_upstream`)
Given any generated answer, quiz, or concept, `trace_upstream` traverses the lineage graph in reverse topological order, returning all ancestral input files, timestamps, and model transformations that contributed to the output.

### 3.2 Downstream Impact Analysis (`trace_downstream`)
When a video is edited, re-transcribed, or re-captioned, `trace_downstream` computes the exact transitive closure of all downstream concepts, embeddings, search indexes, and user study decks that must be invalidated and re-indexed.

---

## 4. Compliance & Auditability
Data lineage records are fully serializable and exportable alongside GDPR Right-to-Explanation requests, verifying that responses were generated strictly from verified lecture data without unauthorized external training ingestion.
