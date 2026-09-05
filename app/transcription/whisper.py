"""High-throughput ASR with faster-whisper + Silero VAD + anti-hallucination guards."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class HighThroughputTranscriber:
    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        batch_size: int = 16,
    ):
        # Lazy import so Groq/Deepgram deployments do not require GPU wheels at import time
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=4,
            num_workers=1,
        )
        self.batched_pipeline = BatchedInferencePipeline(model=self.model)
        self.batch_size = batch_size

    def transcribe_lecture(
        self,
        audio_path: str,
        language: Optional[str] = "en",
    ) -> List[Dict[str, Any]]:
        """
        Batched transcription with Silero VAD segmentation and hallucination safeguards.
        Returns list of segments with start/end/text/words.
        """
        vad_parameters = {
            "threshold": 0.5,
            "min_speech_duration_ms": 250,
            "max_speech_duration_s": 30.0,
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 200,
        }

        segments, _info = self.batched_pipeline.transcribe(
            audio_path,
            batch_size=self.batch_size,
            vad_filter=True,
            vad_parameters=vad_parameters,
            # Critical: disable historical conditioning to stop cascade hallucinations
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            temperature=[0.0, 0.2, 0.4],
            word_timestamps=True,
            language=language,
        )

        structured: List[Dict[str, Any]] = []
        for segment in segments:
            structured.append(
                {
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": segment.text.strip(),
                    "words": [
                        {
                            "word": w.word,
                            "start": round(w.start, 2),
                            "end": round(w.end, 2),
                            "probability": round(w.probability, 2),
                        }
                        for w in (segment.words or [])
                    ],
                }
            )
        return structured
