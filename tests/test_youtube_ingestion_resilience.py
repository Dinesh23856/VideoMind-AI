"""Focused unit & integration tests for YouTube ingestion resilience and HTTP 429 handling.

Validates:
1. Successful subtitle retrieval (REAL classification, DOWNLOADED status)
2. HTTP 429 rate-limit detection and classification (RATE_LIMITED, NOT_AVAILABLE)
3. Bounded exponential retry behavior (terminates cleanly, no infinite loop)
4. Non-retryable errors fail-fast without redundant retries
5. Pipeline graceful fallback from failed subtitles to audio stream & ASR
6. Strict SSRF validation on incoming URLs
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yt_dlp.utils

from app.ingestion.youtube import (
    harvest_youtube_captions,
    is_rate_limited_error,
    parse_vtt_or_srt_to_segments,
)
from app.core.errors import SSRFSecurityError


class TestRateLimitDetection:
    def test_is_rate_limited_error_true(self):
        err_429 = yt_dlp.utils.DownloadError("ERROR: Unable to download video subtitles for 'en': HTTP Error 429: Too Many Requests")
        assert is_rate_limited_error(err_429) is True

        err_too_many = Exception("HTTP 429 Too Many Requests")
        assert is_rate_limited_error(err_too_many) is True

        err_rate_limit = Exception("rate-limit exceeded on timedtext")
        assert is_rate_limited_error(err_rate_limit) is True

    def test_is_rate_limited_error_false(self):
        err_private = yt_dlp.utils.DownloadError("ERROR: Private video. Sign in if you've been granted access.")
        assert is_rate_limited_error(err_private) is False

        err_not_found = yt_dlp.utils.DownloadError("ERROR: Video unavailable")
        assert is_rate_limited_error(err_not_found) is False


class TestHarvestYouTubeCaptions:
    @patch("app.ingestion.youtube.validate_external_url")
    @patch("yt_dlp.YoutubeDL")
    def test_successful_subtitle_retrieval(self, mock_ydl_cls, mock_validate, tmp_path):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        video_id = "dQw4w9WgXcQ"
        sub_file = tmp_path / f"{video_id}.en.vtt"
        sub_file.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:04.000\nHello world", encoding="utf-8")

        mock_ydl = MagicMock()
        mock_ydl.extract_info.return_value = {
            "id": video_id,
            "title": "Never Gonna Give You Up",
            "subtitles": {"en": [{"ext": "vtt", "url": "https://example.com/en.vtt"}]},
            "automatic_captions": {},
        }
        mock_ydl_cls.return_value.__enter__.return_value = mock_ydl

        sub_path, info = harvest_youtube_captions(url, str(tmp_path), max_retries=1, backoff_base_sec=0.0)

        assert sub_path == str(sub_file)
        assert info is not None
        assert info["subtitles_status"] == "DOWNLOADED"
        assert info["subtitles_classification"] == "REAL"
        assert info["title"] == "Never Gonna Give You Up"

    @patch("app.ingestion.youtube.validate_external_url")
    @patch("yt_dlp.YoutubeDL")
    def test_http_429_classification_and_bounded_retry(self, mock_ydl_cls, mock_validate, tmp_path):
        url = "https://www.youtube.com/watch?v=RATE_LIMIT_TEST"
        video_id = "RATE_LIMIT_TEST"

        mock_ydl = MagicMock()
        mock_ydl.extract_info.return_value = {
            "id": video_id,
            "title": "Throttled Video",
            "subtitles": {"en": [{"ext": "vtt", "url": "https://example.com/timedtext"}]},
            "automatic_captions": {},
        }
        mock_ydl.download.side_effect = yt_dlp.utils.DownloadError(
            "ERROR: Unable to download video subtitles for 'en': HTTP Error 429: Too Many Requests"
        )
        mock_ydl_cls.return_value.__enter__.return_value = mock_ydl

        # Request 2 retries -> 3 attempts total (initial + 2 retries)
        sub_path, info = harvest_youtube_captions(
            url, str(tmp_path), max_retries=2, backoff_base_sec=0.0
        )

        assert sub_path is None
        assert info is not None
        assert info["subtitles_status"] == "RATE_LIMITED"
        assert info["subtitles_classification"] == "NOT_AVAILABLE"
        assert "429" in info["subtitles_error"]
        # Exactly 3 download attempts made before terminating
        assert mock_ydl.download.call_count == 3

    @patch("app.ingestion.youtube.validate_external_url")
    @patch("yt_dlp.YoutubeDL")
    def test_no_infinite_retry_loop(self, mock_ydl_cls, mock_validate, tmp_path):
        url = "https://www.youtube.com/watch?v=LOOP_TEST"
        mock_ydl = MagicMock()
        mock_ydl.extract_info.side_effect = yt_dlp.utils.DownloadError("HTTP Error 429: Too Many Requests")
        mock_ydl_cls.return_value.__enter__.return_value = mock_ydl

        max_retries = 3
        sub_path, info = harvest_youtube_captions(
            url, str(tmp_path), max_retries=max_retries, backoff_base_sec=0.0
        )

        assert sub_path is None
        assert info["subtitles_status"] == "RATE_LIMITED"
        assert info["subtitles_classification"] == "NOT_AVAILABLE"
        # Initial attempt (1) + retries (3) = 4 calls, never infinite
        assert mock_ydl.extract_info.call_count == max_retries + 1

    @patch("app.ingestion.youtube.validate_external_url")
    @patch("yt_dlp.YoutubeDL")
    def test_non_retryable_error_fails_fast(self, mock_ydl_cls, mock_validate, tmp_path):
        url = "https://www.youtube.com/watch?v=PRIVATE_TEST"
        mock_ydl = MagicMock()
        mock_ydl.extract_info.side_effect = yt_dlp.utils.DownloadError("ERROR: Private video")
        mock_ydl_cls.return_value.__enter__.return_value = mock_ydl

        sub_path, info = harvest_youtube_captions(
            url, str(tmp_path), max_retries=3, backoff_base_sec=0.0
        )

        assert sub_path is None
        assert info is None
        # Fails immediately on first attempt without retrying
        assert mock_ydl.extract_info.call_count == 1

    @patch("app.ingestion.youtube.validate_external_url")
    @patch("yt_dlp.YoutubeDL")
    def test_no_subtitles_available_on_video(self, mock_ydl_cls, mock_validate, tmp_path):
        url = "https://www.youtube.com/watch?v=NO_SUBS"
        mock_ydl = MagicMock()
        mock_ydl.extract_info.return_value = {
            "id": "NO_SUBS",
            "title": "Music Video Without Captions",
            "subtitles": {},
            "automatic_captions": {},
        }
        mock_ydl_cls.return_value.__enter__.return_value = mock_ydl

        sub_path, info = harvest_youtube_captions(url, str(tmp_path), max_retries=1, backoff_base_sec=0.0)

        assert sub_path is None
        assert info is not None
        assert info["subtitles_status"] == "NOT_FOUND"
        assert info["subtitles_classification"] == "NOT_AVAILABLE"
        # Download should not even be called if no English tracks exist
        assert mock_ydl.download.call_count == 0

    def test_ssrf_rejection(self, tmp_path):
        with pytest.raises(SSRFSecurityError):
            harvest_youtube_captions("http://127.0.0.1:8000/internal", str(tmp_path))

        with pytest.raises(SSRFSecurityError):
            harvest_youtube_captions("http://169.254.169.254/latest/meta-data", str(tmp_path))


class TestVttParsing:
    def test_parse_vtt_to_segments(self, tmp_path):
        vtt_content = """WEBVTT
Kind: captions
Language: en

00:00:01.500 --> 00:00:04.200
In this lecture, we introduce neural networks.

00:00:04.500 --> 00:00:08.000
<c>Specifically</c>, backpropagation computes gradients.
"""
        vtt_path = tmp_path / "test.vtt"
        vtt_path.write_text(vtt_content, encoding="utf-8")

        segments = parse_vtt_or_srt_to_segments(str(vtt_path))
        assert len(segments) == 2
        assert segments[0]["start"] == 1.5
        assert segments[0]["end"] == 4.2
        assert "introduce neural networks" in segments[0]["text"]
        assert segments[1]["start"] == 4.5
        assert segments[1]["end"] == 8.0
        assert "backpropagation computes gradients" in segments[1]["text"]


class TestPipelineFallbackOn429:
    @patch("app.orchestration.tasks.temporary_media_workspace")
    @patch("app.orchestration.tasks.harvest_youtube_captions")
    @patch("app.orchestration.tasks.get_transcriber")
    @patch("app.orchestration.tasks.demux_audio_to_wav")
    @patch("yt_dlp.YoutubeDL")
    def test_pipeline_falls_back_to_audio_on_subtitle_429(
        self, mock_ydl_cls, mock_demux, mock_get_transcriber, mock_harvest, mock_workspace, tmp_path
    ):
        """
        When harvest_youtube_captions returns (None, info) because subtitles are rate-limited,
        process_video_job MUST gracefully fall back to downloading audio and running ASR.
        """
        from contextlib import contextmanager
        from app.orchestration.tasks import process_video_job

        @contextmanager
        def fake_workspace():
            yield str(tmp_path)

        mock_workspace.side_effect = fake_workspace

        # Simulate 429 rate-limited subtitle failure
        mock_harvest.return_value = (
            None,
            {
                "id": "vid429",
                "title": "Machine Learning Lecture",
                "subtitles_status": "RATE_LIMITED",
                "subtitles_classification": "NOT_AVAILABLE",
            },
        )

        # Mock audio download in tasks.py
        mock_ydl = MagicMock()
        mock_ydl.extract_info.return_value = {
            "id": "vid429",
            "title": "Machine Learning Lecture",
        }
        mock_ydl_cls.return_value.__enter__.return_value = mock_ydl

        # Create a mock WAV file in scratch when yt-dlp runs
        def fake_extract_info(source, download=True):
            # Write a dummy wav file so candidates search succeeds
            wav = tmp_path / "vid429.wav"
            wav.write_bytes(b"RIFFdummywav")
            return {"id": "vid429", "title": "Machine Learning Lecture"}

        mock_ydl.extract_info.side_effect = fake_extract_info
        mock_demux.side_effect = lambda src, dst: src

        # Mock transcriber
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe_lecture.return_value = [
            {"start": 0.0, "end": 10.0, "text": "Fallback audio transcription of lecture.", "words": []}
        ]
        mock_get_transcriber.return_value = mock_transcriber

        # Execute process_video_job directly
        job_result = process_video_job(
            job_id="test_fallback_job_429",
            source="https://www.youtube.com/watch?v=vid429",
            source_type="url",
            title="Machine Learning Lecture",
            output_dir=str(tmp_path / "artifacts"),
        )

        assert job_result["status"] == "COMPLETED"
        assert job_result["title"] == "Machine Learning Lecture"
        # Verified: Transcriber was called as fallback
        mock_transcriber.transcribe_lecture.assert_called_once()
