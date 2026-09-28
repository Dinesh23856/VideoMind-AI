"""Deterministic audio demuxing via ffmpeg + ephemeral workspace hygiene."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from typing import Generator, Optional


@contextmanager
def temporary_media_workspace(prefix: str = "media_proc_") -> Generator[str, None, None]:
    """
    Create an isolated POSIX scratch directory that is guaranteed to be
    recursively purged on exit or unhandled exception.
    """
    scratch_dir = tempfile.mkdtemp(prefix=prefix)
    try:
        yield scratch_dir
    finally:
        if os.path.exists(scratch_dir):
            shutil.rmtree(scratch_dir, ignore_errors=True)


def demux_audio_to_wav(
    input_path: str,
    output_path: Optional[str] = None,
    sample_rate: int = 16000,
) -> str:
    """
    Demux container (MP4/MKV/MOV/WEBM/...) to mono 16 kHz PCM 16-bit LE WAV.
    Bypasses video decoding entirely for speed and low memory use.
    If the file is already a WAV and ffmpeg is unavailable, it is used directly.
    """
    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = f"{base}_16k_mono.wav"

    ffmpeg_bin = shutil.which("ffmpeg")

    # If it is already a WAV and ffmpeg is not installed or output path is input path
    if input_path.lower().endswith(".wav"):
        if not ffmpeg_bin or os.path.abspath(input_path) == os.path.abspath(output_path):
            if os.path.abspath(input_path) != os.path.abspath(output_path):
                shutil.copyfile(input_path, output_path)
            return output_path

    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg executable was not found on system PATH. Required to decode media containers.")

    cmd = [
        ffmpeg_bin,
        "-v",
        "error",
        "-y",
        "-i",
        input_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        output_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg demux failed: {result.stderr.strip()}")
    except FileNotFoundError:
        if input_path.lower().endswith(".wav"):
            shutil.copyfile(input_path, output_path)
            return output_path
        raise RuntimeError("ffmpeg executable not found.")

    if not os.path.exists(output_path):
        raise RuntimeError("ffmpeg reported success but output WAV is missing")
    return output_path

