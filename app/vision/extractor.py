"""
Intelligent Video Keyframe Extraction, Temporal Scene Detection, and Visual Intelligence.

Replaces fixed-interval and byte-size comparisons with:
- TemporalSceneDetector: Content-aware scene cut and transition detection
- PerceptualHasher: Deterministic dHash/pHash deduplication with Hamming distance
- VisualContentClassifier: Feature- and token-based visual categorization (slide, code, formula, etc.)
- EnhancedOCREngine: Tesseract bounding boxes, normalization, and duplicate suppression
- PyAV / ffmpeg video decoding with robust fallback
- Strict capability truthfulness (REAL visual algorithms, HEURISTIC semantic inference, NOT_AVAILABLE OCR fallback)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from app.domain.models import VisualFrameContext
from app.vision.classifier import VisualClassificationResult, VisualContentClassifier
from app.vision.ocr import EnhancedOCREngine, OCRResult
from app.vision.perceptual import PerceptualHasher
from app.vision.scene_detector import SceneBoundary, TemporalSceneDetector

try:
    import av
    HAS_PYAV = True
except ImportError:
    HAS_PYAV = False


class VisionProvider(ABC):
    """Abstract interface for video keyframe extraction and visual understanding."""

    @abstractmethod
    def extract_keyframes(
        self,
        video_path: str,
        output_dir: str,
        max_frames: int = 50,
    ) -> List[VisualFrameContext]:
        """Extract salient visual frames and context from video."""
        pass


class DifferentialVisionExtractor(VisionProvider):
    """
    Production-grade visual intelligence & keyframe extraction engine.
    Performs perceptual hashing, temporal scene segmentation, content classification,
    and OCR extraction without relying on raw byte-size or file-size heuristics.
    """

    def __init__(
        self,
        sample_interval_seconds: float = 3.0,
        similarity_threshold: float = 0.15,
        scene_transition_threshold: float = 0.25,
        min_scene_duration_sec: float = 2.0,
        ocr_engine: Optional[EnhancedOCREngine] = None,
        hasher: Optional[PerceptualHasher] = None,
        classifier: Optional[VisualContentClassifier] = None,
        scene_detector: Optional[TemporalSceneDetector] = None,
    ):
        self.sample_interval = sample_interval_seconds
        self.similarity_threshold = similarity_threshold
        self.hasher = hasher or PerceptualHasher()
        self.ocr_engine = ocr_engine or EnhancedOCREngine()
        self.classifier = classifier or VisualContentClassifier()
        self.scene_detector = scene_detector or TemporalSceneDetector(
            transition_threshold=scene_transition_threshold,
            min_scene_duration_sec=min_scene_duration_sec,
            hasher=self.hasher,
        )

    def extract_keyframes(
        self,
        video_path: str,
        output_dir: str,
        max_frames: int = 50,
    ) -> List[VisualFrameContext]:
        """Extract salient visual frames from video."""
        _, keyframes = self.extract_scenes_and_keyframes(video_path, output_dir, max_frames=max_frames)
        return keyframes

    def extract_scenes_and_keyframes(
        self,
        video_path: str,
        output_dir: str,
        max_frames: int = 50,
    ) -> Tuple[List[SceneBoundary], List[VisualFrameContext]]:
        """
        Extracts both temporal scene boundaries and representative keyframes
        with complete perceptual and OCR metadata.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if not os.path.exists(video_path):
            return [], []

        # 1. Decode candidate frames across the video
        candidates = self._decode_candidate_frames(video_path, out_path)
        if not candidates:
            return [], []

        # 2. Detect Scene Boundaries
        video_id = Path(video_path).stem
        scenes = self.scene_detector.detect_scenes(candidates, video_id=video_id)

        # 3. Intelligent Keyframe Selection per Scene
        selected_candidates: List[Tuple[float, Path, Image.Image, str, str]] = []
        # Item: (timestamp, file_path, PIL Image, scene_id, selection_reason)

        for scene in scenes:
            # Find candidate frames within this scene
            scene_frames = [
                c for c in candidates
                if scene.start_time <= c[0] <= scene.end_time
            ]
            if not scene_frames:
                continue

            # Pick the sharpest frame as representative of this scene
            best_frame = scene_frames[0]
            max_sharpness = -1.0
            for ts, fpath, img in scene_frames:
                sharpness = self._calculate_sharpness(img)
                if sharpness > max_sharpness:
                    max_sharpness = sharpness
                    best_frame = (ts, fpath, img)

            selected_candidates.append(
                (best_frame[0], best_frame[1], best_frame[2], scene.scene_id, "scene_representative_sharpness")
            )

        # If scene count was lower than max_frames / 2 and we have more candidates,
        # add perceptually unique candidates that don't duplicate existing ones
        if len(selected_candidates) < max_frames:
            existing_hashes = [self.hasher.compute_dhash(c[2]) for c in selected_candidates]
            for ts, fpath, img in candidates:
                if len(selected_candidates) >= max_frames:
                    break
                curr_hash = self.hasher.compute_dhash(img)
                # Check Hamming distance against all currently selected
                is_duplicate = any(
                    self.hasher.hamming_distance(curr_hash, eh) <= 10
                    for eh in existing_hashes
                )
                if not is_duplicate:
                    # Find matching scene
                    m_scene = next((s for s in scenes if s.start_time <= ts <= s.end_time), None)
                    s_id = m_scene.scene_id if m_scene else f"scn_{uuid.uuid4().hex[:8]}"
                    selected_candidates.append(
                        (ts, fpath, img, s_id, "perceptually_distinct")
                    )
                    existing_hashes.append(curr_hash)

        # Enforce max_frames budget
        selected_candidates.sort(key=lambda x: x[0])
        if len(selected_candidates) > max_frames:
            # Subsample evenly across time to respect keyframe budget
            indices = np.linspace(0, len(selected_candidates) - 1, max_frames, dtype=int)
            selected_candidates = [selected_candidates[i] for i in indices]

        # 4. Process selected keyframes: Perceptual Hashing, OCR, Classification
        contexts: List[VisualFrameContext] = []
        last_ocr_text = ""

        for ts, fpath, img, scene_id, reason in selected_candidates:
            frame_id = f"frm_{uuid.uuid4().hex[:8]}"
            dhash_val = self.hasher.compute_dhash(img)

            # OCR Extraction
            ocr_res = self.ocr_engine.extract_ocr(img)
            # Consecutive duplicate OCR suppression
            if self.ocr_engine.is_duplicate_ocr(ocr_res.normalized_text, last_ocr_text):
                ocr_text = ""
                bboxes = []
            else:
                ocr_text = ocr_res.normalized_text
                bboxes = [b.to_dict() for b in ocr_res.bounding_boxes]
                if ocr_text:
                    last_ocr_text = ocr_text

            # Visual Content Classification
            cls_res = self.classifier.classify(img, ocr_text=ocr_text)

            # Link representative frame ID back to scene if matched
            for scn in scenes:
                if scn.scene_id == scene_id and not scn.representative_frame_id:
                    scn.representative_frame_id = frame_id

            contexts.append(
                VisualFrameContext(
                    frame_id=frame_id,
                    timestamp=round(ts, 2),
                    frame_type=cls_res.frame_type,
                    description=cls_res.description,
                    ocr_text=ocr_text,
                    image_path=str(fpath),
                    confidence=cls_res.confidence,
                    bounding_boxes=bboxes,
                    perceptual_hash=dhash_val,
                    scene_id=scene_id,
                    selection_reason=reason,
                    classification="HEURISTIC",
                )
            )

        return scenes, contexts

    def _decode_candidate_frames(
        self,
        video_path: str,
        out_path: Path,
    ) -> List[Tuple[float, Path, Image.Image]]:
        """
        Decodes video frames into (timestamp_sec, file_path, PIL_Image) tuples.
        Attempts PyAV first for in-process memory-efficient decoding;
        falls back to ffmpeg CLI if PyAV is not installed or encounters an error.
        """
        candidates: List[Tuple[float, Path, Image.Image]] = []

        if HAS_PYAV:
            try:
                candidates = self._decode_with_pyav(video_path, out_path)
                if candidates:
                    return candidates
            except Exception:
                candidates = []

        # Fallback to ffmpeg CLI
        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin:
            try:
                candidates = self._decode_with_ffmpeg(video_path, out_path, ffmpeg_bin)
            except Exception:
                candidates = []

        return candidates

    def _decode_with_pyav(
        self,
        video_path: str,
        out_path: Path,
    ) -> List[Tuple[float, Path, Image.Image]]:
        """Extracts candidate frames using PyAV."""
        candidates: List[Tuple[float, Path, Image.Image]] = []
        container = av.open(video_path)
        stream = container.streams.video[0]

        next_target_time = 0.0
        frame_idx = 0

        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            time_sec = float(frame.pts * stream.time_base)
            if time_sec >= next_target_time:
                img = frame.to_image().convert("RGB")
                fpath = out_path / f"frame_{frame_idx:04d}_{round(time_sec, 2)}s.jpg"
                img.save(fpath, format="JPEG", quality=85)
                candidates.append((round(time_sec, 2), fpath, img))
                frame_idx += 1
                next_target_time = time_sec + self.sample_interval

        container.close()
        return candidates

    def _decode_with_ffmpeg(
        self,
        video_path: str,
        out_path: Path,
        ffmpeg_bin: str,
    ) -> List[Tuple[float, Path, Image.Image]]:
        """Extracts candidate frames using ffmpeg CLI."""
        frame_pattern = str(out_path / "frame_%04d.jpg")
        cmd = [
            ffmpeg_bin,
            "-v", "error",
            "-y",
            "-i", video_path,
            "-vf", f"fps=1/{self.sample_interval},scale=960:-1",
            "-q:v", "3",
            frame_pattern,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if res.returncode != 0:
            return []

        frame_files = sorted(list(out_path.glob("frame_*.jpg")))
        candidates = []
        for idx, fpath in enumerate(frame_files):
            ts = round(idx * self.sample_interval, 2)
            try:
                img = Image.open(fpath).convert("RGB")
                candidates.append((ts, fpath, img))
            except Exception:
                continue

        return candidates

    def _calculate_sharpness(self, img: Image.Image) -> float:
        """Calculates image sharpness using variance of the Laplacian filter."""
        gray = img.convert("L").resize((160, 120), Image.Resampling.BILINEAR)
        arr = np.array(gray, dtype=np.float32)
        # Discrete Laplacian kernel
        lap = (
            np.roll(arr, 1, axis=0) + np.roll(arr, -1, axis=0) +
            np.roll(arr, 1, axis=1) + np.roll(arr, -1, axis=1) -
            4 * arr
        )
        return float(np.var(lap))


VideoKeyframeExtractor = DifferentialVisionExtractor


def extract_keyframes(video_path: str, output_dir: str) -> List[VisualFrameContext]:
    """Module-level convenience wrapper for keyframe extraction."""
    extractor = DifferentialVisionExtractor()
    return extractor.extract_keyframes(video_path, output_dir)
