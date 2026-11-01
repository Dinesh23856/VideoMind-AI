"""
Temporal Scene & Shot Boundary Detection Engine.

Provides content-aware video scene and slide transition detection using:
- Perceptual hash Hamming delta
- Color and luminance histogram divergence
- Keyframe nomination and temporal interval partitioning

Truth Classification:
- Classification: REAL (deterministic mathematical video content transition analysis).
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from app.vision.perceptual import PerceptualHasher


@dataclass
class SceneBoundary:
    scene_id: str
    video_id: str
    start_time: float
    end_time: float
    start_ms: int
    end_ms: int
    duration_sec: float
    representative_frame_id: Optional[str] = None
    confidence: float = 0.90
    detection_method: str = "perceptual_delta"
    classification: str = "REAL"
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "video_id": self.video_id,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_sec": round(self.duration_sec, 3),
            "representative_frame_id": self.representative_frame_id,
            "confidence": round(self.confidence, 3),
            "detection_method": self.detection_method,
            "classification": self.classification,
            "provenance": self.provenance,
        }


class TemporalSceneDetector:
    """
    Lightweight, local-first temporal scene and slide boundary detector.
    Analyzes frame-to-frame perceptual differences and color transitions.
    """

    def __init__(
        self,
        transition_threshold: float = 0.25,
        min_scene_duration_sec: float = 2.0,
        hasher: Optional[PerceptualHasher] = None,
    ):
        self.transition_threshold = transition_threshold
        self.min_scene_duration_sec = min_scene_duration_sec
        self.hasher = hasher or PerceptualHasher()

    @staticmethod
    def get_classification() -> str:
        return "REAL"

    def compute_frame_difference(
        self,
        img1: Image.Image,
        img2: Image.Image,
    ) -> float:
        """
        Calculates a normalized visual difference score [0.0, 1.0] between two frames.
        Combines:
        1. dHash Hamming distance (structural/edge gradient shift)
        2. Color histogram distance (palette/background change)
        """
        # 1. Perceptual dHash delta
        h1 = self.hasher.compute_dhash(img1)
        h2 = self.hasher.compute_dhash(img2)
        dist = self.hasher.hamming_distance(h1, h2)
        dhash_diff = min(1.0, dist / 32.0)  # 32 bits = 50% bit flip

        # 2. Downsampled color histogram delta
        t1 = img1.resize((64, 48), Image.Resampling.BILINEAR)
        t2 = img2.resize((64, 48), Image.Resampling.BILINEAR)
        arr1 = np.array(t1, dtype=np.float32) / 255.0
        arr2 = np.array(t2, dtype=np.float32) / 255.0

        pixel_diff = float(np.mean(np.abs(arr1 - arr2)))

        # Weighted combination: 60% perceptual structure, 40% pixel/color delta
        combined = 0.6 * dhash_diff + 0.4 * min(1.0, pixel_diff * 3.0)
        return float(np.clip(combined, 0.0, 1.0))

    def detect_scenes(
        self,
        frames: List[Any],
        video_id: str = "video",
        total_duration: Optional[float] = None,
    ) -> List[SceneBoundary]:
        """
        Partitions an ordered sequence of timestamped frames into distinct scenes.
        Accepts tuples of format:
        - (timestamp_sec, Image, frame_id)
        - (timestamp_sec, file_path_or_id, Image)
        - (timestamp_sec, Image)
        Returns a list of SceneBoundary objects.
        """
        if not frames:
            return []

        # Normalize frames into (timestamp_sec, PIL_Image, frame_id)
        normalized_frames: List[Tuple[float, Image.Image, str]] = []
        for idx, item in enumerate(frames):
            ts = float(item[0])
            if len(item) == 3:
                if isinstance(item[1], Image.Image):
                    img = item[1]
                    fid = str(item[2])
                elif isinstance(item[2], Image.Image):
                    img = item[2]
                    fid = str(item[1])
                else:
                    img = self.hasher._load_image(item[1])
                    fid = str(item[2])
            elif len(item) == 2:
                img = item[1] if isinstance(item[1], Image.Image) else self.hasher._load_image(item[1])
                fid = f"frm_{idx}"
            else:
                continue
            normalized_frames.append((ts, img, fid))

        if not normalized_frames:
            return []

        # Sort by timestamp
        sorted_frames = sorted(normalized_frames, key=lambda x: x[0])
        scene_boundaries: List[SceneBoundary] = []

        curr_scene_start = sorted_frames[0][0]
        curr_scene_frames = [sorted_frames[0]]

        for i in range(1, len(sorted_frames)):
            prev_ts, prev_img, _ = sorted_frames[i - 1]
            curr_ts, curr_img, _ = sorted_frames[i]

            diff = self.compute_frame_difference(prev_img, curr_img)
            time_since_start = curr_ts - curr_scene_start

            # Check if visual transition occurs AND satisfies minimum scene duration
            if diff >= self.transition_threshold and time_since_start >= self.min_scene_duration_sec:
                # Flush completed scene
                scene_end = curr_ts
                sc_id = f"scene_{video_id}_{int(curr_scene_start * 1000)}"
                rep_frame_id = self._select_best_frame(curr_scene_frames)

                scene_boundaries.append(
                    SceneBoundary(
                        scene_id=sc_id,
                        video_id=video_id,
                        start_time=curr_scene_start,
                        end_time=scene_end,
                        start_ms=int(curr_scene_start * 1000),
                        end_ms=int(scene_end * 1000),
                        duration_sec=scene_end - curr_scene_start,
                        representative_frame_id=rep_frame_id,
                        confidence=0.92,
                        detection_method="perceptual_delta",
                        classification="REAL",
                        provenance={
                            "frames_count": len(curr_scene_frames),
                            "transition_delta": round(diff, 3),
                        },
                    )
                )

                curr_scene_start = curr_ts
                curr_scene_frames = [sorted_frames[i]]
            else:
                curr_scene_frames.append(sorted_frames[i])

        # Final scene
        if curr_scene_frames:
            last_ts = sorted_frames[-1][0]
            scene_end = total_duration if total_duration and total_duration > last_ts else (last_ts + 2.0)
            sc_id = f"scene_{video_id}_{int(curr_scene_start * 1000)}"
            rep_frame_id = self._select_best_frame(curr_scene_frames)

            scene_boundaries.append(
                SceneBoundary(
                    scene_id=sc_id,
                    video_id=video_id,
                    start_time=curr_scene_start,
                    end_time=scene_end,
                    start_ms=int(curr_scene_start * 1000),
                    end_ms=int(scene_end * 1000),
                    duration_sec=scene_end - curr_scene_start,
                    representative_frame_id=rep_frame_id,
                    confidence=0.92,
                    detection_method="perceptual_delta",
                    classification="REAL",
                    provenance={"frames_count": len(curr_scene_frames)},
                )
            )

        return scene_boundaries

    def _select_best_frame(self, frames: List[Tuple[float, Image.Image, str]]) -> str:
        """
        Picks the representative frame for a scene.
        Prefers the frame with highest sharpness/edge energy (not blurry or mid-transition).
        """
        if not frames:
            return ""
        if len(frames) == 1:
            return frames[0][2]

        # Score by Laplacian variance on downsampled grayscale
        best_score = -1.0
        best_id = frames[0][2]

        for _, img, f_id in frames:
            thumb = img.convert("L").resize((120, 90), Image.Resampling.BILINEAR)
            arr = np.array(thumb, dtype=np.float32)
            # Edge variance
            h_diff = np.abs(arr[1:, :] - arr[:-1, :])
            v_diff = np.abs(arr[:, 1:] - arr[:, :-1])
            sharpness = float(np.var(h_diff) + np.var(v_diff))
            if sharpness > best_score:
                best_score = sharpness
                best_id = f_id

        return best_id

    detect_scenes_from_frames = detect_scenes
