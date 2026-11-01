"""
Multimodal visual intelligence: perceptual hashing, temporal scene detection,
OCR, visual classification, local VLM, and timeline alignment.
"""

from .classifier import VisualClassificationResult, VisualContentClassifier
from .events import MultimodalEventExtractor, MultimodalVideoEvent
from .extractor import (
    DifferentialVisionExtractor,
    VideoKeyframeExtractor,
    VisionProvider,
    extract_keyframes,
)
from .ocr import EnhancedOCREngine, OCRBoundingBox, OCRResult
from .perceptual import PerceptualHasher
from .scene_detector import SceneBoundary, TemporalSceneDetector
from .timeline import (
    align_visual_frames_to_transcript,
    build_multimodal_timeline,
    find_multimodal_context_at,
)
from .vlm import LocalVLMInterface, VLMResponse

__all__ = [
    # Perceptual Hashing
    "PerceptualHasher",
    # Scene Detection
    "SceneBoundary",
    "TemporalSceneDetector",
    # OCR
    "EnhancedOCREngine",
    "OCRBoundingBox",
    "OCRResult",
    # Classification
    "VisualContentClassifier",
    "VisualClassificationResult",
    # Local VLM
    "LocalVLMInterface",
    "VLMResponse",
    # Multimodal Events
    "MultimodalVideoEvent",
    "MultimodalEventExtractor",
    # Extraction
    "VisionProvider",
    "DifferentialVisionExtractor",
    "VideoKeyframeExtractor",
    "extract_keyframes",
    # Timeline
    "align_visual_frames_to_transcript",
    "build_multimodal_timeline",
    "find_multimodal_context_at",
]
