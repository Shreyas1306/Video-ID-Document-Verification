"""
Preprocessing Module
====================
Video → Frame Extraction → Document Detection → Cropping → Perspective Correction
→ Normalized Document Frames

Submodules:
    - frame_extractor:       Video decoding and key-frame sampling
    - document_detector:     Document candidate detection (OpenCV contour & YOLO)
    - document_cropper:      Bounding-box / contour-based cropping
    - perspective_corrector: Homography-based four-point perspective transform
"""

from src.preprocessing.candidate_scorer import CandidateScorer
from src.preprocessing.document_cropper import DocumentCropper
from src.preprocessing.document_detector import (
    BaseDocumentDetector,
    OpenCVContourDetector,
    YOLODocumentDetector,
    get_document_detector,
    run_detection_pipeline,
)
from src.preprocessing.frame_extractor import FrameExtractor
from src.preprocessing.perspective_corrector import (
    PerspectiveCorrector,
    estimate_document_corners,
    order_points,
    run_perspective_pipeline,
)
from src.preprocessing.temporal_document_tracker import TemporalDocumentTracker

__all__ = [
    "FrameExtractor",
    "BaseDocumentDetector",
    "OpenCVContourDetector",
    "YOLODocumentDetector",
    "get_document_detector",
    "run_detection_pipeline",
    "CandidateScorer",
    "TemporalDocumentTracker",
    "DocumentCropper",
    "PerspectiveCorrector",
    "estimate_document_corners",
    "order_points",
    "run_perspective_pipeline",
]

