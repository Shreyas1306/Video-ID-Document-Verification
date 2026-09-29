"""
Unit Tests for HybridDocumentDetector Component
===============================================
Validates:
  1. Carrier paper triggers learned UNet fallback.
  2. Pristine grayscale card stays on OpenCV fast path (Pristine Card Exemption).
  3. Nested / container candidate triggers learned fallback.
  4. Aspect ratio failure triggers learned fallback.
  5. Normal valid card stays on OpenCV fast path.
  6. Learned fallback produces valid 4-corner quadrilateral.
  7. Missing learned prediction is handled gracefully without crashing.
  8. Output schema remains compatible with TemporalDocumentTracker & PerspectiveCorrector.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock
import numpy as np
import pytest

from src.preprocessing.candidate_scorer import CandidateScorer
from src.preprocessing.document_detector import OpenCVContourDetector
from src.preprocessing.hybrid_document_detector import (
    HybridDocumentDetector,
    MIDV500Segmenter,
    VariantCAmbiguityGate,
)
from src.preprocessing.temporal_document_tracker import TemporalDocumentTracker


@pytest.fixture
def scorer():
    return CandidateScorer()


@pytest.fixture
def gate():
    return VariantCAmbiguityGate()


def test_carrier_paper_triggers_learned_fallback(scorer, gate):
    """Verify that a carrier paper candidate (area=0.38, aspect=1.42) triggers UNet fallback."""
    h, w = 1000, 1000  # Frame area = 1,000,000
    corners = np.array([[100, 100], [800, 100], [800, 600], [100, 600]], dtype=np.float32)
    carrier_cand = {
        "corners": corners,
        "area": 380000.0,  # 38% of frame
        "aspect_ratio": 1.42,
        "static_score": 0.77,
        "confidence": 0.77,
        "edge_quality": 0.70,
        "encloses_count": 0,
    }
    opencv_res = {
        "detected": True,
        "corners": corners.tolist(),
        "confidence": 0.77,
        "static_score": 0.77,
        "all_detections": [carrier_cand],
    }

    decision = gate.evaluate(opencv_res, (h, w), scorer=scorer)
    assert decision["invoke_unet"] is True
    assert "carrier_area_exceeded" in decision["reason"]
    assert decision["signals"]["area_ratio"] == 0.38


def test_pristine_grayscale_card_stays_on_opencv_fast_path(scorer, gate):
    """
    Verify that a pristine card with static_score=0.64 (due to grayscale edge contrast)
    is exempted and stays on the OpenCV fast path.
    """
    h, w = 1000, 1600  # Frame area = 1,600,000
    # Pristine card: area = 152,000 (9.5%), aspect = 1.62, rectangularity = 0.95
    corners = np.array([[200, 300], [700, 300], [700, 609], [200, 609]], dtype=np.float32)
    pristine_cand = {
        "corners": corners,
        "area": 151500.0,
        "aspect_ratio": 1.62,
        "static_score": 0.64,  # Lower edge contrast in grayscale
        "confidence": 0.64,
        "edge_quality": 0.50,
        "encloses_count": 0,
    }
    opencv_res = {
        "detected": True,
        "corners": corners.tolist(),
        "confidence": 0.64,
        "static_score": 0.64,
        "all_detections": [pristine_cand],
    }

    decision = gate.evaluate(opencv_res, (h, w), scorer=scorer)
    assert decision["invoke_unet"] is False
    assert decision["reason"] == "confident_opencv_candidate"
    assert decision["signals"]["is_pristine"] is True


def test_nested_container_candidate_triggers_fallback(scorer, gate):
    """Verify that an enclosing container bezel triggers UNet fallback."""
    h, w = 1000, 1000
    corners = np.array([[50, 50], [900, 50], [900, 600], [50, 600]], dtype=np.float32)
    container_cand = {
        "corners": corners,
        "area": 250000.0,
        "aspect_ratio": 1.58,
        "static_score": 0.85,
        "confidence": 0.85,
        "encloses_count": 2,  # Encloses inner quads
    }
    opencv_res = {
        "detected": True,
        "corners": corners.tolist(),
        "confidence": 0.85,
        "static_score": 0.85,
        "all_detections": [container_cand],
    }

    decision = gate.evaluate(opencv_res, (h, w), scorer=scorer)
    assert decision["invoke_unet"] is True
    assert decision["reason"] == "encloses_inner_quad_container"


def test_aspect_ratio_failure_triggers_fallback(scorer, gate):
    """Verify that a candidate with severe aspect ratio deviation triggers UNet fallback."""
    h, w = 1000, 1000
    corners = np.array([[100, 100], [900, 100], [900, 300], [100, 300]], dtype=np.float32)
    distorted_cand = {
        "corners": corners,
        "area": 160000.0,
        "aspect_ratio": 2.45,  # Far from ID-1 (1.586) or ID-3 (1.420)
        "static_score": 0.72,
        "confidence": 0.72,
        "encloses_count": 0,
    }
    opencv_res = {
        "detected": True,
        "corners": corners.tolist(),
        "confidence": 0.72,
        "static_score": 0.72,
        "all_detections": [distorted_cand],
    }

    decision = gate.evaluate(opencv_res, (h, w), scorer=scorer)
    assert decision["invoke_unet"] is True
    assert "aspect_ratio_deviation" in decision["reason"]


def test_normal_valid_card_stays_on_opencv(scorer, gate):
    """Verify that a normal high-confidence ID-1 card stays on the OpenCV fast path."""
    h, w = 1000, 1600
    corners = np.array([[200, 300], [700, 300], [700, 615], [200, 615]], dtype=np.float32)
    card_cand = {
        "corners": corners,
        "area": 150000.0,
        "aspect_ratio": 1.586,
        "static_score": 0.78,
        "confidence": 0.78,
        "encloses_count": 0,
    }
    opencv_res = {
        "detected": True,
        "corners": corners.tolist(),
        "confidence": 0.78,
        "static_score": 0.78,
        "all_detections": [card_cand],
    }

    decision = gate.evaluate(opencv_res, (h, w), scorer=scorer)
    assert decision["invoke_unet"] is False
    assert decision["reason"] == "confident_opencv_candidate"


def test_learned_fallback_produces_valid_four_corners():
    """Verify that when fallback is triggered, HybridDocumentDetector returns valid 4 corners."""
    # Mock OpenCV detector returning carrier paper
    mock_cv = MagicMock()
    mock_cv.detect_candidates.return_value = [{
        "corners": np.array([[50, 50], [900, 50], [900, 700], [50, 700]], dtype=np.float32),
        "area": 400000.0,
        "aspect_ratio": 1.42,
        "static_score": 0.76,
        "confidence": 0.76,
    }]
    mock_cv.detect_frame.return_value = {
        "detected": True,
        "corners": [[50, 50], [900, 50], [900, 700], [50, 700]],
        "confidence": 0.76,
        "static_score": 0.76,
        "all_detections": mock_cv.detect_candidates.return_value,
    }

    # Mock Segmenter returning true card corners inside the sheet
    mock_seg = MagicMock()
    card_quad = np.array([[200, 250], [600, 250], [600, 500], [200, 500]], dtype=np.float32)
    mock_seg.process_frame.return_value = {
        "detected": True,
        "corners": card_quad,
        "confidence": 0.94,
        "bbox": [200, 250, 600, 500],
        "method": "approx_poly_dp",
    }

    detector = HybridDocumentDetector(
        opencv_detector=mock_cv,
        segmenter=mock_seg,
    )

    dummy_frame = np.full((1000, 1000, 3), 200, dtype=np.uint8)
    res = detector.detect_frame(dummy_frame, frame_id="frame_001")

    assert res["detected"] is True
    assert res["localization_method"] == "midv500_unet"
    assert "carrier_area_exceeded" in res["gate_reason"]
    assert len(res["corners"]) == 4
    assert res["confidence"] == 0.94
    mock_seg.process_frame.assert_called_once()


def test_missing_learned_prediction_handled_gracefully():
    """Verify that when UNet produces no detection or throws an exception, negative result is returned."""
    mock_cv = MagicMock()
    mock_cv.detect_candidates.return_value = []
    mock_cv.detect_frame.return_value = {
        "detected": False,
        "corners": None,
        "confidence": 0.0,
        "all_detections": [],
    }

    mock_seg = MagicMock()
    # Segmenter also fails
    mock_seg.process_frame.return_value = {
        "detected": False,
        "corners": None,
        "confidence": 0.0,
        "bbox": None,
    }

    detector = HybridDocumentDetector(
        opencv_detector=mock_cv,
        segmenter=mock_seg,
    )

    dummy_frame = np.full((500, 500, 3), 128, dtype=np.uint8)
    res = detector.detect_frame(dummy_frame, frame_id="frame_empty")

    assert res["detected"] is False
    assert res["corners"] is None
    assert res["localization_method"] == "midv500_unet"
    assert res["reason"] == "learned_segmentation_failed"


def test_output_schema_compatibility():
    """Verify that HybridDocumentDetector produces an output schema matching TemporalDocumentTracker."""
    mock_cv = MagicMock()
    card_corners = np.array([[100, 100], [500, 100], [500, 350], [100, 350]], dtype=np.float32)
    mock_cv.detect_candidates.return_value = [{
        "corners": card_corners,
        "area": 100000.0,
        "aspect_ratio": 1.60,
        "static_score": 0.78,
        "confidence": 0.78,
    }]
    mock_cv.detect_frame.return_value = {
        "detected": True,
        "bbox": [100, 100, 500, 350],
        "corners": card_corners.tolist(),
        "confidence": 0.78,
        "static_score": 0.78,
        "all_detections": mock_cv.detect_candidates.return_value,
    }

    detector = HybridDocumentDetector(opencv_detector=mock_cv)
    dummy_frame = np.full((1000, 1000, 3), 128, dtype=np.uint8)
    res = detector.detect_frame(dummy_frame, frame_id="schema_check")

    required_keys = {
        "frame_id",
        "detected",
        "bbox",
        "corners",
        "confidence",
        "localization_method",
        "gate_reason",
        "candidate_count",
        "static_score",
        "temporal_score",
        "signals",
        "reason",
        "all_detections",
    }
    assert required_keys.issubset(set(res.keys()))
    assert isinstance(res["corners"], list)
    assert len(res["corners"]) == 4
    for pt in res["corners"]:
        assert len(pt) == 2
