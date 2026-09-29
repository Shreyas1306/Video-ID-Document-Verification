"""
Unit and Integration Tests for Two-Tier Hybrid Localization Experiment
======================================================================
Tests:
- AmbiguityGate initialization and configuration properties
- Fallback decisions across Conservative, Balanced, and Aggressive gates
- Missing OpenCV detection (must trigger UNet unconditionally)
- Carrier sheet detection (excess area ratio triggers fallback)
- Aspect ratio distortion (aspect score deviation triggers fallback)
- Container nesting (carrier/bezel enclosing inner quad triggers fallback)
- Zero ground-truth leakage verification (gate must not accept or use ground truth)
- HybridDocumentLocalizer end-to-end routing (OpenCV confident vs UNet fallback)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pytest

from scripts.experiments.evaluate_hybrid_localization import (
    AmbiguityGate,
    HybridDocumentLocalizer,
    get_standard_gate_configurations,
)
from src.preprocessing.candidate_scorer import CandidateScorer


class MockSegmenter:
    """Mock UNet segmenter to verify routing without heavy inference."""

    def __init__(self, return_corners: Optional[np.ndarray] = None, detected: bool = True):
        self.call_count = 0
        self.detected = detected
        if return_corners is not None:
            self.return_corners = return_corners
        else:
            self.return_corners = np.array(
                [[100, 100], [400, 100], [400, 300], [100, 300]], dtype=np.float32
            )

    def process_frame(self, img: np.ndarray):
        self.call_count += 1
        return {
            "detected": self.detected,
            "corners": self.return_corners,
            "confidence": 0.985,
            "inference_time_ms": 1500.0,
            "mask": np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8),
        }


def test_standard_gate_configurations():
    """Verify that all standard gate configurations initialize with valid properties."""
    gates = get_standard_gate_configurations()
    assert len(gates) >= 4
    names = [g.name for g in gates]
    assert "failure_only" in names
    assert "conservative" in names
    assert "balanced" in names
    assert "aggressive" in names

    for g in gates:
        assert isinstance(g.min_static_score, float)
        assert isinstance(g.max_area_ratio, float)
        assert isinstance(g.min_aspect_score, float)
        assert len(g.description) > 0


def test_gate_missing_opencv_detection_triggers_unet():
    """When OpenCV detects nothing, all gates must unconditionally invoke UNet."""
    empty_opencv_res = {
        "detected": False,
        "corners": None,
        "confidence": 0.0,
        "static_score": 0.0,
        "all_detections": [],
    }
    frame_shape = (1080, 1920)

    for gate in get_standard_gate_configurations():
        decision = gate.evaluate(empty_opencv_res, frame_shape)
        assert decision["invoke_unet"] is True
        assert decision["reason"] == "opencv_not_detected"
        assert decision["signals"]["detected"] is False


def test_gate_confident_clean_card_bypasses_unet():
    """A clean, well-proportioned ID card candidate should pass through without triggering UNet."""
    scorer = CandidateScorer()
    # Synthetic clean ID-1 candidate: aspect = 85.6/54.0 = 1.585, area = 25% of frame
    h, w = 1000, 1600
    corners = np.array([[300, 300], [1100, 300], [1100, 800], [300, 800]], dtype=np.float32)
    clean_cand = {
        "corners": corners,
        "area": 400000.0,  # 25% of 1.6M
        "aspect_ratio": 1.60,
        "static_score": 0.85,
        "confidence": 0.85,
        "edge_quality": 0.80,
        "encloses_count": 0,
        "is_nested": False,
    }
    opencv_res = {
        "detected": True,
        "corners": corners.tolist(),
        "confidence": 0.85,
        "static_score": 0.85,
        "all_detections": [clean_cand],
    }

    gate = AmbiguityGate(name="balanced", min_static_score=0.70, max_area_ratio=0.48, min_aspect_score=0.75)
    decision = gate.evaluate(opencv_res, (h, w), scorer=scorer)

    assert decision["invoke_unet"] is False
    assert decision["reason"] == "confident_opencv_candidate"
    assert decision["signals"]["detected"] is True
    assert decision["signals"]["static_score"] == 0.85
    assert decision["signals"]["area_ratio"] == 0.25


def test_gate_carrier_sheet_area_exceeded_triggers_unet():
    """An oversized candidate (carrier sheet / desk > 48% area) must trigger UNet fallback."""
    scorer = CandidateScorer()
    h, w = 1000, 1000  # Frame area = 1,000,000
    # Candidate area = 650,000 (area_ratio = 0.65)
    corners = np.array([[50, 50], [950, 50], [950, 750], [50, 750]], dtype=np.float32)
    carrier_cand = {
        "corners": corners,
        "area": 630000.0,
        "aspect_ratio": 1.28,
        "static_score": 0.72,
        "confidence": 0.72,
        "edge_quality": 0.70,
        "encloses_count": 0,
    }
    opencv_res = {
        "detected": True,
        "corners": corners.tolist(),
        "confidence": 0.72,
        "static_score": 0.72,
        "all_detections": [carrier_cand],
    }

    gate = AmbiguityGate(name="balanced", min_static_score=0.70, max_area_ratio=0.48, min_aspect_score=0.75)
    decision = gate.evaluate(opencv_res, (h, w), scorer=scorer)

    assert decision["invoke_unet"] is True
    assert "area_ratio_carrier_sheet" in decision["reason"]
    assert decision["signals"]["area_ratio"] > 0.48


def test_gate_aspect_ratio_deviation_triggers_unet():
    """A candidate whose aspect ratio severely deviates from standard ID cards must trigger UNet."""
    scorer = CandidateScorer()
    h, w = 1000, 1600
    # Distorted elongated candidate: aspect ratio = 2.4 (aspect score < 0.30)
    corners = np.array([[100, 400], [1300, 400], [1300, 900], [100, 900]], dtype=np.float32)
    distorted_cand = {
        "corners": corners,
        "area": 250000.0,
        "aspect_ratio": 2.40,
        "static_score": 0.75,
        "confidence": 0.75,
        "edge_quality": 0.75,
        "encloses_count": 0,
    }
    opencv_res = {
        "detected": True,
        "corners": corners.tolist(),
        "confidence": 0.75,
        "static_score": 0.75,
        "all_detections": [distorted_cand],
    }

    gate = AmbiguityGate(name="balanced", min_static_score=0.70, max_area_ratio=0.48, min_aspect_score=0.75)
    decision = gate.evaluate(opencv_res, (h, w), scorer=scorer)

    assert decision["invoke_unet"] is True
    assert "aspect_ratio_deviation" in decision["reason"]


def test_gate_container_detection_triggers_unet():
    """A candidate marked as an outer container (enclosing an inner quad) must trigger UNet."""
    scorer = CandidateScorer()
    h, w = 1000, 1600
    corners = np.array([[200, 200], [1000, 200], [1000, 700], [200, 700]], dtype=np.float32)
    container_cand = {
        "corners": corners,
        "area": 400000.0,
        "aspect_ratio": 1.60,
        "static_score": 0.75,
        "confidence": 0.75,
        "encloses_count": 1,  # Outer container!
        "nesting_delta": -0.30,
    }
    opencv_res = {
        "detected": True,
        "corners": corners.tolist(),
        "confidence": 0.75,
        "static_score": 0.75,
        "all_detections": [container_cand],
    }

    gate = AmbiguityGate(name="balanced", check_container=True)
    decision = gate.evaluate(opencv_res, (h, w), scorer=scorer)

    assert decision["invoke_unet"] is True
    assert "container" in decision["reason"]
    assert decision["signals"]["is_container"] is True


def test_no_ground_truth_leakage_in_gate():
    """
    Formally verify that AmbiguityGate.evaluate only inspects inference-time features
    and does NOT accept, request, or reference ground truth coordinates or labels.
    """
    import inspect

    gate = AmbiguityGate(name="balanced")
    sig = inspect.signature(gate.evaluate)
    param_names = list(sig.parameters.keys())

    # Only opencv_res, frame_shape, and optional scorer are accepted
    assert "pts_gt" not in param_names
    assert "ground_truth" not in param_names
    assert "gt" not in param_names
    assert "label" not in param_names
    assert "ann" not in param_names
    assert set(param_names).issubset({"self", "opencv_res", "frame_shape", "scorer"})


def test_hybrid_localizer_bypasses_unet_on_confident_frame():
    """Verify that when the gate is confident, UNet is never called."""
    mock_seg = MockSegmenter()
    gate = AmbiguityGate(name="balanced", min_static_score=0.70)
    localizer = HybridDocumentLocalizer(gate=gate, segmenter=mock_seg)

    # Synthetic frame with a clean card
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    # Synthetic pre-computed confident result
    pre_confident = {
        "detected": True,
        "corners": [[100, 100], [500, 100], [500, 350], [100, 350]],
        "confidence": 0.88,
        "static_score": 0.88,
        "all_detections": [{
            "corners": np.array([[100, 100], [500, 100], [500, 350], [100, 350]], dtype=np.float32),
            "area": 100000.0,
            "aspect_ratio": 1.60,
            "static_score": 0.88,
            "edge_quality": 0.85,
            "encloses_count": 0,
        }],
    }

    # Custom tracker returning confident result
    class ConfidentTracker:
        def process_frame(self, *args, **kwargs):
            return pre_confident

    res = localizer.process_frame(img, frame_id="frame_01", tracker=ConfidentTracker())

    assert res["detected"] is True
    assert res["source"] == "opencv"
    assert res["invoked_unet"] is False
    assert mock_seg.call_count == 0  # UNet was bypassed!
    assert res["unet_time_ms"] == 0.0


def test_hybrid_localizer_invokes_unet_on_ambiguous_frame():
    """Verify that when the gate flags ambiguity, UNet is invoked and returns corners."""
    mock_seg = MockSegmenter()
    gate = AmbiguityGate(name="balanced", min_static_score=0.70)
    localizer = HybridDocumentLocalizer(gate=gate, segmenter=mock_seg)

    img = np.zeros((600, 800, 3), dtype=np.uint8)
    # Tracker returns detection failure
    class EmptyTracker:
        def process_frame(self, *args, **kwargs):
            return {"detected": False, "corners": None, "confidence": 0.0, "all_detections": []}

    res = localizer.process_frame(img, frame_id="frame_02", tracker=EmptyTracker())

    assert res["detected"] is True
    assert res["source"] == "unet"
    assert res["invoked_unet"] is True
    assert mock_seg.call_count == 1  # UNet was called!
    assert res["corners"] is not None
    assert res["unet_time_ms"] > 0.0
