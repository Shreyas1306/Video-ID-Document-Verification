"""
Unit and Integration Tests for Refined Hybrid Ambiguity Gate Experiment
========================================================================
Tests:
- RefinedAmbiguityGate initialization and variant definitions
- Carrier paper detection via area threshold calibration (area > 0.35)
- Carrier paper detection via compound signature (area > 0.30 and aspect < 1.50)
- Pristine card exemption preventing false triggers on clean grayscale cards
- Total detection failure triggers UNet unconditionally
- Zero ground-truth leakage verification
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pytest

from scripts.experiments.evaluate_hybrid_gate_refinement import (
    RefinedAmbiguityGate,
    get_refinement_gate_variants,
)
from src.preprocessing.candidate_scorer import CandidateScorer


def test_refinement_gate_variants_initialization():
    """Verify that all refinement gate variants initialize with valid properties."""
    variants = get_refinement_gate_variants()
    assert len(variants) == 4
    names = [v.name for v in variants]
    assert "baseline_balanced" in names
    assert "variant_a_area_035" in names
    assert "variant_b_compound_sig" in names
    assert "variant_c_dual_path" in names

    for v in variants:
        assert isinstance(v.min_static_score, float)
        assert isinstance(v.max_area_ratio, float)
        assert len(v.description) > 0


def test_carrier_sheet_caught_by_variant_a_area_cutoff():
    """Verify that a carrier sheet contour (area=0.38, aspect=1.42) triggers Variant A."""
    scorer = CandidateScorer()
    h, w = 1000, 1000  # Frame area = 1,000,000
    # Candidate area = 380,000 (area_ratio = 0.38)
    corners = np.array([[100, 100], [800, 100], [800, 600], [100, 600]], dtype=np.float32)
    carrier_cand = {
        "corners": corners,
        "area": 380000.0,
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

    variants = {v.name: v for v in get_refinement_gate_variants()}
    gate_a = variants["variant_a_area_035"]
    decision = gate_a.evaluate(opencv_res, (h, w), scorer=scorer)

    assert decision["invoke_unet"] is True
    assert "carrier_area_exceeded" in decision["reason"]
    assert decision["signals"]["area_ratio"] == 0.38


def test_carrier_sheet_caught_by_variant_b_compound_signature():
    """Verify that a carrier sheet with area=0.36 and aspect=1.42 triggers compound signature."""
    scorer = CandidateScorer()
    h, w = 1000, 1000
    corners = np.array([[100, 100], [750, 100], [750, 560], [100, 560]], dtype=np.float32)
    carrier_cand = {
        "corners": corners,
        "area": 360000.0,
        "aspect_ratio": 1.42,
        "static_score": 0.76,
        "confidence": 0.76,
        "encloses_count": 0,
    }
    opencv_res = {
        "detected": True,
        "corners": corners.tolist(),
        "confidence": 0.76,
        "static_score": 0.76,
        "all_detections": [carrier_cand],
    }

    variants = {v.name: v for v in get_refinement_gate_variants()}
    gate_b = variants["variant_b_compound_sig"]
    decision = gate_b.evaluate(opencv_res, (h, w), scorer=scorer)

    assert decision["invoke_unet"] is True
    assert "compound_carrier_signature" in decision["reason"]


def test_pristine_card_exemption_prevents_false_trigger():
    """
    Verify that in Variant C, a pristine card with static_score=0.64 (due to grayscale edge contrast)
    is exempted and passes through OpenCV without triggering UNet.
    """
    scorer = CandidateScorer()
    h, w = 1000, 1600
    # Pristine card: area = 9.5% (152,000 px), aspect = 1.65, rectangularity = 0.95
    corners = np.array([[200, 300], [700, 300], [700, 603], [200, 603]], dtype=np.float32)
    pristine_cand = {
        "corners": corners,
        "area": 151500.0,
        "aspect_ratio": 1.65,
        "static_score": 0.64,  # Lower edge contrast
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

    variants = {v.name: v for v in get_refinement_gate_variants()}
    gate_c = variants["variant_c_dual_path"]
    decision = gate_c.evaluate(opencv_res, (h, w), scorer=scorer)

    assert decision["invoke_unet"] is False
    assert decision["reason"] == "confident_opencv_candidate"
    assert decision["signals"]["is_pristine"] is True


def test_empty_detection_triggers_unconditionally():
    """Verify that when OpenCV detects nothing, all refined variants trigger UNet."""
    empty_res = {"detected": False, "corners": None, "confidence": 0.0, "all_detections": []}
    for gate in get_refinement_gate_variants():
        decision = gate.evaluate(empty_res, (1080, 1920))
        assert decision["invoke_unet"] is True
        assert decision["reason"] == "opencv_not_detected"


def test_no_ground_truth_leakage_in_refined_gate():
    """Verify that RefinedAmbiguityGate.evaluate takes strictly zero ground truth arguments."""
    import inspect

    gate = RefinedAmbiguityGate(name="test")
    sig = inspect.signature(gate.evaluate)
    param_names = list(sig.parameters.keys())

    assert "pts_gt" not in param_names
    assert "ground_truth" not in param_names
    assert "gt" not in param_names
    assert "label" not in param_names
    assert set(param_names).issubset({"self", "opencv_res", "frame_shape", "scorer"})
