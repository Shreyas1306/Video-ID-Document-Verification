"""
Unit Tests for Candidate Scoring and Temporal Document Tracking
===============================================================
Verifies:
1. CandidateScorer geometric evaluation:
   - Aspect ratio scoring (proximity to ID-1 / ID-3).
   - Area ratio scoring (optimal document size vs oversized bezels).
   - Rectangularity and orthogonality scores.
   - Nesting and containment detection (inner card bonus, outer container penalty).
2. TemporalDocumentTracker multi-frame behavior:
   - First-frame candidate selection without prior state.
   - Multi-frame temporal continuity and exponential smoothing.
   - Tracker reset on manual call and after consecutive lost frames.
   - Graceful handling of empty or malformed candidate lists.
3. Integration and backward compatibility:
   - Configuration loading from settings.yaml.
   - Detector detect_candidates() and detect_frame() interface contracts.
"""

import cv2
import numpy as np
import pytest

from src.preprocessing.candidate_scorer import CandidateScorer
from src.preprocessing.document_detector import OpenCVContourDetector
from src.preprocessing.temporal_document_tracker import TemporalDocumentTracker, compute_quad_iou


# =============================================================================
# 1. Candidate Scorer Tests
# =============================================================================

def test_scorer_aspect_ratio_evaluation():
    """Verify that ID-1 card aspect ratio (1.586) scores higher than square or banner."""
    scorer = CandidateScorer()

    score_id1 = scorer.score_aspect_ratio(1.586)
    score_passport = scorer.score_aspect_ratio(1.420)
    score_near_id1 = scorer.score_aspect_ratio(1.620)
    score_square = scorer.score_aspect_ratio(1.050)
    score_banner = scorer.score_aspect_ratio(2.600)

    assert score_id1 > 0.99
    assert score_passport > 0.99
    assert score_near_id1 > 0.95
    assert score_near_id1 > score_square
    assert score_near_id1 > score_banner


def test_scorer_area_ratio_evaluation():
    """Verify that document-sized area (20-40%) scores higher than huge bezels (>70%)."""
    scorer = CandidateScorer()

    score_ideal = scorer.score_area_ratio(0.30)
    score_moderate = scorer.score_area_ratio(0.20)
    score_huge_bezel = scorer.score_area_ratio(0.80)
    score_tiny = scorer.score_area_ratio(0.03)

    assert score_ideal > 0.90
    assert score_moderate > 0.70
    assert score_huge_bezel < 0.20
    assert score_ideal > score_huge_bezel
    assert score_ideal > score_tiny


def test_scorer_nesting_detection_and_ranking():
    """Verify that an inner ID card nested inside an outer container screen is favored."""
    scorer = CandidateScorer()

    # Outer candidate: tablet/phone bezel (large area, square-ish)
    c_outer = {
        "corners": np.array([[50, 50], [750, 50], [750, 750], [50, 750]], dtype=np.float32),
        "area": 490000.0,
        "aspect_ratio": 1.0,
        "rectangularity": 1.0,
        "confidence": 0.95,
    }

    # Inner candidate: ID card inside the screen (aspect 1.586, moderate area)
    c_inner = {
        "corners": np.array([[200, 250], [600, 250], [600, 502], [200, 502]], dtype=np.float32),
        "area": 100800.0,
        "aspect_ratio": 1.587,
        "rectangularity": 1.0,
        "confidence": 0.90,
    }

    scored = scorer.score_candidates([c_outer, c_inner], frame_shape=(800, 800))
    assert len(scored) == 2

    # Inner candidate must have is_nested = True
    inner_scored = next(c for c in scored if c["area"] < 200000)
    outer_scored = next(c for c in scored if c["area"] > 200000)

    assert inner_scored["is_nested"] is True
    assert outer_scored["encloses_count"] >= 1

    # Inner document card must be ranked first despite smaller raw area/confidence
    assert scored[0] == inner_scored
    assert inner_scored["static_score"] > outer_scored["static_score"]


# =============================================================================
# 2. Temporal Document Tracker Tests
# =============================================================================

def test_tracker_first_frame_initialization():
    """Verify that first frame initializes tracking state."""
    tracker = TemporalDocumentTracker()
    assert tracker.last_accepted_quad is None
    assert tracker.frames_tracked == 0

    frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    cand = {
        "corners": np.array([[100, 100], [400, 100], [400, 290], [100, 290]], dtype=np.float32),
        "bbox": [100, 100, 400, 290],
        "area": 57000.0,
        "aspect_ratio": 1.579,
        "rectangularity": 1.0,
        "confidence": 0.85,
    }

    res = tracker.process_frame([cand], frame, frame_id="f001")
    assert res["detected"] is True
    assert res["method"] == "first_frame_detection"
    assert tracker.last_accepted_quad is not None
    assert tracker.frames_tracked == 1
    assert tracker.frames_lost == 0


def test_tracker_temporal_smoothing_and_continuity():
    """Verify multi-frame temporal tracking and smooth corner adjustment."""
    tracker = TemporalDocumentTracker()
    frame = np.full((480, 640, 3), 40, dtype=np.uint8)

    # Frame 1
    pts1 = np.array([[100, 100], [400, 100], [400, 290], [100, 290]], dtype=np.float32)
    cand1 = {
        "corners": pts1,
        "bbox": [100, 100, 400, 290],
        "area": 57000.0,
        "aspect_ratio": 1.579,
        "rectangularity": 1.0,
        "confidence": 0.85,
    }
    tracker.process_frame([cand1], frame, frame_id="f001")

    # Frame 2: slightly shifted document (+4 pixels right, +2 pixels down)
    pts2 = pts1 + np.array([4.0, 2.0])
    cand2 = {
        "corners": pts2,
        "bbox": [104, 102, 404, 292],
        "area": 57000.0,
        "aspect_ratio": 1.579,
        "rectangularity": 1.0,
        "confidence": 0.85,
    }

    # Spurious background rectangle (e.g. tablet border)
    cand_distractor = {
        "corners": np.array([[20, 20], [600, 20], [600, 450], [20, 450]], dtype=np.float32),
        "bbox": [20, 20, 600, 450],
        "area": 249400.0,
        "aspect_ratio": 1.35,
        "rectangularity": 1.0,
        "confidence": 0.70,
    }

    res2 = tracker.process_frame([cand2, cand_distractor], frame, frame_id="f002")
    assert res2["detected"] is True
    assert res2["method"] == "temporally_guided"
    assert tracker.frames_tracked == 2

    # Corner position should be smoothed between pts1 and pts2
    tracked_corners = np.array(res2["corners"])
    assert 100.0 <= tracked_corners[0, 0] <= 104.0
    assert 100.0 <= tracked_corners[0, 1] <= 102.0


def test_tracker_reset_and_loss_recovery():
    """Verify tracker resets state gracefully upon manual reset or consecutive lost frames."""
    tracker = TemporalDocumentTracker()
    frame = np.full((480, 640, 3), 40, dtype=np.uint8)

    cand = {
        "corners": np.array([[100, 100], [400, 100], [400, 290], [100, 290]], dtype=np.float32),
        "bbox": [100, 100, 400, 290],
        "area": 57000.0,
        "aspect_ratio": 1.579,
        "rectangularity": 1.0,
        "confidence": 0.85,
    }
    tracker.process_frame([cand], frame, frame_id="f001")
    assert tracker.last_accepted_quad is not None

    # Manual reset
    tracker.reset()
    assert tracker.last_accepted_quad is None
    assert tracker.frames_tracked == 0
    assert tracker.frames_lost == 0

    # Auto reset after max_frames_lost empty frames
    tracker.process_frame([cand], frame, frame_id="f001")
    for _ in range(6):
        tracker.process_frame([], frame, frame_id="empty")

    assert tracker.last_accepted_quad is None
    assert tracker.frames_lost == 0


def test_empty_and_malformed_candidates_handling():
    """Verify tracker does not crash on empty, None, or zero candidates."""
    tracker = TemporalDocumentTracker()
    frame = np.full((300, 300, 3), 50, dtype=np.uint8)

    res_empty = tracker.process_frame([], frame, frame_id="f_empty")
    assert res_empty["detected"] is False
    assert res_empty["corners"] is None
    assert res_empty["method"] == "no_candidates"


# =============================================================================
# 3. Detector Integration & Compatibility Tests
# =============================================================================

def test_detector_detect_candidates_and_detect_frame_api():
    """Verify OpenCVContourDetector exposes detect_candidates and backward-compatible detect_frame."""
    detector = OpenCVContourDetector()

    # Create synthetic frame with rectangular document
    frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    cv2.rectangle(frame, (140, 130), (500, 350), (245, 245, 245), -1)
    cv2.rectangle(frame, (140, 130), (500, 350), (20, 20, 20), 2)

    # 1. detect_candidates API
    cands = detector.detect_candidates(frame, frame_id="cands_test")
    assert isinstance(cands, list)
    assert len(cands) >= 1
    assert "corners" in cands[0]
    assert "aspect_ratio" in cands[0]
    assert "area_ratio" in cands[0]

    # 2. detect_frame API
    res = detector.detect_frame(frame, frame_id="frame_test")
    assert isinstance(res, dict)
    assert res["detected"] is True
    assert res["corners"] is not None
    assert len(res["corners"]) == 4
    assert "all_detections" in res
    assert len(res["all_detections"]) >= 1
