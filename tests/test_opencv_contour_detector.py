"""
Unit and Integration Tests for OpenCV Contour Document Detector (Phase 2 Baseline)
===================================================================================
Tests:
  1. Synthetic rectangular document on plain background
  2. Tilted synthetic document
  3. Document with moderate perspective distortion
  4. Image containing no document
  5. Multiple rectangular objects
  6. Very small candidate
  7. Invalid/empty image inputs
  8. Factory method selection (OpenCV vs YOLO)
  9. End-to-end perspective correction integration
"""

import cv2
import numpy as np
import pytest

from src.preprocessing.document_detector import (
    BaseDocumentDetector,
    OpenCVContourDetector,
    YOLODocumentDetector,
    get_document_detector,
)
from src.preprocessing.perspective_corrector import PerspectiveCorrector, order_points


@pytest.fixture
def detector() -> OpenCVContourDetector:
    return OpenCVContourDetector(
        confidence_threshold=0.35,
        min_area_ratio=0.05,
        max_area_ratio=0.95,
        min_aspect_ratio=1.0,
        max_aspect_ratio=2.8,
    )


# =============================================================================
# Test 1: Synthetic rectangular document on plain background
# =============================================================================
def test_synthetic_rectangular_document_on_plain_background(detector: OpenCVContourDetector):
    """Verify localization of an axis-aligned rectangular document on a contrasting background."""
    h, w = 480, 640
    frame = np.full((h, w, 3), 40, dtype=np.uint8)

    # Draw rectangular document (w=360, h=220 -> aspect=1.636, area=79,200 -> 25.8% of frame)
    x1, y1, x2, y2 = 140, 130, 500, 350
    cv2.rectangle(frame, (x1, y1), (x2, y2), (245, 245, 245), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 20, 20), 2)
    cv2.putText(frame, "IDENTITY CARD", (x1 + 40, y1 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)

    res = detector.detect_frame(frame, frame_id="rect_plain")

    assert res["detected"] is True
    assert res["method"] == "opencv_contour"
    assert res["confidence"] >= 0.70
    assert res["corners"] is not None
    assert len(res["corners"]) == 4

    corners = np.array(res["corners"])
    # Corners are ordered [TL, TR, BR, BL]
    # Check Top-Left
    assert pytest.approx(corners[0][0], abs=6) == x1
    assert pytest.approx(corners[0][1], abs=6) == y1
    # Check Top-Right
    assert pytest.approx(corners[1][0], abs=6) == x2
    assert pytest.approx(corners[1][1], abs=6) == y1
    # Check Bottom-Right
    assert pytest.approx(corners[2][0], abs=6) == x2
    assert pytest.approx(corners[2][1], abs=6) == y2
    # Check Bottom-Left
    assert pytest.approx(corners[3][0], abs=6) == x1
    assert pytest.approx(corners[3][1], abs=6) == y2


# =============================================================================
# Test 2: Tilted synthetic document
# =============================================================================
def test_tilted_synthetic_document(detector: OpenCVContourDetector):
    """Verify localization of a tilted/rotated document."""
    h, w = 500, 600
    frame = np.full((h, w, 3), 35, dtype=np.uint8)

    # Tilted card corners (15 degrees rotation)
    card_w, card_h = 300, 180
    cx, cy = w // 2, h // 2
    angle = np.radians(15.0)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    base_pts = np.array([
        [-card_w / 2, -card_h / 2],
        [card_w / 2, -card_h / 2],
        [card_w / 2, card_h / 2],
        [-card_w / 2, card_h / 2],
    ], dtype=np.float32)

    rot_pts = np.zeros_like(base_pts)
    for i, (bx, by) in enumerate(base_pts):
        rot_pts[i] = [cx + bx * cos_a - by * sin_a, cy + bx * sin_a + by * cos_a]

    int_pts = np.round(rot_pts).astype(np.int32)
    cv2.fillPoly(frame, [int_pts], (240, 240, 245))
    cv2.polylines(frame, [int_pts], isClosed=True, color=(10, 10, 10), thickness=2)
    cv2.putText(frame, "TILTED DOC", (cx - 60, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2)

    res = detector.detect_frame(frame, frame_id="tilted_doc")

    assert res["detected"] is True
    assert res["corners"] is not None
    assert len(res["corners"]) == 4

    corners = np.array(res["corners"])
    # Verify canonical clockwise ordering
    # TL: smallest sum (x+y)
    # BR: largest sum (x+y)
    sums = corners.sum(axis=1)
    assert np.argmin(sums) == 0
    assert np.argmax(sums) == 2


# =============================================================================
# Test 3: Document with moderate perspective distortion
# =============================================================================
def test_document_with_moderate_perspective_distortion(detector: OpenCVContourDetector):
    """Verify localization of a document with perspective shear (trapezoidal view)."""
    h, w = 500, 600
    frame = np.full((h, w, 3), 50, dtype=np.uint8)

    # Trapezoidal perspective polygon simulating camera view at an angle
    quad_corners = np.array([
        [160, 120],  # TL (narrower top)
        [440, 140],  # TR
        [480, 360],  # BR (wider bottom)
        [110, 340],  # BL
    ], dtype=np.int32)

    cv2.fillPoly(frame, [quad_corners], (248, 248, 250))
    cv2.polylines(frame, [quad_corners], isClosed=True, color=(15, 15, 15), thickness=2)
    cv2.putText(frame, "PERSPECTIVE ID", (180, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)

    res = detector.detect_frame(frame, frame_id="perspective_doc")

    assert res["detected"] is True
    assert res["corners"] is not None
    assert len(res["corners"]) == 4

    # Connect directly to perspective correction and verify homography normalization
    corrector = PerspectiveCorrector(target_width=600, target_height=400)
    warp_res = corrector.correct_perspective(image=frame, corners=np.array(res["corners"]))

    assert warp_res["success"] is True
    assert warp_res["method_applied"] == "homography"
    assert warp_res["normalized_image"] is not None
    assert warp_res["normalized_image"].shape == (400, 600, 3)


# =============================================================================
# Test 4: Image containing no document
# =============================================================================
def test_image_containing_no_document(detector: OpenCVContourDetector):
    """Verify that an image without any document cleanly returns detected=False."""
    # Test flat uniform image
    flat_frame = np.full((400, 500, 3), 128, dtype=np.uint8)
    res_flat = detector.detect_frame(flat_frame, frame_id="no_doc_flat")

    assert res_flat["detected"] is False
    assert res_flat["bbox"] is None
    assert res_flat["corners"] is None
    assert res_flat["confidence"] == 0.0
    assert "Insufficient image contrast" in res_flat["reason"]

    # Test random noisy image with no geometric rectangles
    rng = np.random.RandomState(42)
    noise_frame = rng.randint(40, 80, (400, 500, 3), dtype=np.uint8)
    res_noise = detector.detect_frame(noise_frame, frame_id="no_doc_noise")

    assert res_noise["detected"] is False
    assert res_noise["corners"] is None
    assert res_noise["confidence"] == 0.0


# =============================================================================
# Test 5: Multiple rectangular objects
# =============================================================================
def test_multiple_rectangular_objects(detector: OpenCVContourDetector):
    """Verify that when multiple objects are present, the best candidate is correctly selected."""
    h, w = 600, 800
    frame = np.full((h, w, 3), 40, dtype=np.uint8)

    # Object 1: Small square icon (w=60, h=60, area=3600, 1.0 aspect)
    cv2.rectangle(frame, (50, 50), (110, 110), (220, 220, 220), -1)

    # Object 2: Prominent ID document card (w=400, h=250, area=100,000, aspect=1.60)
    card_x1, card_y1, card_x2, card_y2 = 250, 180, 650, 430
    cv2.rectangle(frame, (card_x1, card_y1), (card_x2, card_y2), (245, 245, 248), -1)
    cv2.rectangle(frame, (card_x1, card_y1), (card_x2, card_y2), (10, 10, 10), 2)
    cv2.putText(frame, "CARD CANDIDATE", (card_x1 + 30, card_y1 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 2)

    res = detector.detect_frame(frame, frame_id="multi_objects")

    assert res["detected"] is True
    assert res["corners"] is not None
    # Verify that the detected bounding box corresponds to the card, not the small icon
    bbox = res["bbox"]
    assert pytest.approx(bbox[0], abs=10) == card_x1
    assert pytest.approx(bbox[1], abs=10) == card_y1
    assert pytest.approx(bbox[2], abs=10) == card_x2
    assert pytest.approx(bbox[3], abs=10) == card_y2


# =============================================================================
# Test 6: Very small candidate rejected
# =============================================================================
def test_very_small_candidate_rejected(detector: OpenCVContourDetector):
    """Verify that tiny rectangular noise/stickers are rejected by area constraints."""
    h, w = 500, 500
    frame = np.full((h, w, 3), 40, dtype=np.uint8)

    # Tiny rectangle (w=25, h=20, area=500 -> 0.002 ratio, below min_area_ratio 0.05 and 1000px threshold)
    cv2.rectangle(frame, (200, 200), (225, 220), (240, 240, 240), -1)

    res = detector.detect_frame(frame, frame_id="tiny_candidate")

    assert res["detected"] is False
    assert res["corners"] is None
    assert res["bbox"] is None


# =============================================================================
# Test 7: Invalid and empty image inputs
# =============================================================================
def test_invalid_and_empty_images(detector: OpenCVContourDetector):
    """Verify error handling on malformed image inputs."""
    with pytest.raises(ValueError, match="Invalid frame input"):
        detector.detect_frame(None)

    with pytest.raises(ValueError, match="Invalid frame input"):
        detector.detect_frame("invalid_string")

    with pytest.raises(ValueError, match="Invalid image array dimensions"):
        detector.detect_frame(np.zeros((0, 0, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match="Invalid image array dimensions"):
        detector.detect_frame(np.zeros((100, 100), dtype=np.uint8))  # 2D grayscale


# =============================================================================
# Test 8: Factory method selection and polymorphism
# =============================================================================
def test_factory_method_selection():
    """Verify factory returns appropriate detector instance according to config."""
    det_contour = get_document_detector(method="opencv_contour")
    assert isinstance(det_contour, BaseDocumentDetector)
    assert isinstance(det_contour, OpenCVContourDetector)

    det_yolo = get_document_detector(method="yolo")
    assert isinstance(det_yolo, BaseDocumentDetector)
    assert isinstance(det_yolo, YOLODocumentDetector)

    with pytest.raises(ValueError, match="Unsupported document detection method"):
        get_document_detector(method="unknown_detector")


# =============================================================================
# Test 9: Full Perspective Correction Chain
# =============================================================================
def test_full_perspective_correction_chain(detector: OpenCVContourDetector):
    """
    Verify complete pipeline chain:
      input frame -> detected quadrilateral -> ordered corners -> homography -> normalized document
    """
    h, w = 480, 640
    frame = np.full((h, w, 3), 45, dtype=np.uint8)

    # Slightly tilted card
    pts = np.array([[120, 90], [510, 130], [480, 390], [90, 340]], dtype=np.int32)
    cv2.fillPoly(frame, [pts], (240, 240, 245))
    cv2.polylines(frame, [pts], isClosed=True, color=(10, 10, 10), thickness=2)
    cv2.putText(frame, "FULL CHAIN TEST", (160, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)

    # 1. Detection
    det = detector.detect_frame(frame)
    assert det["detected"] is True
    assert det["corners"] is not None

    # 2. Ordered Corners
    ordered = order_points(det["corners"])
    assert ordered.shape == (4, 2)

    # 3. Homography Warp
    corrector = PerspectiveCorrector(target_width=600, target_height=400)
    warp_result = corrector.correct_perspective(frame, corners=ordered)

    assert warp_result["success"] is True
    assert warp_result["method_applied"] == "homography"
    norm_doc = warp_result["normalized_image"]
    assert norm_doc is not None
    assert norm_doc.shape == (400, 600, 3)

    # Normalized document should have high brightness in the center (from card content)
    center_patch = norm_doc[100:300, 150:450]
    assert np.mean(center_patch) > 200.0
