"""
Unit and Integration Tests for Document Cropping & Perspective Correction (Phase 3)
===================================================================================
Tests:
- Point ordering logic (order_points)
- Successful 4-corner perspective correction and homography warping
- Failed corner detection handling (with fallback resize enabled and disabled)
- Verification of configurable normalized output dimensions
- Side-by-side comparison debug frame generation
"""

from pathlib import Path
from typing import Tuple
import cv2
import numpy as np
import pytest

from src.preprocessing.perspective_corrector import (
    PerspectiveCorrector,
    estimate_document_corners,
    order_points,
)


class TestPointOrdering:
    """Test standard 4-point ordering function."""

    def test_order_points_permuted(self):
        # Coordinates: TL=(10, 10), TR=(100, 15), BR=(95, 80), BL=(12, 75)
        raw_pts = np.array([[95, 80], [10, 10], [12, 75], [100, 15]], dtype=np.float32)
        ordered = order_points(raw_pts)

        # Expected: TL, TR, BR, BL
        np.testing.assert_allclose(ordered[0], [10, 10])
        np.testing.assert_allclose(ordered[1], [100, 15])
        np.testing.assert_allclose(ordered[2], [95, 80])
        np.testing.assert_allclose(ordered[3], [12, 75])


class TestPerspectiveCorrectionSuccess:
    """Test successful perspective correction with homography."""

    @pytest.fixture
    def synthetic_tilted_card_image(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create a tilted rectangular card on a dark surface with known corners."""
        h, w = 300, 400
        canvas = np.full((h, w, 3), 40, dtype=np.uint8)

        # High-contrast bright quad simulating an ID card
        # Ordered: TL=(50, 40), TR=(350, 60), BR=(330, 260), BL=(70, 240)
        src_corners = np.array([[50, 40], [350, 60], [330, 260], [70, 240]], dtype=np.int32)
        cv2.fillPoly(canvas, [src_corners], (245, 245, 248))
        cv2.polylines(canvas, [src_corners], isClosed=True, color=(10, 10, 10), thickness=2)

        # Draw content inside card
        cv2.putText(canvas, "TEST ID CARD", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)

        return canvas, src_corners.astype(np.float32)

    def test_correction_with_explicit_corners(self, synthetic_tilted_card_image):
        img, corners = synthetic_tilted_card_image
        corrector = PerspectiveCorrector(target_width=500, target_height=300)

        result = corrector.correct_perspective(img, corners=corners)

        assert result["success"] is True
        assert result["method_applied"] == "homography"
        assert result["error"] is None
        assert result["normalized_image"] is not None
        assert result["normalized_image"].shape == (300, 500, 3)

    def test_automatic_corner_estimation_and_warp(self, synthetic_tilted_card_image):
        img, expected_corners = synthetic_tilted_card_image
        corrector = PerspectiveCorrector(target_width=600, target_height=400)

        result = corrector.correct_perspective(img)

        # Corner estimator should detect the high-contrast card polygon
        assert result["success"] is True
        assert result["method_applied"] == "homography"
        assert result["normalized_image"].shape == (400, 600, 3)
        assert len(result["corners"]) == 4


class TestPerspectiveCorrectionFailure:
    """Test failure detection and fallback handling when corners are not found."""

    def test_failed_corner_detection_with_fallback(self):
        """Flat uniform image has no distinct corners; should use fallback resize."""
        flat_img = np.full((250, 350, 3), 128, dtype=np.uint8)
        corrector = PerspectiveCorrector(
            target_width=600,
            target_height=400,
            fallback_to_crop=True,
        )

        result = corrector.correct_perspective(flat_img)

        # Must report success as False and method as fallback_resize
        assert result["success"] is False
        assert result["method_applied"] == "fallback_resize"
        assert result["normalized_image"] is not None
        assert result["normalized_image"].shape == (400, 600, 3)
        assert "Corner detection failed" in result["error"]

    def test_failed_corner_detection_without_fallback(self):
        """When fallback is disabled, failed corner detection should return None."""
        flat_img = np.full((250, 350, 3), 128, dtype=np.uint8)
        corrector = PerspectiveCorrector(
            target_width=600,
            target_height=400,
            fallback_to_crop=False,
        )

        result = corrector.correct_perspective(flat_img)

        assert result["success"] is False
        assert result["method_applied"] == "none"
        assert result["normalized_image"] is None
        assert "Corner detection failed" in result["error"]

    def test_invalid_input_handling(self):
        corrector = PerspectiveCorrector()
        res_none = corrector.correct_perspective(None)
        assert res_none["success"] is False
        assert res_none["normalized_image"] is None

        res_empty = corrector.correct_perspective(np.zeros((0, 0, 3), dtype=np.uint8))
        assert res_empty["success"] is False


class TestDebugComparisonGeneration:
    """Test side-by-side debug image creation."""

    def test_create_comparison_debug(self):
        corrector = PerspectiveCorrector(target_width=600, target_height=400)
        orig = np.full((200, 300, 3), 100, dtype=np.uint8)
        norm = np.full((400, 600, 3), 200, dtype=np.uint8)
        corners = [[20.0, 20.0], [280.0, 30.0], [270.0, 180.0], [30.0, 170.0]]

        debug_img = corrector.create_comparison_debug(orig, norm, corners=corners, status_text="HOMOGRAPHY")

        assert isinstance(debug_img, np.ndarray)
        # Height should equal target_height (400) + header (40) = 440
        assert debug_img.shape[0] == 440
        assert debug_img.shape[1] > 600
