"""
Unit and Integration Tests for Public Dataset Preprocessing Validation
======================================================================
Tests:
1. DLC-2021 VIA annotation quadrilateral parsing.
2. Quadrilateral IoU calculation (identical, disjoint, and partial overlap).
3. Corner error calculation (Euclidean pixel distance under canonical ordering).
4. Document-level metadata preservation in generated manifests.
5. Normalized output generation, dimensions (400x600x3), and attack-type preservation.
"""

import csv
import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from scripts.validate_public_preprocessing import (
    compute_corner_errors,
    compute_quadrilateral_iou,
    parse_via_ground_truth_quad,
)


@pytest.fixture
def sample_ann_path() -> Path:
    return Path("data/public/dlc2021/annotations/alb_id_00.or0001.json")


def test_dlc_annotation_parsing(sample_ann_path: Path):
    """Test that official VIA JSON ground truth is accurately parsed."""
    if not sample_ann_path.exists():
        pytest.skip("DLC-2021 annotations not present on disk")

    pts = parse_via_ground_truth_quad(sample_ann_path, "000001.jpg")
    assert pts is not None, "Failed to parse ground truth quad for 000001.jpg"
    assert pts.shape == (4, 2), f"Expected shape (4, 2), got {pts.shape}"
    assert np.all(np.isfinite(pts)), "Coordinates contain non-finite numbers"

    # Non-existent frame should return None gracefully
    missing_pts = parse_via_ground_truth_quad(sample_ann_path, "non_existent_frame.jpg")
    assert missing_pts is None


def test_quadrilateral_iou_metrics():
    """Test polygon IoU calculation under controlled geometric configurations."""
    # Identical squares
    sq1 = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    sq2 = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    assert abs(compute_quadrilateral_iou(sq1, sq2) - 1.0) < 1e-4

    # Disjoint squares
    sq3 = np.array([[200, 200], [300, 200], [300, 300], [200, 300]], dtype=np.float32)
    assert compute_quadrilateral_iou(sq1, sq3) == 0.0

    # 50% horizontal shift: overlap is 50 x 100 = 5000, union is 15000 -> IoU = 1/3 ~ 0.3333
    sq4 = np.array([[50, 0], [150, 0], [150, 100], [50, 100]], dtype=np.float32)
    iou = compute_quadrilateral_iou(sq1, sq4)
    assert 0.30 <= iou <= 0.35, f"Expected ~0.333, got {iou}"


def test_corner_error_calculation():
    """Test corner error calculation with known pixel offsets."""
    pts_gt = np.array([[10, 10], [110, 10], [110, 70], [10, 70]], dtype=np.float32)
    # Shift each point by +3 on x, +4 on y -> Euclidean distance = 5.0 pixels
    pts_pred = pts_gt + np.array([3.0, 4.0])

    mean_err, med_err, max_err, errors = compute_corner_errors(pts_pred, pts_gt)
    assert abs(mean_err - 5.0) < 1e-4
    assert abs(med_err - 5.0) < 1e-4
    assert abs(max_err - 5.0) < 1e-4
    assert len(errors) == 4


def test_document_level_metadata_preservation():
    """Verify that validation frame metrics CSV preserves all required fields and attributes."""
    metrics_csv = Path("outputs/public_preprocessing_validation/frame_metrics.csv")
    if not metrics_csv.exists():
        pytest.skip("Validation metrics CSV not yet generated")

    required_fields = [
        "dataset", "document_id", "video_id", "attack_type", "frame_id", "label",
        "detected", "confidence", "iou", "mean_corner_error", "median_corner_error",
        "max_corner_error", "perspective_success", "raw_path", "detected_path", "normalized_path"
    ]

    with open(metrics_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == required_fields
        rows = list(reader)

    assert len(rows) == 80, f"Expected 80 evaluated frames, got {len(rows)}"

    attack_types_found = set(r["attack_type"] for r in rows)
    assert attack_types_found == {"or", "cc", "cg", "re"}, f"Missing attack modes: {attack_types_found}"

    for r in rows:
        assert r["dataset"] == "DLC-2021"
        assert r["document_id"] in {"alb_id_00", "alb_id_01"}
        if r["attack_type"] == "or":
            assert r["label"] == "REAL"
        else:
            assert r["label"] == "ATTACKED"


def test_normalized_output_generation_and_attack_type_preservation():
    """Verify that normalized frames exist, have valid dimensions, and maintain label segregation."""
    norm_base = Path("data/public/normalized_dlc2021")
    manifest_file = norm_base / "normalized_manifest.csv"
    if not manifest_file.exists():
        pytest.skip("Normalized dataset manifest not yet created")

    real_frames = list((norm_base / "real").glob("*.png"))
    attacked_frames = list((norm_base / "attacked").glob("*.png"))

    assert len(real_frames) > 0, "No real normalized frames found"
    assert len(attacked_frames) > 0, "No attacked normalized frames found"

    # Sample a normalized frame and verify shape
    sample_img = cv2.imread(str(real_frames[0]))
    assert sample_img is not None, "Failed to read normalized frame"
    assert sample_img.shape == (400, 600, 3), f"Expected (400, 600, 3), got {sample_img.shape}"

    # Verify manifest linkage
    with open(manifest_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        manifest_rows = list(reader)

    assert len(manifest_rows) == len(real_frames) + len(attacked_frames)
    for r in manifest_rows:
        assert r["attack_type"] in {"or", "cc", "cg", "re"}
        assert r["label"] in {"REAL", "ATTACKED"}
        norm_path = Path(r["normalized_path"])
        assert norm_path.exists(), f"Normalized frame path missing on disk: {norm_path}"
