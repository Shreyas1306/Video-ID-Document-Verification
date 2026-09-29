"""
Unit and Integration Tests for Public Dataset Normalization Generator
=====================================================================
Tests scripts/build_public_training_dataset.py:
1. Document split invariants (zero leakage, coverage of all 8 identities).
2. Presentation attack mode mapping to binary REAL vs ATTACKED classes.
3. Metadata extraction from clip folder names.
4. Output directory hierarchy structure.
5. Manifest and failure CSV schemas.
"""

from pathlib import Path
import tempfile
import pytest

from scripts.build_public_training_dataset import (
    DOCUMENT_SPLITS,
    MODE_TO_LABEL,
    compute_corner_error,
    compute_quadrilateral_iou,
    extract_clip_metadata,
)
import numpy as np


def test_document_splits_containment_and_zero_leakage():
    """Verify that all 8 document identities are partitioned with zero leakage."""
    expected_ids = {
        "alb_id_00", "alb_id_01", "alb_id_02", "alb_id_03",
        "alb_id_04", "alb_id_05", "aze_passport_00", "aze_passport_01"
    }
    assert set(DOCUMENT_SPLITS.keys()) == expected_ids

    train_ids = {k for k, v in DOCUMENT_SPLITS.items() if v == "train"}
    val_ids = {k for k, v in DOCUMENT_SPLITS.items() if v == "val"}
    test_ids = {k for k, v in DOCUMENT_SPLITS.items() if v == "test"}

    # Disjointness check
    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(val_ids.intersection(test_ids)) == 0

    assert len(train_ids) == 4
    assert len(val_ids) == 2
    assert len(test_ids) == 2


def test_mode_to_label_mapping():
    """Verify that genuine 'or' maps to REAL and presentation attacks map to ATTACKED."""
    assert MODE_TO_LABEL["or"] == "REAL"
    assert MODE_TO_LABEL["cc"] == "ATTACKED"
    assert MODE_TO_LABEL["cg"] == "ATTACKED"
    assert MODE_TO_LABEL["re"] == "ATTACKED"


def test_extract_clip_metadata():
    """Verify regex extraction of document ID, attack mode, and clip ID."""
    doc_id, mode, clip_id = extract_clip_metadata("alb_id_00.cc0001")
    assert doc_id == "alb_id_00"
    assert mode == "cc"
    assert clip_id == "alb_id_00.cc0001"

    doc_id, mode, clip_id = extract_clip_metadata("aze_passport_01.or0001")
    assert doc_id == "aze_passport_01"
    assert mode == "or"
    assert clip_id == "aze_passport_01.or0001"


def test_iou_and_corner_error_math():
    """Verify polygon IoU and corner error calculations on identical and offset quads."""
    pts1 = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    pts2 = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    assert compute_quadrilateral_iou(pts1, pts2) == pytest.approx(1.0, rel=1e-3)
    assert compute_corner_error(pts1, pts2) == pytest.approx(0.0, abs=1e-4)

    # Shifted by 10 px
    pts3 = np.array([[10, 0], [110, 0], [110, 100], [10, 100]], dtype=np.float32)
    iou = compute_quadrilateral_iou(pts1, pts3)
    assert 0.80 < iou < 0.85
    assert compute_corner_error(pts1, pts3) == pytest.approx(10.0, abs=1e-2)


def test_stratified_split_allocation_invariants():
    """Verify that the stratified allocation contains all 8 documents with zero leakage and both modalities."""
    from scripts.build_stratified_dataset import SPLIT_ALLOCATION

    train_ids = set(SPLIT_ALLOCATION["train"])
    val_ids = set(SPLIT_ALLOCATION["val"])
    test_ids = set(SPLIT_ALLOCATION["test"])

    # Disjointness check
    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(val_ids.intersection(test_ids)) == 0

    # Total 8 identities
    all_ids = train_ids | val_ids | test_ids
    assert len(all_ids) == 8

    # Both train and test contain both document modalities (ID and Passport)
    assert any(doc.startswith("alb_id") for doc in train_ids)
    assert any(doc.startswith("aze_passport") for doc in train_ids)
    assert any(doc.startswith("alb_id") for doc in test_ids)
    assert any(doc.startswith("aze_passport") for doc in test_ids)


def test_load_stratified_dataset_if_present():
    """Verify loading stratified dataset via load_partitioned_dataset if directory exists."""
    from src.visual.dataset import is_partitioned_dataset, load_partitioned_dataset

    stratified_dir = Path("outputs/normalized_public_dataset_stratified")
    if not stratified_dir.is_dir():
        pytest.skip("Stratified dataset directory not present on disk")

    assert is_partitioned_dataset(stratified_dir)
    train_s, val_s, test_s = load_partitioned_dataset(stratified_dir)

    assert len(train_s) == 767
    assert len(val_s) == 400
    assert len(test_s) == 360

    # Check that crops exist
    for s in train_s[:10] + val_s[:10] + test_s[:10]:
        assert Path(s["path"]).is_file()

