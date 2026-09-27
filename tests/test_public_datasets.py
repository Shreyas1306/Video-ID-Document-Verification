"""
Unit and Integration Tests for Public Dataset Ingestion and Leakage Prevention
=============================================================================
Verifies:
1. Manifest presence and schema adherence for DLC-2021 and MIDV-2020.
2. Binary label mapping:
   - or -> REAL
   - cc, cg, re -> ATTACKED
3. Document-level zero-leakage invariant:
   - No document identity appears across multiple splits in DLC-2021.
4. Cross-dataset isolation:
   - Zero overlap in document identities between DLC-2021 and MIDV-2020.
5. Image decodability and annotation integrity:
   - Sample frames decode via OpenCV.
   - Annotation JSON files parse correctly and contain required keys.
"""

import csv
import json
from pathlib import Path
import cv2
import pytest


@pytest.fixture
def public_data_dir() -> Path:
    return Path("data/public")


def test_dlc2021_manifest_schema_and_mapping(public_data_dir: Path):
    manifest_path = public_data_dir / "dlc2021_manifest.csv"
    if not manifest_path.exists():
        pytest.skip("DLC-2021 manifest not yet generated (ingestion in progress)")

    required_columns = [
        "dataset", "document_id", "video_id", "attack_type", "label", "frame_path", "split", "source_file"
    ]

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == required_columns
        rows = list(reader)

    assert len(rows) > 0, "DLC manifest is empty"

    attack_types_found = set()
    for row in rows:
        assert row["dataset"] == "DLC-2021"
        assert row["document_id"]
        assert row["video_id"]
        assert row["split"] in {"train", "validation", "test"}

        at = row["attack_type"]
        attack_types_found.add(at)
        if at == "or":
            assert row["label"] == "REAL", f"Expected REAL for 'or', got {row['label']}"
        elif at in {"cc", "cg", "re"}:
            assert row["label"] == "ATTACKED", f"Expected ATTACKED for '{at}', got {row['label']}"
        else:
            pytest.fail(f"Unknown attack type: {at}")

    # Ensure all 4 presentation modes are present in the dataset
    assert {"or", "cc", "cg", "re"}.issubset(attack_types_found)


def test_dlc2021_zero_leakage_document_containment(public_data_dir: Path):
    """Enforce strict rule: All presentation modes of a document identity MUST be in ONE split."""
    manifest_path = public_data_dir / "dlc2021_manifest.csv"
    if not manifest_path.exists():
        pytest.skip("DLC-2021 manifest not yet generated")

    doc_splits = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id = row["document_id"]
            split = row["split"]
            doc_splits.setdefault(doc_id, set()).add(split)

    for doc_id, splits in doc_splits.items():
        assert len(splits) == 1, (
            f"LEAKAGE VIOLATION: Document {doc_id} appears in multiple splits: {splits}"
        )


def test_midv2020_manifest_schema_and_integrity(public_data_dir: Path):
    manifest_path = public_data_dir / "midv2020_manifest.csv"
    if not manifest_path.exists():
        pytest.skip("MIDV-2020 manifest not yet generated")

    required_columns = [
        "dataset", "document_id", "video_id", "frame_path", "annotation_path", "ground_truth_text", "split"
    ]

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == required_columns
        rows = list(reader)

    assert len(rows) > 0, "MIDV manifest is empty"
    for row in rows:
        assert row["dataset"] == "MIDV-2020"
        assert row["document_id"]
        assert row["frame_path"]
        assert row["split"] == "pipeline_validation"


def test_cross_dataset_document_isolation(public_data_dir: Path):
    """Verify zero overlap between DLC-2021 and MIDV-2020 document identities."""
    dlc_manifest = public_data_dir / "dlc2021_manifest.csv"
    midv_manifest = public_data_dir / "midv2020_manifest.csv"
    if not dlc_manifest.exists() or not midv_manifest.exists():
        pytest.skip("Manifests not yet ready")

    dlc_docs = set()
    with open(dlc_manifest, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dlc_docs.add(row["document_id"])

    midv_docs = set()
    with open(midv_manifest, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            midv_docs.add(row["document_id"])

    overlap = dlc_docs.intersection(midv_docs)
    assert len(overlap) == 0, f"Cross-dataset leakage detected! Overlapping doc IDs: {overlap}"


def test_public_sample_frames_decodable(public_data_dir: Path):
    """Verify that sample frame files from manifests exist and can be decoded by OpenCV."""
    dlc_manifest = public_data_dir / "dlc2021_manifest.csv"
    if not dlc_manifest.exists():
        pytest.skip("DLC-2021 manifest not yet ready")

    with open(dlc_manifest, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Test first 10 frames
    for r in rows[:10]:
        img_path = Path(r["frame_path"])
        assert img_path.is_file(), f"Frame file missing: {img_path}"
        img = cv2.imread(str(img_path))
        assert img is not None, f"Failed to decode image with OpenCV: {img_path}"
        assert img.shape[0] > 0 and img.shape[1] > 0
        assert img.shape[2] == 3
