"""
Tests for Visual Integrity Dataset, Validation, Grouping, and Leak-Free Splitting
"""

import os
from pathlib import Path
import tempfile
import pytest
from PIL import Image
import torch

from src.visual.dataset import (
    CLASS_TO_LABEL,
    DocumentDataset,
    collect_dataset_samples,
    create_leak_free_splits,
    extract_doc_id,
    get_document_transforms,
    validate_dataset_structure,
)


def test_extract_doc_id():
    """Verify document/video ID extraction from various filename conventions."""
    assert extract_doc_id("doc_001_f02.png") == "doc_001"
    assert extract_doc_id("doc_045_frame_12.jpg") == "doc_045"
    assert extract_doc_id("video_03_f01.png") == "video_03"
    assert extract_doc_id("card_99_frame.png") == "card_99"
    assert extract_doc_id(Path("REAL/doc_100/frame_01.png")) == "doc_100"


def test_validate_dataset_structure_missing_dir():
    """Verify that a non-existent directory reports invalid with clear issue message."""
    report = validate_dataset_structure("non_existent_path_xyz_123")
    assert not report.is_valid
    assert not report.real_dir_exists
    assert not report.attacked_dir_exists
    assert any("does not exist" in issue for issue in report.issues)


def test_validate_dataset_structure_incomplete(tmp_path):
    """Verify that a directory missing REAL or ATTACKED reports invalid."""
    # Create only REAL/
    real_dir = tmp_path / "REAL"
    real_dir.mkdir()
    for i in range(5):
        img = Image.new("RGB", (100, 100), color=(255, 255, 255))
        img.save(real_dir / f"doc_001_f{i}.png")

    report = validate_dataset_structure(tmp_path)
    assert not report.is_valid
    assert report.real_dir_exists
    assert not report.attacked_dir_exists
    assert any("ATTACKED" in issue for issue in report.issues)


def test_validate_dataset_structure_valid(tmp_path):
    """Verify that a properly structured directory passes validation."""
    real_dir = tmp_path / "REAL"
    att_dir = tmp_path / "ATTACKED"
    real_dir.mkdir()
    att_dir.mkdir()

    for d in [1, 2]:
        for f in range(3):
            img = Image.new("RGB", (100, 100), color=(200, 200, 200))
            img.save(real_dir / f"doc_real_{d:02d}_f{f:02d}.png")
            img.save(att_dir / f"doc_att_{d:02d}_f{f:02d}.png")

    report = validate_dataset_structure(tmp_path, min_samples_per_class=3)
    assert report.is_valid
    assert report.real_count == 6
    assert report.attacked_count == 6
    assert report.unique_groups == 4
    assert len(report.issues) == 0


def test_create_leak_free_splits():
    """Verify that train, val, and test splits have ZERO shared document IDs."""
    samples = []
    # Create 10 real documents and 10 attacked documents with 4 frames each
    for d in range(1, 11):
        doc_id = f"doc_real_{d:02d}"
        for f in range(1, 5):
            samples.append({
                "path": f"/dummy/{doc_id}_f{f}.png",
                "label": 0,
                "class_name": "REAL",
                "doc_id": doc_id,
            })
    for d in range(1, 11):
        doc_id = f"doc_att_{d:02d}"
        for f in range(1, 5):
            samples.append({
                "path": f"/dummy/{doc_id}_f{f}.png",
                "label": 1,
                "class_name": "ATTACKED",
                "doc_id": doc_id,
            })

    train_s, val_s, test_s = create_leak_free_splits(
        samples, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42
    )

    train_docs = {s["doc_id"] for s in train_s}
    val_docs = {s["doc_id"] for s in val_s}
    test_docs = {s["doc_id"] for s in test_s}

    # CRITICAL LEAKAGE ASSERTIONS
    assert len(train_docs.intersection(val_docs)) == 0, "Leakage detected between train and val!"
    assert len(train_docs.intersection(test_docs)) == 0, "Leakage detected between train and test!"
    assert len(val_docs.intersection(test_docs)) == 0, "Leakage detected between val and test!"

    # Total samples conserved
    assert len(train_s) + len(val_s) + len(test_s) == len(samples)


def test_document_transforms_and_dataset(tmp_path):
    """Verify document transforms and DocumentDataset output tensor shapes."""
    img_path = tmp_path / "test_doc_f01.png"
    img = Image.new("RGB", (300, 200), color=(128, 128, 128))
    img.save(img_path)

    sample = [{
        "path": str(img_path),
        "label": 0,
        "doc_id": "test_doc",
        "class_name": "REAL",
    }]

    transform = get_document_transforms(image_size=(224, 224), is_training=True)
    dataset = DocumentDataset(sample, transform=transform)

    assert len(dataset) == 1
    batch = dataset[0]

    assert "image" in batch
    assert "label" in batch
    assert "doc_id" in batch

    # Check tensor shape (3, 224, 224)
    tensor = batch["image"]
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert batch["label"].item() == 0
    assert batch["doc_id"] == "test_doc"


def test_audit_project_datasets():
    """Verify that project dataset audit correctly assesses all data directories."""
    from src.visual.dataset_auditor import audit_project_datasets

    records = audit_project_datasets()
    assert "data/raw/" in records
    assert "data/sample/" in records
    assert "data/synthetic/" in records

    # data/raw is empty -> not suitable
    assert records["data/raw/"]["suitable_for_training"] == "NO"

    # data/sample is single-doc video -> not suitable
    assert records["data/sample/"]["suitable_for_training"] == "NO"

    # data/synthetic is balanced with 250 images -> suitable
    assert records["data/synthetic/"]["suitable_for_training"] == "YES"
    assert records["data/synthetic/"]["real_samples"] == 125
    assert records["data/synthetic/"]["attacked_samples"] == 125


def test_prepare_visual_dataset_and_manifest(tmp_path):
    """Verify visual dataset partitioning and manifest.csv generation on mock data."""
    import csv
    from src.visual.dataset_preparer import prepare_visual_dataset

    # Create a small mock source dataset
    src_dir = tmp_path / "mock_synthetic"
    real_dir = src_dir / "REAL"
    att_dir = src_dir / "ATTACKED"
    real_dir.mkdir(parents=True)
    att_dir.mkdir(parents=True)

    for d in range(1, 7):
        for f in range(1, 4):
            img = Image.new("RGB", (100, 100), color=(200, 200, 200))
            img.save(real_dir / f"doc_real_{d:03d}_f{f:02d}.png")
            img.save(att_dir / f"doc_att_{d:03d}_f{f:02d}.png")

    target_dir = tmp_path / "mock_visual_dataset"
    summary = prepare_visual_dataset(
        source_dir=src_dir,
        target_dir=target_dir,
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
        random_seed=42,
    )

    assert summary["total_samples"] == 36  # 12 docs * 3 frames
    assert summary["leakage_check"]["is_leak_free"] is True
    assert (target_dir / "manifest.csv").is_file()

    # Validate manifest format
    with open(target_dir / "manifest.csv", mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 36
        for r in rows:
            assert r["label"] in {"REAL", "ATTACKED"}
            assert r["split"] in {"train", "val", "test"}
            assert r["source_group"] == r["video_id"]
            assert (target_dir.parent.parent / r["path"]).is_file() or (target_dir / r["path"]).is_file() or Path(r["path"]).exists()


def test_is_and_load_partitioned_dataset(tmp_path):
    """Verify partitioned dataset detection, loading, and zero-leakage validation."""
    from src.visual.dataset import is_partitioned_dataset, load_partitioned_dataset

    part_dir = tmp_path / "part_dataset"
    assert not is_partitioned_dataset(part_dir)

    for split in ["train", "val", "test"]:
        for cls in ["REAL", "ATTACKED"]:
            (part_dir / split / cls).mkdir(parents=True)

    assert is_partitioned_dataset(part_dir)

    # Populate with disjoint documents
    doc_splits = {
        "train": ["doc_tr_01", "doc_tr_02"],
        "val": ["doc_val_01"],
        "test": ["doc_te_01"],
    }
    for split, docs in doc_splits.items():
        for doc in docs:
            for f in range(2):
                img = Image.new("RGB", (64, 64), color=(100, 100, 100))
                img.save(part_dir / split / "REAL" / f"{doc}.or_f{f:02d}.png")
                img.save(part_dir / split / "ATTACKED" / f"{doc}.cc_f{f:02d}.png")

    train_s, val_s, test_s = load_partitioned_dataset(part_dir)
    assert len(train_s) == 8  # 2 docs * 2 classes * 2 frames
    assert len(val_s) == 4    # 1 doc * 2 classes * 2 frames
    assert len(test_s) == 4   # 1 doc * 2 classes * 2 frames

    train_docs = {s["doc_id"] for s in train_s}
    val_docs = {s["doc_id"] for s in val_s}
    test_docs = {s["doc_id"] for s in test_s}
    assert train_docs.isdisjoint(val_docs)
    assert train_docs.isdisjoint(test_docs)
    assert val_docs.isdisjoint(test_docs)

    report = validate_dataset_structure(part_dir, min_samples_per_class=2)
    assert report.is_valid
    assert report.real_count == 8
    assert report.attacked_count == 8
    assert report.total_count == 16

