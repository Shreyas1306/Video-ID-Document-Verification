"""
Visual Document Integrity Dataset Preparation & Manifest Generator
===================================================================
Prepares a partitioned, leak-free visual dataset for EfficientNet transfer learning:
  - Partitions into train (70%), val (15%), and test (15%) splits.
  - Groups strictly by document/video ID (source_group) to guarantee 0% train/val/test leakage.
  - Organizes files into:
      data/visual_dataset/
        ├── train/ (real/, attacked/)
        ├── val/   (real/, attacked/)
        └── test/  (real/, attacked/)
  - Generates canonical CSV manifest:
      data/visual_dataset/manifest.csv
"""

import csv
import logging
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union

from src.visual.dataset import (
    CLASS_TO_LABEL,
    collect_dataset_samples,
    create_leak_free_splits,
    extract_doc_id,
    validate_dataset_structure,
)

logger = logging.getLogger(__name__)


def prepare_visual_dataset(
    source_dir: Union[str, Path] = "data/synthetic",
    target_dir: Union[str, Path] = "data/visual_dataset",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    clean_target: bool = True,
) -> Dict[str, Any]:
    """
    Partition source dataset into train/val/test folders and create manifest.csv.

    Args:
        source_dir: Root of source dataset containing REAL/ and ATTACKED/ subdirectories.
        target_dir: Destination root for visual_dataset.
        train_ratio: Target fraction for training set (default 0.70).
        val_ratio: Target fraction for validation set (default 0.15).
        test_ratio: Target fraction for test set (default 0.15).
        random_seed: Deterministic seed for reproducible document-level split.
        clean_target: If True, clears existing target directory before creating new splits.

    Returns:
        Structured summary dictionary with split counts, document groups, and manifest path.
    """
    src_path = Path(source_dir).resolve()
    dst_path = Path(target_dir).resolve()

    # 1. Validate source dataset
    report = validate_dataset_structure(src_path)
    if not report.is_valid:
        raise ValueError(
            f"Source dataset at '{src_path}' is invalid for preparation: {'; '.join(report.issues)}"
        )

    # 2. Collect all samples
    samples = collect_dataset_samples(src_path)
    if not samples:
        raise ValueError(f"No valid image samples found in '{src_path}'.")

    # 3. Create document-level leak-free splits
    train_samples, val_samples, test_samples = create_leak_free_splits(
        samples=samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
    )

    # Leakage verification
    train_docs = {str(s["doc_id"]) for s in train_samples}
    val_docs = {str(s["doc_id"]) for s in val_samples}
    test_docs = {str(s["doc_id"]) for s in test_samples}

    leakage_train_val = train_docs.intersection(val_docs)
    leakage_train_test = train_docs.intersection(test_docs)
    leakage_val_test = val_docs.intersection(test_docs)

    if leakage_train_val or leakage_train_test or leakage_val_test:
        raise RuntimeError(
            f"Leakage detected during dataset splitting! "
            f"Train-Val: {leakage_train_val}, Train-Test: {leakage_train_test}, Val-Test: {leakage_val_test}"
        )

    # 4. Prepare target directory structure
    if clean_target and dst_path.exists():
        shutil.rmtree(dst_path)

    split_dirs = {
        "train": {
            "REAL": dst_path / "train" / "real",
            "ATTACKED": dst_path / "train" / "attacked",
        },
        "val": {
            "REAL": dst_path / "val" / "real",
            "ATTACKED": dst_path / "val" / "attacked",
        },
        "test": {
            "REAL": dst_path / "test" / "real",
            "ATTACKED": dst_path / "test" / "attacked",
        },
    }

    for split_name, class_paths in split_dirs.items():
        for class_path in class_paths.values():
            class_path.mkdir(parents=True, exist_ok=True)

    # 5. Copy files and build manifest rows
    manifest_rows: List[Dict[str, str]] = []
    split_assignments = [
        ("train", train_samples),
        ("val", val_samples),
        ("test", test_samples),
    ]

    stats = {
        "train": {"REAL": 0, "ATTACKED": 0, "docs": len(train_docs)},
        "val": {"REAL": 0, "ATTACKED": 0, "docs": len(val_docs)},
        "test": {"REAL": 0, "ATTACKED": 0, "docs": len(test_docs)},
    }

    for split_name, split_sample_list in split_assignments:
        for s in split_sample_list:
            src_file = Path(s["path"])
            class_name = str(s["class_name"])  # "REAL" or "ATTACKED"
            doc_id = str(s["doc_id"])
            sample_id = src_file.stem

            target_folder = split_dirs[split_name][class_name]
            dst_file = target_folder / src_file.name

            # Copy image file
            shutil.copy2(src_file, dst_file)

            # Update stats
            stats[split_name][class_name] += 1

            # Relative path inside project
            try:
                rel_path = dst_file.relative_to(dst_path.parent.parent)
            except Exception:
                rel_path = dst_file

            manifest_rows.append({
                "sample_id": sample_id,
                "source": str(src_file),
                "source_group": doc_id,
                "path": str(rel_path).replace("\\", "/"),
                "label": class_name,
                "document_type": "mock_identity_card",
                "video_id": doc_id,
                "split": split_name,
            })

    # 6. Write manifest.csv
    manifest_file = dst_path / "manifest.csv"
    fieldnames = [
        "sample_id",
        "source",
        "source_group",
        "path",
        "label",
        "document_type",
        "video_id",
        "split",
    ]

    with open(manifest_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "source_directory": str(src_path),
        "target_directory": str(dst_path),
        "manifest_path": str(manifest_file),
        "total_samples": len(manifest_rows),
        "total_documents": len(train_docs) + len(val_docs) + len(test_docs),
        "splits": {
            "train": {
                "total": stats["train"]["REAL"] + stats["train"]["ATTACKED"],
                "real": stats["train"]["REAL"],
                "attacked": stats["train"]["ATTACKED"],
                "documents": stats["train"]["docs"],
                "percentage": round((stats["train"]["REAL"] + stats["train"]["ATTACKED"]) / len(manifest_rows) * 100, 1),
            },
            "val": {
                "total": stats["val"]["REAL"] + stats["val"]["ATTACKED"],
                "real": stats["val"]["REAL"],
                "attacked": stats["val"]["ATTACKED"],
                "documents": stats["val"]["docs"],
                "percentage": round((stats["val"]["REAL"] + stats["val"]["ATTACKED"]) / len(manifest_rows) * 100, 1),
            },
            "test": {
                "total": stats["test"]["REAL"] + stats["test"]["ATTACKED"],
                "real": stats["test"]["REAL"],
                "attacked": stats["test"]["ATTACKED"],
                "documents": stats["test"]["docs"],
                "percentage": round((stats["test"]["REAL"] + stats["test"]["ATTACKED"]) / len(manifest_rows) * 100, 1),
            },
        },
        "leakage_check": {
            "train_val_overlap": len(leakage_train_val),
            "train_test_overlap": len(leakage_train_test),
            "val_test_overlap": len(leakage_val_test),
            "is_leak_free": True,
        },
    }

    logger.info(
        f"Visual dataset prepared at '{dst_path}': "
        f"Train={summary['splits']['train']['total']} ({summary['splits']['train']['percentage']}%), "
        f"Val={summary['splits']['val']['total']} ({summary['splits']['val']['percentage']}%), "
        f"Test={summary['splits']['test']['total']} ({summary['splits']['test']['percentage']}%)"
    )

    return summary
