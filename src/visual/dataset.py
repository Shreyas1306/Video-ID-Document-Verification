"""
Visual Document Integrity Dataset & Split Management
=====================================================
Dataset loader, document/video-level grouping, leak-free train/val/test splitting,
and document-appropriate image augmentations for REAL vs ATTACKED classification.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}

# Binary class mapping
CLASS_TO_LABEL: Dict[str, int] = {
    "REAL": 0,
    "ATTACKED": 1,
}
LABEL_TO_CLASS: Dict[int, str] = {
    0: "REAL",
    1: "ATTACKED",
}


@dataclass
class DatasetValidationReport:
    """Structured report detailing the validity of a dataset directory."""
    is_valid: bool
    dataset_dir: str
    real_dir_exists: bool
    attacked_dir_exists: bool
    real_count: int
    attacked_count: int
    total_count: int
    unique_groups: int
    issues: List[str] = field(default_factory=list)
    summary: str = ""

    def __str__(self) -> str:
        lines = [
            f"Dataset Validation Report for: {self.dataset_dir}",
            f"Status: {'VALID' if self.is_valid else 'INVALID'}",
            f"  - REAL directory exists: {self.real_dir_exists} ({self.real_count} images)",
            f"  - ATTACKED directory exists: {self.attacked_dir_exists} ({self.attacked_count} images)",
            f"  - Total images: {self.total_count}",
            f"  - Unique document/video groups: {self.unique_groups}",
        ]
        if self.issues:
            lines.append("  - Issues detected:")
            for issue in self.issues:
                lines.append(f"      * {issue}")
        else:
            lines.append("  - No issues detected. Dataset is suitable for training.")
        return "\n".join(lines)


def extract_doc_id(file_path: Union[str, Path]) -> str:
    """
    Extract document or video group identifier from an image file path.

    Handles conventions:
      - 'doc_001_f01.png' -> 'doc_001'
      - 'doc001_frame02.jpg' -> 'doc001'
      - 'video_04_frame_0003.png' -> 'video_04'
      - 'REAL/doc_005/frame_01.png' -> 'doc_005'
      - Fallback: stem up to last underscore, or entire stem if no separator.
    """
    path = Path(file_path)
    stem = path.stem

    # 1. If inside a nested folder structure like REAL/doc_001/frame_01.png
    if path.parent.name not in {"REAL", "ATTACKED", "real", "attacked", ".", ""}:
        return path.parent.name

    # 2. If dot in stem (DLC-2021 clip convention: <doc_id>.<clip_code>_<frame>)
    if "." in stem:
        return stem.split(".")[0]

    # 3. Regex matching common patterns: doc_xxx, video_xxx, subject_xxx
    match = re.match(r"^([a-zA-Z]+[-_]\d+)", stem)
    if match:
        return match.group(1)

    # 4. Fallback: split by _frame, _f, _t
    frame_split = re.split(r"[_-](?:frame|f|t|\d+$)", stem, flags=re.IGNORECASE)
    if len(frame_split) > 1 and frame_split[0]:
        return frame_split[0]

    # 5. Fallback: split on last underscore if present
    if "_" in stem:
        parts = stem.rsplit("_", 1)
        if parts[0]:
            return parts[0]

    return stem


def is_partitioned_dataset(dataset_dir: Union[str, Path]) -> bool:
    """Check if the dataset directory contains pre-partitioned train/val/test splits."""
    base_path = Path(dataset_dir)
    if not base_path.is_dir():
        return False
    if (base_path / "manifest.csv").is_file():
        return True
    return (
        (base_path / "train").is_dir()
        and (base_path / "val").is_dir()
        and (base_path / "test").is_dir()
    )


def load_partitioned_dataset(
    dataset_dir: Union[str, Path],
) -> Tuple[List[Dict[str, Union[str, int]]], List[Dict[str, Union[str, int]]], List[Dict[str, Union[str, int]]]]:
    """
    Load pre-partitioned train, val, and test sample lists.

    Reads manifest.csv if present, otherwise scans train/, val/, and test/ subdirectories.
    Guarantees zero document identity overlap across splits.
    """
    import csv

    base_path = Path(dataset_dir).resolve()
    manifest_file = base_path / "manifest.csv"

    train_samples: List[Dict[str, Union[str, int]]] = []
    val_samples: List[Dict[str, Union[str, int]]] = []
    test_samples: List[Dict[str, Union[str, int]]] = []

    if manifest_file.is_file():
        with open(manifest_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                split = str(row.get("split", "")).lower().strip()
                label_str = str(row.get("label", "")).upper().strip()
                if label_str not in CLASS_TO_LABEL:
                    continue

                crop_rel = row.get("crop_path") or row.get("path") or ""
                img_path = (base_path / crop_rel).resolve()
                doc_id = (
                    row.get("document_id")
                    or row.get("source_group")
                    or extract_doc_id(img_path)
                )

                item = {
                    "path": str(img_path),
                    "label": CLASS_TO_LABEL[label_str],
                    "class_name": label_str,
                    "doc_id": doc_id,
                }

                if split == "train":
                    train_samples.append(item)
                elif split == "val":
                    val_samples.append(item)
                elif split == "test":
                    test_samples.append(item)
    else:
        # Fallback to subdirectory scanning
        split_map = {
            "train": train_samples,
            "val": val_samples,
            "test": test_samples,
        }
        for split_name, sample_list in split_map.items():
            split_dir = base_path / split_name
            if not split_dir.is_dir():
                continue
            for class_name, label in CLASS_TO_LABEL.items():
                candidates = [split_dir / class_name, split_dir / class_name.lower()]
                seen_dirs = set()
                for class_dir in candidates:
                    if class_dir.is_dir():
                        resolved_dir = class_dir.resolve()
                        if resolved_dir not in seen_dirs:
                            seen_dirs.add(resolved_dir)
                            for img_file in sorted(class_dir.rglob("*")):
                                if img_file.is_file() and img_file.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                                    sample_list.append({
                                        "path": str(img_file.resolve()),
                                        "label": label,
                                        "class_name": class_name,
                                        "doc_id": extract_doc_id(img_file),
                                    })

    # Verification: Ensure zero document ID leakage
    train_ids = {s["doc_id"] for s in train_samples}
    val_ids = {s["doc_id"] for s in val_samples}
    test_ids = {s["doc_id"] for s in test_samples}

    assert len(train_ids.intersection(val_ids)) == 0, f"Leakage between train and val: {train_ids & val_ids}"
    assert len(train_ids.intersection(test_ids)) == 0, f"Leakage between train and test: {train_ids & test_ids}"
    assert len(val_ids.intersection(test_ids)) == 0, f"Leakage between val and test: {val_ids & test_ids}"

    logger.info(
        f"Partitioned dataset loaded: Train={len(train_samples)} frames ({len(train_ids)} docs), "
        f"Val={len(val_samples)} frames ({len(val_ids)} docs), "
        f"Test={len(test_samples)} frames ({len(test_ids)} docs)"
    )

    return train_samples, val_samples, test_samples


def validate_dataset_structure(
    dataset_dir: Union[str, Path],
    real_subdir: str = "REAL",
    attacked_subdir: str = "ATTACKED",
    min_samples_per_class: int = 5,
) -> DatasetValidationReport:
    """
    Inspect the dataset directory and report whether it is suitable for training.

    Checks:
      1. Dataset directory exists.
      2. If partitioned (train/val/test or manifest.csv): validates splits and zero leakage.
      3. Otherwise: REAL and ATTACKED subdirectories exist and meet count thresholds.
      4. Document group counts.
    """
    base_path = Path(dataset_dir)
    issues: List[str] = []

    if not base_path.exists():
        return DatasetValidationReport(
            is_valid=False,
            dataset_dir=str(base_path),
            real_dir_exists=False,
            attacked_dir_exists=False,
            real_count=0,
            attacked_count=0,
            total_count=0,
            unique_groups=0,
            issues=[f"Dataset directory '{base_path}' does not exist."],
            summary="Dataset directory not found.",
        )

    # Check for partitioned dataset
    if is_partitioned_dataset(base_path):
        try:
            train_s, val_s, test_s = load_partitioned_dataset(base_path)
            all_s = train_s + val_s + test_s
            real_count = sum(1 for s in all_s if s["label"] == CLASS_TO_LABEL["REAL"])
            attacked_count = sum(1 for s in all_s if s["label"] == CLASS_TO_LABEL["ATTACKED"])
            unique_groups = len({s["doc_id"] for s in all_s})

            if len(train_s) == 0:
                issues.append("Partitioned dataset has 0 training samples.")
            if len(val_s) == 0:
                issues.append("Partitioned dataset has 0 validation samples.")
            if len(test_s) == 0:
                issues.append("Partitioned dataset has 0 test samples.")
            if real_count < min_samples_per_class:
                issues.append(
                    f"Insufficient REAL samples: found {real_count}, minimum required is {min_samples_per_class}."
                )
            if attacked_count < min_samples_per_class:
                issues.append(
                    f"Insufficient ATTACKED samples: found {attacked_count}, minimum required is {min_samples_per_class}."
                )

            return DatasetValidationReport(
                is_valid=len(issues) == 0,
                dataset_dir=str(base_path),
                real_dir_exists=True,
                attacked_dir_exists=True,
                real_count=real_count,
                attacked_count=attacked_count,
                total_count=len(all_s),
                unique_groups=unique_groups,
                issues=issues,
                summary=(
                    f"Partitioned dataset contains {real_count} REAL and {attacked_count} ATTACKED images "
                    f"across {unique_groups} document groups (Train: {len(train_s)}, Val: {len(val_s)}, Test: {len(test_s)})."
                ),
            )
        except Exception as e:
            issues.append(f"Failed to load partitioned dataset: {e}")
            return DatasetValidationReport(
                is_valid=False,
                dataset_dir=str(base_path),
                real_dir_exists=False,
                attacked_dir_exists=False,
                real_count=0,
                attacked_count=0,
                total_count=0,
                unique_groups=0,
                issues=issues,
                summary=f"Error validating partitioned dataset: {e}",
            )

    real_path = base_path / real_subdir
    attacked_path = base_path / attacked_subdir

    real_exists = real_path.is_dir()
    attacked_exists = attacked_path.is_dir()

    if not real_exists:
        issues.append(f"Subdirectory '{real_subdir}/' does not exist in '{base_path}'.")
    if not attacked_exists:
        issues.append(f"Subdirectory '{attacked_subdir}/' does not exist in '{base_path}'.")

    real_files: List[Path] = []
    attacked_files: List[Path] = []

    if real_exists:
        real_files = [
            p for p in real_path.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
    if attacked_exists:
        attacked_files = [
            p for p in attacked_path.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]

    real_count = len(real_files)
    attacked_count = len(attacked_files)
    total_count = real_count + attacked_count

    if real_count < min_samples_per_class:
        issues.append(
            f"Insufficient REAL samples: found {real_count}, minimum required is {min_samples_per_class}."
        )
    if attacked_count < min_samples_per_class:
        issues.append(
            f"Insufficient ATTACKED samples: found {attacked_count}, minimum required is {min_samples_per_class}."
        )

    # Check document grouping
    all_files = real_files + attacked_files
    unique_groups = len({extract_doc_id(p) for p in all_files})

    if total_count > 0 and unique_groups < 2:
        issues.append(
            f"Only {unique_groups} unique document group detected. Multiple document/video groups "
            f"are required to prevent train/test leakage."
        )

    is_valid = len(issues) == 0
    summary = (
        f"Dataset contains {real_count} REAL and {attacked_count} ATTACKED images "
        f"across {unique_groups} document groups."
    )

    return DatasetValidationReport(
        is_valid=is_valid,
        dataset_dir=str(base_path),
        real_dir_exists=real_exists,
        attacked_dir_exists=attacked_exists,
        real_count=real_count,
        attacked_count=attacked_count,
        total_count=total_count,
        unique_groups=unique_groups,
        issues=issues,
        summary=summary,
    )


def collect_dataset_samples(
    dataset_dir: Union[str, Path],
    real_subdir: str = "REAL",
    attacked_subdir: str = "ATTACKED",
) -> List[Dict[str, Union[str, int]]]:
    """
    Collect all image paths, class labels, and document IDs from dataset directory.
    """
    base_path = Path(dataset_dir)
    samples: List[Dict[str, Union[str, int]]] = []

    for class_name, label in CLASS_TO_LABEL.items():
        subdir = real_subdir if class_name == "REAL" else attacked_subdir
        class_path = base_path / subdir
        if not class_path.is_dir():
            continue

        for file_path in sorted(class_path.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                doc_id = extract_doc_id(file_path)
                samples.append({
                    "path": str(file_path.resolve()),
                    "label": label,
                    "class_name": class_name,
                    "doc_id": doc_id,
                })

    return samples


def create_leak_free_splits(
    samples: List[Dict[str, Union[str, int]]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[List[Dict[str, Union[str, int]]], List[Dict[str, Union[str, int]]], List[Dict[str, Union[str, int]]]]:
    """
    Split samples into train, validation, and test sets at the DOCUMENT/VIDEO level.

    Guarantees:
      - All frames originating from the same document ID belong exclusively to one split.
      - Zero frame-level train/test leakage.
      - Stratification by class at the document level where feasible.
    """
    if not samples:
        return [], [], []

    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-4:
        raise ValueError(f"Split ratios must sum to 1.0 (got {total_ratio})")

    rng = np.random.RandomState(random_seed)

    # Group samples by (doc_id, class_name)
    group_map: Dict[str, Dict[str, Union[str, int, List]]] = {}
    for item in samples:
        doc_id = str(item["doc_id"])
        if doc_id not in group_map:
            group_map[doc_id] = {
                "doc_id": doc_id,
                "label": item["label"],
                "class_name": item["class_name"],
                "samples": [],
            }
        group_map[doc_id]["samples"].append(item)

    # Separate groups by class for stratified group splitting
    real_groups = [g for g in group_map.values() if g["label"] == CLASS_TO_LABEL["REAL"]]
    attacked_groups = [g for g in group_map.values() if g["label"] == CLASS_TO_LABEL["ATTACKED"]]

    # Shuffle groups deterministically
    rng.shuffle(real_groups)
    rng.shuffle(attacked_groups)

    def _split_groups(groups: List[Dict]):
        n = len(groups)
        if n == 0:
            return [], [], []
        if n == 1:
            return groups, [], []
        if n == 2:
            return [groups[0]], [groups[1]], []

        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        if n_train + n_val >= n:
            n_val = max(1, n - n_train)

        train_g = groups[:n_train]
        val_g = groups[n_train:n_train + n_val]
        test_g = groups[n_train + n_val:]

        if not test_g and len(train_g) > 1:
            # Shift one to test if test is empty and train has headroom
            test_g = [train_g.pop()]

        return train_g, val_g, test_g

    real_train_g, real_val_g, real_test_g = _split_groups(real_groups)
    att_train_g, att_val_g, att_test_g = _split_groups(attacked_groups)

    train_groups = real_train_g + att_train_g
    val_groups = real_val_g + att_val_g
    test_groups = real_test_g + att_test_g

    # Flatten samples from groups
    train_samples = [s for g in train_groups for s in g["samples"]]
    val_samples = [s for g in val_groups for s in g["samples"]]
    test_samples = [s for g in test_groups for s in g["samples"]]

    # Verification: Ensure zero document ID leakage
    train_ids = {g["doc_id"] for g in train_groups}
    val_ids = {g["doc_id"] for g in val_groups}
    test_ids = {g["doc_id"] for g in test_groups}

    assert len(train_ids.intersection(val_ids)) == 0, "Leakage between train and val!"
    assert len(train_ids.intersection(test_ids)) == 0, "Leakage between train and test!"
    assert len(val_ids.intersection(test_ids)) == 0, "Leakage between val and test!"

    logger.info(
        f"Leak-free split created: Train={len(train_samples)} frames ({len(train_ids)} docs), "
        f"Val={len(val_samples)} frames ({len(val_ids)} docs), "
        f"Test={len(test_samples)} frames ({len(test_ids)} docs)"
    )

    return train_samples, val_samples, test_samples


def get_document_transforms(
    image_size: Tuple[int, int] = (224, 224),
    is_training: bool = True,
) -> Callable:
    """
    Get image preprocessing and augmentation appropriate for identity documents.

    Key considerations:
      - Document images possess directional text and layout semantics.
      - NO RandomHorizontalFlip is applied (flipping text creates unnatural artifacts).
      - Controlled subtle tilt (±5 degrees) mirrors handheld video frame variation.
      - Gentle brightness/contrast jitter mirrors smartphone sensor exposure variations.
      - Standard ImageNet normalization for transfer learning backbones.
    """
    if is_training:
        return T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.RandomRotation(degrees=(-5, 5), interpolation=T.InterpolationMode.BILINEAR),
            T.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.02),
            T.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.98, 1.02)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


class DocumentDataset(Dataset):
    """
    PyTorch Dataset for binary visual document integrity classification.
    """

    def __init__(
        self,
        samples: Sequence[Dict[str, Union[str, int]]],
        transform: Optional[Callable] = None,
    ):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        item = self.samples[idx]
        image_path = str(item["path"])
        label = int(item["label"])
        doc_id = str(item.get("doc_id", ""))

        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return blank placeholder tensor if image is unreadable
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))

        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = T.ToTensor()(image)

        return {
            "image": image_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "doc_id": doc_id,
            "path": image_path,
        }
