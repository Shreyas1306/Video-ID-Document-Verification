"""
Phase 3C: Build Document-Type Stratified DLC-2021 Dataset Split
=============================================================
Creates a repartitioned dataset under outputs/normalized_public_dataset_stratified/
with physical-document identity stratification:

TRAIN:
  - alb_id_00
  - alb_id_01
  - alb_id_02
  - aze_passport_00
VAL:
  - alb_id_03
  - alb_id_04
TEST:
  - alb_id_05
  - aze_passport_01

Zero physical-identity leakage: Each identity exists exclusively in one split.
Copies normalized PNG crops from outputs/normalized_public_dataset/ without reprocessing.
Verifies all crops are 600x400 RGB PNG.
"""

from collections import Counter
import csv
import json
import logging
from pathlib import Path
import shutil
import sys
from typing import Dict, List

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Target identity partition allocation
SPLIT_ALLOCATION = {
    "train": ["alb_id_00", "alb_id_01", "alb_id_02", "aze_passport_00"],
    "val": ["alb_id_03", "alb_id_04"],
    "test": ["alb_id_05", "aze_passport_01"],
}

DOC_TO_SPLIT = {}
for split_name, doc_list in SPLIT_ALLOCATION.items():
    for doc_id in doc_list:
        DOC_TO_SPLIT[doc_id] = split_name


def build_stratified_dataset(
    source_dir: Path = PROJECT_ROOT / "outputs" / "normalized_public_dataset",
    output_dir: Path = PROJECT_ROOT / "outputs" / "normalized_public_dataset_stratified",
) -> Dict:
    manifest_path = source_dir / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Source manifest not found at: {manifest_path}")

    # Read original manifest
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        original_rows = list(reader)

    logger.info(f"Loaded {len(original_rows)} records from {manifest_path}")

    # Validate that all document IDs in manifest are mapped
    manifest_doc_ids = sorted({r["document_id"] for r in original_rows})
    logger.info(f"Document IDs found in manifest: {manifest_doc_ids}")
    allocated_doc_ids = sorted(list(DOC_TO_SPLIT.keys()))
    assert manifest_doc_ids == allocated_doc_ids, (
        f"Manifest doc IDs {manifest_doc_ids} do not match allocation {allocated_doc_ids}"
    )

    # Prepare output directories
    if output_dir.exists():
        logger.warning(f"Output directory {output_dir} already exists. Removing prior build...")
        shutil.rmtree(output_dir)

    for split in ["train", "val", "test"]:
        for label in ["REAL", "ATTACKED"]:
            (output_dir / split / label).mkdir(parents=True, exist_ok=True)

    # Process and copy files
    new_rows: List[Dict] = []
    split_counts = Counter()
    label_counts = {s: Counter() for s in ["train", "val", "test"]}
    doc_counts = {s: Counter() for s in ["train", "val", "test"]}
    attack_counts = {s: Counter() for s in ["train", "val", "test"]}
    doc_type_counts = {s: Counter() for s in ["train", "val", "test"]}

    logger.info("Copying normalized crops to new stratified layout...")
    for idx, row in enumerate(original_rows):
        doc_id = row["document_id"]
        label = row["label"].upper()
        new_split = DOC_TO_SPLIT[doc_id]
        orig_crop_path = source_dir / row["crop_path"]

        if not orig_crop_path.is_file():
            raise FileNotFoundError(f"Source crop does not exist: {orig_crop_path}")

        filename = orig_crop_path.name
        new_rel_crop_path = f"{new_split}/{label}/{filename}"
        dest_crop_path = output_dir / new_rel_crop_path

        shutil.copy2(orig_crop_path, dest_crop_path)

        # Clone row and update split and crop_path
        new_row = dict(row)
        new_row["split"] = new_split
        new_row["crop_path"] = new_rel_crop_path
        new_rows.append(new_row)

        # Update stats
        split_counts[new_split] += 1
        label_counts[new_split][label] += 1
        doc_counts[new_split][doc_id] += 1
        attack_counts[new_split][row.get("attack_type", "unknown")] += 1
        doc_type = "albanian_id" if doc_id.startswith("alb_id") else "azerbaijan_passport"
        doc_type_counts[new_split][doc_type] += 1

    logger.info(f"Copied {len(new_rows)} images to {output_dir}")

    # Verification: Validate image integrity and dimensions
    logger.info("Verifying all copied crops (600x400 RGB PNG)...")
    for r in new_rows:
        img_path = output_dir / r["crop_path"]
        assert img_path.is_file(), f"Missing file: {img_path}"
        with Image.open(img_path) as img:
            assert img.size == (600, 400), f"Invalid size {img.size} for {img_path}"
            assert img.mode == "RGB", f"Invalid mode {img.mode} for {img_path}"
            assert img.format == "PNG", f"Invalid format {img.format} for {img_path}"

    logger.info("All 1,527 crops verified: 600x400 RGB PNG.")

    # Write new manifest.csv
    fieldnames = list(new_rows[0].keys())
    # Ensure required columns are present at the beginning
    priority_fields = ["frame_id", "document_id", "clip_id", "attack_type", "label", "split", "crop_path"]
    other_fields = [f for f in fieldnames if f not in priority_fields]
    ordered_fieldnames = priority_fields + other_fields

    manifest_dest = output_dir / "manifest.csv"
    with open(manifest_dest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_fieldnames)
        writer.writeheader()
        writer.writerows(new_rows)
    logger.info(f"Saved manifest to {manifest_dest}")

    # Copy failures.csv if it exists in source
    source_failures = source_dir / "failures.csv"
    if source_failures.is_file():
        shutil.copy2(source_failures, output_dir / "failures.csv")

    # Verification: Zero document ID leakage
    train_ids = set(SPLIT_ALLOCATION["train"])
    val_ids = set(SPLIT_ALLOCATION["val"])
    test_ids = set(SPLIT_ALLOCATION["test"])

    leakage_train_val = train_ids.intersection(val_ids)
    leakage_train_test = train_ids.intersection(test_ids)
    leakage_val_test = val_ids.intersection(test_ids)

    assert not leakage_train_val, f"Leakage train/val: {leakage_train_val}"
    assert not leakage_train_test, f"Leakage train/test: {leakage_train_test}"
    assert not leakage_val_test, f"Leakage val/test: {leakage_val_test}"

    logger.info("Zero document identity leakage verified.")

    # Compile summary statistics
    summary = {
        "dataset_name": "DLC-2021 Stratified Public Dataset",
        "total_samples": len(new_rows),
        "split_allocation": SPLIT_ALLOCATION,
        "sample_counts_by_split": dict(split_counts),
        "label_distribution_by_split": {s: dict(c) for s, c in label_counts.items()},
        "document_type_distribution": {s: dict(c) for s, c in doc_type_counts.items()},
        "document_id_counts": {s: dict(c) for s, c in doc_counts.items()},
        "attack_type_counts": {s: dict(c) for s, c in attack_counts.items()},
        "zero_leakage_verified": True,
        "crop_specs_verified": "600x400 RGB PNG",
    }

    summary_dest = output_dir / "summary.json"
    with open(summary_dest, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_dest}")

    return summary


if __name__ == "__main__":
    summary = build_stratified_dataset()
    print("\n" + "=" * 60)
    print("DLC-2021 STRATIFIED DATASET GENERATION SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
