"""
Build Public Training Dataset: Normalized DLC-2021 Dataset Generator
====================================================================
Processes raw DLC-2021 video frames through the validated Hybrid Variant C
document localization detector (OpenCV V2 + Variant C Ambiguity Gate + MIDV-500 UNet)
and PerspectiveCorrector to generate clean, rectified document crops for
training the EfficientNet-B0 visual integrity classifier.

Features:
- Strict zero-leakage document-level split enforcement:
    Train:      alb_id_00, alb_id_01, alb_id_02, alb_id_03 (800 frames)
    Validation: alb_id_04, alb_id_05 (400 frames)
    Test:       aze_passport_00, aze_passport_01 (400 frames, held-out)
- Sequential frame-by-frame processing with per-clip TemporalDocumentTracker
- Direct full-frame 4-corner homography to 600x400 normalized frames
- Preserves all metadata: document_id, attack_type, label (REAL vs ATTACKED), clip_id
- Computes and logs localization IoU and corner error against official VIA annotations
- Comprehensive output logging: manifest.csv, failures.csv, summary.json
"""

import argparse
import csv
import json
import logging
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from shapely.geometry import Polygon

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.candidate_scorer import CandidateScorer
from src.preprocessing.document_detector import OpenCVContourDetector
from src.preprocessing.hybrid_document_detector import (
    HybridDocumentDetector,
    MIDV500Segmenter,
    VariantCAmbiguityGate,
)
from src.preprocessing.perspective_corrector import PerspectiveCorrector, order_points
from src.preprocessing.temporal_document_tracker import TemporalDocumentTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("BuildPublicDataset")

# Canonical Document-Level Split Definitions (Zero Leakage)
DOCUMENT_SPLITS = {
    "alb_id_00": "train",
    "alb_id_01": "train",
    "alb_id_02": "train",
    "alb_id_03": "train",
    "alb_id_04": "val",
    "alb_id_05": "val",
    "aze_passport_00": "test",
    "aze_passport_01": "test",
}

# Presentation attack mode to binary classification label
MODE_TO_LABEL = {
    "or": "REAL",
    "cc": "ATTACKED",
    "cg": "ATTACKED",
    "re": "ATTACKED",
}


def parse_via_ground_truth_quad(ann_file: Path, frame_filename: str) -> Optional[np.ndarray]:
    """Extract ground-truth 4-corner polygon from VIA annotation JSON."""
    if not ann_file.exists():
        return None
    try:
        with open(ann_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        meta = data.get("_via_img_metadata", {})
        for _, item in meta.items():
            if item.get("filename") == frame_filename:
                regions = item.get("regions", [])
                if not regions:
                    return None
                shape = regions[0].get("shape_attributes", {})
                if shape.get("name") == "polygon":
                    xs = shape.get("all_points_x", [])
                    ys = shape.get("all_points_y", [])
                    if len(xs) == 4 and len(ys) == 4:
                        pts = np.column_stack([xs, ys]).astype(np.float32)
                        return order_points(pts)
        return None
    except Exception as e:
        logger.debug(f"Failed parsing ground truth for {frame_filename}: {e}")
        return None


def compute_quadrilateral_iou(pts1: np.ndarray, pts2: np.ndarray) -> float:
    """Compute polygon Intersection over Union (IoU)."""
    try:
        p1 = Polygon(pts1)
        p2 = Polygon(pts2)
        if not p1.is_valid:
            p1 = p1.buffer(0)
        if not p2.is_valid:
            p2 = p2.buffer(0)
        inter = p1.intersection(p2).area
        union = p1.union(p2).area
        return float(inter / union) if union > 0 else 0.0
    except Exception:
        return 0.0


def compute_corner_error(pts_pred: np.ndarray, pts_gt: np.ndarray) -> float:
    """Compute mean Euclidean corner error in pixels."""
    p_pred = order_points(pts_pred)
    p_gt = order_points(pts_gt)
    return float(np.mean(np.linalg.norm(p_pred - p_gt, axis=1)))


def extract_clip_metadata(clip_folder_name: str) -> Tuple[str, str, str]:
    """
    Parse clip folder name (e.g. 'alb_id_00.cc0001') into:
    (document_id, attack_type, clip_id)
    """
    m = re.match(r"^([a-zA-Z0-9_]+)\.([a-z]{2})\d+$", clip_folder_name)
    if m:
        doc_id = m.group(1)
        mode = m.group(2)
        return doc_id, mode, clip_folder_name
    # Fallback heuristic
    parts = clip_folder_name.split(".")
    doc_id = parts[0] if len(parts) > 0 else "unknown"
    mode = parts[1][:2] if len(parts) > 1 else "unknown"
    return doc_id, mode, clip_folder_name


def build_normalized_dataset(
    dlc_dir: Path,
    output_dir: Path,
    weights_path: Path,
    max_frames_per_clip: Optional[int] = None,
    target_width: int = 600,
    target_height: int = 400,
    overwrite: bool = False,
    smoke_test: bool = False,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Generate normalized crops from DLC-2021 video frames.
    """
    frames_dir = dlc_dir / "frames"
    ann_dir = dlc_dir / "annotations"

    if not frames_dir.exists():
        raise FileNotFoundError(f"DLC frames directory not found: {frames_dir}")

    # Prepare output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        for label in ["REAL", "ATTACKED"]:
            (output_dir / split / label).mkdir(parents=True, exist_ok=True)

    # Initialize components
    logger.info("Initializing HybridDocumentDetector (Variant C Dual-Path Pristine)...")
    scorer = CandidateScorer()
    opencv_detector = OpenCVContourDetector()
    gate = VariantCAmbiguityGate()
    segmenter = MIDV500Segmenter(weights_path=weights_path, device=device, lazy_load=True)

    corrector = PerspectiveCorrector(
        target_width=target_width,
        target_height=target_height,
        corner_refinement=True,
    )

    clip_folders = sorted([f for f in frames_dir.iterdir() if f.is_dir()])
    logger.info(f"Discovered {len(clip_folders)} DLC-2021 clip folders in {frames_dir}")

    if smoke_test:
        logger.info("SMOKE TEST MODE ENABLED: Processing 2 frames per clip across all 32 clips (64 total frames).")
        max_frames_per_clip = 2

    manifest_rows: List[Dict[str, Any]] = []
    failure_rows: List[Dict[str, Any]] = []

    total_processed = 0
    total_succeeded = 0
    total_failed = 0
    unet_invocations = 0

    stats = {
        "by_split": {"train": 0, "val": 0, "test": 0},
        "by_label": {"REAL": 0, "ATTACKED": 0},
        "by_mode": {"or": 0, "cc": 0, "cg": 0, "re": 0},
        "by_method": {"opencv_v2": 0, "midv500_unet": 0},
    }

    t_start = time.perf_counter()

    for clip_idx, clip_folder in enumerate(clip_folders, 1):
        clip_name = clip_folder.name
        doc_id, mode, clip_id = extract_clip_metadata(clip_name)
        split = DOCUMENT_SPLITS.get(doc_id, "train")
        label = MODE_TO_LABEL.get(mode, "ATTACKED")
        ann_file = ann_dir / f"{clip_name}.json"

        # Per-clip temporal tracker to avoid bleeding state across different videos
        clip_tracker = TemporalDocumentTracker()
        hybrid_detector = HybridDocumentDetector(
            opencv_detector=opencv_detector,
            scorer=scorer,
            tracker=clip_tracker,
            segmenter=segmenter,
            gate=gate,
            weights_path=weights_path,
            lazy_load_segmenter=True,
        )

        frame_files = sorted(list(clip_folder.glob("*.jpg")))
        if max_frames_per_clip is not None:
            # Sample uniformly or take first N
            if len(frame_files) > max_frames_per_clip:
                indices = np.linspace(0, len(frame_files) - 1, max_frames_per_clip, dtype=int)
                frame_files = [frame_files[i] for i in indices]

        logger.info(
            f"[{clip_idx}/{len(clip_folders)}] Processing {clip_name} ({len(frame_files)} frames) "
            f"-> Split: {split}, Label: {label}"
        )

        for frame_file in frame_files:
            fname = frame_file.name
            frame_id = f"{clip_name}_{fname}"
            total_processed += 1

            img = cv2.imread(str(frame_file))
            if img is None:
                failure_rows.append({
                    "frame_id": frame_id,
                    "source_path": str(frame_file),
                    "document_id": doc_id,
                    "clip_id": clip_id,
                    "attack_type": mode,
                    "split": split,
                    "label": label,
                    "reason": "image_decode_failed",
                })
                total_failed += 1
                continue

            pts_gt = parse_via_ground_truth_quad(ann_file, fname)

            # Run Hybrid Document Localization
            t0 = time.perf_counter()
            det_res = hybrid_detector.detect_frame(img, frame_id=frame_id)
            det_time_ms = (time.perf_counter() - t0) * 1000.0

            detected = bool(det_res.get("detected", False))
            corners = det_res.get("corners")
            loc_method = det_res.get("localization_method", "none")
            gate_reason = det_res.get("gate_reason", "none")

            if loc_method == "midv500_unet":
                unet_invocations += 1

            if not detected or corners is None:
                failure_rows.append({
                    "frame_id": frame_id,
                    "source_path": str(frame_file),
                    "document_id": doc_id,
                    "clip_id": clip_id,
                    "attack_type": mode,
                    "split": split,
                    "label": label,
                    "reason": f"localization_failed: {det_res.get('reason', 'no_corners')}",
                })
                total_failed += 1
                continue

            # Run Perspective Correction to 600x400
            pts_array = np.array(corners, dtype=np.float32)
            corr_res = corrector.correct_perspective(img, corners=pts_array, frame_id=frame_id)

            if not corr_res.get("success", False) or corr_res.get("normalized_image") is None:
                failure_rows.append({
                    "frame_id": frame_id,
                    "source_path": str(frame_file),
                    "document_id": doc_id,
                    "clip_id": clip_id,
                    "attack_type": mode,
                    "split": split,
                    "label": label,
                    "reason": f"perspective_correction_failed: {corr_res.get('error', 'unknown')}",
                })
                total_failed += 1
                continue

            warped = corr_res["normalized_image"]

            # Save normalized image crop
            crop_filename = f"{clip_name}_{fname.replace('.jpg', '.png')}"
            crop_rel_path = f"{split}/{label}/{crop_filename}"
            crop_abs_path = output_dir / split / label / crop_filename

            if overwrite or not crop_abs_path.exists():
                cv2.imwrite(str(crop_abs_path), warped)

            # Compute QA metrics against ground truth if available
            iou_gt = 0.0
            corner_err_gt = float("nan")
            if pts_gt is not None:
                iou_gt = compute_quadrilateral_iou(pts_array, pts_gt)
                corner_err_gt = compute_corner_error(pts_array, pts_gt)

            manifest_rows.append({
                "frame_id": frame_id,
                "crop_path": crop_rel_path,
                "split": split,
                "label": label,
                "document_id": doc_id,
                "clip_id": clip_id,
                "attack_type": mode,
                "frame_filename": fname,
                "source_frame_path": str(frame_file).replace("\\", "/"),
                "localization_method": loc_method,
                "confidence": round(float(det_res.get("confidence", 1.0)), 4),
                "gate_reason": gate_reason,
                "iou_with_gt": round(iou_gt, 4),
                "corner_error_with_gt": round(corner_err_gt, 2) if not math.isnan(corner_err_gt) else "",
                "latency_ms": round(det_time_ms, 1),
                "width": target_width,
                "height": target_height,
            })

            total_succeeded += 1
            stats["by_split"][split] += 1
            stats["by_label"][label] += 1
            stats["by_mode"][mode] += 1
            stats["by_method"][loc_method] += 1

    total_time = time.perf_counter() - t_start

    # Export manifest.csv
    manifest_csv = output_dir / "manifest.csv"
    fieldnames = [
        "frame_id", "crop_path", "split", "label", "document_id", "clip_id",
        "attack_type", "frame_filename", "source_frame_path", "localization_method",
        "confidence", "gate_reason", "iou_with_gt", "corner_error_with_gt",
        "latency_ms", "width", "height"
    ]
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    # Export failures.csv
    failures_csv = output_dir / "failures.csv"
    fail_fields = ["frame_id", "source_path", "document_id", "clip_id", "attack_type", "split", "label", "reason"]
    with open(failures_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fail_fields)
        writer.writeheader()
        writer.writerows(failure_rows)

    # Export summary.json
    summary = {
        "total_source_frames": total_processed,
        "successfully_normalized": total_succeeded,
        "failed_frames": total_failed,
        "success_rate_pct": round((total_succeeded / total_processed * 100.0) if total_processed > 0 else 0.0, 2),
        "unet_invocation_count": unet_invocations,
        "unet_invocation_pct": round((unet_invocations / total_processed * 100.0) if total_processed > 0 else 0.0, 2),
        "elapsed_seconds": round(total_time, 2),
        "target_resolution": [target_width, target_height],
        "splits": stats["by_split"],
        "labels": stats["by_label"],
        "modes": stats["by_mode"],
        "methods": stats["by_method"],
        "document_split_mapping": DOCUMENT_SPLITS,
    }

    summary_json = output_dir / "summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 70)
    logger.info("PUBLIC DATASET NORMALIZATION COMPLETE")
    logger.info(f"Total processed:       {total_processed}")
    logger.info(f"Successfully cropped:  {total_succeeded} ({summary['success_rate_pct']}%)")
    logger.info(f"Failed frames:         {total_failed}")
    logger.info(f"UNet invocations:      {unet_invocations} ({summary['unet_invocation_pct']}%)")
    logger.info(f"Train / Val / Test:    {stats['by_split']['train']} / {stats['by_split']['val']} / {stats['by_split']['test']}")
    logger.info(f"REAL / ATTACKED:       {stats['by_label']['REAL']} / {stats['by_label']['ATTACKED']}")
    logger.info(f"Modes (or/cc/cg/re):   {stats['by_mode']['or']} / {stats['by_mode']['cc']} / {stats['by_mode']['cg']} / {stats['by_mode']['re']}")
    logger.info(f"Output directory:      {output_dir}")
    logger.info(f"Manifest path:         {manifest_csv}")
    logger.info(f"Failures path:         {failures_csv}")
    logger.info("=" * 70)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate normalized DLC-2021 dataset using Hybrid Variant C.")
    parser.add_argument("--dlc-dir", type=Path, default=Path("data/public/dlc2021"), help="DLC-2021 dataset directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/normalized_public_dataset"), help="Output directory for normalized crops.")
    parser.add_argument("--weights-path", type=Path, default=Path("models/weights/midv500_unet_resnet34.pth"), help="Path to MIDV-500 UNet weights.")
    parser.add_argument("--max-frames-per-clip", type=int, default=None, help="Max frames to sample per clip (None for all 50).")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick smoke test (2 frames per clip = 64 frames).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing image crops.")
    parser.add_argument("--device", type=str, default="cpu", help="PyTorch device ('cpu' or 'cuda').")
    args = parser.parse_args()

    build_normalized_dataset(
        dlc_dir=args.dlc_dir,
        output_dir=args.output_dir,
        weights_path=args.weights_path,
        max_frames_per_clip=args.max_frames_per_clip,
        overwrite=args.overwrite,
        smoke_test=args.smoke_test,
        device=args.device,
    )


if __name__ == "__main__":
    main()
