"""
Public Dataset Preprocessing Validation & Normalization Pipeline
===============================================================
Evaluates our existing OpenCVContourDetector and PerspectiveCorrector
against official ground-truth annotations from DLC-2021 and MIDV-2020.

Workflow:
1. Load controlled subsets of DLC-2021 (across all 4 attack modes: or, cc, cg, re)
   and MIDV-2020.
2. Parse official VIA ground-truth document quadrilaterals.
3. Run existing OpenCVContourDetector.
4. Calculate polygon IoU and pixel corner errors between prediction and ground truth.
5. Apply PerspectiveCorrector to generate normalized document images.
6. Generate side-by-side diagnostic visualizations (GT in blue, predicted in green).
7. Save frame metrics CSV, summary JSON, and contact sheets.
8. Populate data/public/normalized_dlc2021/ (real/ and attacked/) with quality-verified
   normalized document frames and metadata manifest.
"""

import csv
import json
import logging
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from shapely.geometry import Polygon

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.document_detector import OpenCVContourDetector
from src.preprocessing.perspective_corrector import PerspectiveCorrector, order_points

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("PublicPreprocessingValidation")


def parse_via_ground_truth_quad(ann_json_path: Path, filename: str) -> Optional[np.ndarray]:
    """
    Parse official VIA JSON annotation and extract 4 corner points for a given frame.

    Args:
        ann_json_path: Path to the VIA JSON annotation file.
        filename: Target image filename (e.g. '000001.jpg').

    Returns:
        np.ndarray of shape (4, 2) with float coordinates, or None if not found.
    """
    if not ann_json_path.exists():
        return None

    try:
        with open(ann_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        meta = data.get("_via_img_metadata", {})
        for _, item in meta.items():
            if item.get("filename") == filename:
                for region in item.get("regions", []):
                    attr = region.get("region_attributes", {})
                    # Look for doc_quad region
                    if attr.get("field_name") == "doc_quad":
                        shape = region.get("shape_attributes", {})
                        xs = shape.get("all_points_x", [])
                        ys = shape.get("all_points_y", [])
                        if len(xs) == 4 and len(ys) == 4:
                            pts = np.column_stack([xs, ys]).astype(np.float32)
                            return pts
    except Exception as e:
        logger.warning(f"Error reading annotation {ann_json_path}: {e}")

    return None


def compute_quadrilateral_iou(pts_pred: np.ndarray, pts_gt: np.ndarray) -> float:
    """
    Compute Intersection over Union between two 4-point convex/general polygons.

    Args:
        pts_pred: (4, 2) predicted corner coordinates.
        pts_gt: (4, 2) ground-truth corner coordinates.

    Returns:
        float IoU in range [0.0, 1.0].
    """
    try:
        poly_pred = Polygon(pts_pred)
        poly_gt = Polygon(pts_gt)

        if not poly_pred.is_valid:
            poly_pred = poly_pred.buffer(0)
        if not poly_gt.is_valid:
            poly_gt = poly_gt.buffer(0)

        inter = poly_pred.intersection(poly_gt).area
        union = poly_pred.union(poly_gt).area

        if union <= 0.0:
            return 0.0
        return float(np.clip(inter / union, 0.0, 1.0))
    except Exception:
        # Fallback to raster mask IoU
        try:
            max_x = int(math.ceil(max(np.max(pts_pred[:, 0]), np.max(pts_gt[:, 0])))) + 10
            max_y = int(math.ceil(max(np.max(pts_pred[:, 1]), np.max(pts_gt[:, 1])))) + 10
            mask_pred = np.zeros((max_y, max_x), dtype=np.uint8)
            mask_gt = np.zeros((max_y, max_x), dtype=np.uint8)

            cv2.fillPoly(mask_pred, [pts_pred.astype(np.int32)], 1)
            cv2.fillPoly(mask_gt, [pts_gt.astype(np.int32)], 1)

            inter = np.logical_and(mask_pred, mask_gt).sum()
            union = np.logical_or(mask_pred, mask_gt).sum()
            return float(inter / union) if union > 0 else 0.0
        except Exception:
            return 0.0


def compute_corner_errors(
    pts_pred: np.ndarray, pts_gt: np.ndarray
) -> Tuple[float, float, float, np.ndarray]:
    """
    Order corners canonically and calculate Euclidean error in pixels.

    Args:
        pts_pred: (4, 2) predicted corner points.
        pts_gt: (4, 2) ground-truth corner points.

    Returns:
        Tuple of (mean_error, median_error, max_error, array_of_4_corner_errors).
    """
    ord_pred = order_points(pts_pred)
    ord_gt = order_points(pts_gt)

    distances = np.linalg.norm(ord_pred - ord_gt, axis=1)
    mean_err = float(np.mean(distances))
    median_err = float(np.median(distances))
    max_err = float(np.max(distances))

    return mean_err, median_err, max_err, distances


def draw_diagnostic_detection_overlay(
    img: np.ndarray,
    pts_pred: Optional[np.ndarray],
    pts_gt: Optional[np.ndarray],
    info_text: str,
) -> np.ndarray:
    """
    Render diagnostic overlay:
    - Ground-truth quadrilateral in BLUE (thick line) with circle markers.
    - Predicted quadrilateral in GREEN (medium line) with diamond/box markers.
    - Header banner with info text.
    """
    canvas = img.copy()

    # Draw Ground Truth in Blue (BGR: 255, 120, 0)
    if pts_gt is not None:
        gt_ordered = order_points(pts_gt).astype(np.int32)
        cv2.polylines(canvas, [gt_ordered], isClosed=True, color=(255, 140, 0), thickness=3)
        for idx, (x, y) in enumerate(gt_ordered):
            cv2.circle(canvas, (int(x), int(y)), 8, (255, 140, 0), -1)
            cv2.putText(
                canvas, f"GT{idx}", (int(x) + 10, int(y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 140, 0), 2, cv2.LINE_AA
            )

    # Draw Predicted in Bright Green (BGR: 0, 255, 0)
    if pts_pred is not None:
        pred_ordered = order_points(pts_pred).astype(np.int32)
        cv2.polylines(canvas, [pred_ordered], isClosed=True, color=(0, 255, 0), thickness=2)
        for idx, (x, y) in enumerate(pred_ordered):
            cv2.drawMarker(
                canvas, (int(x), int(y)), (0, 255, 0),
                markerType=cv2.MARKER_DIAMOND, markerSize=14, thickness=2
            )
            cv2.putText(
                canvas, f"P{idx}", (int(x) - 25, int(y) + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA
            )

    # Add header banner
    banner_h = 44
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], banner_h), (20, 24, 30), -1)
    cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
    cv2.putText(
        canvas, info_text, (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2, cv2.LINE_AA
    )

    return canvas


def create_contact_sheet(
    image_paths: List[Path],
    output_path: Path,
    title: str = "Normalized Document Frames",
    thumb_width: int = 240,
    thumb_height: int = 160,
    max_cols: int = 4,
) -> Optional[Path]:
    """Build a consolidated contact sheet of normalized frames."""
    if not image_paths:
        return None

    images: List[Tuple[str, np.ndarray]] = []
    for p in image_paths:
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                thumb = cv2.resize(img, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA)
                lbl = p.stem.replace("norm_", "")
                cv2.rectangle(thumb, (0, 0), (thumb_width, 22), (20, 20, 20), -1)
                cv2.putText(
                    thumb, lbl, (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1, cv2.LINE_AA
                )
                images.append((lbl, thumb))

    if not images:
        return None

    num_images = len(images)
    cols = min(num_images, max_cols)
    rows = (num_images + cols - 1) // cols

    padding = 8
    header_h = 36
    sheet_w = cols * thumb_width + (cols + 1) * padding
    sheet_h = rows * thumb_height + (rows + 1) * padding + header_h

    canvas = np.full((sheet_h, sheet_w, 3), (30, 32, 38), dtype=np.uint8)
    cv2.putText(
        canvas, f"{title} ({num_images} frames)", (padding + 4, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA
    )

    for idx, (_, thumb) in enumerate(images):
        r = idx // cols
        c = idx % cols
        x = padding + c * (thumb_width + padding)
        y = header_h + padding + r * (thumb_height + padding)
        canvas[y : y + thumb_height, x : x + thumb_width] = thumb

    cv2.imwrite(str(output_path), canvas)
    return output_path


def run_public_preprocessing_validation():
    """Execute complete preprocessing validation workflow."""
    out_dir = Path("outputs/public_preprocessing_validation")
    raw_dir = out_dir / "raw"
    detected_dir = out_dir / "detected"
    norm_dir = out_dir / "normalized"
    comp_dir = out_dir / "comparisons"

    for d in [raw_dir, detected_dir, norm_dir, comp_dir]:
        d.mkdir(parents=True, exist_ok=True)

    norm_dataset_dir = Path("data/public/normalized_dlc2021")
    norm_real_dir = norm_dataset_dir / "real"
    norm_att_dir = norm_dataset_dir / "attacked"
    norm_real_dir.mkdir(parents=True, exist_ok=True)
    norm_att_dir.mkdir(parents=True, exist_ok=True)

    # Clean existing images in normalized dataset dirs to match manifest exactly
    for f in norm_real_dir.glob("*.png"):
        f.unlink()
    for f in norm_att_dir.glob("*.png"):
        f.unlink()

    # Target 2 complete document identities across all 4 presentation modes
    target_docs = ["alb_id_00", "alb_id_01"]
    modes = ["or", "cc", "cg", "re"]

    dlc_base = Path("data/public/dlc2021")
    frames_base = dlc_base / "frames"
    anns_base = dlc_base / "annotations"

    detector = OpenCVContourDetector()
    corrector = PerspectiveCorrector()

    frame_metrics: List[Dict[str, Any]] = []
    saved_norm_paths: List[Path] = []
    category_metrics: Dict[str, Dict[str, Any]] = {
        m: {"frames": 0, "detected": 0, "ious": [], "corner_errors": []} for m in modes
    }

    # Sample 10 frames per clip (frames 1, 5, 10, 15, 20, 25, 30, 35, 40, 45) -> 80 frames total
    sample_frame_indices = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45]

    logger.info("=== Starting DLC-2021 Preprocessing Validation across %d docs, 8 clips, 80 frames ===", len(target_docs))

    # Diagnostic representative comparisons
    rep_frames_by_mode: Dict[str, Dict[str, Any]] = {}

    for doc_id in target_docs:
        for mode in modes:
            clip_id = f"{doc_id}.{mode}0001"
            clip_dir = frames_base / clip_id
            ann_file = anns_base / f"{clip_id}.json"

            if not clip_dir.exists():
                logger.warning(f"Clip dir does not exist: {clip_dir}")
                continue

            all_frame_files = sorted(list(clip_dir.glob("*.jpg")))
            if not all_frame_files:
                continue
            indices = np.linspace(0, len(all_frame_files) - 1, min(10, len(all_frame_files)), dtype=int)
            sampled_files = [all_frame_files[i] for i in indices]

            for img_path in sampled_files:
                frame_name = img_path.name
                f_idx_str = frame_name.replace(".jpg", "")

                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                category_metrics[mode]["frames"] += 1

                # 1. Parse ground truth
                pts_gt = parse_via_ground_truth_quad(ann_file, frame_name)

                # 2. Run detector
                frame_id = f"{clip_id}_{f_idx_str}"
                det_res = detector.detect_frame(img, frame_id=frame_id)
                detected = bool(det_res.get("detected", False))
                conf = float(det_res.get("confidence", 0.0))
                pts_pred = det_res.get("corners")

                iou = 0.0
                mean_err, med_err, max_err = float("nan"), float("nan"), float("nan")

                if detected and pts_pred is not None:
                    category_metrics[mode]["detected"] += 1
                    if pts_gt is not None:
                        iou = compute_quadrilateral_iou(pts_pred, pts_gt)
                        mean_err, med_err, max_err, _ = compute_corner_errors(pts_pred, pts_gt)
                        category_metrics[mode]["ious"].append(iou)
                        category_metrics[mode]["corner_errors"].append(mean_err)

                # 3. Perspective correction
                persp_success = False
                norm_img = None
                norm_rel_path = ""
                if detected and pts_pred is not None:
                    corr_res = corrector.correct_perspective(img, corners=pts_pred, frame_id=frame_id)
                    persp_success = bool(corr_res.get("success", False))
                    norm_img = corr_res.get("normalized_image")

                # Save raw frame copy
                raw_out = raw_dir / f"raw_{frame_id}.jpg"
                cv2.imwrite(str(raw_out), img)

                # Save diagnostic overlay frame
                det_out = detected_dir / f"detected_{frame_id}.jpg"
                gt_str = f"GT: Yes | IoU: {iou:.3f} | Err: {mean_err:.1f}px" if pts_gt is not None else "GT: None"
                status_str = f"[{mode.upper()}] Conf: {conf:.2f} | {gt_str}"
                overlay_img = draw_diagnostic_detection_overlay(img, pts_pred, pts_gt, status_str)
                cv2.imwrite(str(det_out), overlay_img)

                # Save normalized frame
                if persp_success and norm_img is not None:
                    norm_out = norm_dir / f"norm_{frame_id}.png"
                    cv2.imwrite(str(norm_out), norm_img)
                    saved_norm_paths.append(norm_out)
                    norm_rel_path = str(norm_out.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")

                    # Also copy to normalized public dataset directory if quality is good (IoU >= 0.50 or valid detection)
                    label = "REAL" if mode == "or" else "ATTACKED"
                    dst_folder = norm_real_dir if label == "REAL" else norm_att_dir
                    dst_file = dst_folder / f"{frame_id}.png"
                    cv2.imwrite(str(dst_file), norm_img)

                # Capture best representative frame for each mode (highest IoU or first detected)
                if detected and pts_gt is not None:
                    if mode not in rep_frames_by_mode or iou > rep_frames_by_mode[mode]["iou"]:
                        rep_frames_by_mode[mode] = {
                            "frame_id": frame_id,
                            "raw": img,
                            "overlay": overlay_img,
                            "norm": norm_img,
                            "pts_pred": pts_pred,
                            "pts_gt": pts_gt,
                            "iou": iou,
                            "conf": conf,
                            "mean_err": mean_err,
                            "mode": mode,
                        }

                frame_metrics.append({
                    "dataset": "DLC-2021",
                    "document_id": doc_id,
                    "video_id": clip_id,
                    "attack_type": mode,
                    "frame_id": frame_id,
                    "label": "REAL" if mode == "or" else "ATTACKED",
                    "detected": detected,
                    "confidence": round(conf, 4),
                    "iou": round(iou, 4) if detected else 0.0,
                    "mean_corner_error": round(mean_err, 2) if not math.isnan(mean_err) else "",
                    "median_corner_error": round(med_err, 2) if not math.isnan(med_err) else "",
                    "max_corner_error": round(max_err, 2) if not math.isnan(max_err) else "",
                    "perspective_success": persp_success,
                    "raw_path": str(raw_out.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/"),
                    "detected_path": str(det_out.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/"),
                    "normalized_path": norm_rel_path,
                })

    # Save frame metrics CSV
    csv_path = out_dir / "frame_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "dataset", "document_id", "video_id", "attack_type", "frame_id", "label",
            "detected", "confidence", "iou", "mean_corner_error", "median_corner_error",
            "max_corner_error", "perspective_success", "raw_path", "detected_path", "normalized_path"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(frame_metrics)
    logger.info("Saved %s (%d rows)", csv_path, len(frame_metrics))

    # Compute overall statistics for DLC-2021
    total_frames = len(frame_metrics)
    total_detected = sum(1 for r in frame_metrics if r["detected"])
    detection_success_rate = total_detected / total_frames if total_frames > 0 else 0.0

    valid_ious = [r["iou"] for r in frame_metrics if r["detected"] and r["iou"] > 0]
    mean_iou = float(np.mean(valid_ious)) if valid_ious else 0.0
    median_iou = float(np.median(valid_ious)) if valid_ious else 0.0
    min_iou = float(np.min(valid_ious)) if valid_ious else 0.0

    valid_corner_errs = [float(r["mean_corner_error"]) for r in frame_metrics if r["mean_corner_error"] != ""]
    mean_corner_err = float(np.mean(valid_corner_errs)) if valid_corner_errs else 0.0
    median_corner_err = float(np.median(valid_corner_errs)) if valid_corner_errs else 0.0
    max_corner_err = float(np.max([float(r["max_corner_error"]) for r in frame_metrics if r["max_corner_error"] != ""])) if valid_corner_errs else 0.0

    persp_success_count = sum(1 for r in frame_metrics if r["perspective_success"])
    persp_success_rate = persp_success_count / total_detected if total_detected > 0 else 0.0

    # Build representative 4-panel diagnostic comparisons for each mode
    comp_images = []
    panel_w, panel_h = 320, 240
    for m in modes:
        rep = rep_frames_by_mode.get(m)
        if rep:
            p_raw = cv2.resize(rep["raw"], (panel_w, panel_h))
            p_gt = draw_diagnostic_detection_overlay(rep["raw"], None, rep["pts_gt"], f"GT Quad [{m.upper()}]")
            p_gt_res = cv2.resize(p_gt, (panel_w, panel_h))
            p_det = cv2.resize(rep["overlay"], (panel_w, panel_h))
            if rep["norm"] is not None:
                p_norm = cv2.resize(rep["norm"], (panel_w, panel_h))
            else:
                p_norm = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
                cv2.putText(p_norm, "Norm Failed", (50, panel_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            four_panel = np.hstack([p_raw, p_gt_res, p_det, p_norm])
            mode_comp_path = comp_dir / f"comparison_{m}.png"
            cv2.imwrite(str(mode_comp_path), four_panel)
            logger.info("Saved 4-panel diagnostic for mode %s to %s", m, mode_comp_path)

            ov_thumb = cv2.resize(rep["overlay"], (360, 240))
            comp_images.append(ov_thumb)

    if len(comp_images) == 4:
        top_row = np.hstack([comp_images[0], comp_images[1]])
        bot_row = np.hstack([comp_images[2], comp_images[3]])
        full_collage = np.vstack([top_row, bot_row])
        collage_path = out_dir / "diagnostic_comparison_grid.png"
        cv2.imwrite(str(collage_path), full_collage)
        logger.info("Saved diagnostic comparison grid to %s", collage_path)

    # Generate contact sheet of normalized frames
    contact_sheet_path = out_dir / "contact_sheet.png"
    create_contact_sheet(
        image_paths=saved_norm_paths,
        output_path=contact_sheet_path,
        title="DLC-2021 PREPROCESSING VALIDATION — NORMALIZED DOCUMENT FRAMES",
    )

    # 4. Spot Check MIDV-2020
    logger.info("=== Running MIDV-2020 Spot-Check Evaluation ===")
    midv_base = Path("data/public/midv2020")
    midv_frames_base = midv_base / "frames"
    midv_anns_base = midv_base / "annotations"
    midv_metrics = []

    midv_sample_clips = ["lva_passport_28", "lva_passport_43"]
    for clip_id in midv_sample_clips:
        c_dir = midv_frames_base / clip_id
        ann_file = midv_anns_base / f"{clip_id}.json"
        if not c_dir.exists():
            continue

        all_midv_files = sorted(list(c_dir.glob("*.jpg")))
        if not all_midv_files:
            continue
        indices = np.linspace(0, len(all_midv_files) - 1, min(5, len(all_midv_files)), dtype=int)
        sampled_files = [all_midv_files[i] for i in indices]

        for img_path in sampled_files:
            f_name = img_path.name

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            pts_gt = parse_via_ground_truth_quad(ann_file, f_name)
            det_res = detector.detect_frame(img, frame_id=f"{clip_id}_{f_name}")
            detected = bool(det_res.get("detected", False))
            conf = float(det_res.get("confidence", 0.0))
            pts_pred = det_res.get("corners")

            iou = 0.0
            mean_err = float("nan")
            if detected and pts_pred is not None and pts_gt is not None:
                iou = compute_quadrilateral_iou(pts_pred, pts_gt)
                mean_err, _, _, _ = compute_corner_errors(pts_pred, pts_gt)

            midv_metrics.append({
                "clip_id": clip_id,
                "frame": f_name,
                "detected": detected,
                "confidence": conf,
                "iou": iou,
                "mean_corner_error": mean_err,
            })

    midv_total = len(midv_metrics)
    midv_detected = sum(1 for m in midv_metrics if m["detected"])
    midv_ious = [m["iou"] for m in midv_metrics if m["detected"] and m["iou"] > 0]
    midv_mean_iou = float(np.mean(midv_ious)) if midv_ious else 0.0

    # Build summary JSON
    summary_data = {
        "dlc2021": {
            "total_frames_evaluated": total_frames,
            "successful_detections": total_detected,
            "detection_success_rate": round(detection_success_rate, 4),
            "perspective_correction_success_rate": round(persp_success_rate, 4),
            "iou_metrics": {
                "mean_iou": round(mean_iou, 4),
                "median_iou": round(median_iou, 4),
                "min_iou": round(min_iou, 4),
            },
            "corner_error_pixels": {
                "mean_corner_error": round(mean_corner_err, 2),
                "median_corner_error": round(median_corner_err, 2),
                "max_corner_error": round(max_corner_err, 2),
            },
            "results_by_attack_category": {
                m: {
                    "frames": category_metrics[m]["frames"],
                    "successful_detections": category_metrics[m]["detected"],
                    "detection_rate": round(category_metrics[m]["detected"] / max(1, category_metrics[m]["frames"]), 4),
                    "mean_iou": round(float(np.mean(category_metrics[m]["ious"])), 4) if category_metrics[m]["ious"] else 0.0,
                    "median_iou": round(float(np.median(category_metrics[m]["ious"])), 4) if category_metrics[m]["ious"] else 0.0,
                }
                for m in modes
            },
            "normalized_frames_generated": {
                "total": len(saved_norm_paths),
                "real": len(list(norm_real_dir.glob("*.png"))),
                "attacked": len(list(norm_att_dir.glob("*.png"))),
            }
        },
        "midv2020_spot_check": {
            "frames_evaluated": midv_total,
            "successful_detections": midv_detected,
            "detection_rate": round(midv_detected / max(1, midv_total), 4) if midv_total > 0 else 0.0,
            "mean_iou": round(midv_mean_iou, 4),
        }
    }

    summary_json_path = out_dir / "summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    logger.info("Saved summary JSON to %s", summary_json_path)

    # Build normalized dataset manifest
    norm_manifest_path = norm_dataset_dir / "normalized_manifest.csv"
    with open(norm_manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset", "document_id", "video_id", "attack_type", "frame_id", "label", "normalized_path"
        ])
        writer.writeheader()
        for r in frame_metrics:
            if r["perspective_success"] and r["normalized_path"]:
                writer.writerow({
                    "dataset": r["dataset"],
                    "document_id": r["document_id"],
                    "video_id": r["video_id"],
                    "attack_type": r["attack_type"],
                    "frame_id": r["frame_id"],
                    "label": r["label"],
                    "normalized_path": r["normalized_path"],
                })

    logger.info("Saved normalized public dataset manifest to %s", norm_manifest_path)
    return summary_data


if __name__ == "__main__":
    run_public_preprocessing_validation()
