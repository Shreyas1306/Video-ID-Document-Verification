"""
Public Dataset Preprocessing Validation V2: Candidate Scoring & Temporal Tracking
=================================================================================
Evaluates the enhanced localization pipeline combining:
1. OpenCVContourDetector (candidate generation)
2. CandidateScorer (aspect ratio, area, rectangularity, edge quality, nesting analysis)
3. TemporalDocumentTracker (multi-frame continuity, corner smoothing, drift prevention)
4. PerspectiveCorrector (homography normalization)

Compares frame-by-frame against the V1 baseline detector on identical frame samples
from DLC-2021 (or, cc, cg, re) and MIDV-2020.
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validate_public_preprocessing import (
    compute_corner_errors,
    compute_quadrilateral_iou,
    create_contact_sheet,
    parse_via_ground_truth_quad,
)
from src.preprocessing.candidate_scorer import CandidateScorer
from src.preprocessing.document_detector import OpenCVContourDetector
from src.preprocessing.perspective_corrector import PerspectiveCorrector, order_points
from src.preprocessing.temporal_document_tracker import TemporalDocumentTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("PreprocessingValidationV2")


def draw_v1_v2_diagnostic_overlay(
    img: np.ndarray,
    pts_gt: Optional[np.ndarray],
    pts_baseline: Optional[np.ndarray],
    pts_v2: Optional[np.ndarray],
    info_text: str,
) -> np.ndarray:
    """
    Render diagnostic overlay comparing Baseline vs V2 vs Ground Truth:
    - Ground Truth: Blue contour (BGR: 255, 140, 0) with circular markers.
    - Baseline (V1): Orange contour (BGR: 0, 165, 255) with cross markers.
    - V2 (Enhanced): Green contour (BGR: 0, 255, 0) with diamond markers.
    - Info banner on top.
    """
    canvas = img.copy()

    # 1. Ground Truth in Blue
    if pts_gt is not None:
        gt_ordered = order_points(pts_gt).astype(np.int32)
        cv2.polylines(canvas, [gt_ordered], isClosed=True, color=(255, 140, 0), thickness=3)
        for idx, (x, y) in enumerate(gt_ordered):
            cv2.circle(canvas, (int(x), int(y)), 7, (255, 140, 0), -1)

    # 2. Baseline (V1) in Orange/Red
    if pts_baseline is not None:
        base_ordered = order_points(pts_baseline).astype(np.int32)
        cv2.polylines(canvas, [base_ordered], isClosed=True, color=(0, 140, 255), thickness=2)
        for idx, (x, y) in enumerate(base_ordered):
            cv2.drawMarker(canvas, (int(x), int(y)), (0, 140, 255), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)

    # 3. V2 (Enhanced) in Bright Green
    if pts_v2 is not None:
        v2_ordered = order_points(pts_v2).astype(np.int32)
        cv2.polylines(canvas, [v2_ordered], isClosed=True, color=(0, 255, 0), thickness=2)
        for idx, (x, y) in enumerate(v2_ordered):
            cv2.drawMarker(canvas, (int(x), int(y)), (0, 255, 0), markerType=cv2.MARKER_DIAMOND, markerSize=14, thickness=2)

    # Banner
    banner_h = 44
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], banner_h), (20, 24, 30), -1)
    cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
    cv2.putText(
        canvas, info_text, (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (240, 240, 240), 2, cv2.LINE_AA
    )
    return canvas


def run_public_preprocessing_validation_v2():
    out_dir = Path("outputs/public_preprocessing_validation_v2")
    raw_dir = out_dir / "raw"
    detected_dir = out_dir / "detected"
    norm_dir = out_dir / "normalized"
    comp_dir = out_dir / "comparisons"

    for d in [raw_dir, detected_dir, norm_dir, comp_dir]:
        d.mkdir(parents=True, exist_ok=True)

    target_docs = ["alb_id_00", "alb_id_01"]
    modes = ["or", "cc", "cg", "re"]

    dlc_base = Path("data/public/dlc2021")
    frames_base = dlc_base / "frames"
    anns_base = dlc_base / "annotations"

    detector = OpenCVContourDetector()
    scorer = CandidateScorer()
    corrector = PerspectiveCorrector()

    frame_metrics: List[Dict[str, Any]] = []
    saved_norm_paths: List[Path] = []

    category_stats = {
        m: {
            "frames": 0,
            "v1_detected": 0, "v1_ious": [], "v1_corner_errs": [], "v1_persp": 0,
            "v2_detected": 0, "v2_ious": [], "v2_corner_errs": [], "v2_persp": 0,
        }
        for m in modes
    }

    rep_frames_by_mode: Dict[str, Dict[str, Any]] = {}

    logger.info("=== Running Preprocessing Validation V2 across %d docs, 8 clips, 80 frames ===", len(target_docs))

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

            # 10 evenly spaced frames per clip
            indices = np.linspace(0, len(all_frame_files) - 1, min(10, len(all_frame_files)), dtype=int)
            sampled_files = [all_frame_files[i] for i in indices]

            # Initialize a fresh TemporalDocumentTracker for each video clip
            tracker = TemporalDocumentTracker(scorer=scorer)

            for img_path in sampled_files:
                frame_name = img_path.name
                f_idx_str = frame_name.replace(".jpg", "")
                frame_id = f"{clip_id}_{f_idx_str}"

                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                category_stats[mode]["frames"] += 1

                # Ground truth
                pts_gt = parse_via_ground_truth_quad(ann_file, frame_name)

                # --- 1. Baseline (V1) Evaluation ---
                candidates = detector.detect_candidates(img, frame_id=frame_id)
                v1_detected = False
                v1_conf = 0.0
                pts_v1 = None
                v1_iou = 0.0
                v1_mean_err = float("nan")

                if candidates:
                    # Baseline sorting: confidence descending, area descending
                    sorted_baseline = sorted(candidates, key=lambda c: (c["confidence"], c["area"]), reverse=True)
                    best_v1 = sorted_baseline[0]
                    if best_v1["confidence"] >= detector.confidence_threshold:
                        v1_detected = True
                        v1_conf = float(best_v1["confidence"])
                        pts_v1 = best_v1["corners"]
                        category_stats[mode]["v1_detected"] += 1
                        if pts_gt is not None:
                            v1_iou = compute_quadrilateral_iou(pts_v1, pts_gt)
                            v1_mean_err, _, _, _ = compute_corner_errors(pts_v1, pts_gt)
                            category_stats[mode]["v1_ious"].append(v1_iou)
                            category_stats[mode]["v1_corner_errs"].append(v1_mean_err)

                # --- 2. V2 Evaluation (Candidate Scorer + Temporal Tracker) ---
                track_res = tracker.process_frame(candidates, img, frame_id=frame_id)
                v2_detected = bool(track_res.get("detected", False))
                v2_conf = float(track_res.get("confidence", 0.0))
                v2_static = float(track_res.get("static_score", 0.0))
                v2_temporal = float(track_res.get("temporal_score", 0.0))
                v2_method = str(track_res.get("method", "none"))
                pts_v2 = np.array(track_res["corners"], dtype=np.float32) if v2_detected and track_res["corners"] else None

                v2_iou = 0.0
                v2_mean_err = float("nan")
                v2_med_err = float("nan")
                v2_max_err = float("nan")

                if v2_detected and pts_v2 is not None:
                    category_stats[mode]["v2_detected"] += 1
                    if pts_gt is not None:
                        v2_iou = compute_quadrilateral_iou(pts_v2, pts_gt)
                        v2_mean_err, v2_med_err, v2_max_err, _ = compute_corner_errors(pts_v2, pts_gt)
                        category_stats[mode]["v2_ious"].append(v2_iou)
                        category_stats[mode]["v2_corner_errs"].append(v2_mean_err)

                # Perspective correction for V2
                persp_v2_success = False
                norm_img = None
                norm_rel_path = ""
                if v2_detected and pts_v2 is not None:
                    corr_res = corrector.correct_perspective(img, corners=pts_v2, frame_id=frame_id)
                    persp_v2_success = bool(corr_res.get("success", False))
                    norm_img = corr_res.get("normalized_image")
                    if persp_v2_success:
                        category_stats[mode]["v2_persp"] += 1

                # Save raw frame
                raw_out = raw_dir / f"raw_{frame_id}.jpg"
                cv2.imwrite(str(raw_out), img)

                # Save diagnostic overlay
                det_out = detected_dir / f"detected_{frame_id}.jpg"
                gt_txt = f"GT: Blue | V1 IoU: {v1_iou:.2f} | V2 IoU: {v2_iou:.2f} ({v2_method})"
                overlay_img = draw_v1_v2_diagnostic_overlay(img, pts_gt, pts_v1, pts_v2, gt_txt)
                cv2.imwrite(str(det_out), overlay_img)

                # Save normalized frame
                if persp_v2_success and norm_img is not None:
                    norm_out = norm_dir / f"norm_{frame_id}.png"
                    cv2.imwrite(str(norm_out), norm_img)
                    saved_norm_paths.append(norm_out)
                    norm_rel_path = str(norm_out.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")

                # Best representative frame for diagnostic collage
                if v2_detected and pts_gt is not None:
                    if mode not in rep_frames_by_mode or v2_iou > rep_frames_by_mode[mode]["v2_iou"]:
                        rep_frames_by_mode[mode] = {
                            "frame_id": frame_id,
                            "raw": img,
                            "overlay": overlay_img,
                            "norm": norm_img,
                            "pts_gt": pts_gt,
                            "pts_v1": pts_v1,
                            "pts_v2": pts_v2,
                            "v1_iou": v1_iou,
                            "v2_iou": v2_iou,
                            "v2_mean_err": v2_mean_err,
                            "mode": mode,
                        }

                frame_metrics.append({
                    "dataset": "DLC-2021",
                    "document_id": doc_id,
                    "video_id": clip_id,
                    "attack_type": mode,
                    "frame_id": frame_id,
                    "label": "REAL" if mode == "or" else "ATTACKED",
                    # Baseline metrics
                    "baseline_detected": v1_detected,
                    "baseline_confidence": round(v1_conf, 4),
                    "baseline_iou": round(v1_iou, 4),
                    "baseline_mean_corner_error": round(v1_mean_err, 2) if not math.isnan(v1_mean_err) else "",
                    # V2 metrics
                    "v2_detected": v2_detected,
                    "v2_confidence": round(v2_conf, 4),
                    "v2_static_score": round(v2_static, 4),
                    "v2_temporal_score": round(v2_temporal, 4),
                    "v2_method": v2_method,
                    "v2_iou": round(v2_iou, 4),
                    "v2_mean_corner_error": round(v2_mean_err, 2) if not math.isnan(v2_mean_err) else "",
                    "v2_median_corner_error": round(v2_med_err, 2) if not math.isnan(v2_med_err) else "",
                    "v2_max_corner_error": round(v2_max_err, 2) if not math.isnan(v2_max_err) else "",
                    "v2_perspective_success": persp_v2_success,
                    "raw_path": str(raw_out.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/"),
                    "detected_path": str(det_out.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/"),
                    "normalized_path": norm_rel_path,
                })

    # Save frame metrics CSV
    csv_path = out_dir / "frame_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "dataset", "document_id", "video_id", "attack_type", "frame_id", "label",
            "baseline_detected", "baseline_confidence", "baseline_iou", "baseline_mean_corner_error",
            "v2_detected", "v2_confidence", "v2_static_score", "v2_temporal_score", "v2_method",
            "v2_iou", "v2_mean_corner_error", "v2_median_corner_error", "v2_max_corner_error",
            "v2_perspective_success", "raw_path", "detected_path", "normalized_path"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(frame_metrics)
    logger.info("Saved %s (%d rows)", csv_path, len(frame_metrics))

    # Overall Metrics Computation
    total_frames = len(frame_metrics)

    # V1 Overall
    v1_det_count = sum(1 for r in frame_metrics if r["baseline_detected"])
    v1_det_rate = v1_det_count / total_frames if total_frames > 0 else 0.0
    v1_ious = [r["baseline_iou"] for r in frame_metrics if r["baseline_detected"] and r["baseline_iou"] > 0]
    v1_mean_iou = float(np.mean(v1_ious)) if v1_ious else 0.0
    v1_med_iou = float(np.median(v1_ious)) if v1_ious else 0.0
    v1_min_iou = float(np.min(v1_ious)) if v1_ious else 0.0
    v1_errs = [float(r["baseline_mean_corner_error"]) for r in frame_metrics if r["baseline_mean_corner_error"] != ""]
    v1_mean_err = float(np.mean(v1_errs)) if v1_errs else 0.0
    v1_med_err = float(np.median(v1_errs)) if v1_errs else 0.0

    # V2 Overall
    v2_det_count = sum(1 for r in frame_metrics if r["v2_detected"])
    v2_det_rate = v2_det_count / total_frames if total_frames > 0 else 0.0
    v2_ious = [r["v2_iou"] for r in frame_metrics if r["v2_detected"] and r["v2_iou"] > 0]
    v2_mean_iou = float(np.mean(v2_ious)) if v2_ious else 0.0
    v2_med_iou = float(np.median(v2_ious)) if v2_ious else 0.0
    v2_min_iou = float(np.min(v2_ious)) if v2_ious else 0.0
    v2_errs = [float(r["v2_mean_corner_error"]) for r in frame_metrics if r["v2_mean_corner_error"] != ""]
    v2_mean_err = float(np.mean(v2_errs)) if v2_errs else 0.0
    v2_med_err = float(np.median(v2_errs)) if v2_errs else 0.0
    v2_persp_count = sum(1 for r in frame_metrics if r["v2_perspective_success"])
    v2_persp_rate = v2_persp_count / v2_det_count if v2_det_count > 0 else 0.0

    # Diagnostic 4-panel comparisons
    panel_w, panel_h = 320, 240
    collage_tiles = []
    for m in modes:
        rep = rep_frames_by_mode.get(m)
        if rep:
            p_raw = cv2.resize(rep["raw"], (panel_w, panel_h))
            p_gt = draw_v1_v2_diagnostic_overlay(rep["raw"], rep["pts_gt"], None, None, f"GT Ground Truth [{m.upper()}]")
            p_gt_res = cv2.resize(p_gt, (panel_w, panel_h))
            p_overlay = cv2.resize(rep["overlay"], (panel_w, panel_h))
            if rep["norm"] is not None:
                p_norm = cv2.resize(rep["norm"], (panel_w, panel_h))
            else:
                p_norm = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
                cv2.putText(p_norm, "Norm Failed", (40, panel_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            four_panel = np.hstack([p_raw, p_gt_res, p_overlay, p_norm])
            mode_comp_path = comp_dir / f"comparison_{m}.png"
            cv2.imwrite(str(mode_comp_path), four_panel)
            logger.info("Saved 4-panel diagnostic for mode %s to %s", m, mode_comp_path)

            ov_thumb = cv2.resize(rep["overlay"], (360, 240))
            collage_tiles.append(ov_thumb)

    if len(collage_tiles) == 4:
        top_row = np.hstack([collage_tiles[0], collage_tiles[1]])
        bot_row = np.hstack([collage_tiles[2], collage_tiles[3]])
        full_collage = np.vstack([top_row, bot_row])
        collage_path = out_dir / "diagnostic_comparison_grid.png"
        cv2.imwrite(str(collage_path), full_collage)
        logger.info("Saved diagnostic comparison grid to %s", collage_path)

    # Contact sheet
    contact_sheet_path = out_dir / "contact_sheet.png"
    create_contact_sheet(
        image_paths=saved_norm_paths,
        output_path=contact_sheet_path,
        title="DLC-2021 V2 PREPROCESSING VALIDATION — NORMALIZED DOCUMENT FRAMES",
    )

    # MIDV-2020 Spot Check
    logger.info("=== Running MIDV-2020 Spot-Check Evaluation V2 ===")
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

        midv_tracker = TemporalDocumentTracker(scorer=scorer)

        for img_path in sampled_files:
            f_name = img_path.name
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            pts_gt = parse_via_ground_truth_quad(ann_file, f_name)
            cands = detector.detect_candidates(img, frame_id=f"{clip_id}_{f_name}")
            track_res = midv_tracker.process_frame(cands, img, frame_id=f"{clip_id}_{f_name}")

            detected = bool(track_res.get("detected", False))
            conf = float(track_res.get("confidence", 0.0))
            pts_v2 = np.array(track_res["corners"], dtype=np.float32) if detected and track_res["corners"] else None

            iou = 0.0
            mean_err = float("nan")
            if detected and pts_v2 is not None and pts_gt is not None:
                iou = compute_quadrilateral_iou(pts_v2, pts_gt)
                mean_err, _, _, _ = compute_corner_errors(pts_v2, pts_gt)

            midv_metrics.append({
                "clip_id": clip_id,
                "frame": f_name,
                "detected": detected,
                "confidence": conf,
                "iou": iou,
                "mean_corner_error": mean_err,
            })

    midv_total = len(midv_metrics)
    midv_det = sum(1 for m in midv_metrics if m["detected"])
    midv_ious = [m["iou"] for m in midv_metrics if m["detected"] and m["iou"] > 0]
    midv_mean_iou = float(np.mean(midv_ious)) if midv_ious else 0.0

    # Build Comprehensive Summary JSON
    summary_data = {
        "overall_comparison_v1_vs_v2": {
            "total_frames_evaluated": total_frames,
            "v1_baseline": {
                "detection_success_rate": round(v1_det_rate, 4),
                "mean_iou": round(v1_mean_iou, 4),
                "median_iou": round(v1_med_iou, 4),
                "min_iou": round(v1_min_iou, 4),
                "mean_corner_error_px": round(v1_mean_err, 2),
                "median_corner_error_px": round(v1_med_err, 2),
            },
            "v2_enhanced": {
                "detection_success_rate": round(v2_det_rate, 4),
                "perspective_correction_success_rate": round(v2_persp_rate, 4),
                "mean_iou": round(v2_mean_iou, 4),
                "median_iou": round(v2_med_iou, 4),
                "min_iou": round(v2_min_iou, 4),
                "mean_corner_error_px": round(v2_mean_err, 2),
                "median_corner_error_px": round(v2_med_err, 2),
            },
            "delta_v2_minus_v1": {
                "detection_rate_delta": round(v2_det_rate - v1_det_rate, 4),
                "mean_iou_delta": round(v2_mean_iou - v1_mean_iou, 4),
                "median_iou_delta": round(v2_med_iou - v1_med_iou, 4),
                "mean_corner_error_delta_px": round(v2_mean_err - v1_mean_err, 2),
            }
        },
        "results_by_attack_category": {
            m: {
                "frames": category_stats[m]["frames"],
                "v1_baseline": {
                    "detected": category_stats[m]["v1_detected"],
                    "detection_rate": round(category_stats[m]["v1_detected"] / max(1, category_stats[m]["frames"]), 4),
                    "mean_iou": round(float(np.mean(category_stats[m]["v1_ious"])), 4) if category_stats[m]["v1_ious"] else 0.0,
                    "median_iou": round(float(np.median(category_stats[m]["v1_ious"])), 4) if category_stats[m]["v1_ious"] else 0.0,
                    "mean_corner_error_px": round(float(np.mean(category_stats[m]["v1_corner_errs"])), 2) if category_stats[m]["v1_corner_errs"] else 0.0,
                },
                "v2_enhanced": {
                    "detected": category_stats[m]["v2_detected"],
                    "detection_rate": round(category_stats[m]["v2_detected"] / max(1, category_stats[m]["frames"]), 4),
                    "mean_iou": round(float(np.mean(category_stats[m]["v2_ious"])), 4) if category_stats[m]["v2_ious"] else 0.0,
                    "median_iou": round(float(np.median(category_stats[m]["v2_ious"])), 4) if category_stats[m]["v2_ious"] else 0.0,
                    "mean_corner_error_px": round(float(np.mean(category_stats[m]["v2_corner_errs"])), 2) if category_stats[m]["v2_corner_errs"] else 0.0,
                    "perspective_success_rate": round(category_stats[m]["v2_persp"] / max(1, category_stats[m]["v2_detected"]), 4),
                }
            }
            for m in modes
        },
        "midv2020_spot_check": {
            "frames_evaluated": midv_total,
            "successful_detections": midv_det,
            "detection_rate": round(midv_det / max(1, midv_total), 4) if midv_total > 0 else 0.0,
            "mean_iou": round(midv_mean_iou, 4),
        }
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    logger.info("Saved summary JSON to %s", summary_path)
    return summary_data


if __name__ == "__main__":
    run_public_preprocessing_validation_v2()
