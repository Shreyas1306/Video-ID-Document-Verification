"""
OCR and Text Verification Pipeline Validation Script
=====================================================
Validates Phase 4:
  Normalized Document Frames
  -> OCR
  -> Text Extraction
  -> Key Field Extraction
  -> Field Presence/Format Checks
  -> Cross-Frame Text Consistency
  -> OCR Confidence & Verification Summary

Outputs structured diagnostics and visual artifacts under:
outputs/ocr_validation/
  ├── raw/
  ├── annotated/
  ├── results.json
  ├── summary.json
  └── contact_sheet.png
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.text.ocr_engine import OCREngine
from src.text.field_parser import FieldParser
from src.text.text_consistency import TextConsistencyAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def annotate_frame_with_ocr(
    image: np.ndarray,
    ocr_result: Dict[str, Any],
    parsed_fields: Dict[str, Any],
    frame_id: str,
) -> np.ndarray:
    """
    Draw text bounding polygons, line labels, confidences, and summary badge on frame.
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]

    # Draw detected line bounding boxes
    for line in ocr_result.get("lines", []):
        box = line.get("box", [])
        text = line.get("text", "")
        conf = line.get("confidence", 0.0)

        if len(box) == 4:
            pts = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts], isClosed=True, color=(0, 220, 100), thickness=2)
            # Label near top-left of box
            x_min = min(p[0] for p in box)
            y_min = min(p[1] for p in box)
            label = f"{text[:22]} ({conf:.2f})"
            cv2.putText(
                annotated,
                label,
                (max(5, x_min), max(15, y_min - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (0, 70, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                label,
                (max(5, x_min), max(15, y_min - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (180, 255, 200),
                1,
                cv2.LINE_AA,
            )

    # Top overlay header banner
    banner_h = 42
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (25, 25, 30), -1)
    cv2.addWeighted(overlay, 0.78, annotated, 0.22, 0, annotated)

    avg_conf = ocr_result.get("average_confidence", 0.0)
    lines_count = len(ocr_result.get("lines", []))
    header_text = f"{frame_id} | OCR Lines: {lines_count} | Avg Conf: {avg_conf:.3f}"
    cv2.putText(
        annotated,
        header_text,
        (10, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # Fields summary in header
    fields_found = [
        k for k, present in parsed_fields.get("field_presence", {}).items() if present
    ]
    fields_text = f"Fields Detected: {', '.join(fields_found) if fields_found else 'None'}"
    cv2.putText(
        annotated,
        fields_text,
        (10, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (120, 240, 160) if fields_found else (100, 100, 240),
        1,
        cv2.LINE_AA,
    )

    return annotated


def create_ocr_contact_sheet(
    annotated_images: List[np.ndarray],
    frame_labels: List[str],
    output_path: Path,
    cols: int = 2,
) -> Path:
    """
    Generate a visual contact sheet grid of annotated OCR frames.
    """
    if not annotated_images:
        return output_path

    n = len(annotated_images)
    rows = (n + cols - 1) // cols

    thumb_w, thumb_h = 480, 320
    canvas_w = cols * thumb_w
    canvas_h = rows * thumb_h + 50
    canvas = np.full((canvas_h, canvas_w, 3), 32, dtype=np.uint8)

    # Title header
    cv2.putText(
        canvas,
        "OCR & TEXT VERIFICATION VALIDATION - MULTI-FRAME SEQUENCE",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )

    for idx, (img, lbl) in enumerate(zip(annotated_images, frame_labels)):
        r = idx // cols
        c = idx % cols
        x = c * thumb_w
        y = 50 + r * thumb_h

        resized = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        canvas[y : y + thumb_h, x : x + thumb_w] = resized
        # Cell border
        cv2.rectangle(canvas, (x, y), (x + thumb_w, y + thumb_h), (60, 60, 65), 2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    return output_path


def run_ocr_validation(
    normalized_dir: Union[str, Path] = "outputs/preprocessing_validation/normalized",
    output_dir: Union[str, Path] = "outputs/ocr_validation",
    config_path: Union[str, Path] = "config/settings.yaml",
) -> Dict[str, Any]:
    """
    Execute full OCR and Text Verification validation on normalized frames.
    """
    norm_path = Path(normalized_dir).resolve()
    out_path = Path(output_dir).resolve()
    cfg_path = Path(config_path).resolve()

    if not norm_path.is_dir():
        raise FileNotFoundError(f"Normalized frames directory does not exist: {norm_path}")

    # Discover normalized frames
    frame_files = sorted([f for f in norm_path.glob("*.png")] + [f for f in norm_path.glob("*.jpg")])
    if not frame_files:
        raise ValueError(f"No normalized frame images found in: {norm_path}")

    # Prepare output directories
    raw_dir = out_path / "raw"
    annotated_dir = out_path / "annotated"
    raw_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Initializing OCR engine from configuration: {cfg_path}")
    ocr_engine = OCREngine(config_path=cfg_path)
    field_parser = FieldParser(config_path=cfg_path)
    consistency_analyzer = TextConsistencyAnalyzer()

    logger.info(f"Active OCR Engine: {ocr_engine.active_engine}")
    logger.info(f"Processing {len(frame_files)} normalized frames...")

    per_frame_results: List[Dict[str, Any]] = []
    per_frame_parsed_fields: List[Dict[str, Any]] = []
    annotated_images: List[np.ndarray] = []
    frame_labels: List[str] = []
    confidences: List[float] = []
    successful_ocr_count = 0

    for frame_file in frame_files:
        frame_id = frame_file.stem
        img = cv2.imread(str(frame_file))

        if img is None or img.size == 0:
            logger.warning(f"Failed to read image: {frame_file}")
            continue

        # 1. OCR text extraction
        ocr_res = ocr_engine.extract_text(image=img, frame_id=frame_id)
        lines = ocr_res.get("lines", [])
        avg_conf = ocr_res.get("average_confidence", 0.0)
        confidences.append(avg_conf)

        if len(lines) > 0:
            successful_ocr_count += 1

        # 2. Key field parsing & validation
        parsed = field_parser.parse_fields(raw_text=ocr_res.get("full_text", ""), lines=lines)
        per_frame_parsed_fields.append(parsed)

        # 3. Save raw frame JSON
        raw_json_file = raw_dir / f"{frame_id}.json"
        with open(raw_json_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "frame_id": frame_id,
                    "engine_used": ocr_res.get("engine_used"),
                    "average_confidence": avg_conf,
                    "line_count": len(lines),
                    "lines": lines,
                    "full_text": ocr_res.get("full_text", ""),
                    "parsed_fields": parsed,
                },
                f,
                indent=2,
            )

        # 4. Generate visual annotated frame
        annotated_img = annotate_frame_with_ocr(img, ocr_res, parsed, frame_id)
        annotated_file = annotated_dir / f"annotated_{frame_id}.png"
        cv2.imwrite(str(annotated_file), annotated_img)

        annotated_images.append(annotated_img)
        frame_labels.append(frame_id)

        # 5. Record frame summary entry
        per_frame_results.append({
            "frame_id": frame_id,
            "file": frame_file.name,
            "extracted_text": ocr_res.get("full_text", ""),
            "ocr_confidence": avg_conf,
            "num_text_regions": len(lines),
            "lines": lines,
            "extracted_key_fields": {
                "name": parsed.get("name"),
                "dob": parsed.get("dob"),
                "document_number": parsed.get("document_number"),
                "address": parsed.get("address"),
                "field_presence": parsed.get("field_presence"),
                "format_validity": parsed.get("format_validity"),
            },
            "raw_output_path": str(raw_json_file),
            "annotated_output_path": str(annotated_file),
        })

        logger.info(
            f"Frame {frame_id}: conf={avg_conf:.3f}, lines={len(lines)}, "
            f"name={parsed.get('name')}, dob={parsed.get('dob')}, "
            f"doc={parsed.get('document_number')}, addr={parsed.get('address')}"
        )

    # Cross-frame text consistency analysis
    overall_mean_conf = float(np.mean(confidences)) if confidences else 0.0
    consistency_res = consistency_analyzer.analyze_cross_frame_consistency(
        frame_extractions=per_frame_parsed_fields,
        overall_ocr_confidence=overall_mean_conf,
    )

    # Generate contact sheet
    contact_sheet_path = out_path / "contact_sheet.png"
    create_ocr_contact_sheet(annotated_images, frame_labels, contact_sheet_path)

    # Rationale for engine used
    engine_used = ocr_engine.active_engine
    if engine_used == "easyocr":
        engine_rationale = (
            "EasyOCR was active as the primary robust CPU engine with local PyTorch weights. "
            "PaddleOCR was bypassed/fell back due to native Windows environment execution constraints "
            "and external model-hub connectivity requirements."
        )
    else:
        engine_rationale = "PaddleOCR was used directly as configured."

    # Build summary
    total_frames = len(frame_files)
    summary_data = {
        "input_directory": str(norm_path),
        "total_frames_processed": total_frames,
        "successful_ocr_frames": successful_ocr_count,
        "failed_ocr_frames": total_frames - successful_ocr_count,
        "ocr_success_rate": round(successful_ocr_count / total_frames, 4) if total_frames > 0 else 0.0,
        "ocr_engine_used": engine_used,
        "ocr_engine_rationale": engine_rationale,
        "average_ocr_confidence": round(overall_mean_conf, 4),
        "consensus_fields": {
            "name": consistency_res.get("name"),
            "dob": consistency_res.get("dob"),
            "document_number": consistency_res.get("document_number"),
            "address": consistency_res.get("address"),
        },
        "field_presence_rate": consistency_res.get("field_presence_rate", {}),
        "format_validity": consistency_res.get("format_validity", {}),
        "per_field_consistency": consistency_res.get("per_field_consistency", {}),
        "cross_frame_text_consistency_score": consistency_res.get("text_consistency", 0.0),
        "stable_fields": consistency_res.get("stable_fields", []),
        "problematic_fields": consistency_res.get("problematic_fields", []),
        "output_directory": str(out_path),
        "artifacts": {
            "results_json": str(out_path / "results.json"),
            "summary_json": str(out_path / "summary.json"),
            "raw_dir": str(raw_dir),
            "annotated_dir": str(annotated_dir),
            "contact_sheet": str(contact_sheet_path),
        },
        "disclaimer": consistency_res.get("disclaimer", ""),
    }

    # Save results.json and summary.json
    with open(out_path / "results.json", "w", encoding="utf-8") as f:
        json.dump(per_frame_results, f, indent=2)

    with open(out_path / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    logger.info(f"Saved results to {out_path / 'results.json'}")
    logger.info(f"Saved summary to {out_path / 'summary.json'}")
    logger.info(f"Saved contact sheet to {contact_sheet_path}")

    return summary_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate OCR & Text Verification on normalized frames.")
    parser.add_argument(
        "--normalized-dir",
        type=str,
        default="outputs/preprocessing_validation/normalized",
        help="Path to folder containing normalized document frames.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/ocr_validation",
        help="Path to directory where validation results will be written.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        help="Path to settings.yaml configuration.",
    )
    args = parser.parse_args()

    run_ocr_validation(
        normalized_dir=args.normalized_dir,
        output_dir=args.output_dir,
        config_path=args.config,
    )
