"""
Debug Utility for Document Localization & Perspective Correction
================================================================
Saves:
  1. original frame
  2. frame with detected quadrilateral & corner markers
  3. perspective-corrected document (homography warped)

Usage:
  python scripts/debug_document_localization.py
  python scripts/debug_document_localization.py --input path/to/frame.png --output-dir outputs/debug_localization
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.preprocessing.document_cropper import DocumentCropper
from src.preprocessing.document_detector import OpenCVContourDetector
from src.preprocessing.perspective_corrector import PerspectiveCorrector


def generate_synthetic_tilted_document_frame(
    width: int = 640,
    height: int = 480,
    angle_degrees: float = 12.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a realistic synthetic tilted ID document placed on a darker textured desk surface.

    Returns:
        Tuple of (frame_bgr, ground_truth_corners).
    """
    # Background desk surface
    background = np.full((height, width, 3), 45, dtype=np.uint8)
    # Add subtle texture
    noise = np.random.RandomState(42).randint(-8, 9, (height, width, 3), dtype=np.int16)
    desk = np.clip(background.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Base card dimensions (e.g. 320x200, aspect ratio 1.6)
    card_w, card_h = 320, 200
    card = np.full((card_h, card_w, 3), 245, dtype=np.uint8)

    # Document header band
    cv2.rectangle(card, (0, 0), (card_w, 40), (40, 80, 180), -1)
    cv2.putText(card, "IDENTITY CARD", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Photo placeholder
    cv2.rectangle(card, (20, 55), (100, 160), (180, 180, 190), -1)
    cv2.rectangle(card, (20, 55), (100, 160), (80, 80, 90), 1)
    cv2.putText(card, "PHOTO", (32, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 90), 1)

    # Text fields
    cv2.putText(card, "NAME: JANE DOE", (115, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1)
    cv2.putText(card, "DOB: 15-08-1992", (115, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1)
    cv2.putText(card, "ID NO: MOCK-8842-X", (115, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1)
    cv2.putText(card, "EXP: 2030-12-31", (115, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1)

    # Card border
    cv2.rectangle(card, (0, 0), (card_w - 1, card_h - 1), (20, 20, 20), 2)

    # Warp card onto desk with rotation & perspective tilt
    src_corners = np.array([
        [0, 0],
        [card_w, 0],
        [card_w, card_h],
        [0, card_h],
    ], dtype=np.float32)

    # Target destination points on the desk with perspective distortion
    center_x, center_y = width // 2, height // 2
    rad = np.radians(angle_degrees)
    cos_a, sin_a = np.cos(rad), np.sin(rad)

    # Rotate relative to center and apply perspective shear
    def transform_pt(x: float, y: float) -> Tuple[float, float]:
        dx = x - card_w / 2.0
        dy = y - card_h / 2.0
        # Rotate
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a
        # Perspective depth compression on top edge
        depth_factor = 1.0 + (ry / height) * 0.25
        return center_x + rx * depth_factor, center_y + ry * depth_factor

    dst_corners = np.array([transform_pt(p[0], p[1]) for p in src_corners], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(src_corners, dst_corners)
    cv2.warpPerspective(
        card,
        matrix,
        (width, height),
        dst=desk,
        borderMode=cv2.BORDER_TRANSPARENT,
    )

    return desk, dst_corners


def run_debug(
    input_path: Optional[str] = None,
    output_dir: str = "outputs/debug_localization",
    config_path: Optional[str] = "config/settings.yaml",
) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if input_path and Path(input_path).is_file():
        frame = cv2.imread(str(input_path))
        if frame is None:
            raise ValueError(f"Could not read input image from: {input_path}")
        source_desc = f"Loaded file: {input_path}"
    else:
        frame, _ = generate_synthetic_tilted_document_frame()
        source_desc = "Generated synthetic tilted document on textured background"

    detector = OpenCVContourDetector(config_path=config_path)
    cropper = DocumentCropper()
    corrector = PerspectiveCorrector(config_path=config_path)

    det_result = detector.detect_frame(frame, frame_id="debug_sample")

    # 1. Save original frame
    p_orig = out_path / "1_original_frame.png"
    cv2.imwrite(str(p_orig), frame)

    # 2. Save frame with detected quadrilateral and corners
    if det_result["detected"]:
        annotated = cropper.draw_annotation(
            frame=frame,
            bbox=det_result["bbox"],
            corners=det_result["corners"],
            confidence=det_result["confidence"],
            class_name="identity_document",
        )
    else:
        annotated = frame.copy()
        cv2.putText(
            annotated,
            f"NO DETECTION: {det_result.get('reason', 'Unknown')}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    p_annot = out_path / "2_detected_quadrilateral.png"
    cv2.imwrite(str(p_annot), annotated)

    # 3. Save perspective-corrected document (homography warped)
    if det_result["detected"] and det_result["corners"] is not None:
        corners_arr = np.array(det_result["corners"], dtype=np.float32)
        warp_res = corrector.correct_perspective(image=frame, corners=corners_arr)
        norm_img = warp_res["normalized_image"]
    else:
        norm_img = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.putText(
            norm_img,
            "CORRECTION UNAVAILABLE (NO DETECTION)",
            (50, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    p_norm = out_path / "3_perspective_corrected.png"
    cv2.imwrite(str(p_norm), norm_img)

    # Save diagnostics summary JSON
    summary = {
        "source": source_desc,
        "detected": det_result["detected"],
        "confidence": det_result["confidence"],
        "bbox": det_result["bbox"],
        "corners": det_result["corners"],
        "method": det_result["method"],
        "reason": det_result.get("reason"),
        "artifacts": {
            "original_frame": str(p_orig),
            "detected_quadrilateral": str(p_annot),
            "perspective_corrected": str(p_norm),
        },
    }
    p_json = out_path / "debug_summary.json"
    with open(p_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Document Localization Debug Complete.")
    print(f"  Detected: {det_result['detected']} (Confidence: {det_result['confidence']})")
    print(f"  Original:      {p_orig}")
    print(f"  Quadrilateral: {p_annot}")
    print(f"  Normalized:    {p_norm}")
    print(f"  Summary JSON:  {p_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug Document Localization & Perspective Correction")
    parser.add_argument("--input", type=str, default=None, help="Path to input image (or leave empty for synthetic)")
    parser.add_argument("--output-dir", type=str, default="outputs/debug_localization", help="Output directory")
    parser.add_argument("--config", type=str, default="config/settings.yaml", help="Path to YAML config")
    args = parser.parse_args()

    run_debug(input_path=args.input, output_dir=args.output_dir, config_path=args.config)
