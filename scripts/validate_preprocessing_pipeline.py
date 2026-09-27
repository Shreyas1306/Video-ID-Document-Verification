"""
End-to-End Preprocessing Pipeline Validation
============================================
Executes:
  Video -> Frame Extraction -> Document Detection (OpenCV Contour Baseline)
  -> Four-Corner Localization -> Perspective Correction -> Normalized Frames

Generates:
  outputs/preprocessing_validation/
    ├── frames/
    ├── detected/
    ├── normalized/
    ├── normalized_contact_sheet.png
    ├── metadata.json
    └── summary.json
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.preprocessing.document_cropper import DocumentCropper
from src.preprocessing.document_detector import OpenCVContourDetector
from src.preprocessing.frame_extractor import FrameExtractor
from src.preprocessing.perspective_corrector import PerspectiveCorrector
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("validate_preprocessing")


def create_normalized_contact_sheet(
    normalized_image_paths: List[Path],
    output_path: Path,
    max_cols: int = 5,
    thumb_width: int = 300,
    thumb_height: int = 200,
) -> Optional[Path]:
    """
    Create a clean contact sheet montage of normalized frames in temporal order.
    """
    if not normalized_image_paths:
        logger.warning("No normalized images available for contact sheet.")
        return None

    images: List[Tuple[str, np.ndarray]] = []
    for p in normalized_image_paths:
        img = cv2.imread(str(p))
        if img is not None:
            resized = cv2.resize(img, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA)
            # Add small frame label banner
            label = p.stem.replace("norm_", "")
            cv2.rectangle(resized, (0, 0), (thumb_width, 24), (25, 25, 30), -1)
            cv2.putText(
                resized,
                label,
                (8, 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
            images.append((label, resized))

    if not images:
        return None

    num_images = len(images)
    cols = min(num_images, max_cols)
    rows = (num_images + cols - 1) // cols

    padding = 10
    header_height = 40
    sheet_w = cols * thumb_width + (cols + 1) * padding
    sheet_h = rows * thumb_height + (rows + 1) * padding + header_height

    contact_sheet = np.full((sheet_h, sheet_w, 3), (35, 38, 45), dtype=np.uint8)

    # Title header
    cv2.putText(
        contact_sheet,
        f"PREPROCESSING VALIDATION — NORMALIZED DOCUMENT FRAMES ({num_images} frames in temporal sequence)",
        (padding, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    for idx, (lbl, thumb) in enumerate(images):
        r = idx // cols
        c = idx % cols
        x = padding + c * (thumb_width + padding)
        y = header_height + padding + r * (thumb_height + padding)
        contact_sheet[y : y + thumb_height, x : x + thumb_width] = thumb

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), contact_sheet)
    logger.info(f"Saved normalized contact sheet to: {output_path}")
    return output_path


def run_preprocessing_validation(
    video_path: str = "data/sample/sample_document_video.mp4",
    output_dir: str = "outputs/preprocessing_validation",
    config_path: str = "config/settings.yaml",
) -> Dict[str, Any]:
    """
    Execute full preprocessing pipeline on mock document video and generate verification artifacts.
    """
    video_file = Path(video_path).resolve()
    if not video_file.is_file():
        raise FileNotFoundError(f"Input video file not found at: {video_file}")

    out_root = Path(output_dir).resolve()
    frames_dir = out_root / "frames"
    detected_dir = out_root / "detected"
    normalized_dir = out_root / "normalized"

    frames_dir.mkdir(parents=True, exist_ok=True)
    detected_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting Preprocessing Validation on: {video_file}")
    logger.info(f"Using configuration: {config_path}")

    # Read video properties
    cap = cv2.VideoCapture(str(video_file))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    # Step 1: Frame Extraction
    logger.info("Step 1: Extracting sampled frames...")
    extractor = FrameExtractor(config_path=config_path)
    extraction_res = extractor.extract_from_video(
        video_path=video_file,
        output_dir=frames_dir,
        save_to_disk=True,
    )
    frame_paths = extraction_res.get("frame_paths", [])
    sampled_frames_count = len(frame_paths)
    logger.info(f"Extracted {sampled_frames_count} sampled frames from {total_video_frames} total video frames.")

    # Initialize modules
    detector = OpenCVContourDetector(config_path=config_path)
    cropper = DocumentCropper()
    corrector = PerspectiveCorrector(config_path=config_path)

    metadata: List[Dict[str, Any]] = []
    successful_detections = 0
    failed_detections = 0
    successful_perspective_corrections = 0
    failed_perspective_corrections = 0
    confidences: List[float] = []
    saved_normalized_paths: List[Path] = []

    # Step 2: Process each extracted frame
    for fp_str in frame_paths:
        frame_file_path = Path(fp_str)
        frame_id = frame_file_path.stem

        img = cv2.imread(str(frame_file_path))
        if img is None:
            logger.error(f"Failed to read extracted frame image: {frame_file_path}")
            failed_detections += 1
            metadata.append({
                "frame_id": frame_id,
                "detection_status": False,
                "confidence": 0.0,
                "corners": None,
                "normalized_output_path": None,
                "error": "Failed to read image file from disk",
            })
            continue

        # Detect document candidate and corners
        det_res = detector.detect_frame(frame=img, frame_id=frame_id)
        detected = bool(det_res.get("detected", False))
        conf = float(det_res.get("confidence", 0.0))
        corners = det_res.get("corners")
        bbox = det_res.get("bbox")

        # Save annotated detection frame
        debug_path = detected_dir / f"detected_{frame_id}.png"
        if detected and corners is not None:
            annotated = cropper.draw_annotation(
                frame=img,
                bbox=bbox,
                corners=corners,
                confidence=conf,
                class_name="identity_document",
            )
        else:
            annotated = img.copy()
            cv2.putText(
                annotated,
                f"NO DETECTION: {det_res.get('reason', 'Confidence below threshold')}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
        cv2.imwrite(str(debug_path), annotated)

        norm_out_path_str: Optional[str] = None

        if detected and corners is not None:
            successful_detections += 1
            confidences.append(conf)

            # Step 3: Perspective correction using 4 corners
            corners_arr = np.array(corners, dtype=np.float32)
            warp_res = corrector.correct_perspective(image=img, corners=corners_arr, frame_id=frame_id)

            if warp_res.get("success", False) and warp_res.get("normalized_image") is not None:
                norm_img = warp_res["normalized_image"]
                norm_file = normalized_dir / f"norm_{frame_id}.png"
                cv2.imwrite(str(norm_file), norm_img)
                norm_out_path_str = str(norm_file)
                saved_normalized_paths.append(norm_file)
                successful_perspective_corrections += 1
                logger.info(f"Frame {frame_id}: Detection PASSED (conf={conf:.3f}), Perspective correction PASSED.")
            else:
                failed_perspective_corrections += 1
                logger.warning(f"Frame {frame_id}: Detection PASSED, but perspective correction failed: {warp_res.get('error')}")
        else:
            failed_detections += 1
            logger.info(f"Frame {frame_id}: Detection FAILED. Reason: {det_res.get('reason')}")

        metadata.append({
            "frame_id": frame_id,
            "detection_status": detected,
            "confidence": round(conf, 4),
            "corners": corners,
            "bbox": bbox,
            "normalized_output_path": norm_out_path_str,
            "detected_debug_path": str(debug_path),
        })

    # Metrics calculation
    detection_rate = (
        round(successful_detections / sampled_frames_count, 4)
        if sampled_frames_count > 0
        else 0.0
    )
    perspective_success_rate = (
        round(successful_perspective_corrections / successful_detections, 4)
        if successful_detections > 0
        else 0.0
    )
    avg_confidence = (
        round(float(np.mean(confidences)), 4)
        if confidences
        else 0.0
    )

    # Step 4: Generate contact sheet montage
    contact_sheet_path = out_root / "normalized_contact_sheet.png"
    create_normalized_contact_sheet(
        normalized_image_paths=saved_normalized_paths,
        output_path=contact_sheet_path,
    )

    # Step 5: Save metadata.json and summary.json
    meta_file = out_root / "metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    summary = {
        "input_video": str(video_file),
        "video_properties": {
            "total_frames": total_video_frames,
            "fps": video_fps,
            "resolution": [video_w, video_h],
        },
        "total_frames": total_video_frames,
        "sampled_frames": sampled_frames_count,
        "successful_detections": successful_detections,
        "failed_detections": failed_detections,
        "detection_rate": detection_rate,
        "average_detection_confidence": avg_confidence,
        "successful_perspective_corrections": successful_perspective_corrections,
        "failed_perspective_corrections": failed_perspective_corrections,
        "perspective_success_rate": perspective_success_rate,
        "output_directory": str(out_root),
        "artifacts": {
            "frames_directory": str(frames_dir),
            "detected_directory": str(detected_dir),
            "normalized_directory": str(normalized_dir),
            "contact_sheet": str(contact_sheet_path) if contact_sheet_path.is_file() else None,
            "metadata_json": str(meta_file),
        },
    }

    summary_file = out_root / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("=== Preprocessing Pipeline Validation Summary ===")
    logger.info(f"Input Video:              {video_file.name}")
    logger.info(f"Sampled Frames:           {sampled_frames_count}")
    logger.info(f"Successful Detections:    {successful_detections}/{sampled_frames_count} ({detection_rate * 100:.1f}%)")
    logger.info(f"Perspective Corrections:  {successful_perspective_corrections}/{successful_detections} ({perspective_success_rate * 100:.1f}%)")
    logger.info(f"Average Confidence:       {avg_confidence:.4f}")
    logger.info(f"Summary JSON:             {summary_file}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run End-to-End Preprocessing Pipeline Validation")
    parser.add_argument("--video", type=str, default="data/sample/sample_document_video.mp4", help="Input video path")
    parser.add_argument("--output-dir", type=str, default="outputs/preprocessing_validation", help="Output directory")
    parser.add_argument("--config", type=str, default="config/settings.yaml", help="Configuration YAML path")
    args = parser.parse_args()

    run_preprocessing_validation(
        video_path=args.video,
        output_dir=args.output_dir,
        config_path=args.config,
    )
