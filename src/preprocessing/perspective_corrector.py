"""
Perspective Corrector Module
=============================
Normalizes tilted, perspective-distorted document images into front-facing
canonical rectangular images using contour analysis and homography transforms.

Phase 3 Core Component:
- Modular corner estimation from document bounding regions.
- Robust 4-point polygon approximation and validation.
- Sub-pixel corner refinement.
- Four-point perspective transformation (cv2.getPerspectiveTransform + warpPerspective).
- Explicit failure logging when 4 corners cannot be identified reliably.
- Configurable normalized output dimensions (e.g. 600x400).
- Side-by-side visual comparison debugging (original vs corrected).
- Pluggable design allowing a future deep-learning keypoint/corner detector
  to replace or augment the contour-based estimator.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from src.utils.config_loader import load_config


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order four 2D coordinate points in canonical clockwise sequence:
    [top-left, top-right, bottom-right, bottom-left].

    Args:
        pts: Array of shape (4, 2) representing 4 points.

    Returns:
        Ordered float32 array of shape (4, 2).
    """
    pts = np.asarray(pts, dtype=np.float32).reshape((4, 2))
    rect = np.zeros((4, 2), dtype=np.float32)

    # Top-left has smallest sum (x + y), bottom-right has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Top-right has smallest difference (y - x), bottom-left has largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def estimate_document_corners(
    image: np.ndarray,
    min_area_ratio: float = 0.15,
    refine_subpixel: bool = True,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Estimate the four corners of an identity document within a cropped region
    using edge detection, morphological filtering, and contour approximation.

    Args:
        image: BGR document crop (H, W, C).
        min_area_ratio: Minimum area fraction of contour relative to image area.
        refine_subpixel: Whether to apply cv2.cornerSubPix for higher precision.

    Returns:
        Tuple of:
            - Ordered (4, 2) numpy array of corners (float32), or None if not found.
            - Metadata dictionary with detection diagnostics.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return None, {"found": False, "reason": "Invalid or empty image"}

    h, w = image.shape[:2]
    img_area = h * w
    if img_area < 400:  # Image too small for reliable corner estimation
        return None, {"found": False, "reason": f"Image too small ({w}x{h})"}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if float(np.std(gray)) < 5.0:
        return None, {"found": False, "reason": "Insufficient image contrast (flat/uniform image)"}

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Multi-strategy edge and threshold analysis to handle diverse lighting and contrast
    candidates: List[Tuple[float, np.ndarray]] = []  # (area, approx_points)

    threshold_methods = [
        # Strategy 1: Canny edge detector
        lambda: cv2.Canny(blurred, 40, 150),
        # Strategy 2: Otsu thresholding
        lambda: cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        # Strategy 3: Adaptive thresholding
        lambda: cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
    ]

    for method in threshold_methods:
        try:
            binary = method()
            # Close gaps along document perimeter
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

            contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            # Sort contours by area descending
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

            for cnt in sorted_contours:
                area = cv2.contourArea(cnt)
                if area < (img_area * min_area_ratio) or area > (img_area * 0.98):
                    continue

                peri = cv2.arcLength(cnt, True)
                if peri <= 0:
                    continue

                # Search across multiple approximation tolerances
                for eps_factor in [0.015, 0.02, 0.03, 0.04, 0.05]:
                    approx = cv2.approxPolyDP(cnt, eps_factor * peri, True)
                    if len(approx) == 4 and cv2.isContourConvex(approx):
                        candidates.append((area, approx.reshape((4, 2))))
                        break
        except Exception:
            continue

    if not candidates:
        return None, {
            "found": False,
            "reason": "Could not identify a convex 4-corner quadrilateral exceeding minimum area threshold.",
        }

    # Select the largest candidate quadrilateral
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_area, best_corners = candidates[0]

    ordered_corners = order_points(best_corners)

    # Optional sub-pixel refinement
    if refine_subpixel:
        try:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
            refined = cv2.cornerSubPix(
                gray,
                ordered_corners.copy(),
                (3, 3),
                (-1, -1),
                criteria,
            )
            # Only use refined if corners remained valid finite numbers
            if np.all(np.isfinite(refined)):
                ordered_corners = refined
        except Exception:
            pass  # Fall back to unrefined corners

    return ordered_corners, {
        "found": True,
        "corners": ordered_corners.tolist(),
        "contour_area": float(best_area),
        "area_ratio": round(float(best_area) / img_area, 4),
    }


class PerspectiveCorrector:
    """
    Normalizes perspective-distorted document images to standard rectangular dimensions.
    """

    def __init__(
        self,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        corner_refinement: Optional[bool] = None,
        fallback_to_crop: Optional[bool] = None,
        min_contour_area_ratio: Optional[float] = None,
        config_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the PerspectiveCorrector with configuration or explicit parameters.
        """
        try:
            cfg = load_config(config_path)
            persp_cfg = cfg.get("perspective_correction", {})
        except Exception:
            persp_cfg = {}

        self.target_width = int(target_width or persp_cfg.get("target_width", 600))
        self.target_height = int(target_height or persp_cfg.get("target_height", 400))
        self.corner_refinement = (
            corner_refinement
            if corner_refinement is not None
            else persp_cfg.get("corner_refinement", True)
        )
        self.fallback_to_crop = (
            fallback_to_crop
            if fallback_to_crop is not None
            else persp_cfg.get("fallback_to_crop", True)
        )
        self.min_contour_area_ratio = float(
            min_contour_area_ratio
            if min_contour_area_ratio is not None
            else persp_cfg.get("min_contour_area_ratio", 0.15)
        )

        # Destination rectangle corners for canonical perspective warp
        self.dst_corners = np.array(
            [
                [0, 0],
                [self.target_width - 1, 0],
                [self.target_width - 1, self.target_height - 1],
                [0, self.target_height - 1],
            ],
            dtype=np.float32,
        )

    def correct_perspective(
        self,
        image: np.ndarray,
        corners: Optional[np.ndarray] = None,
        frame_id: Optional[Union[str, int]] = None,
    ) -> Dict[str, Any]:
        """
        Apply perspective correction to a document image.

        Args:
            image: Cropped BGR document image (H, W, C).
            corners: Optional pre-computed (4, 2) corner points. If None, automatic estimation is used.
            frame_id: Optional frame identifier for logging.

        Returns:
            Dictionary containing:
                - frame_id: identifier
                - success: bool
                - method_applied: "homography" | "fallback_resize" | "none"
                - normalized_image: np.ndarray (target_height, target_width, 3) or None
                - corners: List of 4 points or None
                - error: str or None
        """
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return {
                "frame_id": frame_id,
                "success": False,
                "method_applied": "none",
                "normalized_image": None,
                "corners": None,
                "error": "Invalid or empty input image provided.",
            }

        src_corners = corners
        diag_info: Dict[str, Any] = {}

        if src_corners is None:
            src_corners, diag_info = estimate_document_corners(
                image,
                min_area_ratio=self.min_contour_area_ratio,
                refine_subpixel=self.corner_refinement,
            )

        if src_corners is not None and len(src_corners) == 4:
            try:
                ordered_src = order_points(src_corners)
                # Compute Homography matrix
                matrix = cv2.getPerspectiveTransform(ordered_src, self.dst_corners)
                # Warp to canonical rectangle
                warped = cv2.warpPerspective(
                    image,
                    matrix,
                    (self.target_width, self.target_height),
                    flags=cv2.INTER_LINEAR,
                )

                return {
                    "frame_id": frame_id,
                    "success": True,
                    "method_applied": "homography",
                    "normalized_image": warped,
                    "corners": ordered_src.tolist(),
                    "error": None,
                    "diagnostics": diag_info,
                }
            except Exception as e:
                err_msg = f"Homography computation failed: {e}"
        else:
            err_msg = diag_info.get("reason", "Four document corners could not be reliably determined.")

        # Failed corner detection branch
        if self.fallback_to_crop:
            # Fallback: Resize the raw bounding box to target dimensions, flagging failure
            fallback_img = cv2.resize(
                image,
                (self.target_width, self.target_height),
                interpolation=cv2.INTER_LINEAR,
            )
            return {
                "frame_id": frame_id,
                "success": False,
                "method_applied": "fallback_resize",
                "normalized_image": fallback_img,
                "corners": None,
                "error": f"Corner detection failed: {err_msg}. Used fallback bounding-box resize.",
                "diagnostics": diag_info,
            }

        return {
            "frame_id": frame_id,
            "success": False,
            "method_applied": "none",
            "normalized_image": None,
            "corners": None,
            "error": f"Corner detection failed: {err_msg}.",
            "diagnostics": diag_info,
        }

    def create_comparison_debug(
        self,
        original_image: np.ndarray,
        normalized_image: Optional[np.ndarray],
        corners: Optional[List[List[float]]] = None,
        status_text: str = "CORRECTED",
    ) -> np.ndarray:
        """
        Create a side-by-side visual comparison image:
        [Original with Corners/Polygon]  |  [Normalized Document View].

        Args:
            original_image: Input cropped image.
            normalized_image: Output normalized document image.
            corners: List of 4 corner coordinates (optional).
            status_text: Status banner string.

        Returns:
            Side-by-side BGR image.
        """
        h_target = self.target_height
        orig_copy = original_image.copy()

        # Draw detected corners and connecting polygon on original image
        if corners is not None and len(corners) == 4:
            pts = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(orig_copy, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            for i, pt in enumerate(corners):
                cx, cy = int(round(pt[0])), int(round(pt[1]))
                cv2.circle(orig_copy, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(
                    orig_copy,
                    f"C{i+1}",
                    (cx + 6, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 255),
                    1,
                )
        else:
            # Indicate failure on original
            cv2.putText(
                orig_copy,
                "NO 4-CORNERS FOUND",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
            )

        # Resize original view to match height of target
        orig_h, orig_w = orig_copy.shape[:2]
        scale = h_target / max(1, orig_h)
        new_w = max(1, int(round(orig_w * scale)))
        orig_resized = cv2.resize(orig_copy, (new_w, h_target))

        # Prepare normalized panel
        if normalized_image is not None and normalized_image.size > 0:
            norm_panel = normalized_image.copy()
        else:
            norm_panel = np.zeros((h_target, self.target_width, 3), dtype=np.uint8)
            cv2.putText(
                norm_panel,
                "CORRECTION FAILED",
                (self.target_width // 4, h_target // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        # Header bar for visual clarity
        header_h = 40
        canvas_w = new_w + self.target_width + 10  # 10px divider
        canvas = np.full((h_target + header_h, canvas_w, 3), (35, 35, 40), dtype=np.uint8)

        # Title text
        cv2.putText(
            canvas,
            f"ORIGINAL REGION (CORNERS)",
            (15, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1,
        )
        norm_title = f"NORMALIZED OUTPUT ({self.target_width}x{self.target_height}) - [{status_text}]"
        cv2.putText(
            canvas,
            norm_title,
            (new_w + 25, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (100, 240, 100) if "HOMOGRAPHY" in status_text.upper() else (100, 200, 255),
            1,
        )

        # Place images
        canvas[header_h:, :new_w] = orig_resized
        canvas[header_h:, new_w + 10 :] = norm_panel

        return canvas


def run_perspective_pipeline(
    crops_dir_or_paths: Union[str, Path, List[Union[str, Path]]],
    output_normalized_dir: Optional[Union[str, Path]] = None,
    output_debug_dir: Optional[Union[str, Path]] = None,
    config_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Process document crops, apply homography perspective correction,
    and save normalized rectangular frames and visual debug comparisons.

    Args:
        crops_dir_or_paths: Folder containing Phase 2 crops or list of crop paths.
        output_normalized_dir: Destination folder for normalized frames.
        output_debug_dir: Destination folder for comparison debug images.
        config_path: Path to configuration YAML.

    Returns:
        Summary dictionary with processing metrics and per-frame records.
    """
    corrector = PerspectiveCorrector(config_path=config_path)

    # Collect crop files
    crop_files: List[Path] = []
    if isinstance(crops_dir_or_paths, (str, Path)):
        p = Path(crops_dir_or_paths).resolve()
        if p.is_dir():
            crop_files = sorted(
                [f for f in p.glob("*.*") if f.suffix.lower() in [".png", ".jpg", ".jpeg"]]
            )
        elif p.is_file():
            crop_files = [p]
    else:
        crop_files = [Path(f).resolve() for f in crops_dir_or_paths]

    if not crop_files:
        raise ValueError(f"No valid crop images found at: {crops_dir_or_paths}")

    norm_dir = Path(output_normalized_dir or "outputs/normalized").resolve()
    debug_dir = Path(output_debug_dir or "outputs/debug_perspective").resolve()
    norm_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    successful_homography_count = 0
    fallback_count = 0
    failed_count = 0

    for crop_path in crop_files:
        frame_id = crop_path.stem
        img = cv2.imread(str(crop_path))

        if img is None:
            records.append({
                "frame_id": frame_id,
                "file": crop_path.name,
                "success": False,
                "method_applied": "none",
                "error": "Failed to read image file from disk",
            })
            failed_count += 1
            continue

        result = corrector.correct_perspective(image=img, frame_id=frame_id)
        result["file"] = crop_path.name

        method = result["method_applied"]
        if method == "homography":
            successful_homography_count += 1
            status_tag = "HOMOGRAPHY"
        elif method == "fallback_resize":
            fallback_count += 1
            status_tag = "FALLBACK RESIZE"
        else:
            failed_count += 1
            status_tag = "FAILED"

        # Save normalized frame
        if result["normalized_image"] is not None:
            norm_file = norm_dir / f"norm_{frame_id}.png"
            cv2.imwrite(str(norm_file), result["normalized_image"])
            result["normalized_path"] = str(norm_file)
        else:
            result["normalized_path"] = None

        # Save comparison debug image
        comparison_img = corrector.create_comparison_debug(
            original_image=img,
            normalized_image=result["normalized_image"],
            corners=result["corners"],
            status_text=status_tag,
        )
        debug_file = debug_dir / f"debug_persp_{frame_id}.png"
        cv2.imwrite(str(debug_file), comparison_img)
        result["debug_path"] = str(debug_file)

        # Strip heavy numpy array before JSON serialization
        record_entry = {k: v for k, v in result.items() if k != "normalized_image"}
        records.append(record_entry)

    total = len(crop_files)
    summary = {
        "target_dimensions": f"{corrector.target_width}x{corrector.target_height}",
        "total_crops_processed": total,
        "homography_success_count": successful_homography_count,
        "fallback_resize_count": fallback_count,
        "failed_count": failed_count,
        "homography_success_rate": round(successful_homography_count / total, 3) if total > 0 else 0.0,
        "output_normalized_directory": str(norm_dir),
        "output_debug_directory": str(debug_dir),
        "results": records,
    }

    # Save summary
    summary_file = norm_dir.parent / "perspective_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
