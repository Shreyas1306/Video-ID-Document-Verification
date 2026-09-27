"""
Document Detector Module
========================
Provides modular document localization interfaces and implementations:
  1. OpenCVContourDetector: Fast, robust geometric contour-based candidate detector
     (baseline for ID cards and rectangular documents).
  2. YOLODocumentDetector: Deep learning object detector using Ultralytics YOLOv8
     (for custom/fine-tuned identity document models).

Architecture:
  BaseDocumentDetector (Interface)
  ├── OpenCVContourDetector  (Active baseline)
  └── YOLODocumentDetector   (Deep learning / fine-tuned detector)

Pipeline:
  Frame -> Preprocessing -> Edge Detection -> Contour Detection ->
  Quadrilateral Filtering -> Geometric Candidate Ranking ->
  Four Corners -> Perspective Correction
"""

from abc import ABC, abstractmethod
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from src.preprocessing.perspective_corrector import order_points
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)


class BaseDocumentDetector(ABC):
    """
    Abstract Base Class defining the contract for document detectors.
    """

    @abstractmethod
    def detect_frame(
        self,
        frame: np.ndarray,
        frame_id: Optional[Union[str, int]] = None,
    ) -> Dict[str, Any]:
        """
        Run document detection on a single frame.

        Args:
            frame: BGR image as a 3D numpy array (H, W, 3).
            frame_id: Optional identifier or index for the frame.

        Returns:
            Structured dictionary with detection status, bbox, 4 corners, and confidence.
        """
        pass


class OpenCVContourDetector(BaseDocumentDetector):
    """
    OpenCV contour-based document candidate detector.
    Identifies rectangular document regions in frames and returns four canonical corner points.
    """

    def __init__(
        self,
        confidence_threshold: Optional[float] = None,
        min_area_ratio: Optional[float] = None,
        max_area_ratio: Optional[float] = None,
        min_aspect_ratio: Optional[float] = None,
        max_aspect_ratio: Optional[float] = None,
        refine_subpixel: Optional[bool] = None,
        config_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the OpenCV contour detector with configurable geometric constraints.
        """
        try:
            cfg = load_config(config_path)
            det_cfg = cfg.get("document_detection", {})
        except Exception:
            det_cfg = {}

        self.confidence_threshold = (
            float(confidence_threshold)
            if confidence_threshold is not None
            else float(det_cfg.get("confidence_threshold", 0.35))
        )
        self.min_area_ratio = (
            float(min_area_ratio)
            if min_area_ratio is not None
            else float(det_cfg.get("min_area_ratio", 0.05))
        )
        self.max_area_ratio = (
            float(max_area_ratio)
            if max_area_ratio is not None
            else float(det_cfg.get("max_area_ratio", 0.95))
        )
        self.min_aspect_ratio = (
            float(min_aspect_ratio)
            if min_aspect_ratio is not None
            else float(det_cfg.get("min_aspect_ratio", 1.0))
        )
        self.max_aspect_ratio = (
            float(max_aspect_ratio)
            if max_aspect_ratio is not None
            else float(det_cfg.get("max_aspect_ratio", 2.8))
        )
        self.refine_subpixel = (
            bool(refine_subpixel)
            if refine_subpixel is not None
            else bool(det_cfg.get("refine_subpixel", True))
        )
        try:
            from src.preprocessing.candidate_scorer import CandidateScorer
            self.scorer = CandidateScorer(config_path=config_path)
        except Exception:
            self.scorer = None

    def detect_candidates(
        self,
        frame: np.ndarray,
        frame_id: Optional[Union[str, int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract all candidate quadrilaterals from the frame with full metadata.

        Args:
            frame: BGR image array (H, W, 3).
            frame_id: Optional frame identifier.

        Returns:
            List of candidate dictionaries with corners, bbox, area, aspect_ratio, and confidence.
        """
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError(f"Invalid frame input: expected numpy.ndarray, got {type(frame)}")

        if frame.size == 0 or len(frame.shape) != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"Invalid image array dimensions: expected (H, W, 3) non-empty array, got {frame.shape if hasattr(frame, 'shape') else 'invalid'}"
            )

        h, w = frame.shape[:2]
        img_area = float(h * w)

        if img_area < 400.0:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if float(np.std(gray)) < 5.0:
            return []

        candidates = self._find_quadrilateral_candidates(frame, gray, h, w, img_area)
        for c in candidates:
            c["area_ratio"] = float(c["area"] / max(1.0, img_area))
            c.setdefault("edge_quality", 0.5)

        return candidates

    def detect_frame(
        self,
        frame: np.ndarray,
        frame_id: Optional[Union[str, int]] = None,
    ) -> Dict[str, Any]:
        """
        Detect candidate document quadrilateral in a frame.

        Args:
            frame: BGR image array (H, W, 3).
            frame_id: Optional frame identifier.

        Returns:
            Dictionary containing:
                - frame_id: identifier
                - detected: bool
                - bbox: [x1, y1, x2, y2] or None
                - corners: [[x, y], [x, y], [x, y], [x, y]] or None
                - confidence: float score in [0.0, 1.0]
                - class_name: "identity_document" or None
                - class_id: 0 or None
                - method: "opencv_contour"
                - all_detections: list of candidate detections

        Raises:
            ValueError: If frame is invalid, empty, or not 3-channel BGR.
        """
        candidates = self.detect_candidates(frame, frame_id)

        if not candidates:
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if float(np.std(gray)) < 5.0:
                return self._empty_result(frame_id, "Insufficient image contrast (flat/uniform image)")
            return self._empty_result(frame_id, "No reliable document detected (no quadrilateral candidate found)")

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Rank candidates via CandidateScorer if enabled, else baseline sorting
        if self.scorer is not None and self.scorer.enabled:
            candidates = self.scorer.score_candidates(candidates, (h, w))
            best = candidates[0]
        else:
            candidates.sort(key=lambda c: (c["confidence"], c["area"]), reverse=True)
            best = candidates[0]

        if best["confidence"] < self.confidence_threshold:
            return self._empty_result(
                frame_id,
                f"No reliable document detected (confidence {best['confidence']:.3f} below threshold {self.confidence_threshold:.3f})"
            )

        best_corners = best["corners"]
        if not isinstance(best_corners, np.ndarray):
            best_corners = np.asarray(best_corners, dtype=np.float32).reshape((4, 2))

        # Sub-pixel corner refinement if requested
        if self.refine_subpixel:
            try:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
                refined = cv2.cornerSubPix(
                    gray,
                    best_corners.copy(),
                    (3, 3),
                    (-1, -1),
                    criteria,
                )
                if np.all(np.isfinite(refined)):
                    best_corners = refined
            except Exception:
                pass

        formatted_all = [
            {
                "bbox": c["bbox"],
                "corners": c["corners"].tolist() if isinstance(c["corners"], np.ndarray) else c["corners"],
                "confidence": c["confidence"],
                "area": c["area"],
                "aspect_ratio": c.get("aspect_ratio"),
                "rectangularity": c.get("rectangularity"),
                "area_ratio": c.get("area_ratio"),
                "static_score": c.get("static_score"),
            }
            for c in candidates
        ]

        return {
            "frame_id": frame_id,
            "detected": True,
            "bbox": best["bbox"],
            "corners": best_corners.tolist() if isinstance(best_corners, np.ndarray) else best_corners,
            "confidence": best["confidence"],
            "class_name": "identity_document",
            "class_id": 0,
            "method": "opencv_contour",
            "all_detections": formatted_all,
        }

    def _find_quadrilateral_candidates(
        self,
        frame: np.ndarray,
        gray: np.ndarray,
        h: int,
        w: int,
        img_area: float,
    ) -> List[Dict[str, Any]]:
        """Extract and evaluate quadrilateral candidate contours across multiple threshold strategies."""
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        binaries: List[np.ndarray] = []

        # Strategy 1: Automatic Canny
        try:
            med = float(np.median(blurred))
            low = int(max(20, (1.0 - 0.33) * med))
            high = int(min(220, (1.0 + 0.33) * med))
            canny = cv2.Canny(blurred, low, high)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            binaries.append(cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel, iterations=2))
        except Exception:
            pass

        # Strategy 2: Fixed Canny
        try:
            canny_fixed = cv2.Canny(blurred, 40, 150)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            binaries.append(cv2.morphologyEx(canny_fixed, cv2.MORPH_CLOSE, kernel, iterations=2))
        except Exception:
            pass

        # Strategy 3: Otsu Thresholding (standard + inverted)
        try:
            _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            binaries.append(cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=2))
            binaries.append(cv2.morphologyEx(255 - otsu, cv2.MORPH_CLOSE, kernel, iterations=2))
        except Exception:
            pass

        # Strategy 4: Adaptive Gaussian Thresholding
        try:
            adapt = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            binaries.append(cv2.morphologyEx(adapt, cv2.MORPH_CLOSE, kernel, iterations=2))
        except Exception:
            pass

        raw_candidates: List[Dict[str, Any]] = []

        for binary in binaries:
            for mode in [cv2.RETR_EXTERNAL, cv2.RETR_LIST]:
                try:
                    contours, _ = cv2.findContours(binary, mode, cv2.CHAIN_APPROX_SIMPLE)
                except Exception:
                    continue

                for cnt in contours:
                    area = float(cv2.contourArea(cnt))
                    if area < (self.min_area_ratio * img_area) or area > (self.max_area_ratio * img_area):
                        continue
                    if area < 1000.0:  # Absolute minimum area in pixels
                        continue

                    peri = float(cv2.arcLength(cnt, True))
                    if peri <= 0.0:
                        continue

                    for eps_factor in [0.015, 0.02, 0.025, 0.03, 0.04, 0.05]:
                        approx = cv2.approxPolyDP(cnt, eps_factor * peri, True)
                        if len(approx) == 4 and cv2.isContourConvex(approx):
                            pts = approx.reshape((4, 2)).astype(np.float32)
                            ordered = order_points(pts)

                            cand = self._validate_and_score_quadrilateral(ordered, cnt, area, h, w, img_area)
                            if cand is not None:
                                raw_candidates.append(cand)
                            break

        # Deduplicate candidates with high IoU
        deduped: List[Dict[str, Any]] = []
        for cand in sorted(raw_candidates, key=lambda x: x["confidence"], reverse=True):
            is_dup = False
            for existing in deduped:
                if self._compute_bbox_iou(cand["bbox"], existing["bbox"]) > 0.70:
                    is_dup = True
                    break
            if not is_dup:
                deduped.append(cand)

        return deduped

    def _validate_and_score_quadrilateral(
        self,
        pts: np.ndarray,
        cnt: np.ndarray,
        area: float,
        img_h: int,
        img_w: int,
        img_area: float,
    ) -> Optional[Dict[str, Any]]:
        """Validate candidate geometry against physical document constraints and compute confidence."""
        # 1. Boundary constraint: Never silently treat entire frame as document
        margin = 4.0
        border_pts_count = sum(
            1 for p in pts if (p[0] <= margin or p[0] >= (img_w - margin) or p[1] <= margin or p[1] >= (img_h - margin))
        )
        if border_pts_count >= 4 and (area / img_area) >= 0.88:
            return None

        # 2. Side lengths & Aspect ratio
        w_top = float(np.linalg.norm(pts[1] - pts[0]))
        w_bot = float(np.linalg.norm(pts[2] - pts[3]))
        h_left = float(np.linalg.norm(pts[3] - pts[0]))
        h_right = float(np.linalg.norm(pts[2] - pts[1]))

        mean_w = 0.5 * (w_top + w_bot)
        mean_h = 0.5 * (h_left + h_right)

        min_dim = max(1.0, min(mean_w, mean_h))
        max_dim = max(mean_w, mean_h)
        aspect = max_dim / min_dim

        if aspect < self.min_aspect_ratio or aspect > self.max_aspect_ratio:
            return None

        # 3. Interior angles (orthogonality)
        e0 = pts[1] - pts[0]
        e1 = pts[2] - pts[1]
        e2 = pts[3] - pts[2]
        e3 = pts[0] - pts[3]

        def cos_angle(u: np.ndarray, v: np.ndarray) -> float:
            norm = np.linalg.norm(u) * np.linalg.norm(v)
            return float(abs(np.dot(u, v) / max(1e-6, norm)))

        cos_angles = [
            cos_angle(e0, -e3),  # Top-Left angle
            cos_angle(e1, -e0),  # Top-Right angle
            cos_angle(e2, -e1),  # Bottom-Right angle
            cos_angle(e3, -e2),  # Bottom-Left angle
        ]

        if max(cos_angles) > 0.70:
            # Overly skewed / non-rectangular polygon
            return None

        # 4. Rectangularity vs rotated minimum bounding box
        rect = cv2.minAreaRect(cnt)
        rect_area = float(rect[1][0] * rect[1][1])
        s_rect = min(1.0, area / max(1.0, rect_area))
        if s_rect < 0.70:
            return None

        # Orthogonality score
        s_ortho = 1.0 - float(np.mean(cos_angles))

        # Aspect score: ID-1 card baseline aspect ratio ~1.58
        s_aspect = max(0.0, 1.0 - 0.35 * abs(aspect - 1.58))

        # Area score: prefers balanced frame coverage (10% to 75%)
        area_fraction = area / img_area
        if area_fraction < 0.15:
            s_area = min(1.0, area_fraction / 0.15)
        else:
            s_area = max(0.0, 1.0 - max(0.0, area_fraction - 0.70) / 0.25)

        confidence = round(float(0.40 * s_rect + 0.35 * s_ortho + 0.15 * s_aspect + 0.10 * s_area), 4)
        confidence = max(0.0, min(1.0, confidence))

        x1 = max(0, min(img_w - 1, int(round(float(np.min(pts[:, 0]))))))
        y1 = max(0, min(img_h - 1, int(round(float(np.min(pts[:, 1]))))))
        x2 = max(0, min(img_w - 1, int(round(float(np.max(pts[:, 0]))))))
        y2 = max(0, min(img_h - 1, int(round(float(np.max(pts[:, 1]))))))

        return {
            "corners": pts,
            "bbox": [x1, y1, x2, y2],
            "area": area,
            "confidence": confidence,
            "aspect_ratio": round(aspect, 3),
            "rectangularity": round(s_rect, 3),
        }

    def _compute_bbox_iou(self, b1: List[int], b2: List[int]) -> float:
        """Compute Intersection over Union between two bounding boxes."""
        ix1 = max(b1[0], b2[0])
        iy1 = max(b1[1], b2[1])
        ix2 = min(b1[2], b2[2])
        iy2 = min(b1[3], b2[3])

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter_area = iw * ih

        b1_area = (b1[2] - b1[0]) * (b1[3] - b1[1])
        b2_area = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union_area = b1_area + b2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def _empty_result(self, frame_id: Optional[Union[str, int]], reason: str) -> Dict[str, Any]:
        """Return standardized negative detection result."""
        return {
            "frame_id": frame_id,
            "detected": False,
            "bbox": None,
            "corners": None,
            "confidence": 0.0,
            "class_name": None,
            "class_id": None,
            "method": "opencv_contour",
            "all_detections": [],
            "reason": reason,
        }


class YOLODocumentDetector(BaseDocumentDetector):
    """
    YOLO-based detector for localizing identity documents within video frames.
    Retained for future learned/fine-tuned document models.
    """

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        target_classes: Optional[List[Union[str, int]]] = None,
        config_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the YOLO document detector.

        Args:
            model_name_or_path: YOLO model weights (e.g. 'yolov8n.pt' or custom checkpoint).
            confidence_threshold: Minimum confidence score to accept detection.
            iou_threshold: Non-maximum suppression IoU threshold.
            target_classes: Optional list of class names or integer IDs to filter on.
            config_path: Optional path to YAML configuration file.
        """
        try:
            cfg = load_config(config_path)
            det_cfg = cfg.get("document_detection", {})
        except Exception:
            det_cfg = {}

        self.model_path = model_name_or_path or det_cfg.get("model_name", "yolov8n.pt")
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else det_cfg.get("confidence_threshold", 0.25)
        )
        self.iou_threshold = (
            iou_threshold
            if iou_threshold is not None
            else det_cfg.get("iou_threshold", 0.45)
        )
        self.target_classes = target_classes or det_cfg.get("target_classes", None)

        from ultralytics import YOLO
        self.model = YOLO(self.model_path)

    def detect_frame(
        self,
        frame: np.ndarray,
        frame_id: Optional[Union[str, int]] = None,
    ) -> Dict[str, Any]:
        """
        Run YOLO detection on a single frame.

        Args:
            frame: BGR image as a 3D numpy array (H, W, 3).
            frame_id: Optional identifier or index for the frame.

        Returns:
            Dictionary containing detection status, bbox, corners, and confidence.
        """
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError(f"Invalid frame input: expected numpy.ndarray, got {type(frame)}")

        if frame.size == 0 or len(frame.shape) != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"Invalid image array dimensions: expected (H, W, 3) non-empty array, got {frame.shape if hasattr(frame, 'shape') else 'invalid'}"
            )

        h, w, _ = frame.shape

        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        all_candidates: List[Dict[str, Any]] = []

        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                xyxy_arr = boxes.xyxy.cpu().numpy()
                conf_arr = boxes.conf.cpu().numpy()
                cls_arr = boxes.cls.cpu().numpy().astype(int)
                names = result.names

                for i in range(len(boxes)):
                    raw_cls_id = int(cls_arr[i])
                    cls_name = names.get(raw_cls_id, str(raw_cls_id))
                    conf = float(conf_arr[i])

                    if self.target_classes is not None:
                        if (raw_cls_id not in self.target_classes) and (cls_name not in self.target_classes):
                            continue

                    x1 = max(0, min(w - 1, int(round(xyxy_arr[i][0]))))
                    y1 = max(0, min(h - 1, int(round(xyxy_arr[i][1]))))
                    x2 = max(0, min(w - 1, int(round(xyxy_arr[i][2]))))
                    y2 = max(0, min(h - 1, int(round(xyxy_arr[i][3]))))

                    if x2 > x1 and y2 > y1:
                        corners = [
                            [float(x1), float(y1)],
                            [float(x2), float(y1)],
                            [float(x2), float(y2)],
                            [float(x1), float(y2)],
                        ]
                        candidate = {
                            "bbox": [x1, y1, x2, y2],
                            "corners": corners,
                            "confidence": round(conf, 4),
                            "class_id": raw_cls_id,
                            "class_name": cls_name,
                            "area": (x2 - x1) * (y2 - y1),
                        }
                        all_candidates.append(candidate)

        if all_candidates:
            best = max(all_candidates, key=lambda c: c["confidence"])
            return {
                "frame_id": frame_id,
                "detected": True,
                "bbox": best["bbox"],
                "corners": best["corners"],
                "confidence": best["confidence"],
                "class_name": best["class_name"],
                "class_id": best["class_id"],
                "method": "yolo",
                "all_detections": all_candidates,
            }

        return {
            "frame_id": frame_id,
            "detected": False,
            "bbox": None,
            "corners": None,
            "confidence": 0.0,
            "class_name": None,
            "class_id": None,
            "method": "yolo",
            "all_detections": [],
        }


def get_document_detector(
    method: Optional[str] = None,
    config_path: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> BaseDocumentDetector:
    """
    Factory function to instantiate the configured document detector.

    Args:
        method: "opencv_contour" (baseline) or "yolo". If None, reads from config.
        config_path: Path to configuration YAML.
        **kwargs: Detector-specific override arguments.

    Returns:
        Instance of BaseDocumentDetector.
    """
    try:
        cfg = load_config(config_path)
        det_cfg = cfg.get("document_detection", {})
    except Exception:
        det_cfg = {}

    selected_method = str(method or det_cfg.get("method", "opencv_contour")).lower().strip()

    if selected_method in ["opencv_contour", "contour", "opencv"]:
        return OpenCVContourDetector(config_path=config_path, **kwargs)
    elif selected_method in ["yolo", "yolov8"]:
        return YOLODocumentDetector(config_path=config_path, **kwargs)
    else:
        raise ValueError(
            f"Unsupported document detection method: '{selected_method}'. "
            f"Supported methods are: 'opencv_contour', 'yolo'."
        )


def run_detection_pipeline(
    frames_dir_or_paths: Union[str, Path, List[Union[str, Path]]],
    output_crops_dir: Optional[Union[str, Path]] = None,
    output_debug_dir: Optional[Union[str, Path]] = None,
    method: Optional[str] = None,
    model_name_or_path: Optional[str] = None,
    confidence_threshold: Optional[float] = None,
    config_path: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Run Phase 2 Document Detection pipeline on extracted frames.

    Args:
        frames_dir_or_paths: Directory containing frames or a list of frame file paths.
        output_crops_dir: Directory where cropped document regions will be stored.
        output_debug_dir: Directory where debug annotated frames will be stored.
        method: Detection method ("opencv_contour" or "yolo").
        model_name_or_path: Model checkpoint or name (for YOLO detector).
        confidence_threshold: Confidence threshold for detection.
        config_path: Path to configuration YAML.

    Returns:
        Structured detection summary dictionary.
    """
    from src.preprocessing.document_cropper import DocumentCropper

    detector_kwargs: Dict[str, Any] = dict(kwargs)
    if confidence_threshold is not None:
        detector_kwargs["confidence_threshold"] = confidence_threshold
    if model_name_or_path is not None:
        detector_kwargs["model_name_or_path"] = model_name_or_path

    detector = get_document_detector(
        method=method,
        config_path=config_path,
        **detector_kwargs,
    )
    cropper = DocumentCropper()

    # Collect frame file paths
    frame_files: List[Path] = []
    if isinstance(frames_dir_or_paths, (str, Path)):
        p = Path(frames_dir_or_paths).resolve()
        if p.is_dir():
            frame_files = sorted(
                [f for f in p.glob("*.*") if f.suffix.lower() in [".png", ".jpg", ".jpeg"]]
            )
        elif p.is_file():
            frame_files = [p]
    else:
        frame_files = [Path(f).resolve() for f in frames_dir_or_paths]

    if not frame_files:
        raise ValueError(f"No valid image frames found at source: {frames_dir_or_paths}")

    # Prepare output directories
    crops_dir = Path(output_crops_dir or "outputs/crops").resolve()
    debug_dir = Path(output_debug_dir or "outputs/debug_detections").resolve()
    crops_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    per_frame_results: List[Dict[str, Any]] = []
    detected_count = 0

    for idx, frame_path in enumerate(frame_files):
        frame_id = frame_path.stem
        frame = cv2.imread(str(frame_path))

        if frame is None:
            per_frame_results.append({
                "frame_id": frame_id,
                "file": frame_path.name,
                "detected": False,
                "error": "Failed to read image file",
            })
            continue

        det_result = detector.detect_frame(frame=frame, frame_id=frame_id)
        det_result["file"] = frame_path.name

        if det_result["detected"] and det_result["bbox"] is not None:
            detected_count += 1

            # 1. Save cropped region
            cropped = cropper.crop_region(
                frame=frame,
                bbox=det_result["bbox"],
                corners=det_result.get("corners"),
            )
            if cropped is not None:
                crop_path = crops_dir / f"crop_{frame_id}.png"
                cropper.save_crop_image(cropped, crop_path)
                det_result["crop_path"] = str(crop_path)
            else:
                det_result["crop_path"] = None

            # 2. Save annotated debug frame
            annotated = cropper.draw_annotation(
                frame=frame,
                bbox=det_result["bbox"],
                corners=det_result.get("corners"),
                confidence=det_result["confidence"],
                class_name=det_result["class_name"],
            )
            debug_path = debug_dir / f"debug_{frame_id}.png"
            cv2.imwrite(str(debug_path), annotated)
            det_result["debug_path"] = str(debug_path)
        else:
            # Handle no detection gracefully
            det_result["crop_path"] = None
            debug_path = debug_dir / f"debug_{frame_id}.png"
            debug_img = frame.copy()
            cv2.putText(
                debug_img,
                "NO DETECTION",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            cv2.imwrite(str(debug_path), debug_img)
            det_result["debug_path"] = str(debug_path)

        per_frame_results.append(det_result)

    total_frames = len(frame_files)
    detection_rate = round(detected_count / total_frames, 3) if total_frames > 0 else 0.0

    detector_desc = (
        f"OpenCV contour-based quadrilateral candidate detector ({type(detector).__name__})"
        if isinstance(detector, OpenCVContourDetector)
        else f"Ultralytics YOLO ({detector.model_path})"
    )

    summary = {
        "detector_type": type(detector).__name__,
        "detector_description": detector_desc,
        "confidence_threshold": detector.confidence_threshold,
        "total_frames_processed": total_frames,
        "frames_detected": detected_count,
        "frames_undetected": total_frames - detected_count,
        "detection_rate": detection_rate,
        "output_crops_directory": str(crops_dir),
        "output_debug_directory": str(debug_dir),
        "results": per_frame_results,
    }

    # Save summary to disk
    summary_file = crops_dir.parent / "detection_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
