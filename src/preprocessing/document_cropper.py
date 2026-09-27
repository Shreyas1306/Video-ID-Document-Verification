"""
Document Cropper Module
=======================
Handles cropping of detected document bounding regions and drawing visual debug annotations.

Phase 2 Core Component:
- Crops detected document areas with configurable boundary padding.
- Clamps crop coordinates safely within frame limits.
- Generates annotated debug frames with visual bounding boxes and confidence tags.
- Saves cropped regions to disk for downstream pipeline stages.
"""

from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
import cv2
import numpy as np


class DocumentCropper:
    """
    Crops detected document bounding boxes from frames and generates debug visualizations.
    """

    def __init__(self, default_padding: float = 0.02):
        """
        Initialize the DocumentCropper.

        Args:
            default_padding: Fraction of width/height to add as safety margin around bbox.
        """
        self.default_padding = max(0.0, float(default_padding))

    def crop_region(
        self,
        frame: np.ndarray,
        bbox: Optional[List[int]] = None,
        padding_ratio: Optional[float] = None,
        corners: Optional[Union[List[Any], np.ndarray]] = None,
    ) -> Optional[np.ndarray]:
        """
        Extract the sub-image defined by bbox or corners with optional padding.

        Args:
            frame: Full BGR frame image (H, W, C).
            bbox: [x1, y1, x2, y2] bounding box coordinates.
            padding_ratio: Fraction of box width/height to pad outwards.
            corners: Optional (4, 2) corner points. If bbox is None, bbox is derived from corners.

        Returns:
            Cropped image as a numpy array, or None if bbox/corners is invalid/missing.
        """
        if frame is None:
            return None

        if bbox is None and corners is not None and len(corners) == 4:
            pts = np.asarray(corners, dtype=np.float32)
            x1 = int(np.floor(np.min(pts[:, 0])))
            y1 = int(np.floor(np.min(pts[:, 1])))
            x2 = int(np.ceil(np.max(pts[:, 0])))
            y2 = int(np.ceil(np.max(pts[:, 1])))
            bbox = [x1, y1, x2, y2]

        if bbox is None or len(bbox) != 4:
            return None

        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            return None

        h, w = frame.shape[:2]
        pad = self.default_padding if padding_ratio is None else max(0.0, float(padding_ratio))

        box_w = x2 - x1
        box_h = y2 - y1
        pad_x = int(round(box_w * pad))
        pad_y = int(round(box_h * pad))

        # Clamp padded bounds safely to image boundaries
        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(w, x2 + pad_x)
        crop_y2 = min(h, y2 + pad_y)

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return None

        cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        return cropped.copy()

    def draw_annotation(
        self,
        frame: np.ndarray,
        bbox: Optional[List[int]] = None,
        confidence: float = 0.0,
        class_name: Optional[str] = None,
        color: Tuple[int, int, int] = (0, 220, 0),
        thickness: int = 2,
        corners: Optional[Union[List[Any], np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Create a copy of frame with visual bounding box or quadrilateral polygon and confidence badge overlay.

        Args:
            frame: Base frame image.
            bbox: [x1, y1, x2, y2] coordinates.
            confidence: Confidence score of detection.
            class_name: Predicted class label name.
            color: BGR tuple for rectangle and badge.
            thickness: Line width for bounding box / polygon.
            corners: Optional (4, 2) corner points. If provided, renders quadrilateral polygon and corner dots.

        Returns:
            Annotated frame copy.
        """
        annotated = frame.copy()
        if bbox is None and corners is None:
            return annotated

        # If corners are provided, draw exact quadrilateral polygon and corner circles
        if corners is not None and len(corners) == 4:
            pts = np.asarray(corners, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=thickness)
            corner_labels = ["TL", "TR", "BR", "BL"]
            for i, pt in enumerate(corners):
                cx, cy = int(round(pt[0])), int(round(pt[1]))
                cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)
                lbl = corner_labels[i] if i < len(corner_labels) else f"C{i+1}"
                cv2.putText(
                    annotated,
                    lbl,
                    (cx + 4, cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            if corners is None:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        elif corners is not None:
            pts = np.asarray(corners, dtype=np.float32)
            x1 = int(np.floor(np.min(pts[:, 0])))
            y1 = int(np.floor(np.min(pts[:, 1])))
            x2 = int(np.ceil(np.max(pts[:, 0])))
            y2 = int(np.ceil(np.max(pts[:, 1])))
        else:
            return annotated

        # Create informative label tag
        tag = f"{class_name or 'document'}: {confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1

        (text_w, text_h), baseline = cv2.getTextSize(tag, font, font_scale, font_thickness)
        badge_y1 = max(0, y1 - text_h - baseline - 6)
        badge_y2 = y1
        badge_x2 = min(frame.shape[1], x1 + text_w + 8)

        # Background badge for high-contrast readability
        cv2.rectangle(annotated, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
        cv2.putText(
            annotated,
            tag,
            (x1 + 4, badge_y2 - baseline - 2),
            font,
            font_scale,
            (0, 0, 0),  # Black text for contrast on colored badge
            font_thickness,
            cv2.LINE_AA,
        )

        return annotated

    def save_crop_image(self, cropped: np.ndarray, file_path: Union[str, Path]) -> bool:
        """
        Save cropped document image to disk.

        Args:
            cropped: Cropped numpy array image.
            file_path: Output file path.

        Returns:
            True if write succeeded, False otherwise.
        """
        if cropped is None or cropped.size == 0:
            return False

        out_file = Path(file_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)
        return cv2.imwrite(str(out_file), cropped)
