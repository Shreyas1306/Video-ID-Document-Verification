"""
Temporal Document Tracker Module
================================
Stateful temporal tracking and multi-frame consistency for identity document localization.

Maintains temporal continuity across consecutive video frames by:
1. Evaluating geometric displacement (centroid and corner shifts relative to previous frame).
2. Verifying area ratio and aspect ratio stability across frames.
3. Calculating polygon IoU between consecutive frame candidates.
4. Fusing static candidate scores with temporal transition plausibility.
5. Applying exponential corner smoothing to suppress minor camera jitter.
6. Gracefully resetting state when tracking confidence drops below minimum thresholds.
"""

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from shapely.geometry import Polygon

from src.preprocessing.candidate_scorer import CandidateScorer
from src.preprocessing.perspective_corrector import order_points
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)


def compute_quad_iou(pts1: np.ndarray, pts2: np.ndarray) -> float:
    """Compute polygon IoU between two quadrilaterals."""
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


class TemporalDocumentTracker:
    """
    Stateful document tracker enforcing temporal smoothness and multi-frame consistency.
    """

    def __init__(
        self,
        scorer: Optional[CandidateScorer] = None,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the temporal tracker.

        Args:
            scorer: Optional CandidateScorer instance.
            config: Optional configuration dictionary.
            config_path: Optional path to settings.yaml.
        """
        cfg = config
        if cfg is None:
            try:
                full_cfg = load_config(config_path)
                cfg = full_cfg.get("document_candidate_scoring", {})
            except Exception:
                cfg = {}

        self.scorer = scorer if scorer is not None else CandidateScorer(config=cfg, config_path=config_path)
        self.temporal_enabled = bool(cfg.get("temporal_enabled", True))

        weights = cfg.get("weights", {})
        self.w_temporal = float(weights.get("temporal", 0.40)) if self.temporal_enabled else 0.0

        t_cfg = cfg.get("temporal", {})
        self.max_corner_disp = float(t_cfg.get("max_corner_displacement", 0.20))
        self.max_center_disp = float(t_cfg.get("max_center_displacement", 0.15))
        self.max_area_change = float(t_cfg.get("max_area_change", 0.50))
        self.min_tracking_score = float(t_cfg.get("min_tracking_score", 0.35))
        self.smoothing_alpha = float(t_cfg.get("smoothing_alpha", 0.70))
        self.max_frames_lost = int(t_cfg.get("max_frames_lost", 5))

        # Stateful tracking tracking variables
        self.last_accepted_quad: Optional[np.ndarray] = None
        self.last_accepted_area: float = 0.0
        self.last_accepted_aspect: float = 0.0
        self.tracking_confidence: float = 0.0
        self.frames_tracked: int = 0
        self.frames_lost: int = 0
        self.history: List[Dict[str, Any]] = []

    def reset(self) -> None:
        """Reset internal tracking state for a new video or after track loss."""
        self.last_accepted_quad = None
        self.last_accepted_area = 0.0
        self.last_accepted_aspect = 0.0
        self.tracking_confidence = 0.0
        self.frames_tracked = 0
        self.frames_lost = 0
        self.history.clear()

    def compute_temporal_similarity(
        self,
        cand_corners: np.ndarray,
        cand_area: float,
        cand_aspect: float,
        frame_shape: Tuple[int, int],
    ) -> float:
        """
        Compute temporal similarity score [0.0, 1.0] between a candidate and the last accepted quad.
        """
        if self.last_accepted_quad is None:
            return 1.0

        h, w = frame_shape[:2]
        diag = float(math.sqrt(h * h + w * w))

        cand_ord = order_points(cand_corners)
        prev_ord = self.last_accepted_quad

        # 1. Centroid displacement
        c_cand = np.mean(cand_ord, axis=0)
        c_prev = np.mean(prev_ord, axis=0)
        dist_center = float(np.linalg.norm(c_cand - c_prev)) / max(1.0, diag)
        s_center = max(0.0, 1.0 - (dist_center / max(1e-4, self.max_center_disp)))

        # 2. Corner displacement
        corner_dists = np.linalg.norm(cand_ord - prev_ord, axis=1) / max(1.0, diag)
        mean_corner_dist = float(np.mean(corner_dists))
        s_corner = max(0.0, 1.0 - (mean_corner_dist / max(1e-4, self.max_corner_disp)))

        # 3. Area stability
        if self.last_accepted_area > 0 and cand_area > 0:
            ratio = min(cand_area, self.last_accepted_area) / max(cand_area, self.last_accepted_area)
            s_area = ratio if ratio >= (1.0 - self.max_area_change) else 0.5 * ratio
        else:
            s_area = 0.5

        # 4. Aspect ratio stability
        if self.last_accepted_aspect > 0 and cand_aspect > 0:
            aspect_diff = abs(cand_aspect - self.last_accepted_aspect)
            s_aspect = max(0.0, 1.0 - 2.5 * aspect_diff)
        else:
            s_aspect = 0.5

        # 5. Polygon IoU
        s_iou = compute_quad_iou(cand_ord, prev_ord)

        # Composite temporal score
        temporal_score = (
            0.35 * s_iou
            + 0.30 * s_corner
            + 0.15 * s_center
            + 0.10 * s_area
            + 0.10 * s_aspect
        )
        return float(np.clip(temporal_score, 0.0, 1.0))

    def process_frame(
        self,
        candidates: List[Dict[str, Any]],
        frame: np.ndarray,
        frame_id: Optional[Union[str, int]] = None,
        grad_mag: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Process quadrilateral candidates for the current frame, applying static scoring
        and temporal continuity guidance.

        Args:
            candidates: List of candidate dictionaries extracted from contour detector.
            frame: Current BGR frame.
            frame_id: Optional frame identifier.
            grad_mag: Optional pre-computed gradient magnitude.

        Returns:
            Dictionary with tracking result, selected corners, bbox, and confidence.
        """
        h, w = frame.shape[:2]

        if not candidates:
            self.frames_lost += 1
            if self.frames_lost > self.max_frames_lost:
                self.reset()
            return {
                "frame_id": frame_id,
                "detected": False,
                "bbox": None,
                "corners": None,
                "confidence": 0.0,
                "static_score": 0.0,
                "temporal_score": 0.0,
                "method": "no_candidates",
                "all_detections": [],
                "reason": "No candidates provided",
            }

        # Step 1: Static Candidate Scoring
        scored = self.scorer.score_candidates(candidates, (h, w), grad_mag)

        # Step 2: Temporal Guidance & Fusion
        evaluated_candidates = []
        is_tracking_active = self.temporal_enabled and (self.last_accepted_quad is not None)

        for c in scored:
            cand = dict(c)
            corners = cand["corners"]
            area = cand["area"]
            aspect = cand["aspect_ratio"]
            static_score = cand["static_score"]

            if is_tracking_active:
                temp_score = self.compute_temporal_similarity(corners, area, aspect, (h, w))
                fused_score = (1.0 - self.w_temporal) * static_score + self.w_temporal * temp_score
            else:
                temp_score = 1.0
                fused_score = static_score

            cand["temporal_score"] = round(temp_score, 4)
            cand["fused_score"] = round(float(np.clip(fused_score, 0.0, 1.0)), 4)
            evaluated_candidates.append(cand)

        # Sort by fused score descending
        evaluated_candidates.sort(key=lambda x: x["fused_score"], reverse=True)
        best = evaluated_candidates[0]

        # Step 3: Acceptance & State Update
        if best["fused_score"] >= self.min_tracking_score:
            best_corners = order_points(best["corners"])

            # Apply exponential smoothing if tracking active
            if is_tracking_active:
                smoothed_corners = (
                    self.smoothing_alpha * best_corners
                    + (1.0 - self.smoothing_alpha) * self.last_accepted_quad
                )
                self.last_accepted_quad = smoothed_corners
                method = "temporally_guided"
            else:
                self.last_accepted_quad = best_corners
                method = "first_frame_detection"

            self.last_accepted_area = float(best["area"])
            self.last_accepted_aspect = float(best["aspect_ratio"])
            self.tracking_confidence = float(best["fused_score"])
            self.frames_tracked += 1
            self.frames_lost = 0

            # Bounding box
            x1 = max(0, min(w - 1, int(round(float(np.min(self.last_accepted_quad[:, 0]))))))
            y1 = max(0, min(h - 1, int(round(float(np.min(self.last_accepted_quad[:, 1]))))))
            x2 = max(0, min(w - 1, int(round(float(np.max(self.last_accepted_quad[:, 0]))))))
            y2 = max(0, min(h - 1, int(round(float(np.max(self.last_accepted_quad[:, 1]))))))

            res = {
                "frame_id": frame_id,
                "detected": True,
                "bbox": [x1, y1, x2, y2],
                "corners": self.last_accepted_quad.tolist(),
                "confidence": round(self.tracking_confidence, 4),
                "static_score": best["static_score"],
                "temporal_score": best["temporal_score"],
                "class_name": "identity_document",
                "class_id": 0,
                "method": method,
                "all_detections": [
                    {
                        "bbox": c["bbox"],
                        "corners": c["corners"].tolist() if isinstance(c["corners"], np.ndarray) else c["corners"],
                        "confidence": c["fused_score"],
                        "area": c["area"],
                        "aspect_ratio": c.get("aspect_ratio"),
                        "static_score": c.get("static_score"),
                        "temporal_score": c.get("temporal_score"),
                    }
                    for c in evaluated_candidates
                ],
                "reason": None,
            }
            self.history.append({"frame_id": frame_id, "confidence": self.tracking_confidence})
            return res
        else:
            # Score below threshold
            self.frames_lost += 1
            if self.frames_lost > self.max_frames_lost:
                self.reset()

            return {
                "frame_id": frame_id,
                "detected": False,
                "bbox": None,
                "corners": None,
                "confidence": round(float(best["fused_score"]), 4),
                "static_score": best["static_score"],
                "temporal_score": best["temporal_score"],
                "method": "rejected_low_confidence",
                "all_detections": [
                    {
                        "bbox": c["bbox"],
                        "corners": c["corners"].tolist() if isinstance(c["corners"], np.ndarray) else c["corners"],
                        "confidence": c["fused_score"],
                        "area": c["area"],
                    }
                    for c in evaluated_candidates
                ],
                "reason": f"Best candidate score ({best['fused_score']:.3f}) below threshold ({self.min_tracking_score:.3f})",
            }
