"""
Document Candidate Scorer Module
================================
Multi-criteria geometric and appearance ranking for identity document candidates.

Evaluates candidates extracted from the contour detection stage using:
1. Aspect Ratio Plausibility: Proximity to ISO/IEC 7810 ID-1 (1.586) or ID-3 passport (1.420).
2. Frame Area Fraction: Document-like area fraction (15% - 55% frame coverage vs full screen / desk).
3. Rectangularity & Orthogonality: Good adherence to quadrilateral geometry.
4. Edge Gradient Quality: Boundary sharpness along candidate edges.
5. Nesting & Containment Analysis: Demotes outer containers (screens, bezels, carrier sheets)
   in favor of enclosed document quadrilaterals.
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.preprocessing.perspective_corrector import order_points
from src.utils.config_loader import load_config


class CandidateScorer:
    """
    Ranks and scores quadrilateral document candidates using multi-factor geometric heuristics.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the candidate scorer.

        Args:
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

        self.enabled = bool(cfg.get("enabled", True))
        self.target_aspects = [float(a) for a in cfg.get("target_aspect_ratios", [1.586, 1.420])]

        weights = cfg.get("weights", {})
        self.w_aspect = float(weights.get("aspect_ratio", 0.30))
        self.w_area = float(weights.get("area", 0.25))
        self.w_rect = float(weights.get("rectangularity", 0.20))
        self.w_edge = float(weights.get("edge_quality", 0.15))
        self.w_nesting = float(weights.get("nesting_bonus", 0.10))

    def score_aspect_ratio(self, aspect: float) -> float:
        """
        Compute aspect ratio plausibility score.
        Rewards proximity to standard identity document aspect ratios (ID-1: 1.586, ID-3: 1.420).
        """
        if aspect <= 0.0:
            return 0.0
        # Normalise orientation so aspect is >= 1.0
        norm_aspect = aspect if aspect >= 1.0 else 1.0 / aspect
        min_dist = min(abs(norm_aspect - target) for target in self.target_aspects)
        # Gaussian decay around closest target aspect ratio
        return float(np.exp(-3.5 * (min_dist ** 2)))

    def score_area_ratio(self, area_ratio: float) -> float:
        """
        Compute frame area fraction score.
        Documents typically occupy 15% to 55% of the frame. Outer bezels (>65%) or tiny
        artifacts (<8%) are heavily penalised.
        """
        if area_ratio <= 0.0 or area_ratio >= 1.0:
            return 0.0
        # Optimal center at 0.32 with standard deviation 0.16
        center = 0.32
        sigma = 0.16
        score = float(np.exp(-((area_ratio - center) ** 2) / (2.0 * (sigma ** 2))))
        # Additional penalty if excessively large (e.g. tablet screen or carrier sheet > 60%)
        if area_ratio > 0.60:
            penalty = max(0.0, 1.0 - (area_ratio - 0.60) / 0.30)
            score *= penalty
        return float(np.clip(score, 0.0, 1.0))

    def score_rectangularity(self, cand: Dict[str, Any]) -> float:
        """
        Score rectangularity and corner orthogonality of the candidate.
        """
        rect_score = float(cand.get("rectangularity", 1.0))

        corners = cand.get("corners")
        if corners is None or len(corners) != 4:
            return float(np.clip(rect_score, 0.0, 1.0))

        pts = np.asarray(corners, dtype=np.float32).reshape((4, 2))
        e0 = pts[1] - pts[0]
        e1 = pts[2] - pts[1]
        e2 = pts[3] - pts[2]
        e3 = pts[0] - pts[3]

        def cos_angle(u: np.ndarray, v: np.ndarray) -> float:
            norm = float(np.linalg.norm(u) * np.linalg.norm(v))
            return float(abs(np.dot(u, v) / max(1e-6, norm)))

        cos_angles = [
            cos_angle(e0, -e3),
            cos_angle(e1, -e0),
            cos_angle(e2, -e1),
            cos_angle(e3, -e2),
        ]
        s_ortho = float(max(0.0, 1.0 - float(np.mean(cos_angles))))
        return float(np.clip(0.6 * rect_score + 0.4 * s_ortho, 0.0, 1.0))

    def compute_edge_quality(
        self,
        corners: np.ndarray,
        grad_mag: np.ndarray,
        samples_per_edge: int = 15,
    ) -> float:
        """
        Sample gradient magnitude along candidate perimeter to quantify edge sharpness.
        """
        if grad_mag is None or grad_mag.size == 0 or len(corners) != 4:
            return 0.5

        pts = np.asarray(corners, dtype=np.float32).reshape((4, 2))
        h, w = grad_mag.shape[:2]

        edge_samples = []
        for i in range(4):
            p1 = pts[i]
            p2 = pts[(i + 1) % 4]
            for t in np.linspace(0.1, 0.9, samples_per_edge):
                pt = p1 + t * (p2 - p1)
                x = int(round(np.clip(pt[0], 0, w - 1)))
                y = int(round(np.clip(pt[1], 0, h - 1)))
                edge_samples.append(float(grad_mag[y, x]))

        if not edge_samples:
            return 0.5

        mean_grad = float(np.mean(edge_samples))
        # Normalize: typical gradient in 0-255 scale
        norm_grad = float(np.clip(mean_grad / 80.0, 0.0, 1.0))
        return norm_grad

    def detect_nesting_and_containment(self, candidates: List[Dict[str, Any]]) -> None:
        """
        Detect geometric containment between candidates.
        If Candidate A is fully inside Candidate B:
          - A is marked is_nested = True, receives nesting bonus.
          - B is marked as container, receives container penalty.
        """
        for c in candidates:
            c.setdefault("is_nested", False)
            c.setdefault("encloses_count", 0)

        n = len(candidates)
        if n < 2:
            return

        for i in range(n):
            c1 = candidates[i]
            pts1 = np.asarray(c1["corners"], dtype=np.float32).reshape((4, 2))
            for j in range(n):
                if i == j:
                    continue
                c2 = candidates[j]
                pts2 = np.asarray(c2["corners"], dtype=np.float32).reshape((4, 2))

                # Check if all 4 corners of c1 are inside polygon c2
                cnt2 = pts2.astype(np.int32)
                inside = all(
                    cv2.pointPolygonTest(cnt2, (float(p[0]), float(p[1])), False) >= 0
                    for p in pts1
                )
                if inside and c1["area"] < c2["area"] * 0.85:
                    c1["is_nested"] = True
                    c1["nested_parent_idx"] = j
                    c2["encloses_count"] = c2.get("encloses_count", 0) + 1

    def score_candidates(
        self,
        candidates: List[Dict[str, Any]],
        frame_shape: Tuple[int, int],
        grad_mag: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Score and rank a list of quadrilateral candidates.

        Args:
            candidates: List of candidate dictionaries.
            frame_shape: (height, width) of the source frame.
            grad_mag: Optional precomputed gradient magnitude map.

        Returns:
            Scored candidates sorted in descending order of total score.
        """
        if not candidates:
            return []

        h, w = frame_shape[:2]
        img_area = float(h * w)

        # Detect nesting relationships
        self.detect_nesting_and_containment(candidates)

        scored_candidates = []
        for c in candidates:
            cand = dict(c)
            # Ensure corners are numpy array
            if not isinstance(cand["corners"], np.ndarray):
                cand["corners"] = np.asarray(cand["corners"], dtype=np.float32).reshape((4, 2))

            corners = cand["corners"]
            area = float(cand.get("area", cv2.contourArea(corners.astype(np.int32))))
            cand["area"] = area
            area_ratio = float(cand.get("area_ratio", area / max(1.0, img_area)))
            cand["area_ratio"] = area_ratio

            # Aspect ratio
            if "aspect_ratio" not in cand:
                w_top = float(np.linalg.norm(corners[1] - corners[0]))
                w_bot = float(np.linalg.norm(corners[2] - corners[3]))
                h_left = float(np.linalg.norm(corners[3] - corners[0]))
                h_right = float(np.linalg.norm(corners[2] - corners[1]))
                mean_w = max(1.0, 0.5 * (w_top + w_bot))
                mean_h = max(1.0, 0.5 * (h_left + h_right))
                cand["aspect_ratio"] = round(max(mean_w, mean_h) / min(mean_w, mean_h), 3)

            aspect = float(cand["aspect_ratio"])

            # Compute individual scores
            s_aspect = self.score_aspect_ratio(aspect)
            s_area = self.score_area_ratio(area_ratio)
            s_rect = self.score_rectangularity(cand)

            if "edge_quality" in cand:
                s_edge = float(cand["edge_quality"])
            elif grad_mag is not None:
                s_edge = self.compute_edge_quality(corners, grad_mag)
            else:
                s_edge = 0.5
            cand["edge_quality"] = round(s_edge, 3)

            # Nesting adjustment
            nesting_delta = 0.0
            if cand.get("is_nested", False):
                # An inner candidate with good document aspect ratio receives a substantial boost
                if s_aspect > 0.5:
                    nesting_delta += 0.20
            if cand.get("encloses_count", 0) > 0:
                # A candidate enclosing another candidate is heavily penalized (likely carrier screen/bezel)
                nesting_delta -= 0.30

            total_static = (
                self.w_aspect * s_aspect
                + self.w_area * s_area
                + self.w_rect * s_rect
                + self.w_edge * s_edge
                + self.w_nesting * nesting_delta
            )
            total_static = float(np.clip(total_static, 0.0, 1.0))

            cand["score_aspect"] = round(s_aspect, 4)
            cand["score_area"] = round(s_area, 4)
            cand["score_rect"] = round(s_rect, 4)
            cand["score_edge"] = round(s_edge, 4)
            cand["nesting_delta"] = round(nesting_delta, 4)
            cand["static_score"] = round(total_static, 4)
            # Retain composite confidence
            cand["confidence"] = round(total_static, 4)

            scored_candidates.append(cand)

        # Sort by total score descending
        scored_candidates.sort(key=lambda x: x["static_score"], reverse=True)
        return scored_candidates

    def select_best_candidate(
        self,
        candidates: List[Dict[str, Any]],
        frame_shape: Tuple[int, int],
        grad_mag: Optional[np.ndarray] = None,
        min_score: float = 0.30,
    ) -> Optional[Dict[str, Any]]:
        """
        Score and select the single best candidate exceeding minimum score threshold.
        """
        scored = self.score_candidates(candidates, frame_shape, grad_mag)
        if not scored:
            return None
        best = scored[0]
        if best["static_score"] < min_score:
            return None
        return best
