"""
Temporal Consistency Analyzer
=============================
Evaluate cross-frame visual consistency of normalized identity document frames
extracted from a single video sequence.

Approach:
  - Generates L2-normalized deep visual embeddings using DocumentFeatureExtractor.
  - Preserves temporal frame order.
  - Calculates consecutive frame cosine similarities and pairwise similarity matrix.
  - Flags abrupt visual anomalies or document identity shifts.
  - Aggregates frame-to-frame similarities into a calibrated temporal consistency score in [0, 1].
  - Handles missing/insufficient frames and prevents cross-video frame pollution.
"""

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import cv2
import numpy as np

from src.visual.feature_extractor import DocumentFeatureExtractor

logger = logging.getLogger(__name__)


@dataclass
class TransitionAnomaly:
    """Represents an anomalous frame-to-frame visual shift."""
    frame_a_idx: int
    frame_b_idx: int
    similarity: float
    threshold: float
    flagged: bool = True


class TemporalConsistencyAnalyzer:
    """
    Analyzes temporal consistency across a sequence of document frames.
    """

    def __init__(
        self,
        feature_extractor: Optional[DocumentFeatureExtractor] = None,
        consistency_threshold: float = 0.85,
        checkpoint_path: Optional[Union[str, Path]] = "models/weights/best_integrity_model.pth",
        device: Optional[str] = None,
    ):
        self.consistency_threshold = consistency_threshold
        if feature_extractor is not None:
            self.feature_extractor = feature_extractor
        else:
            self.feature_extractor = DocumentFeatureExtractor(
                checkpoint_path=checkpoint_path,
                device=device,
            )

    def analyze_sequence(
        self,
        frames: Sequence[Union[str, Path, np.ndarray]],
        frame_ids: Optional[Sequence[Union[int, str]]] = None,
        video_ids: Optional[Sequence[str]] = None,
        debug_output_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a temporal sequence of document frames.

        Args:
            frames: Sequence of image paths or numpy arrays.
            frame_ids: Optional sequence of frame indices or labels.
            video_ids: Optional sequence of video IDs for provenance verification.
            debug_output_path: Optional path to save debug visualization.

        Returns:
            Structured dictionary containing similarity metrics and consistency score.
        """
        # 1. Provenance Check: Avoid comparing frames from different videos
        if video_ids is not None and len(video_ids) > 1:
            unique_vids = set(video_ids)
            if len(unique_vids) > 1:
                logger.error(f"Cross-video pollution detected! Multiple video IDs provided: {unique_vids}")
                return {
                    "status": "video_id_mismatch",
                    "error": f"Multiple video IDs detected in sequence: {list(unique_vids)}",
                    "num_frames": len(frames),
                    "temporal_consistency_score": 0.0,
                    "aggregate_score": 0.0,
                    "is_consistent": False,
                    "consecutive_similarities": [],
                    "anomalous_transitions": [],
                }

        # 2. Filter valid frames while preserving original indices
        valid_frames = []
        valid_indices = []

        for i, frame in enumerate(frames):
            if frame is None:
                continue
            if isinstance(frame, np.ndarray) and frame.size == 0:
                continue
            valid_frames.append(frame)
            valid_indices.append(frame_ids[i] if frame_ids is not None else i)

        num_valid = len(valid_frames)

        # 3. Handle Insufficient Frame Edge Cases (< 2 frames)
        if num_valid < 2:
            reason = "Sequence contains 0 valid frames." if num_valid == 0 else "Sequence contains only 1 valid frame (minimum 2 required)."
            logger.warning(f"Insufficient frames for temporal consistency: {reason}")
            return {
                "status": "insufficient_frames",
                "reason": reason,
                "num_frames": num_valid,
                "temporal_consistency_score": 0.0,
                "aggregate_score": 0.0,
                "mean_consecutive_similarity": 0.0,
                "min_consecutive_similarity": 0.0,
                "mean_pairwise_similarity": 0.0,
                "is_consistent": False,
                "consecutive_similarities": [],
                "anomalous_transitions": [],
            }

        # 4. Extract Deep Embeddings in Temporal Order
        try:
            embeddings = self.feature_extractor.extract_batch(valid_frames, normalize=True)
        except Exception as e:
            logger.error(f"Error during feature extraction: {e}")
            return {
                "status": "error",
                "error": str(e),
                "num_frames": num_valid,
                "temporal_consistency_score": 0.0,
                "aggregate_score": 0.0,
                "is_consistent": False,
                "consecutive_similarities": [],
                "anomalous_transitions": [],
            }

        # 5. Compute Consecutive Frame Similarities: cos(e_t, e_{t+1})
        consecutive_sims: List[float] = []
        anomalous_transitions: List[Dict[str, Any]] = []

        for t in range(num_valid - 1):
            e_t = embeddings[t]
            e_next = embeddings[t + 1]
            dot_sim = float(np.dot(e_t, e_next))
            # Clamp to [-1.0, 1.0], clip negative to 0.0 for visual consistency
            sim = max(0.0, min(1.0, dot_sim))
            consecutive_sims.append(round(sim, 4))

            if sim < self.consistency_threshold:
                anomalous_transitions.append(asdict(TransitionAnomaly(
                    frame_a_idx=valid_indices[t],
                    frame_b_idx=valid_indices[t + 1],
                    similarity=round(sim, 4),
                    threshold=self.consistency_threshold,
                )))

        # 6. Compute Pairwise Similarity Matrix
        pairwise_matrix = np.dot(embeddings, embeddings.T)
        pairwise_matrix = np.clip(pairwise_matrix, 0.0, 1.0)

        # Average over all distinct pairs (i != j)
        off_diag_mask = ~np.eye(num_valid, dtype=bool)
        mean_pairwise = float(np.mean(pairwise_matrix[off_diag_mask])) if num_valid > 1 else 1.0

        mean_consecutive = float(np.mean(consecutive_sims))
        min_consecutive = float(np.min(consecutive_sims))

        # 7. Calibrated Temporal Consistency Score
        # Stable baseline: 70% weight on consecutive stability, 30% weight on overall pairwise stability,
        # with penalty if any single transition sharply violates consistency_threshold.
        base_score = 0.65 * mean_consecutive + 0.35 * mean_pairwise
        if min_consecutive < self.consistency_threshold:
            penalty_factor = (min_consecutive / self.consistency_threshold) ** 1.5
            temporal_consistency_score = base_score * penalty_factor
        else:
            temporal_consistency_score = base_score

        temporal_consistency_score = round(max(0.0, min(1.0, temporal_consistency_score)), 4)
        is_consistent = temporal_consistency_score >= self.consistency_threshold and len(anomalous_transitions) == 0

        result = {
            "status": "success",
            "num_frames": num_valid,
            "frame_indices": valid_indices,
            "temporal_consistency_score": temporal_consistency_score,
            "aggregate_score": temporal_consistency_score,  # Alias for backward compatibility
            "mean_consecutive_similarity": round(mean_consecutive, 4),
            "min_consecutive_similarity": round(min_consecutive, 4),
            "mean_pairwise_similarity": round(mean_pairwise, 4),
            "consistency_threshold": self.consistency_threshold,
            "is_consistent": is_consistent,
            "consecutive_similarities": consecutive_sims,
            "anomalous_transitions": anomalous_transitions,
            "pairwise_matrix": np.round(pairwise_matrix, 4).tolist(),
        }

        # 8. Optional Debug Visualization
        if debug_output_path:
            self._render_debug_visualization(
                valid_frames=valid_frames,
                valid_indices=valid_indices,
                consecutive_sims=consecutive_sims,
                threshold=self.consistency_threshold,
                output_path=Path(debug_output_path),
            )

        return result

    def _render_debug_visualization(
        self,
        valid_frames: List[Any],
        valid_indices: List[Any],
        consecutive_sims: List[float],
        threshold: float,
        output_path: Path,
    ) -> None:
        """
        Generate a visual debug image showing the frame progression and transition similarities.
        Uses OpenCV for cross-platform resilience.
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            vis_height = 280
            frame_thumb_w = 160
            frame_thumb_h = 110
            n = len(valid_frames)
            vis_width = max(600, n * (frame_thumb_w + 30) + 40)

            canvas = np.full((vis_height, vis_width, 3), 245, dtype=np.uint8)

            # Title
            cv2.putText(canvas, "Temporal Consistency Transition Analysis", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)

            # Draw thumbnails and transition arrows
            x_offset = 30
            y_offset = 55

            for i in range(n):
                frame = valid_frames[i]
                if isinstance(frame, (str, Path)):
                    img = cv2.imread(str(frame))
                else:
                    img = frame

                if img is not None and img.size > 0:
                    thumb = cv2.resize(img, (frame_thumb_w, frame_thumb_h))
                    canvas[y_offset:y_offset + frame_thumb_h, x_offset:x_offset + frame_thumb_w] = thumb
                    cv2.rectangle(canvas, (x_offset, y_offset), (x_offset + frame_thumb_w, y_offset + frame_thumb_h), (180, 180, 180), 1)

                cv2.putText(canvas, f"F#{valid_indices[i]}", (x_offset + 50, y_offset + frame_thumb_h + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)

                # Draw transition similarity arrow to next frame
                if i < n - 1:
                    sim = consecutive_sims[i]
                    color = (40, 180, 40) if sim >= threshold else (40, 40, 220)
                    arrow_start = (x_offset + frame_thumb_w + 5, y_offset + frame_thumb_h // 2)
                    arrow_end = (x_offset + frame_thumb_w + 25, y_offset + frame_thumb_h // 2)
                    cv2.arrowedLine(canvas, arrow_start, arrow_end, color, 2, tipLength=0.4)

                    sim_text = f"{sim:.2f}"
                    cv2.putText(canvas, sim_text, (x_offset + frame_thumb_w - 5, y_offset + frame_thumb_h // 2 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

                x_offset += frame_thumb_w + 30

            # Overall summary footer
            mean_sim = np.mean(consecutive_sims) if consecutive_sims else 0.0
            status_text = f"Mean Consecutive Sim: {mean_sim:.4f} | Threshold: {threshold:.2f}"
            cv2.putText(canvas, status_text, (20, vis_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)

            cv2.imwrite(str(output_path), canvas)
            logger.info(f"Saved temporal consistency debug visualization to: {output_path}")
        except Exception as e:
            logger.warning(f"Could not render debug visualization: {e}")
