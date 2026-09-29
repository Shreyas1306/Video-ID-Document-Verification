"""
Hybrid Document Detector Module
===============================
Provides an experimental two-tier hybrid document localization backend:
  Tier 1: Fast OpenCV V2 multi-candidate contour detection with CandidateScorer.
  Tier 2: MIDV-500 UNet (ResNet-34) learned document segmentation fallback.
  Gate:   Validated Variant C (Dual-Path Pristine) Ambiguity Gate.

Architecture:
  Input Frame
      │
      ▼
  OpenCV V2 Candidate Generation + CandidateScorer
      │
      ▼
  Variant C Ambiguity Gate Evaluation
      ├── Confident (or Pristine Card) ──► Fast Path: OpenCV Localization
      └── Ambiguous / Carrier / Failure ──► Fallback: MIDV-500 UNet Segmentation
                                                    │
                                                    ▼
                                          Contour-to-Quad Extraction
                                                    │
                                                    ▼
                                          Unified 4-Corner Result
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

try:
    import torch
    import segmentation_models_pytorch as smp
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.preprocessing.candidate_scorer import CandidateScorer
from src.preprocessing.document_detector import BaseDocumentDetector, OpenCVContourDetector
from src.preprocessing.perspective_corrector import order_points
from src.preprocessing.temporal_document_tracker import TemporalDocumentTracker

logger = logging.getLogger(__name__)


@dataclass
class VariantCAmbiguityGate:
    """
    Variant C (Dual-Path Pristine) Ambiguity Gate.
    Evaluates inference-time geometric and sharpness signals to route frames
    between the OpenCV fast path and the MIDV-500 UNet fallback.

    Thresholds (strictly validated from refinement experiment):
      - max_area_ratio = 0.35 (Carrier sheet physical cutoff)
      - compound_carrier_area = 0.30, compound_carrier_max_aspect = 1.50
      - check_container = True (Multi-candidate nesting / container trap)
      - min_aspect_score = 0.75 (Aspect plausibility)
      - pristine_exemption:
          area in [0.06, 0.28], aspect in [1.53, 1.68], rect >= 0.92
          effective_min_static = 0.60
      - default min_static_score = 0.70
    """
    name: str = "variant_c_dual_path"
    min_static_score: float = 0.70
    max_area_ratio: float = 0.35
    compound_carrier_area: Optional[float] = 0.30
    compound_carrier_max_aspect: Optional[float] = 1.50
    pristine_exemption_enabled: bool = True
    pristine_min_static: float = 0.60
    min_aspect_score: float = 0.75
    check_container: bool = True
    description: str = "Variant C: Area 0.35 cutoff + Pristine Card Exemption (rect >= 0.92, aspect in [1.53, 1.68])."

    def evaluate(
        self,
        opencv_res: Optional[Dict[str, Any]],
        frame_shape: Tuple[int, int],
        scorer: Optional[CandidateScorer] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate candidate signals to determine whether learned UNet fallback is invoked.
        Operates strictly on inference-time features (zero GT leakage).
        """
        if opencv_res is None:
            return {
                "invoke_unet": True,
                "reason": "opencv_not_detected",
                "signals": {
                    "detected": False, "static_score": 0.0, "aspect_score": 0.0,
                    "aspect_ratio": 0.0, "area_ratio": 0.0, "edge_score": 0.0,
                    "is_container": False, "rect_score": 0.0, "is_pristine": False,
                },
            }

        detected = bool(opencv_res.get("detected", False))
        corners = opencv_res.get("corners")

        # Fallback 1: Total detection failure
        if not detected or corners is None:
            return {
                "invoke_unet": True,
                "reason": "opencv_not_detected",
                "signals": {
                    "detected": False, "static_score": 0.0, "aspect_score": 0.0,
                    "aspect_ratio": 0.0, "area_ratio": 0.0, "edge_score": 0.0,
                    "is_container": False, "rect_score": 0.0, "is_pristine": False,
                },
            }

        h, w = frame_shape[:2]
        frame_area = float(max(1, h * w))

        all_dets = opencv_res.get("all_detections", [])
        best_cand = all_dets[0] if all_dets else {}

        static_score = float(opencv_res.get("static_score", best_cand.get("static_score", 0.0)))
        cand_area = float(best_cand.get("area", 0.0))
        if cand_area <= 0.0 and corners is not None:
            cand_area = float(cv2.contourArea(np.array(corners, dtype=np.float32)))
        area_ratio = float(cand_area / frame_area)

        # Aspect ratio
        aspect_ratio = float(best_cand.get("aspect_ratio", 0.0))
        pts = np.array(corners, dtype=np.float32).reshape((4, 2))
        if aspect_ratio <= 0.0 and corners is not None:
            w_top = float(np.linalg.norm(pts[1] - pts[0]))
            w_bot = float(np.linalg.norm(pts[2] - pts[3]))
            h_left = float(np.linalg.norm(pts[3] - pts[0]))
            h_right = float(np.linalg.norm(pts[2] - pts[1]))
            mw = max(1.0, 0.5 * (w_top + w_bot))
            mh = max(1.0, 0.5 * (h_left + h_right))
            aspect_ratio = max(mw, mh) / min(mw, mh)

        if scorer is not None and aspect_ratio > 0.0:
            aspect_score = scorer.score_aspect_ratio(aspect_ratio)
            rect_score = scorer.score_rectangularity({"corners": pts, "rectangularity": 1.0})
        else:
            norm_aspect = aspect_ratio if aspect_ratio >= 1.0 else (1.0 / max(1e-4, aspect_ratio))
            d_id1 = abs(norm_aspect - 1.586)
            d_id3 = abs(norm_aspect - 1.420)
            aspect_score = float(np.exp(-3.5 * (min(d_id1, d_id3) ** 2)))
            rect_score = float(best_cand.get("score_rect", 0.85))

        edge_score = float(best_cand.get("edge_quality", 0.5))

        is_container = False
        if self.check_container:
            is_container = bool(best_cand.get("encloses_count", 0) > 0)
            if not is_container and "nesting_delta" in best_cand:
                is_container = best_cand["nesting_delta"] < 0.0

        # Pristine card test: Standard ID-1 card presentation
        is_pristine = (
            (0.06 <= area_ratio <= 0.28)
            and (1.53 <= aspect_ratio <= 1.68)
            and (rect_score >= 0.92)
        )

        signals = {
            "detected": True,
            "static_score": round(static_score, 4),
            "aspect_ratio": round(aspect_ratio, 3),
            "aspect_score": round(aspect_score, 4),
            "area_ratio": round(area_ratio, 4),
            "rect_score": round(rect_score, 4),
            "edge_score": round(edge_score, 4),
            "is_container": is_container,
            "is_pristine": is_pristine,
        }

        # 1. Unconditional carrier sheet area cutoff
        if area_ratio > self.max_area_ratio:
            return {
                "invoke_unet": True,
                "reason": f"carrier_area_exceeded ({area_ratio:.3f} > {self.max_area_ratio})",
                "signals": signals,
            }

        # 2. Compound carrier paper signature
        if (
            self.compound_carrier_area is not None
            and self.compound_carrier_max_aspect is not None
        ):
            if (
                area_ratio > self.compound_carrier_area
                and aspect_ratio < self.compound_carrier_max_aspect
            ):
                return {
                    "invoke_unet": True,
                    "reason": f"compound_carrier_signature (area={area_ratio:.3f}>{self.compound_carrier_area} and aspect={aspect_ratio:.3f}<{self.compound_carrier_max_aspect})",
                    "signals": signals,
                }

        # 3. Outer container trap
        if is_container:
            return {
                "invoke_unet": True,
                "reason": "encloses_inner_quad_container",
                "signals": signals,
            }

        # 4. Aspect score plausibility
        if aspect_score < self.min_aspect_score:
            return {
                "invoke_unet": True,
                "reason": f"aspect_ratio_deviation ({aspect_score:.3f} < {self.min_aspect_score})",
                "signals": signals,
            }

        # 5. Static score check with pristine exemption
        effective_min_static = self.min_static_score
        if self.pristine_exemption_enabled and is_pristine:
            effective_min_static = self.pristine_min_static

        if static_score < effective_min_static:
            return {
                "invoke_unet": True,
                "reason": f"low_static_score ({static_score:.3f} < {effective_min_static:.3f})",
                "signals": signals,
            }

        return {
            "invoke_unet": False,
            "reason": "confident_opencv_candidate",
            "signals": signals,
        }


class MIDV500Segmenter:
    """
    Learned Document Segmentation wrapper using UNet with ResNet-34 backbone
    pretrained on the MIDV-500 identity document dataset.
    """

    def __init__(
        self,
        weights_path: Union[str, Path] = "models/weights/midv500_unet_resnet34.pth",
        input_size: int = 768,
        device: str = "cpu",
        confidence_threshold: float = 0.50,
        min_area_ratio: float = 0.01,
        lazy_load: bool = False,
    ):
        self.weights_path = Path(weights_path)
        self.input_size = input_size
        self.device_str = device
        self.confidence_threshold = confidence_threshold
        self.min_area_ratio = min_area_ratio
        self.model: Optional[Any] = None

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        if not lazy_load:
            self._load_model()

    def _load_model(self) -> None:
        """Load UNet model architecture and weights."""
        if self.model is not None:
            return

        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch or segmentation_models_pytorch is not installed. "
                "Cannot initialize MIDV500Segmenter."
            )

        if not self.weights_path.exists():
            raise FileNotFoundError(f"MIDV-500 model weights not found at: {self.weights_path}")

        self.device = torch.device(self.device_str)
        logger.info(f"Loading MIDV-500 UNet (ResNet-34) from: {self.weights_path} (device: {self.device})")
        self.model = smp.Unet(encoder_name="resnet34", classes=1, encoder_weights=None)
        ckpt = torch.load(str(self.weights_path), map_location=self.device)
        sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        clean_sd = {k.replace("model.", ""): v for k, v in sd.items()}
        self.model.load_state_dict(clean_sd)
        self.model.to(self.device)
        self.model.eval()

    def preprocess_image(self, img_bgr: np.ndarray) -> Tuple[Any, int, int, int, int]:
        """
        Resize with aspect ratio preserved, pad to multiple of 32, normalize ImageNet.
        """
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        scale = self.input_size / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        img_resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)

        pad_h = (32 - nh % 32) % 32
        pad_w = (32 - nw % 32) % 32
        img_padded = cv2.copyMakeBorder(
            img_resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0
        )

        norm_img = ((img_padded.astype(np.float32) / 255.0) - self.mean) / self.std
        tensor = torch.from_numpy(norm_img.transpose(2, 0, 1)).unsqueeze(0).float()
        return tensor.to(self.device), nh, nw, pad_h, pad_w

    def predict_mask(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Forward pass yielding full-resolution binary mask and probability map.
        Returns:
            mask: uint8 binary mask (0 or 255)
            prob_map: float32 map [0.0, 1.0]
            inference_ms: inference runtime in milliseconds
        """
        self._load_model()
        h, w = img_bgr.shape[:2]
        tensor, nh, nw, pad_h, pad_w = self.preprocess_image(img_bgr)

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = self.model(tensor)
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()
        inf_time_ms = (time.perf_counter() - t0) * 1000.0

        # Unpad
        prob_unpadded = prob[:nh, :nw]
        # Resize to original dimensions
        prob_orig = cv2.resize(prob_unpadded, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = (prob_orig >= self.confidence_threshold).astype(np.uint8) * 255

        return mask, prob_orig, inf_time_ms

    def extract_quadrilateral(
        self, mask: np.ndarray, prob_map: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray], float, str]:
        """
        Derive canonical 4-corner quadrilateral polygon from binary mask.
        Returns:
            detected: bool
            corners: (4, 2) ordered float coordinates or None
            confidence: float mean probability
            method: string method description
        """
        h, w = mask.shape[:2]
        total_pixels = h * w
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return False, None, 0.0, "no_contours"

        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < (total_pixels * self.min_area_ratio):
            return False, None, 0.0, "area_too_small"

        peri = cv2.arcLength(c, True)
        best_quad = None
        method = "approx_poly_dp"

        # Adaptive epsilon search for 4-vertex polygon
        for eps_factor in np.linspace(0.01, 0.08, 35):
            approx = cv2.approxPolyDP(c, eps_factor * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                best_quad = order_points(approx.reshape(4, 2).astype(np.float32))
                method = f"poly_approx_eps_{eps_factor:.3f}"
                break

        # Fallback to minAreaRect if 4-point polygon was not found
        if best_quad is None:
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            best_quad = order_points(box.astype(np.float32))
            method = "min_area_rect_fallback"

        # Confidence: average probability in segmented region
        mask_binary = (mask > 0)
        conf = float(prob_map[mask_binary].mean()) if mask_binary.sum() > 0 else 0.0

        return True, best_quad, float(np.clip(conf, 0.0, 1.0)), method

    def process_frame(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        """
        End-to-end segmentation and quadrilateral extraction for a single frame.
        """
        h, w = img_bgr.shape[:2]
        mask, prob_map, inf_time_ms = self.predict_mask(img_bgr)
        detected, corners, confidence, method = self.extract_quadrilateral(mask, prob_map)

        bbox = None
        if detected and corners is not None:
            x1 = max(0, min(w - 1, int(round(float(np.min(corners[:, 0]))))))
            y1 = max(0, min(h - 1, int(round(float(np.min(corners[:, 1]))))))
            x2 = max(0, min(w - 1, int(round(float(np.max(corners[:, 0]))))))
            y2 = max(0, min(h - 1, int(round(float(np.max(corners[:, 1]))))))
            bbox = [x1, y1, x2, y2]

        return {
            "detected": detected,
            "corners": corners,
            "confidence": confidence,
            "bbox": bbox,
            "method": method,
            "inference_time_ms": inf_time_ms,
            "mask": mask,
        }


class HybridDocumentDetector(BaseDocumentDetector):
    """
    Experimental Hybrid Document Detector combining:
      1. Fast OpenCV V2 multi-candidate contour detector + CandidateScorer.
      2. Validated Variant C Ambiguity Gate.
      3. Learned MIDV-500 UNet segmentation fallback.

    Produces a unified detection result compatible with BaseDocumentDetector,
    TemporalDocumentTracker, and PerspectiveCorrector.
    """

    def __init__(
        self,
        opencv_detector: Optional[OpenCVContourDetector] = None,
        scorer: Optional[CandidateScorer] = None,
        tracker: Optional[TemporalDocumentTracker] = None,
        segmenter: Optional[MIDV500Segmenter] = None,
        gate: Optional[VariantCAmbiguityGate] = None,
        weights_path: Union[str, Path] = "models/weights/midv500_unet_resnet34.pth",
        config_path: Optional[Union[str, Path]] = None,
        lazy_load_segmenter: bool = True,
        update_tracker_on_fallback: bool = False,
    ):
        """
        Initialize the Hybrid Document Detector.

        Args:
            opencv_detector: Optional existing OpenCVContourDetector instance.
            scorer: Optional existing CandidateScorer instance.
            tracker: Optional existing TemporalDocumentTracker instance.
            segmenter: Optional existing MIDV500Segmenter instance.
            gate: Optional existing VariantCAmbiguityGate instance.
            weights_path: Path to MIDV-500 UNet checkpoint.
            config_path: Path to configuration YAML.
            lazy_load_segmenter: If True, defer UNet weight loading until first fallback.
            update_tracker_on_fallback: If True, update tracker state with UNet corners.
        """
        self.scorer = scorer if scorer is not None else CandidateScorer(config_path=config_path)
        self.opencv_detector = (
            opencv_detector
            if opencv_detector is not None
            else OpenCVContourDetector(config_path=config_path)
        )
        self.tracker = tracker
        self.gate = gate if gate is not None else VariantCAmbiguityGate()
        self.weights_path = Path(weights_path)
        self.lazy_load_segmenter = lazy_load_segmenter
        self.update_tracker_on_fallback = update_tracker_on_fallback

        self._segmenter = segmenter

    @property
    def segmenter(self) -> MIDV500Segmenter:
        """Get or lazily instantiate the MIDV-500 UNet segmenter."""
        if self._segmenter is None:
            self._segmenter = MIDV500Segmenter(
                weights_path=self.weights_path,
                lazy_load=False,
            )
        return self._segmenter

    def detect_candidates(
        self,
        frame: np.ndarray,
        frame_id: Optional[Union[str, int]] = None,
    ) -> List[Dict[str, Any]]:
        """Extract OpenCV candidate quadrilaterals."""
        return self.opencv_detector.detect_candidates(frame, frame_id)

    def detect_frame(
        self,
        frame: np.ndarray,
        frame_id: Optional[Union[str, int]] = None,
    ) -> Dict[str, Any]:
        """
        Run hybrid document localization on a single frame.

        Args:
            frame: BGR image array (H, W, 3).
            frame_id: Optional identifier or index for the frame.

        Returns:
            Unified dictionary containing:
              - frame_id
              - detected (bool)
              - bbox ([x1, y1, x2, y2] or None)
              - corners (list of 4 [x, y] coordinates or None)
              - confidence (float)
              - localization_method ("opencv_v2" or "midv500_unet")
              - gate_reason (string reason from Variant C gate)
              - candidate_count (int)
              - static_score (float)
              - temporal_score (float or None)
              - signals (dict of evaluated features)
              - reason (string description if failed, else None)
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError(f"Invalid frame input: expected non-empty numpy.ndarray, got {type(frame)}")

        h, w = frame.shape[:2]

        # Step 1: OpenCV Candidate Generation
        cands = self.opencv_detector.detect_candidates(frame, frame_id)
        candidate_count = len(cands)

        # Step 2: Temporal Tracker or Scorer Evaluation
        if self.tracker is not None:
            opencv_res = self.tracker.process_frame(cands, frame, frame_id=frame_id)
        else:
            opencv_res = self.opencv_detector.detect_frame(frame, frame_id=frame_id)

        # Step 3: Variant C Ambiguity Gate Evaluation
        gate_decision = self.gate.evaluate(opencv_res, (h, w), scorer=self.scorer)
        invoke_unet = gate_decision["invoke_unet"]
        gate_reason = gate_decision["reason"]
        signals = gate_decision["signals"]

        # Step 4: Routing Decision
        if not invoke_unet:
            # FAST PATH: Confident OpenCV Candidate (or Exempted Pristine Card)
            corners = opencv_res.get("corners")
            if isinstance(corners, np.ndarray):
                corners = corners.tolist()

            return {
                "frame_id": frame_id,
                "detected": bool(opencv_res.get("detected", False)),
                "bbox": opencv_res.get("bbox"),
                "corners": corners,
                "confidence": float(opencv_res.get("confidence", 0.0)),
                "localization_method": "opencv_v2",
                "gate_reason": gate_reason,
                "candidate_count": candidate_count,
                "static_score": float(opencv_res.get("static_score", signals.get("static_score", 0.0))),
                "temporal_score": opencv_res.get("temporal_score", None),
                "signals": signals,
                "reason": opencv_res.get("reason", None),
                "all_detections": opencv_res.get("all_detections", []),
            }

        # FALLBACK PATH: Learned MIDV-500 UNet Segmentation
        try:
            seg_res = self.segmenter.process_frame(frame)
        except Exception as e:
            logger.warning(f"UNet fallback execution failed on frame {frame_id}: {e}")
            seg_res = {"detected": False, "corners": None, "confidence": 0.0, "bbox": None}

        if seg_res["detected"] and seg_res["corners"] is not None:
            unet_corners = seg_res["corners"]
            ordered_corners = order_points(unet_corners)

            # Optionally update tracker state if tracking is active
            if self.tracker is not None and self.update_tracker_on_fallback:
                self.tracker.last_accepted_quad = ordered_corners
                self.tracker.last_accepted_area = float(cv2.contourArea(ordered_corners))
                w_c = float(np.linalg.norm(ordered_corners[1] - ordered_corners[0]))
                h_c = float(np.linalg.norm(ordered_corners[3] - ordered_corners[0]))
                self.tracker.last_accepted_aspect = max(w_c, h_c) / max(1.0, min(w_c, h_c))
                self.tracker.tracking_confidence = float(seg_res["confidence"])
                self.tracker.frames_lost = 0
                self.tracker.frames_tracked += 1

            return {
                "frame_id": frame_id,
                "detected": True,
                "bbox": seg_res["bbox"],
                "corners": ordered_corners.tolist(),
                "confidence": float(seg_res["confidence"]),
                "localization_method": "midv500_unet",
                "gate_reason": gate_reason,
                "candidate_count": candidate_count,
                "static_score": float(opencv_res.get("static_score", signals.get("static_score", 0.0))),
                "temporal_score": opencv_res.get("temporal_score", None),
                "signals": signals,
                "reason": None,
                "all_detections": opencv_res.get("all_detections", []),
            }

        # Negative detection fallback
        return {
            "frame_id": frame_id,
            "detected": False,
            "bbox": None,
            "corners": None,
            "confidence": 0.0,
            "localization_method": "midv500_unet",
            "gate_reason": gate_reason,
            "candidate_count": candidate_count,
            "static_score": float(opencv_res.get("static_score", 0.0)),
            "temporal_score": opencv_res.get("temporal_score", None),
            "signals": signals,
            "reason": "learned_segmentation_failed",
            "all_detections": opencv_res.get("all_detections", []),
        }
