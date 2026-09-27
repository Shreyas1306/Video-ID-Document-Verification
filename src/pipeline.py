"""
End-to-End Document Verification Pipeline
=========================================
Connects all visual, temporal, textual, fusion, and risk modules into a unified,
production-scale document verification workflow.

Execution Sequence:
  Video Input
  -> Frame Extraction
  -> Document Detection
  -> Document Cropping
  -> Perspective Correction
  -> Normalized Frames
  -> Visual Analysis
  -> Temporal Analysis
  -> OCR
  -> Text Verification
  -> Evidence Fusion
  -> Risk Assessment
  -> Verification Report

Guarantees:
  - Preserves exact stage ordering.
  - Never silently skips a failed stage.
  - Measures high-resolution processing times for each stage.
  - Saves all intermediate visual, textual, and debug artifacts.
  - Provides independent testability for every stage.
"""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Union

import cv2
import numpy as np

from src.fusion.risk_classifier import RiskClassifier
from src.fusion.score_fusion import EvidenceFusionEngine
from src.preprocessing.candidate_scorer import CandidateScorer
from src.preprocessing.document_cropper import DocumentCropper
from src.preprocessing.document_detector import (
    OpenCVContourDetector,
    get_document_detector,
    run_detection_pipeline,
)
from src.preprocessing.frame_extractor import FrameExtractor
from src.preprocessing.perspective_corrector import (
    PerspectiveCorrector,
    order_points,
    run_perspective_pipeline,
)
from src.preprocessing.temporal_document_tracker import TemporalDocumentTracker
from src.report.report_generator import VerificationReportGenerator
from src.text.ocr_engine import run_text_verification_pipeline
from src.utils.config_loader import load_config
from src.visual.temporal_consistency import TemporalConsistencyAnalyzer
from src.visual.visual_integrity import DocumentIntegrityClassifier

logger = logging.getLogger(__name__)


@dataclass
class StageStatus:
    """Execution status for a single pipeline stage."""
    stage_name: str
    status: str  # "SUCCESS", "WARNING", "FAILED", "SKIPPED"
    duration_seconds: float
    items_processed: int
    message: str = ""
    details: Optional[Dict[str, Any]] = None


class DocumentVerificationPipeline:
    """
    Main orchestration class for the Video-Based Identity Document Verification pipeline.
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = "config/settings.yaml",
        base_output_dir: Optional[Union[str, Path]] = "outputs",
    ):
        self.config_path = str(config_path) if config_path else "config/settings.yaml"
        self.config = load_config(self.config_path)
        self.base_output_dir = Path(base_output_dir or self.config.get("paths", {}).get("output_dir", "outputs"))
        self.report_generator = VerificationReportGenerator()
        self.tracker = TemporalDocumentTracker(config_path=self.config_path)

    # -------------------------------------------------------------------------
    # Stage 1: Frame Extraction
    # -------------------------------------------------------------------------
    def extract_frames(
        self,
        video_path: Union[str, Path],
        output_dir: Path,
    ) -> Dict[str, Any]:
        """Extract ordered, sampled frames from video."""
        extractor = FrameExtractor(config_path=self.config_path)
        return extractor.extract_from_video(
            video_path=video_path,
            output_dir=output_dir,
            save_to_disk=True,
        )

    # -------------------------------------------------------------------------
    # Stages 2 & 3: Document Detection, Tracking & Cropping
    # -------------------------------------------------------------------------
    def detect_and_crop(
        self,
        frame_paths: Sequence[Union[str, Path]],
        output_crops_dir: Path,
        output_debug_dir: Path,
        tracker: Optional[TemporalDocumentTracker] = None,
    ) -> Dict[str, Any]:
        """
        Run document candidate detection, candidate scoring, and temporal tracking across sequential frames.
        
        Saves bounding-box crops and annotated debug frames while retaining the selected full-frame
        4-corner coordinates in each frame's result metadata for direct homography rectification.
        """
        crops_dir = Path(output_crops_dir).resolve()
        debug_dir = Path(output_debug_dir).resolve()
        crops_dir.mkdir(parents=True, exist_ok=True)
        debug_dir.mkdir(parents=True, exist_ok=True)

        frame_files: List[Path] = []
        if isinstance(frame_paths, (str, Path)):
            p = Path(frame_paths).resolve()
            if p.is_dir():
                frame_files = sorted(
                    [f for f in p.glob("*.*") if f.suffix.lower() in [".png", ".jpg", ".jpeg"]]
                )
            elif p.is_file():
                frame_files = [p]
        else:
            frame_files = [Path(f).resolve() for f in frame_paths]

        if not frame_files:
            return {
                "detector_type": "None",
                "detector_description": "No frames provided",
                "confidence_threshold": 0.0,
                "total_frames_processed": 0,
                "frames_detected": 0,
                "frames_undetected": 0,
                "detection_rate": 0.0,
                "output_crops_directory": str(crops_dir),
                "output_debug_directory": str(debug_dir),
                "results": [],
            }

        detector = get_document_detector(config_path=self.config_path)
        cropper = DocumentCropper()
        active_tracker = tracker if tracker is not None else TemporalDocumentTracker(config_path=self.config_path)

        per_frame_results: List[Dict[str, Any]] = []
        detected_count = 0

        for frame_p in frame_files:
            frame_id = frame_p.stem
            frame = cv2.imread(str(frame_p))

            if frame is None:
                per_frame_results.append({
                    "frame_id": frame_id,
                    "file": frame_p.name,
                    "frame_path": str(frame_p),
                    "detected": False,
                    "bbox": None,
                    "corners": None,
                    "confidence": 0.0,
                    "static_score": 0.0,
                    "temporal_score": 0.0,
                    "method": "read_error",
                    "tracking_used": False,
                    "crop_path": None,
                    "debug_path": None,
                    "candidate_count": 0,
                    "all_detections": [],
                    "error": f"Failed to read image file: {frame_p}",
                })
                continue

            if hasattr(detector, "detect_candidates"):
                candidates = detector.detect_candidates(frame, frame_id=frame_id)
            else:
                single_det = detector.detect_frame(frame, frame_id=frame_id)
                if single_det.get("detected") and single_det.get("corners"):
                    cand = {
                        "bbox": single_det["bbox"],
                        "corners": np.array(single_det["corners"], dtype=np.float32),
                        "area": float((single_det["bbox"][2] - single_det["bbox"][0]) * (single_det["bbox"][3] - single_det["bbox"][1])),
                        "aspect_ratio": 1.5,
                        "confidence": single_det["confidence"],
                        "area_ratio": 0.3,
                        "rectangularity": 1.0,
                        "edge_quality": 0.5,
                    }
                    candidates = [cand]
                else:
                    candidates = []

            track_res = active_tracker.process_frame(candidates, frame, frame_id=frame_id)
            is_detected = bool(track_res.get("detected", False))
            bbox = track_res.get("bbox")
            corners = track_res.get("corners")
            method = track_res.get("method", "none")
            tracking_used = bool(method == "temporally_guided")

            crop_path = None
            debug_path = debug_dir / f"debug_{frame_id}.png"

            if is_detected and bbox is not None:
                detected_count += 1
                cropped = cropper.crop_region(frame=frame, bbox=bbox, corners=corners)
                if cropped is not None:
                    c_path = crops_dir / f"crop_{frame_id}.png"
                    cropper.save_crop_image(cropped, c_path)
                    crop_path = str(c_path)

                annotated = cropper.draw_annotation(
                    frame=frame,
                    bbox=bbox,
                    corners=corners,
                    confidence=track_res.get("confidence", 0.0),
                    class_name=track_res.get("class_name", "identity_document"),
                )
                cv2.imwrite(str(debug_path), annotated)
            else:
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

            per_frame_results.append({
                "frame_id": frame_id,
                "file": frame_p.name,
                "frame_path": str(frame_p),
                "detected": is_detected,
                "bbox": bbox,
                "corners": corners,
                "confidence": track_res.get("confidence", 0.0),
                "static_score": track_res.get("static_score", 0.0),
                "temporal_score": track_res.get("temporal_score", 0.0),
                "method": method,
                "tracking_used": tracking_used,
                "crop_path": crop_path,
                "debug_path": str(debug_path),
                "candidate_count": len(candidates),
                "all_detections": track_res.get("all_detections", []),
                "reason": track_res.get("reason"),
            })

        total_frames = len(frame_files)
        detection_rate = round(detected_count / total_frames, 3) if total_frames > 0 else 0.0

        summary = {
            "detector_type": type(detector).__name__,
            "detector_description": f"{type(detector).__name__} with CandidateScorer and TemporalDocumentTracker",
            "confidence_threshold": detector.confidence_threshold,
            "tracking_enabled": active_tracker.temporal_enabled,
            "total_frames_processed": total_frames,
            "frames_detected": detected_count,
            "frames_undetected": total_frames - detected_count,
            "detection_rate": detection_rate,
            "output_crops_directory": str(crops_dir),
            "output_debug_directory": str(debug_dir),
            "results": per_frame_results,
        }

        summary_file = crops_dir.parent / "detection_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary

    # -------------------------------------------------------------------------
    # Stages 4 & 5: Perspective Correction & Normalization
    # -------------------------------------------------------------------------
    def correct_perspective(
        self,
        targets: Sequence[Union[str, Path, Dict[str, Any]]],
        output_normalized_dir: Path,
        output_debug_dir: Path,
    ) -> Dict[str, Any]:
        """
        Rectify detected documents into normalized rectangular images.
        
        If `targets` contains detection result dictionaries with full-frame corners,
        homography perspective correction is applied directly on the full frame
        without bounding-box re-detection.
        
        If `targets` contains file paths (e.g. crop images), falls back to the legacy
        crop-based perspective estimation for backward compatibility.
        """
        norm_dir = Path(output_normalized_dir).resolve()
        debug_dir = Path(output_debug_dir).resolve()
        norm_dir.mkdir(parents=True, exist_ok=True)
        debug_dir.mkdir(parents=True, exist_ok=True)

        if not targets:
            return {
                "target_dimensions": [600, 400],
                "total_crops_processed": 0,
                "total_frames_processed": 0,
                "homography_success_count": 0,
                "homography_success_rate": 0.0,
                "fallback_resize_count": 0,
                "output_normalized_directory": str(norm_dir),
                "output_debug_directory": str(debug_dir),
                "results": [],
            }

        first_item = targets[0]
        if isinstance(first_item, dict):
            corrector = PerspectiveCorrector(config_path=self.config_path)
            results: List[Dict[str, Any]] = []
            homography_count = 0
            fallback_count = 0

            for det in targets:
                frame_id = det.get("frame_id", "unknown")
                frame_path = det.get("frame_path") or det.get("file")
                detected = bool(det.get("detected", False))
                corners = det.get("corners")

                if not detected or corners is None or frame_path is None:
                    results.append({
                        "frame_id": frame_id,
                        "file": det.get("file", f"{frame_id}.png"),
                        "success": False,
                        "method_applied": "none",
                        "normalized_path": None,
                        "debug_path": None,
                        "corners": None,
                        "error": det.get("reason", "Document not detected"),
                    })
                    continue

                frame_p = Path(frame_path)
                frame = cv2.imread(str(frame_p)) if frame_p.is_file() else None
                if frame is None:
                    results.append({
                        "frame_id": frame_id,
                        "file": frame_p.name if frame_p else f"{frame_id}.png",
                        "success": False,
                        "method_applied": "none",
                        "normalized_path": None,
                        "debug_path": None,
                        "corners": None,
                        "error": f"Failed to load frame from {frame_path}",
                    })
                    continue

                corners_arr = np.array(corners, dtype=np.float32).reshape((4, 2))
                corr_res = corrector.correct_perspective(image=frame, corners=corners_arr, frame_id=frame_id)

                success = bool(corr_res.get("success", False))
                norm_img = corr_res.get("normalized_image")

                if success and norm_img is not None:
                    homography_count += 1
                    norm_path = norm_dir / f"norm_{frame_id}.png"
                    cv2.imwrite(str(norm_path), norm_img)

                    debug_comp = corrector.create_comparison_debug(
                        original_image=frame,
                        normalized_image=norm_img,
                        corners=corners_arr,
                        status_text=f"HOMOGRAPHY ({det.get('method', 'tracked')})",
                    )
                    debug_path = debug_dir / f"debug_persp_{frame_id}.png"
                    cv2.imwrite(str(debug_path), debug_comp)

                    results.append({
                        "frame_id": frame_id,
                        "file": frame_p.name,
                        "success": True,
                        "method_applied": corr_res.get("method_applied", "homography"),
                        "normalized_path": str(norm_path),
                        "debug_path": str(debug_path),
                        "corners": corners,
                        "error": None,
                    })
                else:
                    fallback_count += 1
                    results.append({
                        "frame_id": frame_id,
                        "file": frame_p.name,
                        "success": False,
                        "method_applied": corr_res.get("method_applied", "fallback_resize"),
                        "normalized_path": None,
                        "debug_path": None,
                        "corners": corners,
                        "error": corr_res.get("error", "Perspective correction failed"),
                    })

            total = len(targets)
            summary = {
                "target_dimensions": [corrector.target_width, corrector.target_height],
                "total_crops_processed": total,
                "total_frames_processed": total,
                "homography_success_count": homography_count,
                "homography_success_rate": round(homography_count / total, 3) if total > 0 else 0.0,
                "fallback_resize_count": fallback_count,
                "output_normalized_directory": str(norm_dir),
                "output_debug_directory": str(debug_dir),
                "results": results,
            }

            summary_file = norm_dir.parent / "perspective_summary.json"
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

            return summary
        else:
            return run_perspective_pipeline(
                crops_dir_or_paths=targets,
                output_normalized_dir=output_normalized_dir,
                output_debug_dir=output_debug_dir,
                config_path=self.config_path,
            )

    def run(
        self,
        video_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Alias for process_video to support standard runner interfaces."""
        return self.process_video(video_path=video_path, video_output_dir=output_dir)

    # -------------------------------------------------------------------------
    # Stage 6: Visual Analysis (Integrity Classification)
    # -------------------------------------------------------------------------
    def analyze_visual_integrity(
        self,
        norm_paths: Sequence[Union[str, Path]],

    ) -> Dict[str, Any]:
        """Run transfer-learning visual integrity model across normalized frames."""
        visual_model_path = self.config.get("visual_integrity", {}).get(
            "model_checkpoint", "models/weights/best_integrity_model.pth"
        )
        classifier = DocumentIntegrityClassifier(
            checkpoint_path=visual_model_path if Path(visual_model_path).is_file() else None,
            device="cpu",
        )

        predictions = []
        for p in norm_paths:
            pred = classifier.predict_image(p)
            predictions.append({"path": str(p), **pred})

        scores = [p["integrity_score"] for p in predictions]
        avg_score = float(np.mean(scores)) if scores else 0.0

        return {
            "aggregated_score": round(avg_score, 4),
            "frames_evaluated": len(predictions),
            "predictions": predictions,
        }

    # -------------------------------------------------------------------------
    # Stage 7: Temporal Analysis (Embedding Consistency)
    # -------------------------------------------------------------------------
    def analyze_temporal_consistency(
        self,
        norm_paths: Sequence[Union[str, Path]],
        video_id: str,
        debug_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Measure frame-to-frame cosine similarity of visual embeddings."""
        thresh = self.config.get("temporal_consistency", {}).get("consistency_threshold", 0.85)
        visual_model_path = self.config.get("visual_integrity", {}).get(
            "model_checkpoint", "models/weights/best_integrity_model.pth"
        )

        analyzer = TemporalConsistencyAnalyzer(
            checkpoint_path=visual_model_path if Path(visual_model_path).is_file() else None,
            consistency_threshold=thresh,
            device="cpu",
        )

        return analyzer.analyze_sequence(
            frames=norm_paths,
            frame_ids=[Path(p).stem for p in norm_paths],
            video_ids=[video_id] * len(norm_paths),
            debug_output_path=debug_path,
        )

    # -------------------------------------------------------------------------
    # Stages 8 & 9: OCR & Text Consistency Verification
    # -------------------------------------------------------------------------
    def extract_and_verify_text(
        self,
        norm_paths: Sequence[Union[str, Path]],
        output_summary_dir: Path,
    ) -> Dict[str, Any]:
        """Run OCR extraction, field parsing, and cross-frame textual consistency."""
        return run_text_verification_pipeline(
            normalized_frames_or_paths=norm_paths,
            output_summary_dir=output_summary_dir,
            config_path=self.config_path,
        )

    # -------------------------------------------------------------------------
    # Stage 10: Multi-Modal Evidence Fusion
    # -------------------------------------------------------------------------
    def fuse_evidence(
        self,
        visual_score: Optional[float],
        temporal_score: Optional[float],
        ocr_confidence: Optional[float],
        text_consistency: Optional[float],
    ) -> Dict[str, Any]:
        """Combine multi-modal evidence scores with dynamic re-weighting."""
        engine = EvidenceFusionEngine(config_path=self.config_path)
        return engine.fuse(
            visual_integrity=visual_score,
            temporal_consistency=temporal_score,
            ocr_confidence=ocr_confidence,
            text_consistency=text_consistency,
        )

    # -------------------------------------------------------------------------
    # Stage 11: Risk Assessment
    # -------------------------------------------------------------------------
    def assess_risk(
        self,
        fused_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Map fused integrity score to LOW / MEDIUM / HIGH risk classification."""
        classifier = RiskClassifier(config_path=self.config_path)
        return classifier.assess_risk(
            integrity_score=fused_result["fused_integrity_score"],
            evidence=fused_result,
        )

    # -------------------------------------------------------------------------
    # Complete End-to-End Orchestrator
    # -------------------------------------------------------------------------
    def process_video(
        self,
        video_path: Union[str, Path],
        video_output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the complete 12-stage document verification pipeline on a video.
        """
        vid_p = Path(video_path).resolve()
        if not vid_p.is_file():
            raise FileNotFoundError(f"Input video file does not exist: {vid_p}")

        video_stem = vid_p.stem
        out_root = Path(video_output_dir) if video_output_dir else self.base_output_dir / video_stem
        out_root.mkdir(parents=True, exist_ok=True)

        stage_timings: Dict[str, float] = {}
        stage_statuses: Dict[str, Dict[str, Any]] = {}
        artifacts: Dict[str, str] = {}

        logger.info(f"Starting end-to-end document verification for: {vid_p.name}")
        pipeline_start = time.time()

        # -------------------------------------------------------------
        # Stage 1: Frame Extraction
        # -------------------------------------------------------------
        t0 = time.time()
        frames_dir = out_root / "frames"
        artifacts["frames_directory"] = str(frames_dir)
        try:
            extraction_meta = self.extract_frames(vid_p, frames_dir)
            frame_paths = extraction_meta.get("frame_paths", [])
            dur = time.time() - t0
            stage_timings["frame_extraction"] = dur
            stage_statuses["frame_extraction"] = asdict(StageStatus(
                stage_name="frame_extraction",
                status="SUCCESS" if frame_paths else "WARNING",
                duration_seconds=dur,
                items_processed=len(frame_paths),
                message=f"Extracted {len(frame_paths)} frames.",
            ))
        except Exception as e:
            dur = time.time() - t0
            stage_timings["frame_extraction"] = dur
            stage_statuses["frame_extraction"] = asdict(StageStatus(
                stage_name="frame_extraction",
                status="FAILED",
                duration_seconds=dur,
                items_processed=0,
                message=f"Extraction failed: {e}",
            ))
            extraction_meta = {"video_path": str(vid_p), "error": str(e)}
            frame_paths = []

        # -------------------------------------------------------------
        # Stage 2: Document Detection, Tracking & Cropping
        # -------------------------------------------------------------
        t0 = time.time()
        crops_dir = out_root / "crops"
        debug_det_dir = out_root / "debug_detections"
        artifacts["crops_directory"] = str(crops_dir)
        artifacts["debug_detections_directory"] = str(debug_det_dir)

        # Fresh tracker for this video session (ensures zero cross-video pollution)
        tracker = TemporalDocumentTracker(config_path=self.config_path)
        tracker.reset()
        self.tracker = tracker

        if frame_paths:
            try:
                detection_summary = self.detect_and_crop(
                    frame_paths=frame_paths,
                    output_crops_dir=crops_dir,
                    output_debug_dir=debug_det_dir,
                    tracker=tracker,
                )
                detected_count = detection_summary.get("frames_detected", 0)
                dur = time.time() - t0
                stage_timings["document_detection"] = dur
                status_code = "SUCCESS" if detected_count > 0 else "WARNING"
                stage_statuses["document_detection"] = asdict(StageStatus(
                    stage_name="document_detection",
                    status=status_code,
                    duration_seconds=dur,
                    items_processed=detected_count,
                    message=f"Detected & tracked document in {detected_count}/{len(frame_paths)} frames.",
                ))
            except Exception as e:
                dur = time.time() - t0
                stage_timings["document_detection"] = dur
                stage_statuses["document_detection"] = asdict(StageStatus(
                    stage_name="document_detection",
                    status="FAILED",
                    duration_seconds=dur,
                    items_processed=0,
                    message=f"Detection and tracking failed: {e}",
                ))
                detection_summary = {"error": str(e), "total_frames_processed": len(frame_paths), "frames_detected": 0}
        else:
            dur = time.time() - t0
            stage_timings["document_detection"] = dur
            stage_statuses["document_detection"] = asdict(StageStatus(
                stage_name="document_detection",
                status="WARNING",
                duration_seconds=dur,
                items_processed=0,
                message="Skipped detection: No input frames extracted.",
            ))
            detection_summary = {"total_frames_processed": 0, "frames_detected": 0}

        # -------------------------------------------------------------
        # Stage 3: Perspective Correction & Normalization
        # -------------------------------------------------------------
        t0 = time.time()
        norm_dir = out_root / "normalized"
        debug_persp_dir = out_root / "debug_perspective"
        artifacts["normalized_directory"] = str(norm_dir)
        artifacts["debug_perspective_directory"] = str(debug_persp_dir)

        det_results = detection_summary.get("results", [])
        if det_results:
            try:
                perspective_summary = self.correct_perspective(
                    targets=det_results,
                    output_normalized_dir=norm_dir,
                    output_debug_dir=debug_persp_dir,
                )
                norm_paths = [r["normalized_path"] for r in perspective_summary.get("results", []) if r.get("normalized_path")]
                dur = time.time() - t0
                stage_timings["perspective_correction"] = dur
                stage_statuses["perspective_correction"] = asdict(StageStatus(
                    stage_name="perspective_correction",
                    status="SUCCESS" if norm_paths else "WARNING",
                    duration_seconds=dur,
                    items_processed=len(norm_paths),
                    message=f"Normalized {len(norm_paths)} frames via full-frame homography.",
                ))
            except Exception as e:
                dur = time.time() - t0
                stage_timings["perspective_correction"] = dur
                stage_statuses["perspective_correction"] = asdict(StageStatus(
                    stage_name="perspective_correction",
                    status="FAILED",
                    duration_seconds=dur,
                    items_processed=0,
                    message=f"Perspective correction failed: {e}",
                ))
                perspective_summary = {"error": str(e)}
                norm_paths = []
        else:
            dur = time.time() - t0
            stage_timings["perspective_correction"] = dur
            stage_statuses["perspective_correction"] = asdict(StageStatus(
                stage_name="perspective_correction",
                status="WARNING",
                duration_seconds=dur,
                items_processed=0,
                message="Skipped perspective correction: No document detections available.",
            ))
            perspective_summary = {"total_frames_processed": 0, "homography_success_count": 0}
            norm_paths = []


        # -------------------------------------------------------------
        # Stage 4: Visual Analysis (Integrity Classification)
        # -------------------------------------------------------------
        t0 = time.time()
        if norm_paths:
            try:
                visual_summary = self.analyze_visual_integrity(norm_paths)
                dur = time.time() - t0
                stage_timings["visual_analysis"] = dur
                stage_statuses["visual_analysis"] = asdict(StageStatus(
                    stage_name="visual_analysis",
                    status="SUCCESS",
                    duration_seconds=dur,
                    items_processed=len(norm_paths),
                    message=f"Evaluated {len(norm_paths)} frames (Score: {visual_summary['aggregated_score']}).",
                ))
            except Exception as e:
                dur = time.time() - t0
                stage_timings["visual_analysis"] = dur
                stage_statuses["visual_analysis"] = asdict(StageStatus(
                    stage_name="visual_analysis",
                    status="FAILED",
                    duration_seconds=dur,
                    items_processed=0,
                    message=f"Visual integrity analysis failed: {e}",
                ))
                visual_summary = {"aggregated_score": None, "error": str(e)}
        else:
            dur = time.time() - t0
            stage_timings["visual_analysis"] = dur
            stage_statuses["visual_analysis"] = asdict(StageStatus(
                stage_name="visual_analysis",
                status="WARNING",
                duration_seconds=dur,
                items_processed=0,
                message="Visual analysis degraded: No normalized frames available.",
            ))
            visual_summary = {"aggregated_score": None}

        # -------------------------------------------------------------
        # Stage 5: Temporal Consistency Analysis
        # -------------------------------------------------------------
        t0 = time.time()
        temporal_debug_path = out_root / "debug_temporal_consistency.png"
        artifacts["debug_temporal_visualization"] = str(temporal_debug_path)

        if len(norm_paths) >= 2:
            try:
                temporal_summary = self.analyze_temporal_consistency(
                    norm_paths=norm_paths,
                    video_id=video_stem,
                    debug_path=temporal_debug_path,
                )
                dur = time.time() - t0
                stage_timings["temporal_analysis"] = dur
                stage_statuses["temporal_analysis"] = asdict(StageStatus(
                    stage_name="temporal_analysis",
                    status="SUCCESS" if temporal_summary.get("status") == "success" else "WARNING",
                    duration_seconds=dur,
                    items_processed=len(norm_paths),
                    message=f"Temporal analysis complete (Score: {temporal_summary.get('temporal_consistency_score')}).",
                ))
            except Exception as e:
                dur = time.time() - t0
                stage_timings["temporal_analysis"] = dur
                stage_statuses["temporal_analysis"] = asdict(StageStatus(
                    stage_name="temporal_analysis",
                    status="FAILED",
                    duration_seconds=dur,
                    items_processed=0,
                    message=f"Temporal analysis failed: {e}",
                ))
                temporal_summary = {"temporal_consistency_score": None, "error": str(e)}
        else:
            dur = time.time() - t0
            stage_timings["temporal_analysis"] = dur
            stage_statuses["temporal_analysis"] = asdict(StageStatus(
                stage_name="temporal_analysis",
                status="WARNING",
                duration_seconds=dur,
                items_processed=len(norm_paths),
                message=f"Insufficient normalized frames ({len(norm_paths)}) for temporal sequence analysis.",
            ))
            temporal_summary = {
                "temporal_consistency_score": None,
                "reason": "Insufficient frames (< 2) for temporal analysis",
            }

        # -------------------------------------------------------------
        # Stage 6: OCR & Text Verification
        # -------------------------------------------------------------
        t0 = time.time()
        if norm_paths:
            try:
                ocr_summary = self.extract_and_verify_text(norm_paths, out_root)
                dur = time.time() - t0
                stage_timings["ocr_and_text_verification"] = dur
                stage_statuses["ocr_and_text_verification"] = asdict(StageStatus(
                    stage_name="ocr_and_text_verification",
                    status="SUCCESS",
                    duration_seconds=dur,
                    items_processed=len(norm_paths),
                    message=f"OCR extracted fields (Confidence: {ocr_summary.get('ocr_confidence')}, Text Consistency: {ocr_summary.get('text_consistency')}).",
                ))
            except Exception as e:
                dur = time.time() - t0
                stage_timings["ocr_and_text_verification"] = dur
                stage_statuses["ocr_and_text_verification"] = asdict(StageStatus(
                    stage_name="ocr_and_text_verification",
                    status="FAILED",
                    duration_seconds=dur,
                    items_processed=0,
                    message=f"OCR verification failed: {e}",
                ))
                ocr_summary = {"ocr_confidence": None, "text_consistency": None, "error": str(e)}
        else:
            dur = time.time() - t0
            stage_timings["ocr_and_text_verification"] = dur
            stage_statuses["ocr_and_text_verification"] = asdict(StageStatus(
                stage_name="ocr_and_text_verification",
                status="WARNING",
                duration_seconds=dur,
                items_processed=0,
                message="OCR verification degraded: No normalized frames available.",
            ))
            ocr_summary = {"ocr_confidence": None, "text_consistency": None}

        # -------------------------------------------------------------
        # Stage 7: Evidence Fusion
        # -------------------------------------------------------------
        t0 = time.time()
        try:
            fusion_summary = self.fuse_evidence(
                visual_score=visual_summary.get("aggregated_score"),
                temporal_score=temporal_summary.get("temporal_consistency_score"),
                ocr_confidence=ocr_summary.get("ocr_confidence"),
                text_consistency=ocr_summary.get("text_consistency"),
            )
            dur = time.time() - t0
            stage_timings["evidence_fusion"] = dur
            stage_statuses["evidence_fusion"] = asdict(StageStatus(
                stage_name="evidence_fusion",
                status="SUCCESS",
                duration_seconds=dur,
                items_processed=fusion_summary.get("num_sources_available", 0),
                message=f"Fused integrity score: {fusion_summary['fused_integrity_score']} (Coverage: {fusion_summary['evidence_coverage'] * 100:.1f}%).",
            ))
        except Exception as e:
            dur = time.time() - t0
            stage_timings["evidence_fusion"] = dur
            stage_statuses["evidence_fusion"] = asdict(StageStatus(
                stage_name="evidence_fusion",
                status="FAILED",
                duration_seconds=dur,
                items_processed=0,
                message=f"Fusion failed: {e}",
            ))
            fusion_summary = {
                "fused_integrity_score": 0.0,
                "evidence_coverage": 0.0,
                "error": str(e),
            }

        # -------------------------------------------------------------
        # Stage 8: Risk Assessment
        # -------------------------------------------------------------
        t0 = time.time()
        try:
            risk_summary = self.assess_risk(fusion_summary)
            dur = time.time() - t0
            stage_timings["risk_assessment"] = dur
            stage_statuses["risk_assessment"] = asdict(StageStatus(
                stage_name="risk_assessment",
                status="SUCCESS",
                duration_seconds=dur,
                items_processed=1,
                message=f"Risk level determined: {risk_summary['risk_level']}.",
            ))
        except Exception as e:
            dur = time.time() - t0
            stage_timings["risk_assessment"] = dur
            stage_statuses["risk_assessment"] = asdict(StageStatus(
                stage_name="risk_assessment",
                status="FAILED",
                duration_seconds=dur,
                items_processed=0,
                message=f"Risk assessment failed: {e}",
            ))
            risk_summary = {"risk_level": "UNKNOWN", "integrity_score": 0.0, "evidence": {}}

        # -------------------------------------------------------------
        # Stage 9: Report Generation & Serialization
        # -------------------------------------------------------------
        t0 = time.time()
        json_path = out_root / "pipeline_result.json"
        txt_path = out_root / "verification_report.txt"
        artifacts["pipeline_result_json"] = str(json_path)
        artifacts["verification_report_txt"] = str(txt_path)

        final_json = self.report_generator.build_structured_json(
            video_metadata=extraction_meta,
            stage_timings=stage_timings,
            stage_statuses=stage_statuses,
            detection_summary=detection_summary,
            perspective_summary=perspective_summary,
            visual_summary=visual_summary,
            temporal_summary=temporal_summary,
            ocr_summary=ocr_summary,
            fusion_summary=fusion_summary,
            risk_summary=risk_summary,
            output_artifacts=artifacts,
        )

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(final_json, f, indent=2)

        report_text = self.report_generator.build_human_readable_report(
            structured_data=final_json,
            output_path=txt_path,
        )

        dur = time.time() - t0
        stage_timings["report_generation"] = dur
        stage_statuses["report_generation"] = asdict(StageStatus(
            stage_name="report_generation",
            status="SUCCESS",
            duration_seconds=dur,
            items_processed=2,
            message="Generated JSON result and human-readable text report.",
        ))

        total_elapsed = time.time() - pipeline_start
        final_json["total_processing_time_seconds"] = round(total_elapsed, 3)
        final_json["stage_timings_seconds"]["report_generation"] = round(dur, 3)
        final_json["stage_execution_statuses"]["report_generation"] = stage_statuses["report_generation"]

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(final_json, f, indent=2)

        logger.info(f"Pipeline completed in {total_elapsed:.2f} seconds. Risk: {risk_summary.get('risk_level')}")

        return final_json
