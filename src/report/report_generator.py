"""
Verification Report Generator
=============================
Assembles complete structured JSON results and human-readable verification reports
from all pipeline stages.

Provides transparent evidence summaries, stage timings, intermediate artifact
references, and the mandatory legal non-authentication disclaimer.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.fusion.risk_classifier import LEGAL_DISCLAIMER

logger = logging.getLogger(__name__)


class VerificationReportGenerator:
    """
    Builds structured JSON verification summaries and formatted text reports.
    """

    def __init__(self, disclaimer: str = LEGAL_DISCLAIMER):
        self.disclaimer = disclaimer

    def build_structured_json(
        self,
        video_metadata: Dict[str, Any],
        stage_timings: Dict[str, float],
        stage_statuses: Dict[str, Dict[str, Any]],
        detection_summary: Dict[str, Any],
        perspective_summary: Dict[str, Any],
        visual_summary: Dict[str, Any],
        temporal_summary: Dict[str, Any],
        ocr_summary: Dict[str, Any],
        fusion_summary: Dict[str, Any],
        risk_summary: Dict[str, Any],
        output_artifacts: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Assemble the comprehensive structured verification result dictionary.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_pipeline_time = sum(stage_timings.values())

        return {
            "pipeline_version": "1.0.0",
            "timestamp": timestamp,
            "video_metadata": video_metadata,
            "total_processing_time_seconds": round(total_pipeline_time, 3),
            "stage_timings_seconds": {k: round(v, 3) for k, v in stage_timings.items()},
            "stage_execution_statuses": stage_statuses,
            "verification_result": {
                "integrity_score": risk_summary.get("integrity_score"),
                "risk_level": risk_summary.get("risk_level"),
                "evidence": risk_summary.get("evidence", {}),
                "evidence_coverage": fusion_summary.get("evidence_coverage"),
            },
            "evidence_details": {
                "detection": {
                    "model": detection_summary.get("model_used"),
                    "frames_processed": detection_summary.get("total_frames_processed", 0),
                    "frames_detected": detection_summary.get("frames_detected", 0),
                    "detection_rate": detection_summary.get("detection_rate", 0.0),
                    "limitation_analysis": detection_summary.get("limitation_analysis"),
                },
                "perspective_correction": {
                    "target_dimensions": perspective_summary.get("target_dimensions"),
                    "homography_rate": perspective_summary.get("homography_success_rate", 0.0),
                    "homography_success_count": perspective_summary.get("homography_success_count", 0),
                    "fallback_resize_count": perspective_summary.get("fallback_resize_count", 0),
                },
                "visual_integrity": {
                    "aggregated_score": visual_summary.get("aggregated_score"),
                    "frames_evaluated": visual_summary.get("frames_evaluated", 0),
                    "predictions": visual_summary.get("predictions", []),
                },
                "temporal_consistency": {
                    "temporal_score": temporal_summary.get("temporal_consistency_score"),
                    "mean_consecutive_similarity": temporal_summary.get("mean_consecutive_similarity"),
                    "is_consistent": temporal_summary.get("is_consistent"),
                    "anomalous_transitions": temporal_summary.get("anomalous_transitions", []),
                },
                "ocr_and_text": {
                    "engine": ocr_summary.get("ocr_engine"),
                    "extracted_fields": {
                        "name": ocr_summary.get("name"),
                        "dob": ocr_summary.get("dob"),
                        "document_number": ocr_summary.get("document_number"),
                        "address": ocr_summary.get("address"),
                    },
                    "ocr_confidence": ocr_summary.get("ocr_confidence"),
                    "text_consistency": ocr_summary.get("text_consistency"),
                    "field_presence": ocr_summary.get("field_presence", {}),
                    "format_validity": ocr_summary.get("format_validity", {}),
                },
                "evidence_fusion": {
                    "fused_integrity_score": fusion_summary.get("fused_integrity_score"),
                    "evidence_coverage": fusion_summary.get("evidence_coverage"),
                    "available_sources": fusion_summary.get("available_sources", []),
                    "missing_sources": fusion_summary.get("missing_sources", []),
                    "evidence_breakdown": fusion_summary.get("evidence_breakdown", {}),
                },
            },
            "output_artifacts": output_artifacts,
            "disclaimer": self.disclaimer,
        }

    def build_human_readable_report(
        self,
        structured_data: Dict[str, Any],
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        Generate a human-readable verification report.
        """
        v_res = structured_data.get("verification_result", {})
        risk_level = v_res.get("risk_level", "UNKNOWN")
        integrity_score = v_res.get("integrity_score", 0.0)
        coverage = v_res.get("evidence_coverage", 0.0)

        meta = structured_data.get("video_metadata", {})
        details = structured_data.get("evidence_details", {})
        ocr_det = details.get("ocr_and_text", {})
        fields = ocr_det.get("extracted_fields", {})
        timings = structured_data.get("stage_timings_seconds", {})
        total_time = structured_data.get("total_processing_time_seconds", 0.0)

        # Recommendation based on risk level
        if risk_level == "LOW":
            rec = "PASS (LOW RISK): High multi-modal visual and textual consistency. Verified for low-risk automated flows."
        elif risk_level == "MEDIUM":
            rec = "REVIEW (MEDIUM RISK): Ambiguities, partial evidence, or minor anomalies detected. Manual review advised."
        else:
            rec = "FLAG (HIGH RISK): Significant visual, temporal, or textual inconsistencies detected. Escalate for manual fraud investigation."

        lines = [
            "=" * 74,
            "             IDENTITY DOCUMENT VERIFICATION PIPELINE REPORT               ",
            "=" * 74,
            f"Execution Timestamp:     {structured_data.get('timestamp')}",
            f"Input Video:             {meta.get('video_path')}",
            f"Video Duration / FPS:    {meta.get('video_duration')}s ({meta.get('total_frames')} frames @ {meta.get('original_fps')} fps)",
            f"Total Processing Time:   {total_time:.2f} seconds",
            "-" * 74,
            f"FINAL VERIFICATION OUTCOME:  {risk_level} RISK (Score: {integrity_score * 100:.1f}%)",
            f"Evidence Quality / Coverage: {coverage * 100:.1f}%",
            f"Operational Recommendation:  {rec}",
            "-" * 74,
            "EXTRACTED DOCUMENT IDENTITY FIELDS (OCR EVIDENCE):",
            f"  * Document Number:     {fields.get('document_number', 'Not Extracted')}",
            f"  * Holder Name:         {fields.get('name', 'Not Extracted')}",
            f"  * Date of Birth:       {fields.get('dob', 'Not Extracted')}",
            f"  * Address:             {fields.get('address', 'Not Extracted')}",
            "-" * 74,
            "MULTI-MODAL EVIDENCE BREAKDOWN:",
        ]

        ev_dict = v_res.get("evidence", {})
        ev_labels = [
            ("Visual Document Integrity", "visual_integrity"),
            ("Temporal Consistency", "temporal_consistency"),
            ("Character OCR Confidence", "ocr_confidence"),
            ("Cross-Frame Text Consistency", "text_consistency"),
        ]

        for disp, key in ev_labels:
            val = ev_dict.get(key)
            if val is not None:
                val_str = f"{val * 100:.1f}%"
                flag = "PASS" if val >= 0.70 else ("REVIEW" if val >= 0.40 else "FLAG")
            else:
                val_str = "NOT AVAILABLE"
                flag = "MISSING"
            lines.append(f"  * {disp:<30}: {val_str:<14} [{flag}]")

        # Stage Execution Timings Table
        lines.extend([
            "-" * 74,
            "STAGE EXECUTION TIMINGS (SECONDS):",
        ])
        for stage_name, duration in timings.items():
            status_obj = structured_data.get("stage_execution_statuses", {}).get(stage_name, {})
            status_code = status_obj.get("status", "SUCCESS")
            lines.append(f"  * {stage_name:<28}: {duration:6.2f}s  [{status_code}]")

        # Intermediate Artifacts
        artifacts = structured_data.get("output_artifacts", {})
        if artifacts:
            lines.extend([
                "-" * 74,
                "STORED INTERMEDIATE ARTIFACTS:",
            ])
            for art_name, art_path in artifacts.items():
                lines.append(f"  * {art_name:<24}: {art_path}")

        # Legal Notice
        lines.extend([
            "-" * 74,
            "LEGAL & REGULATORY NOTICE:",
            f"  {self.disclaimer}",
            "=" * 74,
        ])

        report_str = "\n".join(lines)

        if output_path is not None:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(report_str + "\n")
            logger.info(f"Saved human-readable report to: {p}")

        return report_str
