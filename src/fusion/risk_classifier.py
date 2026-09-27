"""
Risk Assessment & Verification Reporting
========================================
Maps fused multi-modal document integrity scores to actionable risk levels
(LOW, MEDIUM, HIGH) using configurable engineering heuristics.

Generates structured JSON assessments and human-readable verification reports
with prominent non-authentication disclaimers.

Notice:
  Thresholds (LOW: >= 0.70, MEDIUM: 0.40 - 0.70, HIGH: < 0.40) are engineering
  heuristic defaults, not scientifically validated or legally certified boundaries.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)

DEFAULT_LOW_RISK_THRESHOLD = 0.70
DEFAULT_MEDIUM_RISK_THRESHOLD = 0.40

LEGAL_DISCLAIMER = (
    "DISCLAIMER: This automated system performs document integrity screening based "
    "on visual, temporal, and textual consistency heuristics. It does NOT provide "
    "official government authentication, nor does it guarantee legal validity or "
    "official proof of citizenship or identity."
)


class RiskClassifier:
    """
    Classifies fused integrity scores into LOW, MEDIUM, or HIGH risk categories
    and produces human-readable verification reports.
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        low_risk_threshold: Optional[float] = None,
        medium_risk_threshold: Optional[float] = None,
    ):
        self.low_threshold = DEFAULT_LOW_RISK_THRESHOLD
        self.medium_threshold = DEFAULT_MEDIUM_RISK_THRESHOLD

        if config_path is not None:
            try:
                cfg = load_config(config_path)
                risk_cfg = cfg.get("risk_classification", {})
                thresh = risk_cfg.get("thresholds", {})
                self.low_threshold = float(thresh.get("low_risk", thresh.get("low", self.low_threshold)))
                self.medium_threshold = float(thresh.get("medium_risk", thresh.get("medium", self.medium_threshold)))
            except Exception as e:
                logger.warning(f"Could not load risk classification configuration from {config_path}: {e}")

        if low_risk_threshold is not None:
            self.low_threshold = float(low_risk_threshold)
        if medium_risk_threshold is not None:
            self.medium_threshold = float(medium_risk_threshold)

        if self.low_threshold <= self.medium_threshold:
            raise ValueError(
                f"low_risk_threshold ({self.low_threshold}) must be strictly greater than "
                f"medium_risk_threshold ({self.medium_threshold})"
            )

    def determine_risk_level(self, score: float) -> str:
        """
        Map a normalized integrity score in [0.0, 1.0] to a risk category.
        """
        if score >= self.low_threshold:
            return "LOW"
        elif score >= self.medium_threshold:
            return "MEDIUM"
        else:
            return "HIGH"

    def assess_risk(
        self,
        integrity_score: float,
        evidence: Optional[Dict[str, Any]] = None,
        visual_integrity: Optional[float] = None,
        temporal_consistency: Optional[float] = None,
        ocr_confidence: Optional[float] = None,
        text_consistency: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Produce the structured risk assessment output.

        Output format:
        {
          "integrity_score": ...,
          "risk_level": "LOW/MEDIUM/HIGH",
          "evidence": {
              "visual_integrity": ...,
              "temporal_consistency": ...,
              "ocr_confidence": ...,
              "text_consistency": ...
          }
        }
        """
        # Clamp integrity score to [0.0, 1.0]
        score = max(0.0, min(1.0, float(integrity_score)))
        risk_level = self.determine_risk_level(score)

        # Populate evidence dict
        ev_dict: Dict[str, Optional[float]] = {
            "visual_integrity": None,
            "temporal_consistency": None,
            "ocr_confidence": None,
            "text_consistency": None,
        }

        # 1. From provided evidence dictionary (e.g. from EvidenceFusionEngine)
        if evidence:
            if "evidence_breakdown" in evidence:
                # Structure from EvidenceFusionEngine
                bd = evidence["evidence_breakdown"]
                for k in ev_dict.keys():
                    if k in bd and bd[k].get("status") == "available":
                        ev_dict[k] = bd[k].get("normalized_score")
            else:
                for k in ev_dict.keys():
                    if k in evidence and evidence[k] is not None:
                        ev_dict[k] = round(float(evidence[k]), 4)

        # 2. Override with any explicit kwargs if provided
        if visual_integrity is not None:
            ev_dict["visual_integrity"] = round(float(visual_integrity), 4)
        if temporal_consistency is not None:
            ev_dict["temporal_consistency"] = round(float(temporal_consistency), 4)
        if ocr_confidence is not None:
            ev_dict["ocr_confidence"] = round(float(ocr_confidence), 4)
        if text_consistency is not None:
            ev_dict["text_consistency"] = round(float(text_consistency), 4)

        return {
            "integrity_score": round(score, 4),
            "risk_level": risk_level,
            "evidence": ev_dict,
        }

    def generate_verification_report(
        self,
        assessment: Dict[str, Any],
        document_metadata: Optional[Dict[str, Any]] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        Generate a human-readable verification screening report.
        """
        doc_meta = document_metadata or {}
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        risk_level = assessment.get("risk_level", "UNKNOWN")
        score = assessment.get("integrity_score", 0.0)
        evidence = assessment.get("evidence", {})

        # Operational recommendation
        if risk_level == "LOW":
            rec = "PASS / LOW RISK: Consistent visual and textual evidence. Suitable for low-risk automated processing."
        elif risk_level == "MEDIUM":
            rec = "REVIEW / MEDIUM RISK: Minor anomalies or incomplete evidence detected. Manual secondary inspection recommended."
        else:
            rec = "FLAG / HIGH RISK: Significant visual, temporal, or textual inconsistency detected. Escalation for fraud review recommended."

        lines = [
            "=" * 70,
            "          IDENTITY DOCUMENT INTEGRITY VERIFICATION REPORT          ",
            "=" * 70,
            f"Report Generated:       {timestamp}",
            f"Verification Outcome:   {risk_level} RISK (Integrity Score: {score * 100:.1f}%)",
            f"Recommendation:         {rec}",
            "-" * 70,
            "DOCUMENT METADATA (EXTRACTED VIA OCR EVIDENCE):",
            f"  Document Number:      {doc_meta.get('document_number', 'Not Extracted')}",
            f"  Holder Name:          {doc_meta.get('name', 'Not Extracted')}",
            f"  Date of Birth:        {doc_meta.get('dob', 'Not Extracted')}",
            f"  Address:              {doc_meta.get('address', 'Not Extracted')}",
            "-" * 70,
            "MULTI-MODAL EVIDENCE SUMMARY:",
        ]

        # Multi-modal breakdown
        labels = [
            ("Visual Integrity", "visual_integrity"),
            ("Temporal Consistency", "temporal_consistency"),
            ("OCR Confidence", "ocr_confidence"),
            ("Text Consistency", "text_consistency"),
        ]

        for display_name, key in labels:
            val = evidence.get(key)
            if val is not None:
                val_str = f"{val * 100:.1f}%"
                status = "PASS" if val >= 0.70 else ("REVIEW" if val >= 0.40 else "FLAG")
            else:
                val_str = "NOT AVAILABLE"
                status = "MISSING"
            lines.append(f"  * {display_name:<24}: {val_str:<15} [{status}]")

        lines.extend([
            "-" * 70,
            "THRESHOLD CONFIGURATION (HEURISTIC BENCHMARKS):",
            f"  * Low Risk Boundary:     score >= {self.low_threshold * 100:.0f}%",
            f"  * Medium Risk Boundary:  {self.medium_threshold * 100:.0f}% <= score < {self.low_threshold * 100:.0f}%",
            f"  * High Risk Boundary:    score < {self.medium_threshold * 100:.0f}%",
            "-" * 70,
            "LEGAL & REGULATORY NOTICE:",
            f"  {LEGAL_DISCLAIMER}",
            "=" * 70,
        ])

        report_text = "\n".join(lines)

        if output_path is not None:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                f.write(report_text + "\n")
            logger.info(f"Saved verification report to: {out_p}")

        return report_text


def assess_risk(
    integrity_score: float,
    evidence: Optional[Dict[str, Any]] = None,
    config_path: Optional[Union[str, Path]] = "config/settings.yaml",
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Convenience functional interface for risk assessment.
    """
    classifier = RiskClassifier(config_path=config_path)
    return classifier.assess_risk(integrity_score=integrity_score, evidence=evidence, **kwargs)
