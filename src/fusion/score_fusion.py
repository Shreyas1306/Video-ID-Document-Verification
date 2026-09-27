"""
Evidence Fusion Engine
======================
Combine multi-modal evidence scores from visual, temporal, and textual branches
into a unified document integrity score.

Methodology:
  - Baseline: Configurable weighted average.
  - Normalization: Bounds checking and clipping of all input evidence scores to [0.0, 1.0].
  - Missing Evidence Resilience: Dynamic re-normalization of active weights across
    available sources so effective weights always sum to 1.0.
  - Provenance Tracking: Explicit record of available vs. missing evidence sources
    and an overall evidence coverage / confidence quality indicator.

Notice:
  Default weights (visual: 0.40, temporal: 0.30, ocr: 0.15, text: 0.15) are
  engineering defaults, not scientifically validated values.
"""

from dataclasses import asdict, dataclass
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "visual_integrity": 0.40,
    "temporal_consistency": 0.30,
    "ocr_confidence": 0.15,
    "text_consistency": 0.15,
}

WEIGHT_ALIASES = {
    "visual_weight": "visual_integrity",
    "visual_integrity": "visual_integrity",
    "visual": "visual_integrity",
    "temporal_weight": "temporal_consistency",
    "temporal_consistency": "temporal_consistency",
    "temporal": "temporal_consistency",
    "ocr_weight": "ocr_confidence",
    "ocr_confidence": "ocr_confidence",
    "ocr": "ocr_confidence",
    "text_consistency_weight": "text_consistency",
    "text_consistency": "text_consistency",
    "text": "text_consistency",
}


@dataclass
class EvidenceDimension:
    """Detailed breakdown for an individual evidence source."""
    source_name: str
    raw_score: Optional[float]
    normalized_score: Optional[float]
    base_weight: float
    effective_weight: float
    weighted_contribution: float
    status: str  # "available" | "missing"


class EvidenceFusionEngine:
    """
    Combines visual, temporal, OCR, and text consistency evidence scores.
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.method = "weighted_average"
        self.base_weights = self._load_and_resolve_weights(config_path, weights)

    def _load_and_resolve_weights(
        self,
        config_path: Optional[Union[str, Path]],
        custom_weights: Optional[Dict[str, float]],
    ) -> Dict[str, float]:
        """Resolve weights from custom dict, config file, or engineering defaults."""
        weights = DEFAULT_WEIGHTS.copy()

        if config_path is not None:
            try:
                cfg = load_config(config_path)
                fusion_cfg = cfg.get("fusion", {})
                self.method = fusion_cfg.get("method", "weighted_average")
                raw_weights = fusion_cfg.get("weights", {})
                for k, v in raw_weights.items():
                    standard_key = WEIGHT_ALIASES.get(k.lower(), k)
                    if standard_key in weights and v is not None:
                        weights[standard_key] = float(v)
            except Exception as e:
                logger.warning(f"Could not load fusion configuration from {config_path}: {e}")

        if custom_weights:
            for k, v in custom_weights.items():
                standard_key = WEIGHT_ALIASES.get(k.lower(), k)
                if standard_key in weights and v is not None:
                    weights[standard_key] = float(v)

        # Validate total weight sum > 0
        total = sum(weights.values())
        if total <= 0:
            raise ValueError(f"Total base weights must be positive, got {total}")

        # Normalize base weights to sum to 1.0
        normalized_base = {k: v / total for k, v in weights.items()}
        return normalized_base

    @staticmethod
    def normalize_score(raw_score: Any) -> Optional[float]:
        """
        Safely convert raw score to a normalized float in [0.0, 1.0].
        Returns None if score is missing or invalid.
        """
        if raw_score is None:
            return None

        try:
            val = float(raw_score)
            if math.isnan(val) or math.isinf(val):
                return None
            # If provided as percentage [5.0, 100.0], scale to [0, 1]
            if val >= 5.0 and val <= 100.0:
                val = val / 100.0
            return max(0.0, min(1.0, val))
        except (ValueError, TypeError):
            return None

    def fuse(
        self,
        visual_integrity: Optional[float] = None,
        temporal_consistency: Optional[float] = None,
        ocr_confidence: Optional[float] = None,
        text_consistency: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Fuse multi-modal evidence into a unified integrity score.

        Args:
            visual_integrity: Visual document integrity score in [0, 1].
            temporal_consistency: Temporal cross-frame consistency score in [0, 1].
            ocr_confidence: Character/field OCR confidence score in [0, 1].
            text_consistency: Textual cross-frame consistency score in [0, 1].

        Returns:
            Structured dictionary with fused score, evidence breakdown, and coverage.
        """
        raw_inputs = {
            "visual_integrity": visual_integrity,
            "temporal_consistency": temporal_consistency,
            "ocr_confidence": ocr_confidence,
            "text_consistency": text_consistency,
        }

        # Check for any alias arguments passed in kwargs
        for k, v in kwargs.items():
            std_k = WEIGHT_ALIASES.get(k.lower())
            if std_k and raw_inputs.get(std_k) is None:
                raw_inputs[std_k] = v

        # 1. Normalize all input scores
        norm_scores = {k: self.normalize_score(v) for k, v in raw_inputs.items()}

        available_sources = [k for k, v in norm_scores.items() if v is not None]
        missing_sources = [k for k, v in norm_scores.items() if v is None]

        # 2. Dynamic Weight Re-normalization across available sources
        available_base_weight_sum = sum(self.base_weights[k] for k in available_sources)

        evidence_breakdown: Dict[str, Dict[str, Any]] = {}
        fused_score = 0.0

        if available_base_weight_sum > 0:
            for k, base_w in self.base_weights.items():
                s = norm_scores.get(k)
                if s is not None:
                    # Effective weight re-scaled to sum to 1.0 across available sources
                    eff_w = base_w / available_base_weight_sum
                    contrib = eff_w * s
                    fused_score += contrib
                    dim = EvidenceDimension(
                        source_name=k,
                        raw_score=raw_inputs.get(k),
                        normalized_score=round(s, 4),
                        base_weight=round(base_w, 4),
                        effective_weight=round(eff_w, 4),
                        weighted_contribution=round(contrib, 4),
                        status="available",
                    )
                else:
                    dim = EvidenceDimension(
                        source_name=k,
                        raw_score=None,
                        normalized_score=None,
                        base_weight=round(base_w, 4),
                        effective_weight=0.0,
                        weighted_contribution=0.0,
                        status="missing",
                    )
                evidence_breakdown[k] = asdict(dim)
        else:
            # Zero available evidence
            logger.warning("No valid evidence sources available for fusion.")
            for k, base_w in self.base_weights.items():
                evidence_breakdown[k] = asdict(EvidenceDimension(
                    source_name=k,
                    raw_score=None,
                    normalized_score=None,
                    base_weight=round(base_w, 4),
                    effective_weight=0.0,
                    weighted_contribution=0.0,
                    status="missing",
                ))

        total_base_weight = sum(self.base_weights.values())
        evidence_coverage = round(available_base_weight_sum / total_base_weight, 4) if total_base_weight > 0 else 0.0

        fused_score = round(max(0.0, min(1.0, fused_score)), 4)

        return {
            "fused_integrity_score": fused_score,
            "evidence_coverage": evidence_coverage,
            "fusion_method": self.method,
            "available_sources": available_sources,
            "missing_sources": missing_sources,
            "num_sources_available": len(available_sources),
            "num_sources_total": len(self.base_weights),
            "evidence_breakdown": evidence_breakdown,
            "disclaimer": (
                "Notice: Weights are engineering defaults and fused scores represent "
                "relative verification evidence heuristics, not legal or official proof of authenticity."
            ),
        }


def fuse_evidence(
    visual_integrity: Optional[float] = None,
    temporal_consistency: Optional[float] = None,
    ocr_confidence: Optional[float] = None,
    text_consistency: Optional[float] = None,
    config_path: Optional[Union[str, Path]] = "config/settings.yaml",
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Convenience functional interface for evidence fusion.
    """
    engine = EvidenceFusionEngine(config_path=config_path)
    return engine.fuse(
        visual_integrity=visual_integrity,
        temporal_consistency=temporal_consistency,
        ocr_confidence=ocr_confidence,
        text_consistency=text_consistency,
        **kwargs,
    )
