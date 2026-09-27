"""
Evidence Fusion Module
======================
Combine sub-scores from visual, temporal, and textual branches into a unified
integrity score and risk evaluation.

Submodules:
    - score_fusion:      Weighted / dynamic fusion of sub-scores
    - risk_classifier:   Map fused score to Low / Medium / High risk
"""

from src.fusion.risk_classifier import (
    DEFAULT_LOW_RISK_THRESHOLD,
    DEFAULT_MEDIUM_RISK_THRESHOLD,
    LEGAL_DISCLAIMER,
    RiskClassifier,
    assess_risk,
)
from src.fusion.score_fusion import (
    DEFAULT_WEIGHTS,
    EvidenceDimension,
    EvidenceFusionEngine,
    fuse_evidence,
)

__all__ = [
    "EvidenceFusionEngine",
    "EvidenceDimension",
    "DEFAULT_WEIGHTS",
    "fuse_evidence",
    "RiskClassifier",
    "assess_risk",
    "DEFAULT_LOW_RISK_THRESHOLD",
    "DEFAULT_MEDIUM_RISK_THRESHOLD",
    "LEGAL_DISCLAIMER",
]
