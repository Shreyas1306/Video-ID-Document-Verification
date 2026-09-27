"""Tests for src.fusion (score_fusion + risk_classifier)"""

import sys
from pathlib import Path
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion.score_fusion import EvidenceFusionEngine, fuse_evidence


class TestScoreFusionSmoke:
    """Smoke tests for score_fusion module."""

    def test_module_importable(self):
        from src.fusion import score_fusion  # noqa: F401
        assert score_fusion is not None

    def test_docstring_mentions_weighted_average(self):
        """Initial version should use weighted average only."""
        from src.fusion import score_fusion
        assert "weighted average" in score_fusion.__doc__.lower()


class TestRiskClassifierSmoke:
    """Smoke tests for risk_classifier module."""

    def test_module_importable(self):
        from src.fusion import risk_classifier  # noqa: F401
        assert risk_classifier is not None

    def test_module_has_docstring(self):
        from src.fusion import risk_classifier
        assert risk_classifier.__doc__ is not None


class TestConfigSmoke:
    """Verify config/settings.yaml is loadable and has expected structure."""

    def test_config_loadable(self):
        import yaml
        config_path = PROJECT_ROOT / "config" / "settings.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        assert isinstance(config, dict)

    def test_config_has_required_sections(self):
        import yaml
        config_path = PROJECT_ROOT / "config" / "settings.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        required_sections = [
            "frame_extraction",
            "document_detection",
            "perspective_correction",
            "visual_features",
            "temporal_consistency",
            "ocr",
            "field_parsing",
            "fusion",
            "risk_classification",
            "paths",
        ]
        for section in required_sections:
            assert section in config, f"Missing config section: {section}"

    def test_document_detection_uses_yolo(self):
        """Phase 2 config should define YOLO model settings."""
        import yaml
        config_path = PROJECT_ROOT / "config" / "settings.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        assert "model_name" in config["document_detection"]
        assert "confidence_threshold" in config["document_detection"]

    def test_fusion_uses_weighted_average(self):
        """Initial config should default to weighted_average fusion."""
        import yaml
        config_path = PROJECT_ROOT / "config" / "settings.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        assert config["fusion"]["method"] == "weighted_average"

    def test_backbone_is_frozen(self):
        """Visual features backbone should be frozen."""
        import yaml
        config_path = PROJECT_ROOT / "config" / "settings.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        assert config["visual_features"]["freeze_backbone"] is True


class TestEvidenceFusionFunctional:
    """Functional tests for Phase 7 Evidence Fusion."""

    def test_weight_normalization_and_resolution(self):
        """Verify that weights load from config or defaults and sum to 1.0."""
        engine = EvidenceFusionEngine(config_path=PROJECT_ROOT / "config" / "settings.yaml")
        total_base = sum(engine.base_weights.values())
        np.testing.assert_allclose(total_base, 1.0, atol=1e-5)
        assert engine.base_weights["visual_integrity"] == pytest.approx(0.40, rel=1e-2)
        assert engine.base_weights["temporal_consistency"] == pytest.approx(0.30, rel=1e-2)
        assert engine.base_weights["ocr_confidence"] == pytest.approx(0.15, rel=1e-2)
        assert engine.base_weights["text_consistency"] == pytest.approx(0.15, rel=1e-2)

    def test_score_normalization(self):
        """Verify input score sanitization (clipping, percentage scaling, invalid handling)."""
        engine = EvidenceFusionEngine()
        assert engine.normalize_score(0.85) == 0.85
        assert engine.normalize_score("0.75") == 0.75
        assert engine.normalize_score(90.0) == 0.90  # Percentage scaled to [0, 1]
        assert engine.normalize_score(-0.2) == 0.0
        assert engine.normalize_score(1.5) == 1.0
        assert engine.normalize_score(None) is None
        assert engine.normalize_score("invalid") is None
        assert engine.normalize_score(float("nan")) is None

    def test_full_evidence_fusion_calculation(self):
        """Verify exact calculation when all 4 evidence dimensions are present."""
        custom_weights = {
            "visual_weight": 0.40,
            "temporal_weight": 0.30,
            "ocr_weight": 0.15,
            "text_consistency_weight": 0.15,
        }
        engine = EvidenceFusionEngine(weights=custom_weights)

        # Expected: 0.40*0.80 + 0.30*0.90 + 0.15*0.70 + 0.15*1.00
        # = 0.32 + 0.27 + 0.105 + 0.15 = 0.8450
        result = engine.fuse(
            visual_integrity=0.80,
            temporal_consistency=0.90,
            ocr_confidence=0.70,
            text_consistency=1.00,
        )

        assert result["fused_integrity_score"] == 0.845
        assert result["evidence_coverage"] == 1.0
        assert len(result["available_sources"]) == 4
        assert len(result["missing_sources"]) == 0

        # Check breakdown
        bd = result["evidence_breakdown"]
        assert bd["visual_integrity"]["status"] == "available"
        assert bd["visual_integrity"]["effective_weight"] == 0.40
        assert bd["visual_integrity"]["weighted_contribution"] == 0.32

    def test_missing_evidence_dynamic_reweighting(self):
        """Verify graceful handling and re-weighting when some sources are missing."""
        custom_weights = {
            "visual_weight": 0.40,
            "temporal_weight": 0.30,
            "ocr_weight": 0.15,
            "text_consistency_weight": 0.15,
        }
        engine = EvidenceFusionEngine(weights=custom_weights)

        # Scenario: Video has only 1 frame, so temporal_consistency is None.
        # OCR was illegible, so ocr_confidence is None.
        # Available: visual_integrity (0.80) and text_consistency (1.00).
        # Base weights: visual=0.40, text=0.15. Total available base weight = 0.55.
        # Effective weights:
        # visual: 0.40 / 0.55 = 0.72727...
        # text:   0.15 / 0.55 = 0.27272...
        # Sum of effective weights = 1.0
        # Fused score: 0.72727 * 0.80 + 0.27272 * 1.00 = 0.58181 + 0.27272 = 0.8545
        result = engine.fuse(
            visual_integrity=0.80,
            temporal_consistency=None,
            ocr_confidence=None,
            text_consistency=1.00,
        )

        assert result["fused_integrity_score"] == 0.8545
        assert result["evidence_coverage"] == pytest.approx(0.55, rel=1e-3)
        assert set(result["available_sources"]) == {"visual_integrity", "text_consistency"}
        assert set(result["missing_sources"]) == {"temporal_consistency", "ocr_confidence"}

        bd = result["evidence_breakdown"]
        assert bd["temporal_consistency"]["status"] == "missing"
        assert bd["temporal_consistency"]["effective_weight"] == 0.0
        assert bd["temporal_consistency"]["weighted_contribution"] == 0.0

        eff_sum = bd["visual_integrity"]["effective_weight"] + bd["text_consistency"]["effective_weight"]
        np.testing.assert_allclose(eff_sum, 1.0, atol=1e-4)

    def test_single_evidence_source(self):
        """When only 1 evidence source is available, fused score equals that source's score."""
        engine = EvidenceFusionEngine()
        result = engine.fuse(visual_integrity=0.65)
        assert result["fused_integrity_score"] == 0.65
        assert result["evidence_coverage"] == pytest.approx(0.40, rel=1e-2)
        assert result["num_sources_available"] == 1

    def test_zero_evidence_edge_case(self):
        """When no evidence sources are provided, returns 0.0 with 0.0 coverage."""
        engine = EvidenceFusionEngine()
        result = engine.fuse()
        assert result["fused_integrity_score"] == 0.0
        assert result["evidence_coverage"] == 0.0
        assert len(result["available_sources"]) == 0
        assert len(result["missing_sources"]) == 4

    def test_sample_case_behaviors(self):
        """Validate expected fused score behaviors on characteristic sample cases."""
        engine = EvidenceFusionEngine()

        # Case 1: Pristine Genuine Document (all high)
        case_real = engine.fuse(visual_integrity=0.95, temporal_consistency=0.92, ocr_confidence=0.90, text_consistency=1.0)
        assert case_real["fused_integrity_score"] >= 0.90

        # Case 2: Visual Tamper (altered photo/text patch -> visual drops sharply)
        case_tamper = engine.fuse(visual_integrity=0.15, temporal_consistency=0.88, ocr_confidence=0.85, text_consistency=0.90)
        # Visual weight is 0.40, so low visual drags score down:
        # 0.40*0.15 + 0.30*0.88 + 0.15*0.85 + 0.15*0.90 = 0.06 + 0.264 + 0.1275 + 0.135 = 0.5865
        assert case_tamper["fused_integrity_score"] < 0.60

        # Case 3: Replay / Screen Attack (inconsistent temporal frames -> temporal drops sharply)
        case_replay = engine.fuse(visual_integrity=0.85, temporal_consistency=0.25, ocr_confidence=0.80, text_consistency=0.85)
        assert case_replay["fused_integrity_score"] < 0.67

        # Case 4: Text Splicing / Cross-frame contradiction (text consistency drops to 0.0)
        case_text_anomaly = engine.fuse(visual_integrity=0.90, temporal_consistency=0.90, ocr_confidence=0.70, text_consistency=0.0)
        assert case_text_anomaly["fused_integrity_score"] == pytest.approx(0.735, rel=1e-3)


class TestRiskClassifierFunctional:
    """Functional tests for Phase 8 Risk Assessment."""

    def test_threshold_boundaries(self):
        """Verify strict boundary behavior for LOW, MEDIUM, and HIGH risk thresholds."""
        from src.fusion.risk_classifier import RiskClassifier

        classifier = RiskClassifier(low_risk_threshold=0.70, medium_risk_threshold=0.40)

        # LOW RISK: score >= 0.70
        assert classifier.determine_risk_level(1.00) == "LOW"
        assert classifier.determine_risk_level(0.85) == "LOW"
        assert classifier.determine_risk_level(0.70) == "LOW"

        # MEDIUM RISK: 0.40 <= score < 0.70
        assert classifier.determine_risk_level(0.6999) == "MEDIUM"
        assert classifier.determine_risk_level(0.55) == "MEDIUM"
        assert classifier.determine_risk_level(0.40) == "MEDIUM"

        # HIGH RISK: score < 0.40
        assert classifier.determine_risk_level(0.3999) == "HIGH"
        assert classifier.determine_risk_level(0.20) == "HIGH"
        assert classifier.determine_risk_level(0.00) == "HIGH"

    def test_structured_output_format(self):
        """Verify the exact required output JSON dictionary format."""
        from src.fusion.risk_classifier import RiskClassifier

        classifier = RiskClassifier()
        result = classifier.assess_risk(
            integrity_score=0.8152,
            visual_integrity=0.6664,
            temporal_consistency=0.8881,
            ocr_confidence=0.8812,
            text_consistency=1.0,
        )

        assert "integrity_score" in result
        assert "risk_level" in result
        assert "evidence" in result

        assert result["integrity_score"] == 0.8152
        assert result["risk_level"] == "LOW"

        ev = result["evidence"]
        assert ev["visual_integrity"] == 0.6664
        assert ev["temporal_consistency"] == 0.8881
        assert ev["ocr_confidence"] == 0.8812
        assert ev["text_consistency"] == 1.0

    def test_missing_evidence_in_assessment(self):
        """Verify assessment formatting when one or more evidence items are None."""
        from src.fusion.risk_classifier import RiskClassifier

        classifier = RiskClassifier()
        result = classifier.assess_risk(
            integrity_score=0.55,
            visual_integrity=0.55,
            temporal_consistency=None,  # e.g. single frame video
            ocr_confidence=None,        # e.g. OCR unreadable
            text_consistency=None,
        )

        assert result["risk_level"] == "MEDIUM"
        assert result["integrity_score"] == 0.55
        assert result["evidence"]["visual_integrity"] == 0.55
        assert result["evidence"]["temporal_consistency"] is None
        assert result["evidence"]["ocr_confidence"] is None
        assert result["evidence"]["text_consistency"] is None

    def test_verification_report_generation(self, tmp_path):
        """Verify human-readable report formatting and required legal disclaimer."""
        from src.fusion.risk_classifier import RiskClassifier, LEGAL_DISCLAIMER

        classifier = RiskClassifier()
        assessment = classifier.assess_risk(
            integrity_score=0.8152,
            visual_integrity=0.6664,
            temporal_consistency=0.8881,
            ocr_confidence=0.8812,
            text_consistency=1.0,
        )

        doc_meta = {
            "name": "SAMPLE CITIZEN",
            "dob": "12/04/1998",
            "document_number": "ID-4421-8890",
            "address": "42 ACADEMIC WAY",
        }

        report_file = tmp_path / "verification_report.txt"
        report_text = classifier.generate_verification_report(
            assessment=assessment,
            document_metadata=doc_meta,
            output_path=report_file,
        )

        # File created
        assert report_file.exists()

        # Report content assertions
        assert "IDENTITY DOCUMENT INTEGRITY VERIFICATION REPORT" in report_text
        assert "LOW RISK" in report_text
        assert "SAMPLE CITIZEN" in report_text
        assert "ID-4421-8890" in report_text
        assert "Visual Integrity" in report_text

        # Mandatory disclaimer check
        assert "official government authentication" in report_text
        assert LEGAL_DISCLAIMER in report_text
