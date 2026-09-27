"""
Unit and Integration Tests for OCR & Text Verification (Phase 4)
================================================================
Tests:
- Key-field extraction (Name, DOB, Document Number, Address)
- Field presence verification
- Field format validation rules
- Unavailable field reporting (no hallucination)
- Cross-frame textual consistency scoring
- Structured JSON output contract
"""

from pathlib import Path
import numpy as np
import pytest

from src.text.field_parser import FieldParser
from src.text.text_consistency import TextConsistencyAnalyzer, compute_string_similarity


class TestFieldParser:
    """Test field extraction, presence checking, and format validation."""

    @pytest.fixture
    def parser(self) -> FieldParser:
        return FieldParser()

    def test_complete_synthetic_text_extraction(self, parser: FieldParser):
        sample_text = (
            "MOCK IDENTITY CARD\n"
            "NAME: SAMPLE CITIZEN\n"
            "DOB: 12/04/1998\n"
            "DOC NO: ID-4421-8890\n"
            "ADDR: 42 ACADEMIC WAY"
        )
        result = parser.parse_fields(raw_text=sample_text)

        assert result["name"] == "SAMPLE CITIZEN"
        assert result["dob"] == "12/04/1998"
        assert result["document_number"] == "ID-4421-8890"
        assert result["address"] == "42 ACADEMIC WAY"

        # Field presence checks
        assert result["field_presence"]["name"] is True
        assert result["field_presence"]["dob"] is True
        assert result["field_presence"]["document_number"] is True
        assert result["field_presence"]["address"] is True

        # Format validation checks
        assert result["format_validity"]["name"] is True
        assert result["format_validity"]["dob"] is True
        assert result["format_validity"]["document_number"] is True
        assert result["format_validity"]["address"] is True

    def test_missing_fields_report_unavailable(self, parser: FieldParser):
        """When fields are absent, report None and False presence (no hallucination)."""
        partial_text = "NAME: ALICE MORGAN\nDOC NO: ID-9988-1122"
        result = parser.parse_fields(raw_text=partial_text)

        assert result["name"] == "ALICE MORGAN"
        assert result["document_number"] == "ID-9988-1122"
        assert result["dob"] is None
        assert result["address"] is None

        assert result["field_presence"]["dob"] is False
        assert result["field_presence"]["address"] is False
        assert result["format_validity"]["dob"] is False

    def test_invalid_dob_format_flagged(self, parser: FieldParser):
        invalid_dob_text = "DOB: 99/99/1998"
        result = parser.parse_fields(raw_text=invalid_dob_text)
        assert result["dob"] == "99/99/1998"
        assert result["field_presence"]["dob"] is True
        assert result["format_validity"]["dob"] is False  # Day 99 and Month 99 are invalid


class TestTextConsistency:
    """Test cross-frame text agreement and consensus evaluation."""

    @pytest.fixture
    def analyzer(self) -> TextConsistencyAnalyzer:
        return TextConsistencyAnalyzer()

    def test_identical_cross_frame_consistency(self, analyzer: TextConsistencyAnalyzer):
        frames = [
            {
                "name": "SAMPLE CITIZEN",
                "dob": "12/04/1998",
                "document_number": "ID-4421-8890",
                "address": "42 ACADEMIC WAY",
                "format_validity": {"name": True, "dob": True, "document_number": True, "address": True},
            },
            {
                "name": "SAMPLE CITIZEN",
                "dob": "12/04/1998",
                "document_number": "ID-4421-8890",
                "address": "42 ACADEMIC WAY",
                "format_validity": {"name": True, "dob": True, "document_number": True, "address": True},
            },
            {
                "name": "SAMPLE CITIZEN",
                "dob": "12/04/1998",
                "document_number": "ID-4421-8890",
                "address": "42 ACADEMIC WAY",
                "format_validity": {"name": True, "dob": True, "document_number": True, "address": True},
            },
        ]

        summary = analyzer.analyze_cross_frame_consistency(frames, overall_ocr_confidence=0.92)

        assert summary["name"] == "SAMPLE CITIZEN"
        assert summary["dob"] == "12/04/1998"
        assert summary["document_number"] == "ID-4421-8890"
        assert summary["address"] == "42 ACADEMIC WAY"
        assert summary["text_consistency"] == 1.0
        assert summary["ocr_confidence"] == 0.92
        assert "disclaimer" in summary

    def test_contradictory_field_drops_consistency(self, analyzer: TextConsistencyAnalyzer):
        frames = [
            {
                "name": "SAMPLE CITIZEN",
                "dob": "12/04/1998",
                "document_number": "ID-4421-8890",
                "address": "42 ACADEMIC WAY",
            },
            {
                "name": "ANOTHER PERSON",  # Contradictory name
                "dob": "12/04/1998",
                "document_number": "ID-4421-8890",
                "address": "42 ACADEMIC WAY",
            },
        ]

        summary = analyzer.analyze_cross_frame_consistency(frames)
        assert summary["text_consistency"] < 1.0
        assert summary["per_field_consistency"]["name"] < 0.5
        assert summary["per_field_consistency"]["dob"] == 1.0

    def test_string_similarity_metric(self):
        assert compute_string_similarity("SAMPLE CITIZEN", "SAMPLE CITIZEN") == 1.0
        assert compute_string_similarity("SAMPLE CITIZEN", "sample citizen") == 1.0
        assert compute_string_similarity(None, "SAMPLE") == 0.0
        assert compute_string_similarity(None, None) == 1.0
        # Absorbs delimiter/punctuation differences
        assert compute_string_similarity("ID-4421-8890", "ID 4421 8890") == 1.0
        assert compute_string_similarity("DOB: 12/04/1998", "DOB; 12/04/1998") == 1.0


class TestOCRExtractionEdgeCases:
    """Test robustness against common real-world OCR noise and variations."""

    @pytest.fixture
    def parser(self) -> FieldParser:
        return FieldParser()

    def test_ocr_separator_variations(self, parser: FieldParser):
        """Test OCR output where ':' is misread as ';', '.', or '-'."""
        noisy_text = (
            "NAME; TEST USER\n"
            "DOB; 01/01/2000\n"
            "DOC NO. ID-9988-1122\n"
            "ADDR - PUNE MAHARASHTRA"
        )
        res = parser.parse_fields(raw_text=noisy_text)
        assert res["name"] == "TEST USER"
        assert res["dob"] == "01/01/2000"
        assert res["document_number"] == "ID-9988-1122"
        assert res["address"] == "PUNE MAHARASHTRA"
        assert all(res["field_presence"].values())
        assert all(res["format_validity"].values())

    def test_document_number_noise_cleanup_and_digit_enforcement(self, parser: FieldParser):
        """Stray OCR symbols (+, commas) should be cleaned and non-digit candidates rejected."""
        # Stray plus sign
        text_with_noise = "DOC NO: ID-44+21-8890"
        res = parser.parse_fields(raw_text=text_with_noise)
        assert res["document_number"] == "ID-4421-8890"
        assert res["format_validity"]["document_number"] is True

        # Non-digit text must NOT be accepted as document number
        text_no_digits = "MOCK IDENTITY CARD\nNAME: SAMPLE CITIZEN"
        res2 = parser.parse_fields(raw_text=text_no_digits)
        assert res2["document_number"] is None
        assert res2["field_presence"]["document_number"] is False

    def test_dob_format_validation_bounds(self, parser: FieldParser):
        """Verify boundary checks for day, month, and year."""
        # Valid dates
        assert parser._validate_dob("01/01/2000") is True
        assert parser._validate_dob("15/08/1995") is True
        assert parser._validate_dob("29/02/2024") is True

        # Invalid bounds
        assert parser._validate_dob("32/01/2000") is False  # Day out of range
        assert parser._validate_dob("15/13/2000") is False  # Month out of range
        assert parser._validate_dob("01/01/1850") is False  # Year < 1900
        assert parser._validate_dob("01/01/2099") is False  # Future year

    def test_empty_and_failed_ocr_handling(self, parser: FieldParser):
        """Verify graceful handling of empty or None OCR text."""
        empty_res = parser.parse_fields(raw_text="")
        assert empty_res["name"] is None
        assert empty_res["dob"] is None
        assert empty_res["document_number"] is None
        assert empty_res["address"] is None
        assert not any(empty_res["field_presence"].values())
        assert not any(empty_res["format_validity"].values())

        analyzer = TextConsistencyAnalyzer()
        empty_analysis = analyzer.analyze_cross_frame_consistency([])
        assert empty_analysis["text_consistency"] == 0.0
        assert empty_analysis["ocr_confidence"] == 0.0
        assert not any(empty_analysis["field_presence"].values())


class TestOCREngineInference:
    """Test OCR engine input validation and structured output contract."""

    def test_empty_image_raises_valueerror(self):
        from src.text.ocr_engine import OCREngine
        engine = OCREngine(engine_name="easyocr")
        empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Invalid or empty image"):
            engine.extract_text(empty_img)

    def test_valid_image_returns_structured_dict(self):
        from src.text.ocr_engine import OCREngine
        import cv2

        engine = OCREngine(engine_name="easyocr")
        # Render a clear synthetic card patch
        canvas = np.full((120, 350, 3), 255, dtype=np.uint8)
        cv2.putText(canvas, "NAME: TEST USER", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(canvas, "DOB: 15/08/1995", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        res = engine.extract_text(canvas, frame_id="test_01")
        assert res["frame_id"] == "test_01"
        assert "engine_used" in res
        assert "lines" in res
        assert "average_confidence" in res
        assert "full_text" in res
        assert res["average_confidence"] >= 0.0


def test_ocr_pipeline_validation_run(tmp_path):
    """Integration test verifying complete run_ocr_validation function."""
    from scripts.validate_ocr_pipeline import run_ocr_validation

    norm_dir = Path("outputs/preprocessing_validation/normalized")
    if not norm_dir.is_dir() or not list(norm_dir.glob("*.png")):
        pytest.skip(f"Normalized frames not available at {norm_dir}")

    out_dir = tmp_path / "ocr_validation_test"
    summary = run_ocr_validation(
        normalized_dir=norm_dir,
        output_dir=out_dir,
        config_path="config/settings.yaml",
    )

    assert summary["total_frames_processed"] == 10
    assert summary["successful_ocr_frames"] == 10
    assert summary["ocr_success_rate"] == 1.0
    assert summary["average_ocr_confidence"] > 0.7
    assert summary["cross_frame_text_consistency_score"] > 0.8
    assert summary["consensus_fields"]["name"] == "SAMPLE CITIZEN"

    # Verify generated artifacts
    assert (out_dir / "results.json").is_file()
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "contact_sheet.png").is_file()
    assert (out_dir / "raw").is_dir()
    assert (out_dir / "annotated").is_dir()
    assert len(list((out_dir / "annotated").glob("*.png"))) == 10

