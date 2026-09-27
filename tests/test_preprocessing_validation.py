"""
Integration tests for End-to-End Preprocessing Validation Pipeline.
==================================================================
Tests:
  - Video -> Frame Extraction -> Document Detection -> 4 Corners -> Perspective Correction.
  - Verification of metrics: detection rate, perspective correction rate, confidence.
"""

from pathlib import Path
import pytest
from scripts.validate_preprocessing_pipeline import run_preprocessing_validation

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_VIDEO_PATH = PROJECT_ROOT / "data" / "sample" / "sample_document_video.mp4"
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


def test_preprocessing_validation_end_to_end(tmp_path):
    """Verify complete preprocessing pipeline on sample mock document video."""
    if not SAMPLE_VIDEO_PATH.is_file():
        pytest.skip(f"Sample video not found at {SAMPLE_VIDEO_PATH}")

    output_dir = tmp_path / "preprocessing_validation"

    summary = run_preprocessing_validation(
        video_path=SAMPLE_VIDEO_PATH,
        output_dir=output_dir,
        config_path=CONFIG_PATH,
    )

    # Check summary metrics
    assert summary["sampled_frames"] > 0
    assert summary["successful_detections"] > 0
    assert summary["detection_rate"] > 0.8
    assert summary["perspective_success_rate"] > 0.8
    assert summary["average_detection_confidence"] > 0.8

    # Verify directories and artifacts exist
    assert (output_dir / "frames").is_dir()
    assert (output_dir / "detected").is_dir()
    assert (output_dir / "normalized").is_dir()
    assert (output_dir / "metadata.json").is_file()
    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "normalized_contact_sheet.png").is_file()

    # Verify extracted normalized frames match successful perspective count
    normalized_files = list((output_dir / "normalized").glob("*.png"))
    assert len(normalized_files) == summary["successful_perspective_corrections"]
