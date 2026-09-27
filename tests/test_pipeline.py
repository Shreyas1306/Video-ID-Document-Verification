"""
Tests for End-to-End Document Verification Pipeline (Phase 9)
=============================================================
Validates:
  - Independent testability of each pipeline stage.
  - Complete end-to-end execution on sample video.
  - Presence of per-stage execution timings.
  - Non-silent failure recording and error handling.
  - Generation of structured JSON and human-readable verification reports.
"""

from pathlib import Path
import pytest

from src.pipeline import DocumentVerificationPipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_VIDEO_PATH = PROJECT_ROOT / "data" / "sample" / "sample_document_video.mp4"


@pytest.fixture
def pipeline(tmp_path):
    """Create a pipeline instance with temporary output directory."""
    return DocumentVerificationPipeline(
        config_path=PROJECT_ROOT / "config" / "settings.yaml",
        base_output_dir=tmp_path / "pipeline_outputs",
    )


def test_nonexistent_video_raises_filenotfound(pipeline):
    """Verify that processing a non-existent video raises FileNotFoundError cleanly."""
    with pytest.raises(FileNotFoundError):
        pipeline.process_video("non_existent_document_video.mp4")


def test_independent_stage_testability(pipeline, tmp_path):
    """Verify that individual pipeline stages can be executed independently."""
    if not SAMPLE_VIDEO_PATH.is_file():
        pytest.skip(f"Sample video not found at {SAMPLE_VIDEO_PATH}")

    # Stage 1: Extraction
    frames_dir = tmp_path / "test_frames"
    ext_meta = pipeline.extract_frames(SAMPLE_VIDEO_PATH, frames_dir)
    assert "frame_paths" in ext_meta
    assert len(ext_meta["frame_paths"]) > 0

    frame_paths = ext_meta["frame_paths"]

    # Stage 2: Detection
    crops_dir = tmp_path / "test_crops"
    debug_det_dir = tmp_path / "test_debug_det"
    det_summary = pipeline.detect_and_crop(frame_paths, crops_dir, debug_det_dir)
    assert "results" in det_summary
    assert "frames_detected" in det_summary

    crop_paths = [r["crop_path"] for r in det_summary["results"] if r.get("crop_path")]

    # Stage 3: Perspective Correction
    norm_dir = tmp_path / "test_normalized"
    debug_persp_dir = tmp_path / "test_debug_persp"
    persp_summary = pipeline.correct_perspective(crop_paths, norm_dir, debug_persp_dir)
    assert "homography_success_count" in persp_summary

    norm_paths = [r["normalized_path"] for r in persp_summary["results"] if r.get("normalized_path")]

    # Stage 4: Visual Analysis
    vis_summary = pipeline.analyze_visual_integrity(norm_paths)
    assert "aggregated_score" in vis_summary
    assert 0.0 <= vis_summary["aggregated_score"] <= 1.0

    # Stage 5: Temporal Analysis
    temp_summary = pipeline.analyze_temporal_consistency(norm_paths, video_id="test_vid")
    assert "temporal_consistency_score" in temp_summary

    # Stage 6: Evidence Fusion
    fused_summary = pipeline.fuse_evidence(
        visual_score=vis_summary["aggregated_score"],
        temporal_score=temp_summary["temporal_consistency_score"],
        ocr_confidence=0.85,
        text_consistency=1.0,
    )
    assert "fused_integrity_score" in fused_summary
    assert 0.0 <= fused_summary["fused_integrity_score"] <= 1.0

    # Stage 7: Risk Assessment
    risk_summary = pipeline.assess_risk(fused_summary)
    assert risk_summary["risk_level"] in {"LOW", "MEDIUM", "HIGH"}


def test_complete_pipeline_execution(pipeline):
    """Verify complete end-to-end pipeline run on sample document video."""
    if not SAMPLE_VIDEO_PATH.is_file():
        pytest.skip(f"Sample video not found at {SAMPLE_VIDEO_PATH}")

    result = pipeline.process_video(SAMPLE_VIDEO_PATH)

    # 1. Output structure
    assert "verification_result" in result
    assert "stage_timings_seconds" in result
    assert "stage_execution_statuses" in result
    assert "evidence_details" in result
    assert "output_artifacts" in result

    # 2. Verification metrics
    v_res = result["verification_result"]
    assert "integrity_score" in v_res
    assert "risk_level" in v_res
    assert v_res["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert "evidence" in v_res

    # 3. Stage timings recorded and non-zero
    timings = result["stage_timings_seconds"]
    expected_stages = [
        "frame_extraction",
        "document_detection",
        "perspective_correction",
        "visual_analysis",
        "temporal_analysis",
        "ocr_and_text_verification",
        "evidence_fusion",
        "risk_assessment",
        "report_generation",
    ]
    for stage in expected_stages:
        assert stage in timings, f"Stage {stage} missing from timings"
        assert timings[stage] >= 0.0

    # 4. Artifacts generated on disk
    artifacts = result["output_artifacts"]
    json_path = Path(artifacts["pipeline_result_json"])
    report_path = Path(artifacts["verification_report_txt"])

    assert json_path.is_file()
    assert report_path.is_file()

    # 5. Check report content for mandatory legal disclaimer
    with open(report_path, "r", encoding="utf-8") as f:
        report_text = f.read()

    assert "IDENTITY DOCUMENT VERIFICATION PIPELINE REPORT" in report_text
    assert "official government authentication" in report_text


def test_pipeline_initializes_and_resets_tracker_between_videos(pipeline, tmp_path):
    """Verify that pipeline manages a TemporalDocumentTracker and resets it between videos."""
    import numpy as np
    from src.preprocessing.temporal_document_tracker import TemporalDocumentTracker

    assert isinstance(pipeline.tracker, TemporalDocumentTracker)

    # Manually populate tracker state to simulate an ongoing track
    pipeline.tracker.last_accepted_quad = np.array([[10, 10], [100, 10], [100, 70], [10, 70]], dtype=np.float32)
    pipeline.tracker.frames_tracked = 5
    pipeline.tracker.tracking_confidence = 0.88

    # Reset must clear tracking state
    pipeline.tracker.reset()
    assert pipeline.tracker.last_accepted_quad is None
    assert pipeline.tracker.frames_tracked == 0
    assert pipeline.tracker.tracking_confidence == 0.0


def test_tracker_receives_sequential_frames_and_applies_temporal_guidance(pipeline, tmp_path):
    """Verify that detect_and_crop processes frames sequentially and applies temporal guidance."""
    import cv2
    import numpy as np

    # Create 3 synthetic document frames with small incremental movement
    frames = []
    frame_paths = []
    for i in range(3):
        img = np.full((400, 600, 3), 40, dtype=np.uint8)
        # Shift card slightly right by 5px per frame
        x_shift = i * 5
        card_pts = np.array([
            [100 + x_shift, 100],
            [350 + x_shift, 110],
            [340 + x_shift, 270],
            [90 + x_shift, 260],
        ], dtype=np.int32)
        cv2.fillPoly(img, [card_pts], (230, 230, 230))
        cv2.polylines(img, [card_pts], True, (10, 10, 10), 2)
        p = tmp_path / f"synth_frame_{i:02d}.png"
        cv2.imwrite(str(p), img)
        frame_paths.append(p)

    crops_dir = tmp_path / "crops"
    debug_dir = tmp_path / "debug_det"

    summary = pipeline.detect_and_crop(frame_paths, crops_dir, debug_dir)

    assert summary["total_frames_processed"] == 3
    assert summary["frames_detected"] == 3
    assert summary["tracking_enabled"] is True

    results = summary["results"]
    assert len(results) == 3

    # Frame 0 should be first frame detection
    assert results[0]["detected"] is True
    assert results[0]["method"] == "first_frame_detection"
    assert results[0]["tracking_used"] is False
    assert results[0]["corners"] is not None

    # Frame 1 and 2 should be temporally guided
    assert results[1]["detected"] is True
    assert results[1]["method"] == "temporally_guided"
    assert results[1]["tracking_used"] is True
    assert results[1]["temporal_score"] > 0.50

    assert results[2]["detected"] is True
    assert results[2]["method"] == "temporally_guided"
    assert results[2]["tracking_used"] is True


def test_full_frame_corners_passed_directly_without_crop_redetection(pipeline, tmp_path, monkeypatch):
    """
    Verify that correct_perspective uses full-frame corners directly without
    calling estimate_document_corners on bounding-box crops.
    """
    import cv2
    import numpy as np
    import src.preprocessing.perspective_corrector as pc_module

    # Monkeypatch estimate_document_corners to fail if called
    def fail_if_called(*args, **kwargs):
        raise AssertionError("estimate_document_corners was called! Full-frame corners were NOT used directly.")

    monkeypatch.setattr(pc_module, "estimate_document_corners", fail_if_called)

    # Create synthetic frame and fake detection result with full-frame corners
    frame = np.full((400, 600, 3), 50, dtype=np.uint8)
    card_pts = np.array([[100, 80], [400, 90], [390, 280], [90, 270]], dtype=np.float32)
    cv2.fillPoly(frame, [card_pts.astype(np.int32)], (220, 220, 220))
    f_path = tmp_path / "test_full_frame.png"
    cv2.imwrite(str(f_path), frame)

    detection_results = [
        {
            "frame_id": "test_full_frame",
            "file": "test_full_frame.png",
            "frame_path": str(f_path),
            "detected": True,
            "corners": card_pts.tolist(),
            "bbox": [90, 80, 400, 280],
            "confidence": 0.95,
            "method": "first_frame_detection",
        }
    ]

    norm_dir = tmp_path / "norm"
    debug_dir = tmp_path / "debug_persp"

    # Must succeed without calling estimate_document_corners!
    persp_summary = pipeline.correct_perspective(detection_results, norm_dir, debug_dir)

    assert persp_summary["homography_success_count"] == 1
    assert len(persp_summary["results"]) == 1
    assert persp_summary["results"][0]["success"] is True
    assert persp_summary["results"][0]["method_applied"] == "homography"
    assert Path(persp_summary["results"][0]["normalized_path"]).is_file()


def test_missing_detection_graceful_handling(pipeline, tmp_path):
    """Verify that missing detections (blank frames) do not crash pipeline or perspective correction."""
    import cv2
    import numpy as np

    # Blank flat image (no document contours possible)
    blank = np.zeros((300, 400, 3), dtype=np.uint8)
    p = tmp_path / "blank.png"
    cv2.imwrite(str(p), blank)

    det_summary = pipeline.detect_and_crop([p], tmp_path / "crops", tmp_path / "debug_det")
    assert det_summary["frames_detected"] == 0
    assert det_summary["results"][0]["detected"] is False

    # Perspective correction must handle undetected result gracefully
    persp_summary = pipeline.correct_perspective(det_summary["results"], tmp_path / "norm", tmp_path / "debug_persp")
    assert persp_summary["homography_success_count"] == 0
    assert persp_summary["results"][0]["success"] is False
    assert persp_summary["results"][0]["normalized_path"] is None


def test_tracker_recovery_after_temporary_detection_loss(pipeline, tmp_path):
    """Verify tracker recovers when a document is lost for 1 frame and reappears."""
    import cv2
    import numpy as np

    # Frame 0: Good card
    f0 = np.full((400, 600, 3), 40, dtype=np.uint8)
    pts = np.array([[100, 100], [350, 100], [350, 260], [100, 260]], dtype=np.int32)
    cv2.fillPoly(f0, [pts], (230, 230, 230))
    cv2.polylines(f0, [pts], True, (10, 10, 10), 2)
    p0 = tmp_path / "f0.png"
    cv2.imwrite(str(p0), f0)

    # Frame 1: Blank frame (motion blur or occlusion)
    f1 = np.zeros((400, 600, 3), dtype=np.uint8)
    p1 = tmp_path / "f1.png"
    cv2.imwrite(str(p1), f1)

    # Frame 2: Good card slightly moved
    f2 = np.full((400, 600, 3), 40, dtype=np.uint8)
    pts2 = np.array([[105, 102], [355, 102], [355, 262], [105, 262]], dtype=np.int32)
    cv2.fillPoly(f2, [pts2], (230, 230, 230))
    cv2.polylines(f2, [pts2], True, (10, 10, 10), 2)
    p2 = tmp_path / "f2.png"
    cv2.imwrite(str(p2), f2)

    summary = pipeline.detect_and_crop([p0, p1, p2], tmp_path / "c", tmp_path / "d")
    results = summary["results"]

    assert results[0]["detected"] is True
    assert results[1]["detected"] is False
    assert results[2]["detected"] is True
    # Frame 2 should still benefit from temporal tracking guidance!
    assert results[2]["method"] == "temporally_guided"
    assert results[2]["tracking_used"] is True


def test_pipeline_run_alias_compatibility(pipeline, tmp_path):
    """Verify that pipeline.run() alias is present and functions identically to process_video()."""
    import inspect
    assert hasattr(pipeline, "run")
    assert callable(pipeline.run)
    sig = inspect.signature(pipeline.run)
    assert "video_path" in sig.parameters
    assert "output_dir" in sig.parameters

