"""
Tests for Temporal Consistency Analysis (Phase 6)
=================================================
Validates:
  - Visual embedding generation preserving temporal ordering.
  - Pairwise and consecutive cosine similarity metrics.
  - Consistent vs. deliberately inconsistent document sequences.
  - Insufficient-frame handling (< 2 frames).
  - Cross-video pollution protection.
  - Debug visualization rendering.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest

from src.visual.feature_extractor import DocumentFeatureExtractor
from src.visual.temporal_consistency import TemporalConsistencyAnalyzer


@pytest.fixture(scope="module")
def feature_extractor():
    """
    Shared feature extractor for test suite.
    Uses the synthetic baseline checkpoint to evaluate synthetic test fixtures,
    decoupling the unit test from changes in the active production model weights.
    """
    ckpt = Path("models/weights/best_integrity_model_synthetic.pth")
    if not ckpt.is_file():
        ckpt = Path("models/weights/best_integrity_model.pth")
    return DocumentFeatureExtractor(
        checkpoint_path=str(ckpt) if ckpt.is_file() else None,
        device="cpu",
    )


@pytest.fixture
def analyzer(feature_extractor):
    """Shared TemporalConsistencyAnalyzer instance."""
    return TemporalConsistencyAnalyzer(
        feature_extractor=feature_extractor,
        consistency_threshold=0.85,
    )


def test_feature_extractor_embedding_shapes(feature_extractor):
    """Verify single and batch feature embedding extraction."""
    img = np.full((224, 224, 3), 180, dtype=np.uint8)

    # 1. Single frame extraction
    feat = feature_extractor.extract_frame(img, normalize=True)
    assert feat.shape == (1280,), f"Expected (1280,), got {feat.shape}"
    norm = np.linalg.norm(feat)
    np.testing.assert_allclose(norm, 1.0, atol=1e-5)

    # 2. Batch extraction
    batch = [img, img, img]
    feats = feature_extractor.extract_batch(batch, normalize=True)
    assert feats.shape == (3, 1280)
    batch_norms = np.linalg.norm(feats, axis=1)
    np.testing.assert_allclose(batch_norms, [1.0, 1.0, 1.0], atol=1e-5)


def test_consistent_frame_sequence(analyzer):
    """
    Verify that a sequence of naturally varying frames of the SAME document
    produces high consecutive similarity and is flagged as consistent.
    """
    # Create base card
    base = np.full((300, 450, 3), 230, dtype=np.uint8)
    cv2.rectangle(base, (20, 20), (430, 70), (40, 70, 150), -1)
    cv2.putText(base, "TEST ID CARD", (50, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.rectangle(base, (30, 100), (130, 220), (100, 100, 100), -1)

    # Simulate minor natural handheld variations (subtle rotation and lighting)
    frames = []
    h, w = base.shape[:2]
    center = (w // 2, h // 2)

    for i in range(4):
        angle = (i - 1.5) * 0.8  # Slight tilt (-1.2 to +1.2 deg)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        t_frame = cv2.warpAffine(base, rot_mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
        t_frame = cv2.convertScaleAbs(t_frame, alpha=1.0 + (i - 1.5) * 0.02, beta=(i - 1.5) * 2)
        frames.append(t_frame)

    result = analyzer.analyze_sequence(frames=frames, video_ids=["video_01"] * 4)

    assert result["status"] == "success"
    assert result["num_frames"] == 4
    assert len(result["consecutive_similarities"]) == 3

    # Consecutive similarities should all be very high for identical card variations
    assert result["mean_consecutive_similarity"] >= 0.85
    assert result["temporal_consistency_score"] >= 0.85
    assert len(result["anomalous_transitions"]) == 0
    assert result["is_consistent"] is True


def test_deliberately_inconsistent_synthetic_sequence(analyzer):
    """
    Verify that an abrupt document alteration/swap in the sequence triggers
    an anomalous transition flag and lowers the consistency score.
    """
    # Document A
    doc_a = np.full((300, 450, 3), 240, dtype=np.uint8)
    cv2.putText(doc_a, "ORIGINAL IDENTITY", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Document B (completely different visual content & colors)
    doc_b = np.full((300, 450, 3), 30, dtype=np.uint8)
    cv2.putText(doc_b, "TAMPERED REPLACEMENT", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.circle(doc_b, (225, 150), 80, (0, 0, 255), -1)

    # Sequence: Doc A -> Doc A -> Doc B (tampered frame) -> Doc A
    frames = [doc_a, doc_a.copy(), doc_b, doc_a.copy()]
    frame_ids = [0, 1, 2, 3]

    result = analyzer.analyze_sequence(frames=frames, frame_ids=frame_ids)

    assert result["status"] == "success"
    assert result["num_frames"] == 4
    assert len(result["consecutive_similarities"]) == 3

    # Transitions to and from doc_b should be flagged as anomalous
    assert len(result["anomalous_transitions"]) > 0
    assert result["is_consistent"] is False

    # Check that transition 1 -> 2 (doc_a -> doc_b) was flagged
    flagged_pairs = [(a["frame_a_idx"], a["frame_b_idx"]) for a in result["anomalous_transitions"]]
    assert (1, 2) in flagged_pairs


def test_insufficient_frames_handling(analyzer):
    """Verify handling of empty or single-frame sequences."""
    # 0 frames
    res_zero = analyzer.analyze_sequence(frames=[])
    assert res_zero["status"] == "insufficient_frames"
    assert res_zero["temporal_consistency_score"] == 0.0
    assert res_zero["is_consistent"] is False

    # 1 frame
    single_frame = [np.full((100, 100, 3), 200, dtype=np.uint8)]
    res_one = analyzer.analyze_sequence(frames=single_frame)
    assert res_one["status"] == "insufficient_frames"
    assert res_one["temporal_consistency_score"] == 0.0
    assert res_one["is_consistent"] is False

    # Sequence with None and empty arrays
    corrupt_seq = [None, np.array([]), np.full((50, 50, 3), 100, dtype=np.uint8)]
    res_corrupt = analyzer.analyze_sequence(frames=corrupt_seq)
    assert res_corrupt["status"] == "insufficient_frames"


def test_cross_video_guard(analyzer):
    """Verify that mixing frames from different video IDs is rejected."""
    frames = [
        np.full((100, 100, 3), 200, dtype=np.uint8),
        np.full((100, 100, 3), 200, dtype=np.uint8),
    ]
    # Differing video IDs
    video_ids = ["video_doc_01", "video_doc_02"]

    result = analyzer.analyze_sequence(frames=frames, video_ids=video_ids)
    assert result["status"] == "video_id_mismatch"
    assert "Cross-video" in result["error"] or "Multiple video IDs" in result["error"]
    assert result["temporal_consistency_score"] == 0.0


def test_debug_visualization_generation(analyzer, tmp_path):
    """Verify debug visualization rendering to disk."""
    frame = np.full((100, 150, 3), 220, dtype=np.uint8)
    frames = [frame, frame.copy(), frame.copy()]
    debug_path = tmp_path / "temporal_debug.png"

    result = analyzer.analyze_sequence(
        frames=frames,
        debug_output_path=debug_path,
    )

    assert result["status"] == "success"
    assert debug_path.exists()
    assert debug_path.stat().st_size > 0
