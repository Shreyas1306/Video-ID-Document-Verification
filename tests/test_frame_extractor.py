"""
Unit and Integration Tests for Frame Extraction (Phase 1)
=========================================================
Tests:
- Invalid, missing, and empty video handling with clear error messages
- Generation and decoding of synthetic video
- Sampling interval and max_frames adherence
- Preservation of frame ordering (strictly monotonic frame index sequence)
- Verifying generated frame files on disk and valid image dimensions
- Metadata generation (FPS, total_frames, duration, extracted_count, interval)
"""

import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from src.preprocessing.frame_extractor import FrameExtractor, extract_frames


@pytest.fixture
def sample_video_path(tmp_path: Path) -> Path:
    """Create a temporary synthetic 30-frame MP4 video for deterministic testing."""
    video_file = tmp_path / "test_synth_video.mp4"
    fps = 10.0
    width, height = 320, 240
    total_frames = 30

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_file), fourcc, fps, (width, height))

    try:
        for i in range(total_frames):
            # Create synthetic frames with distinct color and frame number text
            frame = np.full((height, width, 3), (i * 7) % 256, dtype=np.uint8)
            cv2.putText(
                frame,
                f"Frame {i:02d}",
                (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )
            writer.write(frame)
    finally:
        writer.release()

    assert video_file.exists() and video_file.stat().st_size > 0
    return video_file


class TestFrameExtractorErrorHandling:
    """Test graceful handling of invalid, missing, or corrupted videos."""

    def test_missing_video_raises_filenotfound(self, tmp_path: Path):
        extractor = FrameExtractor()
        fake_path = tmp_path / "non_existent_video.mp4"
        with pytest.raises(FileNotFoundError, match="does not exist"):
            extractor.extract_from_video(fake_path)

    def test_empty_video_raises_valueerror(self, tmp_path: Path):
        extractor = FrameExtractor()
        empty_file = tmp_path / "empty_video.mp4"
        empty_file.touch()  # 0 bytes
        with pytest.raises(ValueError, match="is empty"):
            extractor.extract_from_video(empty_file)

    def test_corrupted_video_raises_valueerror(self, tmp_path: Path):
        extractor = FrameExtractor()
        corrupt_file = tmp_path / "corrupt.mp4"
        corrupt_file.write_bytes(b"This is not a real mp4 video container header")
        with pytest.raises(ValueError, match="Failed to open|Invalid or corrupted"):
            extractor.extract_from_video(corrupt_file)


class TestFrameExtractorFunctionality:
    """Test valid video extraction, ordering, sampling, and metadata."""

    def test_uniform_sampling(self, sample_video_path: Path, tmp_path: Path):
        out_dir = tmp_path / "out_uniform"
        extractor = FrameExtractor(max_frames=6, sampling_strategy="uniform")
        meta = extractor.extract_from_video(sample_video_path, output_dir=out_dir)

        # Verify metadata
        assert meta["total_frames"] == 30
        assert meta["original_fps"] == 10.0
        assert meta["video_duration"] == 3.0
        assert meta["number_of_extracted_frames"] == 6
        assert meta["sampling_interval"] > 0
        assert meta["sampling_strategy"] == "uniform"

        # Verify frame indices ordering is strictly increasing
        indices = meta["frame_indices"]
        assert len(indices) == 6
        assert indices == sorted(indices)
        assert len(set(indices)) == len(indices)  # No duplicates

        # Verify output files exist and are readable
        frame_paths = meta["frame_paths"]
        assert len(frame_paths) == 6
        for p in frame_paths:
            path_obj = Path(p)
            assert path_obj.is_file()
            img = cv2.imread(str(path_obj))
            assert img is not None
            assert img.shape == (240, 320, 3)

        # Verify metadata.json was saved in output directory
        meta_json_path = out_dir / "metadata.json"
        assert meta_json_path.is_file()
        with open(meta_json_path, "r", encoding="utf-8") as f:
            saved_meta = json.load(f)
        assert saved_meta["number_of_extracted_frames"] == 6

    def test_interval_sampling(self, sample_video_path: Path, tmp_path: Path):
        out_dir = tmp_path / "out_interval"
        extractor = FrameExtractor(sampling_interval=5, sampling_strategy="interval")
        meta = extractor.extract_from_video(sample_video_path, output_dir=out_dir)

        # 30 frames with interval 5 -> frames at 0, 5, 10, 15, 20, 25 (6 frames)
        assert meta["number_of_extracted_frames"] == 6
        assert meta["frame_indices"] == [0, 5, 10, 15, 20, 25]

        # Verify zero-padded naming preserves sorting
        frame_names = [Path(p).name for p in meta["frame_paths"]]
        assert frame_names == sorted(frame_names)

    def test_convenience_extract_frames_function(self, sample_video_path: Path, tmp_path: Path):
        out_dir = tmp_path / "out_convenience"
        meta = extract_frames(sample_video_path, output_dir=out_dir, max_frames=3)
        assert meta["number_of_extracted_frames"] == 3
        assert len(meta["frame_paths"]) == 3
