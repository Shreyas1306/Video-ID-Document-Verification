"""
Frame Extractor Module
======================
Extracts and saves sequential video frames from local video files (e.g. MP4)
using OpenCV, preserving temporal ordering and generating comprehensive metadata.

Phase 1 Core Component:
- Validates video integrity and container readability.
- Extracts frames at a configurable sampling interval or uniform count.
- Saves frames with zero-padded sequential names to preserve ordering.
- Produces structured metadata (FPS, total frames, duration, extracted count, interval).
- Configurable via config.yaml or runtime parameters.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from src.utils.config_loader import load_config
from src.utils.video_utils import get_video_metadata


class FrameExtractor:
    """
    Extracts frames from video files with configurable sampling strategies
    and deterministic ordering.
    """

    def __init__(
        self,
        max_frames: Optional[int] = None,
        sampling_interval: Optional[int] = None,
        sampling_strategy: Optional[str] = None,
        output_format: Optional[str] = None,
        config_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the FrameExtractor using provided arguments or defaults from config.

        Args:
            max_frames: Maximum number of frames to extract (default from config or 10).
            sampling_interval: Step size between frames (e.g., 5 = extract every 5th frame).
            sampling_strategy: "uniform" (evenly distributed) or "interval" (fixed step).
            output_format: Image format for extracted frames ("png" or "jpg").
            config_path: Path to YAML config file.
        """
        try:
            cfg = load_config(config_path)
            frame_cfg = cfg.get("frame_extraction", {})
            paths_cfg = cfg.get("paths", {})
        except Exception:
            frame_cfg = {}
            paths_cfg = {}

        self.max_frames = max_frames if max_frames is not None else frame_cfg.get("max_frames", 10)
        self.sampling_interval = sampling_interval if sampling_interval is not None else frame_cfg.get("sampling_interval", 1)
        self.sampling_strategy = sampling_strategy if sampling_strategy is not None else frame_cfg.get("sampling_strategy", "uniform")
        self.output_format = (output_format if output_format is not None else frame_cfg.get("output_format", "png")).lstrip(".")
        self.default_output_dir = Path(paths_cfg.get("output_dir", "outputs")) / "frames"

    def calculate_frame_indices(self, total_frames: int) -> Tuple[List[int], int]:
        """
        Compute which zero-indexed frame positions to extract based on strategy.

        Args:
            total_frames: Total number of frames available in video.

        Returns:
            Tuple of (list_of_indices, effective_sampling_interval).
        """
        if total_frames <= 0:
            return [], 0

        if self.sampling_strategy == "interval":
            interval = max(1, int(self.sampling_interval))
            indices = list(range(0, total_frames, interval))
            if self.max_frames and len(indices) > self.max_frames:
                indices = indices[: self.max_frames]
            return indices, interval

        # Default strategy: "uniform"
        target_count = min(total_frames, max(1, int(self.max_frames)))
        if target_count == 1:
            return [0], total_frames

        # Evenly spaced integer positions across the video
        indices = np.linspace(0, total_frames - 1, num=target_count, dtype=int).tolist()
        # Remove duplicates if video has fewer frames than requested
        indices = sorted(list(set(indices)))

        # Effective average interval
        interval = max(1, int(round((total_frames - 1) / max(1, len(indices) - 1)))) if len(indices) > 1 else 1
        return indices, interval

    def extract_from_video(
        self,
        video_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        save_to_disk: bool = True,
    ) -> Dict[str, Any]:
        """
        Read video, sample frames, save them to output_dir, and return comprehensive metadata.

        Args:
            video_path: Path to the local video file.
            output_dir: Target directory to write extracted frames.
            save_to_disk: Whether to write frame images to disk.

        Returns:
            Dictionary with metadata and list of saved frame file paths:
                - video_path: string
                - filename: string
                - original_fps: float
                - total_frames: int
                - video_duration: float (seconds)
                - number_of_extracted_frames: int
                - sampling_interval: int
                - sampling_strategy: str
                - output_directory: str (if save_to_disk is True)
                - frame_paths: List[str]
                - frame_indices: List[int]

        Raises:
            FileNotFoundError: If the video does not exist.
            ValueError: If video is corrupt, empty, or unreadable.
        """
        target_video = Path(video_path).resolve()
        video_info = get_video_metadata(target_video)

        total_frames = video_info["total_frames"]
        fps = video_info["fps"]
        duration = video_info["duration_sec"]

        indices_to_extract, effective_interval = self.calculate_frame_indices(total_frames)
        if not indices_to_extract:
            raise ValueError(f"No frames could be scheduled for extraction from {target_video.name}")

        out_path: Optional[Path] = None
        if save_to_disk:
            out_path = Path(output_dir).resolve() if output_dir else (self.default_output_dir / target_video.stem).resolve()
            out_path.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(target_video))
        if not cap.isOpened():
            raise ValueError(f"Unable to read video stream from: {target_video}")

        extracted_frame_paths: List[str] = []
        extracted_indices: List[int] = []
        extracted_frames_mem: List[np.ndarray] = []

        try:
            current_frame_pos = 0
            index_set = set(indices_to_extract)
            max_target_index = max(indices_to_extract)

            # Sequential reading preserves exact video ordering and avoids seek bugs in variable-framerate streams
            while current_frame_pos <= max_target_index:
                ret, frame = cap.read()
                if not ret or frame is None:
                    # Premature EOF
                    break

                if current_frame_pos in index_set:
                    extracted_indices.append(current_frame_pos)
                    if save_to_disk and out_path is not None:
                        # Order-preserving zero-padded naming
                        frame_filename = f"frame_{len(extracted_indices):04d}_idx_{current_frame_pos:06d}.{self.output_format}"
                        frame_file_path = out_path / frame_filename
                        success = cv2.imwrite(str(frame_file_path), frame)
                        if not success:
                            raise IOError(f"Failed to write frame image to {frame_file_path}")
                        extracted_frame_paths.append(str(frame_file_path))
                    else:
                        extracted_frames_mem.append(frame)

                current_frame_pos += 1

        finally:
            cap.release()

        if not extracted_indices:
            raise ValueError(f"Failed to extract any readable frames from video: {target_video}")

        result_metadata: Dict[str, Any] = {
            "video_path": str(target_video),
            "filename": target_video.name,
            "original_fps": fps,
            "total_frames": total_frames,
            "video_duration": duration,
            "number_of_extracted_frames": len(extracted_indices),
            "sampling_interval": effective_interval,
            "sampling_strategy": self.sampling_strategy,
            "frame_indices": extracted_indices,
            "frame_paths": extracted_frame_paths,
            "output_directory": str(out_path) if out_path else None,
        }

        # Save metadata.json into the output directory for pipeline traceability
        if save_to_disk and out_path is not None:
            metadata_file = out_path / "metadata.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(result_metadata, f, indent=2)

        return result_metadata


def extract_frames(
    video_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    max_frames: Optional[int] = None,
    sampling_interval: Optional[int] = None,
    sampling_strategy: Optional[str] = None,
    config_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to extract frames from a video using the FrameExtractor.
    """
    extractor = FrameExtractor(
        max_frames=max_frames,
        sampling_interval=sampling_interval,
        sampling_strategy=sampling_strategy,
        config_path=config_path,
    )
    return extractor.extract_from_video(video_path=video_path, output_dir=output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract sequential frames from video.")
    parser.add_argument("--video", type=str, required=True, help="Path to local MP4 video.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save extracted frames.")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to extract.")
    parser.add_argument("--interval", type=int, default=None, help="Sampling interval (frame step).")
    parser.add_argument("--strategy", type=str, default="uniform", choices=["uniform", "interval"])
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML.")

    args = parser.parse_args()

    try:
        meta = extract_frames(
            video_path=args.video,
            output_dir=args.output_dir,
            max_frames=args.max_frames,
            sampling_interval=args.interval,
            sampling_strategy=args.strategy,
            config_path=args.config,
        )
        print("\n=== Frame Extraction Successful ===")
        print(f"Video:           {meta['filename']}")
        print(f"FPS:             {meta['original_fps']}")
        print(f"Total frames:    {meta['total_frames']}")
        print(f"Duration:        {meta['video_duration']} s")
        print(f"Extracted:       {meta['number_of_extracted_frames']} frames")
        print(f"Interval:        {meta['sampling_interval']}")
        print(f"Output folder:   {meta['output_directory']}")
        print("===================================\n")
    except Exception as exc:
        print(f"\n[ERROR] Frame extraction failed: {exc}\n")
        raise SystemExit(1)
