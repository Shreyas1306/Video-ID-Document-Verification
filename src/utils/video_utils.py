"""
Video Utilities
===============
Helper functions for inspecting and handling video files using OpenCV.
"""

from pathlib import Path
from typing import Any, Dict, Union
import cv2


def get_video_metadata(video_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Extract technical metadata from a video file.

    Args:
        video_path: Path to the target video file.

    Returns:
        Dictionary containing:
            - path: string path to video
            - width: video width in pixels
            - height: video height in pixels
            - fps: frames per second
            - total_frames: total number of frames in video
            - duration_sec: duration in seconds
            - codec: fourcc codec code

    Raises:
        FileNotFoundError: If the video file does not exist.
        ValueError: If the file is empty or cannot be opened by OpenCV.
    """
    path_obj = Path(video_path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"Video file does not exist: {path_obj}")

    if path_obj.stat().st_size == 0:
        raise ValueError(f"Video file is empty (0 bytes): {path_obj}")

    cap = cv2.VideoCapture(str(path_obj))
    try:
        if not cap.isOpened():
            raise ValueError(
                f"Failed to open video file with OpenCV. The file may be corrupt, "
                f"unsupported, or missing a valid video container: {path_obj}"
            )

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

        if fps <= 0 or total_frames <= 0 or width <= 0 or height <= 0:
            raise ValueError(
                f"Invalid or corrupted video stream properties (fps={fps}, "
                f"total_frames={total_frames}, width={width}, height={height}): {path_obj}"
            )

        duration_sec = round(total_frames / fps, 3)

        return {
            "path": str(path_obj),
            "filename": path_obj.name,
            "width": width,
            "height": height,
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "duration_sec": duration_sec,
            "codec": fourcc_str,
        }
    finally:
        cap.release()
