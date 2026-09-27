"""
Synthetic Sample Video Generator
=================================
Generates a realistic mock identity document video simulating a smartphone
camera recording a document tilting and moving slightly.

Used for testing video reading, frame extraction, and baseline pipeline workflows
without using any real personal identity data.
"""

from pathlib import Path
import cv2
import numpy as np


def generate_sample_document_video(
    output_path: Path,
    num_frames: int = 60,
    fps: float = 15.0,
    resolution: tuple = (640, 480),
) -> Path:
    """
    Synthesize an MP4 video of a moving mock identity document.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = resolution
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Base document dimensions (proportional to standard card aspect ratio ~ 1.58)
    doc_w, doc_h = 380, 240

    try:
        for i in range(num_frames):
            # Dynamic background (wooden table / desk texture)
            frame = np.full((height, width, 3), (210, 220, 230), dtype=np.uint8)

            # Add subtle motion / oscillation to simulate hand movement
            dx = int(18 * np.sin(i * 0.1))
            dy = int(12 * np.cos(i * 0.08))
            angle = 3.5 * np.sin(i * 0.12)  # subtle tilt
            pitch = 0.04 * np.cos(i * 0.15)  # subtle perspective pitch
            yaw = 0.03 * np.sin(i * 0.11)    # subtle perspective yaw

            cx = width // 2 + dx
            cy = height // 2 + dy

            # Card corners in local coordinate system with perspective variation
            pts = np.array(
                [
                    [-doc_w / 2 * (1.0 - yaw), -doc_h / 2 * (1.0 - pitch)],
                    [doc_w / 2 * (1.0 + yaw), -doc_h / 2 * (1.0 - pitch)],
                    [doc_w / 2 * (1.0 + yaw), doc_h / 2 * (1.0 + pitch)],
                    [-doc_w / 2 * (1.0 - yaw), doc_h / 2 * (1.0 + pitch)],
                ],
                dtype=np.float32,
            )

            # Rotation matrix for slight tilt
            rad = np.radians(angle)
            rot_mat = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
            rot_pts = (pts @ rot_mat.T) + np.array([cx, cy])
            int_pts = rot_pts.astype(np.int32)

            # Draw card drop shadow
            shadow_offset = np.array([8, 8])
            cv2.fillPoly(frame, [int_pts + shadow_offset], (160, 170, 180))

            # Draw card body (off-white)
            cv2.fillPoly(frame, [int_pts], (250, 250, 252))
            cv2.polylines(frame, [int_pts], isClosed=True, color=(70, 70, 70), thickness=2)

            # Draw document content on a level card patch, then warp to current tilt
            card_canvas = np.full((doc_h, doc_w, 3), (252, 252, 254), dtype=np.uint8)
            # Header banner
            cv2.rectangle(card_canvas, (0, 0), (doc_w, 40), (45, 90, 170), -1)
            cv2.putText(card_canvas, "MOCK IDENTITY CARD", (60, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            # Photo placeholder box
            cv2.rectangle(card_canvas, (20, 55), (105, 165), (200, 200, 205), -1)
            cv2.rectangle(card_canvas, (20, 55), (105, 165), (120, 120, 130), 1)
            cv2.putText(card_canvas, "PHOTO", (32, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 110), 1)

            # Mock fields
            cv2.putText(card_canvas, "NAME: SAMPLE CITIZEN", (120, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
            cv2.putText(card_canvas, "DOB: 12/04/1998", (120, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
            cv2.putText(card_canvas, "DOC NO: ID-4421-8890", (120, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
            cv2.putText(card_canvas, "ADDR: 42 ACADEMIC WAY", (120, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)

            # Footer watermark
            cv2.putText(card_canvas, "SYNTHETIC RESEARCH PROTOTYPE", (60, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 160), 1)

            # Homography warp card_canvas into the tilted quadrilateral
            rot_pts_f32 = rot_pts.astype(np.float32)
            src_corners = np.array([[0, 0], [doc_w, 0], [doc_w, doc_h], [0, doc_h]], dtype=np.float32)
            H = cv2.getPerspectiveTransform(src_corners, rot_pts_f32)
            warped_card = cv2.warpPerspective(card_canvas, H, (width, height))

            # Mask warped card into the main frame
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [int_pts], 255)
            frame[mask > 0] = warped_card[mask > 0]

            # Write frame
            writer.write(frame)
    finally:
        writer.release()

    return output_path


if __name__ == "__main__":
    out_file = Path("data/sample/sample_document_video.mp4")
    created = generate_sample_document_video(out_file)
    print(f"Sample video created: {created} (Size: {created.stat().st_size} bytes)")
