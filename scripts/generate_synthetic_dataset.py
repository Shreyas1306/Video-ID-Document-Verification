"""
Synthetic Document Dataset Generator
====================================
Generates a controlled benchmark of mock identity documents (REAL vs ATTACKED)
with multiple frames per document ID to simulate smartphone video captures.

Card format:
  - Base dimensions: 600 x 400 (normalized identity card format)
  - Realistic card elements: Header banner, photo portrait, chip graphic,
    text fields (NAME, DOB, DOC NUMBER, EXPIRY).
  - REAL frames: Natural document variations across frames (subtle perspective,
    lighting variations, slight handheld rotation).
  - ATTACKED frames:
      * Photo tampering: cut-and-paste overlay, mismatched resolution boundary.
      * Text alteration: spliced text with digital artifact patches.
      * Screen replay / Print attack: moire patterns, specular screen reflections/glare.
      * Digital tampering: boundary anomalies and inconsistent texture.
"""

from pathlib import Path
import random
from typing import List, Tuple

import cv2
import numpy as np

# Random seed for reproducibility
random.seed(42)
np.random.seed(42)

FIRST_NAMES = ["JAMES", "EMMA", "LIAM", "OLIVIA", "NOAH", "AVA", "ETHAN", "SOPHIA", "LUCAS", "MIA", "ALEX", "PRIYA", "RAHUL", "ANITA", "DAVID"]
LAST_NAMES = ["SMITH", "JOHNSON", "WILLIAMS", "BROWN", "JONES", "GARCIA", "MILLER", "DAVIS", "RODRIGUEZ", "PATEL", "SHARMA", "KUMAR"]
CITIES = ["NEW YORK, NY", "CHICAGO, IL", "MUMBAI, MH", "LONDON, UK", "TORONTO, ON", "SAN FRANCISCO, CA"]


def create_base_real_card(doc_idx: int) -> Tuple[np.ndarray, dict]:
    """Create a pristine normalized mock ID card."""
    width, height = 600, 400
    card = np.full((height, width, 3), 245, dtype=np.uint8)

    # 1. Subtle card background gradient & micro-pattern
    for y in range(height):
        grad = int(245 - 20 * (y / height))
        card[y, :] = (grad, grad + 2, grad + 5)

    # Draw micro-texture grid
    for y in range(10, height, 15):
        cv2.line(card, (10, y), (width - 10, y), (230, 232, 235), 1)

    # 2. Header banner
    cv2.rectangle(card, (15, 15), (width - 15, 70), (45, 85, 150), -1)
    cv2.putText(card, "NATIONAL IDENTITY CARD", (35, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # 3. Microchip graphic
    chip_x, chip_y, chip_w, chip_h = 45, 110, 65, 50
    cv2.rectangle(card, (chip_x, chip_y), (chip_x + chip_w, chip_y + chip_h), (80, 180, 210), -1)
    cv2.rectangle(card, (chip_x, chip_y), (chip_x + chip_w, chip_y + chip_h), (40, 100, 120), 2)
    cv2.line(card, (chip_x + 20, chip_y), (chip_x + 20, chip_y + chip_h), (40, 100, 120), 1)
    cv2.line(card, (chip_x + 45, chip_y), (chip_x + 45, chip_y + chip_h), (40, 100, 120), 1)

    # 4. Portrait photo box
    photo_x, photo_y, photo_w, photo_h = 45, 190, 130, 160
    cv2.rectangle(card, (photo_x, photo_y), (photo_x + photo_w, photo_y + photo_h), (210, 210, 210), -1)
    cv2.rectangle(card, (photo_x, photo_y), (photo_x + photo_w, photo_y + photo_h), (120, 120, 120), 2)

    # Draw mock human silhouette portrait
    head_center = (photo_x + photo_w // 2, photo_y + 60)
    head_color = (130, 140, 160)
    cv2.circle(card, head_center, 30, head_color, -1)
    cv2.ellipse(card, (photo_x + photo_w // 2, photo_y + 145), (48, 55), 0, 180, 360, (100, 110, 130), -1)

    # 5. Metadata fields
    name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    dob = f"{random.randint(1975, 2002)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
    doc_num = f"ID-{doc_idx:03d}-{random.randint(1000, 9999)}"
    address = random.choice(CITIES)

    metadata = {"name": name, "dob": dob, "doc_number": doc_num, "address": address}

    fields = [
        ("NAME", name),
        ("DOB", dob),
        ("DOC NO", doc_num),
        ("ADDRESS", address),
    ]

    field_start_x = 210
    start_y = 135
    for label, val in fields:
        cv2.putText(card, label, (field_start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
        cv2.putText(card, val, (field_start_x, start_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)
        start_y += 55

    # 6. Card border
    cv2.rectangle(card, (2, 2), (width - 3, height - 3), (180, 180, 180), 3)

    return card, metadata


def simulate_handheld_frame_variation(card: np.ndarray, frame_idx: int) -> np.ndarray:
    """Apply realistic natural frame-to-frame variations (tilt, lighting)."""
    h, w = card.shape[:2]

    # Subtle rotation (-3 to +3 degrees)
    angle = (frame_idx - 2) * 1.2
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    transformed = cv2.warpAffine(card, rot_mat, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # Subtle brightness / contrast shift
    alpha = 1.0 + (frame_idx - 2) * 0.03  # Contrast
    beta = (frame_idx - 2) * 4            # Brightness
    adjusted = cv2.convertScaleAbs(transformed, alpha=alpha, beta=beta)

    return adjusted


def apply_attack(card: np.ndarray, attack_type: str) -> np.ndarray:
    """Apply a controlled realistic forgery or presentation attack."""
    tampered = card.copy()
    h, w = tampered.shape[:2]

    if attack_type == "photo_tamper":
        # Spliced photo: paste an altered box over portrait with visible edge artifact
        px, py, pw, ph = 45, 190, 130, 160
        # Draw fake replacement head with obvious digital edge mismatch
        cv2.rectangle(tampered, (px + 4, py + 4), (px + pw - 4, py + ph - 4), (170, 185, 200), -1)
        cv2.circle(tampered, (px + pw // 2, py + 65), 32, (60, 70, 90), -1)
        cv2.ellipse(tampered, (px + pw // 2, py + 150), (45, 50), 0, 180, 360, (50, 60, 80), -1)
        # Splicing artifact line
        cv2.rectangle(tampered, (px + 4, py + 4), (px + pw - 4, py + ph - 4), (255, 50, 50), 1)

    elif attack_type == "text_alteration":
        # Digital alteration: whiteout box over document number with mismatched font and compression noise
        bx, by, bw, bh = 205, 235, 220, 35
        cv2.rectangle(tampered, (bx, by), (bx + bw, by + bh), (255, 255, 255), -1)
        cv2.putText(tampered, "ID-999-XXXX-FORGED", (bx + 5, by + 24), cv2.FONT_HERSHEY_DUPLEX, 0.6, (10, 10, 150), 2)
        # Add high-frequency noise around altered text patch
        noise = np.random.normal(0, 25, (bh, bw, 3)).astype(np.int16)
        patch = np.clip(tampered[by:by + bh, bx:bx + bw].astype(np.int16) + noise, 0, 255).astype(np.uint8)
        tampered[by:by + bh, bx:bx + bw] = patch

    elif attack_type == "screen_replay":
        # Screen replay: severe specular glare and moiré interference lines
        # 1. Specular glare blob
        glare_center = (w // 2 + random.randint(-50, 50), h // 2 + random.randint(-40, 40))
        cv2.circle(tampered, glare_center, 90, (255, 255, 255), -1)
        tampered = cv2.addWeighted(card, 0.65, tampered, 0.35, 0)
        # 2. Moiré horizontal scanlines
        for y in range(0, h, 4):
            tampered[y, :, :] = np.clip(tampered[y, :, :].astype(np.int16) - 45, 0, 255).astype(np.uint8)

    elif attack_type == "print_scan":
        # Print & scan attack: severe color desaturation, halftone noise, slight blur
        gray = cv2.cvtColor(tampered, cv2.COLOR_BGR2GRAY)
        desat = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        tampered = cv2.addWeighted(tampered, 0.3, desat, 0.7, 0)
        # Halftone noise
        noise = np.random.normal(0, 18, tampered.shape).astype(np.int16)
        tampered = np.clip(tampered.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        tampered = cv2.GaussianBlur(tampered, (3, 3), 0.8)

    return tampered


def generate_benchmark_dataset(
    output_base_dir: str = "data/synthetic",
    num_real_docs: int = 25,
    num_attacked_docs: int = 25,
    frames_per_doc: int = 5,
) -> dict:
    """Generate balanced synthetic benchmark dataset."""
    base_path = Path(output_base_dir)
    real_dir = base_path / "REAL"
    attacked_dir = base_path / "ATTACKED"

    real_dir.mkdir(parents=True, exist_ok=True)
    attacked_dir.mkdir(parents=True, exist_ok=True)

    attack_types = ["photo_tamper", "text_alteration", "screen_replay", "print_scan"]
    stats = {"real_frames": 0, "attacked_frames": 0, "real_docs": num_real_docs, "attacked_docs": num_attacked_docs}

    # 1. Generate REAL documents
    print(f"Generating {num_real_docs} REAL documents ({frames_per_doc} frames each)...")
    for doc_idx in range(1, num_real_docs + 1):
        doc_id = f"doc_real_{doc_idx:03d}"
        base_card, _ = create_base_real_card(doc_idx)

        for frame_idx in range(1, frames_per_doc + 1):
            frame = simulate_handheld_frame_variation(base_card, frame_idx)
            frame_path = real_dir / f"{doc_id}_f{frame_idx:02d}.png"
            cv2.imwrite(str(frame_path), frame)
            stats["real_frames"] += 1

    # 2. Generate ATTACKED documents
    print(f"Generating {num_attacked_docs} ATTACKED documents ({frames_per_doc} frames each)...")
    for doc_idx in range(1, num_attacked_docs + 1):
        doc_id = f"doc_att_{doc_idx:03d}"
        base_card, _ = create_base_real_card(doc_idx + 100)
        attack = attack_types[(doc_idx - 1) % len(attack_types)]

        # Apply base attack to document
        attacked_card = apply_attack(base_card, attack)

        for frame_idx in range(1, frames_per_doc + 1):
            frame = simulate_handheld_frame_variation(attacked_card, frame_idx)
            frame_path = attacked_dir / f"{doc_id}_f{frame_idx:02d}.png"
            cv2.imwrite(str(frame_path), frame)
            stats["attacked_frames"] += 1

    print("Dataset generation complete!")
    print(f"  - REAL: {stats['real_frames']} frames ({num_real_docs} unique docs)")
    print(f"  - ATTACKED: {stats['attacked_frames']} frames ({num_attacked_docs} unique docs)")
    print(f"  - Total: {stats['real_frames'] + stats['attacked_frames']} images in '{output_base_dir}'")
    return stats


if __name__ == "__main__":
    generate_benchmark_dataset()
