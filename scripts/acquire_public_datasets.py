"""
Public Dataset Acquisition and Extraction Script
=================================================
Selectively downloads and extracts targeted subsets of:
1. DLC-2021 (8 independent document identities x 4 modes = 32 clips, first 50 frames each)
2. MIDV-2020 (6 independent document video clips + quad annotations + text ground truth)

Streams uncompressed TAR archives over official Smart Engines FTP directly into
target folders without downloading giant 17-88 GB archives to disk.

Output structure:
data/public/
├── dlc2021/
│   ├── annotations/
│   └── frames/
└── midv2020/
    ├── annotations/
    ├── frames/
    └── templates/
"""

import ftplib
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import tarfile
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("DatasetAcquisition")

FTP_HOST = "smartengines.com"

# Target DLC-2021 document identities (8 documents x 4 presentation types = 32 clips)
DLC_DOC_IDS = [
    "alb_id_00", "alb_id_01", "alb_id_02", "alb_id_03",
    "alb_id_04", "alb_id_05", "aze_passport_00", "aze_passport_01"
]
DLC_MODES = ["or0001", "cc0001", "cg0001", "re0001"]

# Build expected clip prefixes: e.g. "alb_id/00.or0001"
DLC_TARGET_CLIPS = set()
for doc_id in DLC_DOC_IDS:
    code, nxx = doc_id.rsplit("_", 1)
    for mode in DLC_MODES:
        DLC_TARGET_CLIPS.add(f"{code}/{nxx}.{mode}")

# Target MIDV-2020 document clips (Latvian Passport - Latin script)
MIDV_DOC_CODE = "lva_passport"
MIDV_TARGET_CLIPS = ["28", "30", "33", "43", "69", "92"]


def acquire_dlc2021(base_dir: Path) -> dict:
    """Stream and extract DLC-2021 targeted subset from clips.tar on FTP."""
    dlc_dir = base_dir / "dlc2021"
    ann_dir = dlc_dir / "annotations"
    frames_dir = dlc_dir / "frames"
    ann_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Check if DLC-2021 is already fully acquired
    existing_anns = list(ann_dir.glob("*.json"))
    existing_clips = [d for d in frames_dir.iterdir() if d.is_dir() and len(list(d.glob("*.jpg"))) >= 50]
    csv_file = dlc_dir / "dlc-2021.csv"
    if len(existing_anns) >= len(DLC_TARGET_CLIPS) and len(existing_clips) >= len(DLC_TARGET_CLIPS) and csv_file.exists():
        total_frames = sum(len(list(d.glob("*.jpg"))) for d in existing_clips)
        logger.info("DLC-2021 subset already acquired (%d clips, %d annotations, %d frames). Skipping re-download.",
                    len(existing_clips), len(existing_anns), total_frames)
        return {
            "annotations_extracted": len(existing_anns),
            "frames_extracted": total_frames,
            "clips_completed": len(existing_clips),
            "target_clips": list(sorted(DLC_TARGET_CLIPS))
        }

    logger.info("Connecting to FTP %s for DLC-2021 acquisition...", FTP_HOST)
    ftp = ftplib.FTP(FTP_HOST, timeout=30)
    ftp.login()
    ftp.cwd("dlc-2021")

    # Ensure dlc-2021.csv is downloaded
    if not csv_file.exists():
        with open(csv_file, "wb") as f:
            ftp.retrbinary("RETR dlc-2021.csv", f.write)
        logger.info("Saved dlc-2021.csv (%d bytes)", csv_file.stat().st_size)

    # Stream clips.tar
    logger.info("Streaming clips.tar for %d targeted clips...", len(DLC_TARGET_CLIPS))
    sock = ftp.transfercmd("RETR clips.tar")
    sock_file = sock.makefile("rb")
    tar = tarfile.open(mode="r|", fileobj=sock_file)

    extracted_annotations = 0
    extracted_frames = 0
    clips_completed = set()

    t0 = time.time()
    for member in tar:
        if not member.isfile():
            continue

        name = member.name  # e.g. clips/annotations/alb_id/00.cc0001.json or clips/images/alb_id/00.cc0001/000001.jpg

        # Check for annotations
        if "clips/annotations/" in name:
            parts = name.replace("clips/annotations/", "").split("/")
            if len(parts) == 2:
                code, json_name = parts
                clip_prefix = f"{code}/{json_name.replace('.json', '')}"
                if clip_prefix in DLC_TARGET_CLIPS:
                    target_file = ann_dir / f"{code}_{json_name}"
                    if not target_file.exists():
                        f_in = tar.extractfile(member)
                        with open(target_file, "wb") as f_out:
                            f_out.write(f_in.read())
                        extracted_annotations += 1

        # Check for frame images
        elif "clips/images/" in name:
            parts = name.replace("clips/images/", "").split("/")
            if len(parts) == 3:
                code, clip_id, frame_name = parts
                clip_prefix = f"{code}/{clip_id}"
                if clip_prefix in DLC_TARGET_CLIPS:
                    clip_out_dir = frames_dir / f"{code}_{clip_id}"
                    clip_out_dir.mkdir(parents=True, exist_ok=True)
                    target_file = clip_out_dir / frame_name
                    
                    # Only retain up to 50 frames per clip
                    existing = len(list(clip_out_dir.glob("*.jpg")))
                    if existing < 50 and not target_file.exists():
                        f_in = tar.extractfile(member)
                        with open(target_file, "wb") as f_out:
                            f_out.write(f_in.read())
                        extracted_frames += 1
                        if existing + 1 >= 50:
                            clips_completed.add(clip_prefix)
                            logger.info("Clip %s completed (50 frames extracted). Total completed: %d/%d",
                                        clip_prefix, len(clips_completed), len(DLC_TARGET_CLIPS))

        # Termination condition: if all 32 clips reached 50 frames
        if len(clips_completed) >= len(DLC_TARGET_CLIPS):
            logger.info("All 32 DLC-2021 target clips fully extracted!")
            break

    sock_file.close()
    sock.close()
    try:
        ftp.close()
    except Exception:
        pass

    logger.info("DLC-2021 extraction finished in %.1f s. Annotations: %d, Frames: %d, Clips: %d",
                time.time() - t0, extracted_annotations, extracted_frames, len(clips_completed))

    return {
        "annotations_extracted": extracted_annotations,
        "frames_extracted": extracted_frames,
        "clips_completed": len(clips_completed),
        "target_clips": list(sorted(DLC_TARGET_CLIPS))
    }


def acquire_midv2020(base_dir: Path) -> dict:
    """Stream and extract MIDV-2020 validation subset from FTP."""
    midv_dir = base_dir / "midv2020"
    ann_dir = midv_dir / "annotations"
    frames_dir = midv_dir / "frames"
    tmpl_dir = midv_dir / "templates"
    ann_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    tmpl_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract template ground-truth text for lva_passport from templates.tar
    tmpl_file = tmpl_dir / f"{MIDV_DOC_CODE}.json"
    if not tmpl_file.exists():
        logger.info("Connecting to FTP %s for templates.tar...", FTP_HOST)
        ftp = ftplib.FTP(FTP_HOST, timeout=30)
        ftp.login()
        ftp.cwd("midv-2020/dataset")
        logger.info("Streaming template annotation %s.json from templates.tar...", MIDV_DOC_CODE)
        sock = ftp.transfercmd("RETR templates.tar")
        sock_file = sock.makefile("rb")
        tar = tarfile.open(mode="r|", fileobj=sock_file)
        for member in tar:
            if member.name == f"annotations/{MIDV_DOC_CODE}.json":
                f_in = tar.extractfile(member)
                with open(tmpl_file, "wb") as f_out:
                    f_out.write(f_in.read())
                logger.info("Extracted %s (%d bytes)", tmpl_file.name, tmpl_file.stat().st_size)
                break
        sock_file.close()
        sock.close()
        try:
            ftp.quit()
        except Exception:
            pass

    # Step 2: Stream clips.tar for annotations and frames of targeted lva_passport clips using a fresh FTP session
    logger.info("Connecting fresh FTP session for MIDV-2020 clips.tar...")
    ftp = ftplib.FTP(FTP_HOST, timeout=60)
    ftp.login()
    ftp.cwd("midv-2020/dataset")
    logger.info("Streaming clips.tar for MIDV-2020 %s target clips: %s...", MIDV_DOC_CODE, MIDV_TARGET_CLIPS)
    sock = ftp.transfercmd("RETR clips.tar")
    sock_file = sock.makefile("rb")
    tar = tarfile.open(mode="r|", fileobj=sock_file)

    extracted_annotations = 0
    extracted_frames = 0
    clips_completed = set()

    t0 = time.time()
    for member in tar:
        if not member.isfile():
            continue

        name = member.name

        # Check for annotations: annotations/lva_passport/28.json
        if name.startswith(f"annotations/{MIDV_DOC_CODE}/"):
            clip_id = name.split("/")[-1].replace(".json", "")
            if clip_id in MIDV_TARGET_CLIPS:
                target_file = ann_dir / f"{MIDV_DOC_CODE}_{clip_id}.json"
                if not target_file.exists():
                    f_in = tar.extractfile(member)
                    with open(target_file, "wb") as f_out:
                        f_out.write(f_in.read())
                    extracted_annotations += 1

        # Check for frame images: images/lva_passport/28/000001.jpg
        elif name.startswith(f"images/{MIDV_DOC_CODE}/"):
            parts = name.split("/")
            if len(parts) == 4:
                clip_id, frame_name = parts[2], parts[3]
                if clip_id in MIDV_TARGET_CLIPS:
                    clip_out_dir = frames_dir / f"{MIDV_DOC_CODE}_{clip_id}"
                    clip_out_dir.mkdir(parents=True, exist_ok=True)
                    target_file = clip_out_dir / frame_name

                    existing = len(list(clip_out_dir.glob("*.jpg")))
                    if existing < 50 and not target_file.exists():
                        f_in = tar.extractfile(member)
                        with open(target_file, "wb") as f_out:
                            f_out.write(f_in.read())
                        extracted_frames += 1
                        if existing + 1 >= 30:  # At least 30 frames per clip
                            clips_completed.add(clip_id)
                            logger.info("MIDV clip %s reached %d frames. Completed: %d/%d",
                                        clip_id, existing + 1, len(clips_completed), len(MIDV_TARGET_CLIPS))

        if len(clips_completed) >= len(MIDV_TARGET_CLIPS):
            logger.info("All MIDV-2020 target clips reached sufficient frames!")
            break

    sock_file.close()
    sock.close()
    try:
        ftp.close()
    except Exception:
        pass

    logger.info("MIDV-2020 extraction finished in %.1f s. Annotations: %d, Frames: %d, Clips: %d",
                time.time() - t0, extracted_annotations, extracted_frames, len(clips_completed))

    return {
        "annotations_extracted": extracted_annotations,
        "frames_extracted": extracted_frames,
        "clips_completed": len(clips_completed),
        "target_clips": MIDV_TARGET_CLIPS
    }


def main():
    base_data = Path("data/public")
    base_data.mkdir(parents=True, exist_ok=True)

    logger.info("=== Starting Public Dataset Selective Acquisition ===")
    dlc_res = acquire_dlc2021(base_data)
    midv_res = acquire_midv2020(base_data)

    logger.info("=== Acquisition Completed Successfully ===")
    logger.info("DLC-2021: %s", dlc_res)
    logger.info("MIDV-2020: %s", midv_res)


if __name__ == "__main__":
    main()
