"""
Pipeline CLI Runner (Phase 9: End-to-End Execution)
===================================================
Executes the complete Video-Based Identity Document Verification pipeline:
  Video Input
  -> Frame Extraction
  -> Document Detection
  -> Document Cropping
  -> Perspective Correction
  -> Normalized Frames
  -> Visual Analysis
  -> Temporal Analysis
  -> OCR
  -> Text Verification
  -> Evidence Fusion
  -> Risk Assessment
  -> Verification Report

Usage:
  python scripts/run_pipeline.py --video data/sample/sample_document_video.mp4
  python scripts/run_pipeline.py --video data/sample/sample_document_video.mp4 --config config/settings.yaml
  python scripts/run_pipeline.py --video data/sample/sample_document_video.mp4 --output-dir outputs/my_run
"""

import argparse
import json
import logging
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import DocumentVerificationPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_pipeline")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Video-Based Identity Document Verification - End-to-End Pipeline CLI"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to the input video file (e.g. MP4).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        help="Path to YAML configuration file (default: config/settings.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base directory where pipeline outputs will be saved.",
    )

    args = parser.parse_args()
    video_path = Path(args.video)

    if not video_path.is_file():
        print(f"\n[ERROR] Video file does not exist: {video_path}\n", file=sys.stderr)
        return 1

    print("\n" + "=" * 70)
    print("      VIDEO-BASED IDENTITY DOCUMENT VERIFICATION PIPELINE       ")
    print("=" * 70)
    print(f"Target Video:    {video_path.resolve()}")
    print(f"Configuration:   {args.config}")
    print("Initiating 12-stage end-to-end verification...\n")

    try:
        pipeline = DocumentVerificationPipeline(
            config_path=args.config,
            base_output_dir=args.output_dir,
        )

        # Run complete pipeline
        result = pipeline.process_video(video_path=video_path)

        # Extract summary details for explicit reporting
        meta = result.get("video_metadata", {})
        det = result.get("evidence_details", {}).get("detection", {})
        persp = result.get("evidence_details", {}).get("perspective_correction", {})
        vis = result.get("evidence_details", {}).get("visual_integrity", {})
        temp = result.get("evidence_details", {}).get("temporal_consistency", {})
        ocr = result.get("evidence_details", {}).get("ocr_and_text", {})
        fusion = result.get("evidence_details", {}).get("evidence_fusion", {})
        verif = result.get("verification_result", {})
        timings = result.get("stage_timings_seconds", {})
        total_time = result.get("total_processing_time_seconds", 0.0)

        # Display structured summary
        print("\n" + "=" * 70)
        print("               PIPELINE EXECUTION SUMMARY & METRICS               ")
        print("=" * 70)
        print(f"1. Frames Processed:          {meta.get('number_of_extracted_frames', 0)} frames extracted from video")
        print(f"2. Detection Statistics:      {det.get('frames_detected', 0)}/{det.get('frames_processed', 0)} detected ({det.get('detection_rate', 0.0) * 100:.1f}%) [Model: {det.get('model')}]")
        print(f"   Perspective Correction:    {persp.get('homography_success_count', 0)} rectified via homography ({persp.get('homography_rate', 0.0) * 100:.1f}%)")

        print(f"3. OCR Extraction Result:     Name: '{ocr.get('extracted_fields', {}).get('name')}', Doc#: '{ocr.get('extracted_fields', {}).get('document_number')}'")
        print(f"   Character OCR Confidence:  {ocr.get('ocr_confidence') * 100:.2f}% (Engine: {ocr.get('engine')})")
        print(f"4. Visual Integrity Score:    {vis.get('aggregated_score', 0.0):.4f} (Evaluated on {vis.get('frames_evaluated', 0)} normalized frames)")
        print(f"5. Temporal Consistency Score:{temp.get('temporal_score', 0.0):.4f} (Mean consecutive sim: {temp.get('mean_consecutive_similarity', 0.0):.4f})")
        print(f"6. Text Consistency Score:    {ocr.get('text_consistency', 0.0):.4f} (Cross-frame field agreement)")
        print(f"7. Fused Integrity Score:     {fusion.get('fused_integrity_score', 0.0):.4f} (Evidence coverage: {fusion.get('evidence_coverage', 0.0) * 100:.1f}%)")
        print(f"8. Final Risk Classification: {verif.get('risk_level')} RISK")
        print(f"9. Total Processing Time:     {total_time:.3f} seconds")

        print("\nStage-by-Stage Processing Timings:")
        for st_name, st_time in timings.items():
            st_status = result.get("stage_execution_statuses", {}).get(st_name, {}).get("status", "SUCCESS")
            print(f"  - {st_name:<28}: {st_time:6.3f}s  [{st_status}]")
        print("=" * 70 + "\n")

        # Print structured JSON output
        print("=" * 70)
        print("             STRUCTURED VERIFICATION RESULT (JSON)                ")
        print("=" * 70)
        print(json.dumps(result["verification_result"], indent=2))
        print("=" * 70 + "\n")

        # Read and display human-readable report
        report_path = result.get("output_artifacts", {}).get("verification_report_txt")
        if report_path and Path(report_path).is_file():
            with open(report_path, "r", encoding="utf-8") as f:
                print(f.read())

        print(f"\n[SUCCESS] Verification complete! All artifacts saved to:")
        for art_name, art_file in result.get("output_artifacts", {}).items():
            print(f"  * {art_name:<30}: {art_file}")
        print()

        return 0

    except Exception as exc:
        print(f"\n[ERROR] Pipeline execution failed: {exc}\n", file=sys.stderr)
        logger.exception("Pipeline failed with unhandled exception")
        return 1


if __name__ == "__main__":
    sys.exit(main())
