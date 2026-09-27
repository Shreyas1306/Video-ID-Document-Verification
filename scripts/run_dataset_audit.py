"""
Dataset Audit and Preparation CLI Entry Point
=============================================
Runs project-wide dataset audit, prepares partitioned visual dataset,
and generates audit reports in outputs/dataset_audit/.
"""

import argparse
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visual.dataset_auditor import run_full_dataset_audit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="Run Dataset Audit & Visual Integrity Preparation")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/dataset_audit",
        help="Directory to save audit reports and charts",
    )
    parser.add_argument(
        "--visual-dataset-dir",
        type=str,
        default="data/visual_dataset",
        help="Directory to save prepared visual dataset and manifest.csv",
    )
    parser.add_argument(
        "--no-prepare",
        action="store_true",
        help="Run audit only without preparing visual_dataset partitions",
    )
    args = parser.parse_args()

    run_full_dataset_audit(
        project_root=PROJECT_ROOT,
        output_dir=args.output_dir,
        prepare_dataset=not args.no_prepare,
        visual_dataset_dir=args.visual_dataset_dir,
    )


if __name__ == "__main__":
    main()
