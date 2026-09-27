"""
Visual Document Integrity Training Script (Phase 5)
===================================================
Orchestrates:
  1. Dataset inspection and validation.
  2. Document/video-level leak-free train/val/test splitting.
  3. Pretrained EfficientNet-B0 baseline with binary classification head.
  4. Training, validation, and best checkpoint saving.
  5. Test set evaluation: Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix.
  6. Saving structured metrics to outputs/metrics/.
"""

import argparse
import logging
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from src.utils.config_loader import load_config
from src.visual.dataset import (
    DocumentDataset,
    collect_dataset_samples,
    create_leak_free_splits,
    get_document_transforms,
    validate_dataset_structure,
)
from src.visual.trainer import IntegrityTrainer
from src.visual.visual_integrity import DocumentIntegrityClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_visual_integrity")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Visual Document Integrity Classifier")
    parser.add_argument("--config", type=str, default="config/settings.yaml", help="Path to config/settings.yaml")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Path to dataset root (containing REAL/ and ATTACKED/)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--validate-only", action="store_true", help="Validate dataset structure and exit without training")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    train_cfg = config.get("visual_integrity_training", {})
    dataset_dir = args.dataset_dir or train_cfg.get("dataset_dir", "data/synthetic")
    real_subdir = train_cfg.get("real_subdir", "REAL")
    attacked_subdir = train_cfg.get("attacked_subdir", "ATTACKED")
    image_size = tuple(train_cfg.get("image_size", [224, 224]))
    batch_size = args.batch_size or train_cfg.get("batch_size", 16)
    num_workers = train_cfg.get("num_workers", 0)
    lr = args.lr or train_cfg.get("learning_rate", 0.0003)
    weight_decay = train_cfg.get("weight_decay", 0.0001)
    epochs = args.epochs or train_cfg.get("epochs", 10)
    random_seed = train_cfg.get("random_seed", 42)
    checkpoint_dir = train_cfg.get("checkpoint_dir", "models/weights")
    metrics_dir = train_cfg.get("metrics_dir", "outputs/metrics")

    split_cfg = train_cfg.get("split_ratio", {"train": 0.7, "val": 0.15, "test": 0.15})
    train_ratio = split_cfg.get("train", 0.7)
    val_ratio = split_cfg.get("val", 0.15)
    test_ratio = split_cfg.get("test", 0.15)

    print("\n============================================================")
    print(" PHASE 5: VISUAL DOCUMENT INTEGRITY CLASSIFICATION")
    print("============================================================")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Checking structure...")

    # Step 1: Inspect & Validate Dataset Structure
    report = validate_dataset_structure(
        dataset_dir=dataset_dir,
        real_subdir=real_subdir,
        attacked_subdir=attacked_subdir,
    )
    print(report)

    if not report.is_valid:
        print("\n[ERROR] Dataset structure is NOT suitable for training.")
        print("Resolve the reported issues before initiating training.")
        sys.exit(1)

    if args.validate_only:
        print("\nValidation completed successfully (--validate-only flag set). Exiting without training.")
        return

    # Step 2: Collect Samples and Create Leak-Free Splits
    print("\nStep 2: Preparing document-level leak-free splits...")
    samples = collect_dataset_samples(
        dataset_dir=dataset_dir,
        real_subdir=real_subdir,
        attacked_subdir=attacked_subdir,
    )

    train_samples, val_samples, test_samples = create_leak_free_splits(
        samples=samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
    )

    print(f"  * Train set: {len(train_samples)} images")
    print(f"  * Val set:   {len(val_samples)} images")
    print(f"  * Test set:  {len(test_samples)} images")

    # Step 3: Create Datasets and DataLoaders
    train_transform = get_document_transforms(image_size=image_size, is_training=True)
    eval_transform = get_document_transforms(image_size=image_size, is_training=False)

    train_dataset = DocumentDataset(train_samples, transform=train_transform)
    val_dataset = DocumentDataset(val_samples, transform=eval_transform)
    test_dataset = DocumentDataset(test_samples, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Step 4: Instantiate Model (EfficientNet-B0 baseline with binary classification head)
    print("\nStep 3: Initializing EfficientNet-B0 transfer learning baseline...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using compute device: {device}")

    classifier = DocumentIntegrityClassifier(
        backbone_name="efficientnet_b0",
        num_classes=2,
        pretrained=True,
        device=device,
    )

    # Step 5: Train & Validate
    print(f"\nStep 4: Training for {epochs} epochs...")
    trainer = IntegrityTrainer(
        model=classifier,
        lr=lr,
        weight_decay=weight_decay,
        checkpoint_dir=checkpoint_dir,
        metrics_dir=metrics_dir,
        device=device,
    )

    training_summary = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
    )

    # Step 6: Load Best Model and Evaluate on Test Set
    print("\nStep 5: Evaluating best checkpoint on unseen test set...")
    best_checkpoint = training_summary["best_checkpoint"]
    classifier.load_checkpoint(best_checkpoint)

    test_metrics = trainer.evaluate(test_loader)

    print("\n============================================================")
    print(" TEST EVALUATION RESULTS")
    print("============================================================")
    print(f"Total Test Samples:  {test_metrics.get('total_test_samples')}")
    print(f"Accuracy:            {test_metrics.get('accuracy') * 100:.2f}%")
    print(f"Macro Precision:     {test_metrics.get('macro_precision') * 100:.2f}%")
    print(f"Macro Recall:        {test_metrics.get('macro_recall') * 100:.2f}%")
    print(f"Macro F1-Score:      {test_metrics.get('macro_f1') * 100:.2f}%")
    if test_metrics.get("roc_auc") is not None:
        print(f"ROC-AUC Score:       {test_metrics.get('roc_auc'):.4f}")
    else:
        print(f"ROC-AUC Score:       N/A (single class in test slice)")

    cm = test_metrics.get("confusion_matrix", {})
    print("\nConfusion Matrix:")
    print(f"  True Negative  (REAL correctly identified):     {cm.get('true_negative_real')}")
    print(f"  False Positive (REAL flagged as ATTACKED):      {cm.get('false_positive_real_as_attacked')}")
    print(f"  False Negative (ATTACKED missed as REAL):       {cm.get('false_negative_attacked_as_real')}")
    print(f"  True Positive  (ATTACKED correctly caught):     {cm.get('true_positive_attacked')}")

    # Step 7: Save Metrics
    metrics_file, cm_file = trainer.save_metrics(
        training_summary=training_summary,
        test_metrics=test_metrics,
    )
    print(f"\nArtifacts saved:")
    print(f"  - Model checkpoint: {best_checkpoint}")
    print(f"  - Metrics summary:  {metrics_file}")
    print(f"  - Confusion matrix: {cm_file}")
    print("============================================================\n")


if __name__ == "__main__":
    main()
