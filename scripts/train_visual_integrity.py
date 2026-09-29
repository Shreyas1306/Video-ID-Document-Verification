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
import shutil
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.config_loader import load_config
from src.visual.dataset import (
    DocumentDataset,
    collect_dataset_samples,
    create_leak_free_splits,
    get_document_transforms,
    is_partitioned_dataset,
    load_partitioned_dataset,
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
    if is_partitioned_dataset(dataset_dir):
        print(f"\nStep 2: Loading pre-partitioned dataset from '{dataset_dir}'...")
        train_samples, val_samples, test_samples = load_partitioned_dataset(dataset_dir)
    else:
        print("\nStep 2: Preparing document-level leak-free splits from flat directory...")
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

    # Step 4: Archive existing checkpoints before overwriting
    target_ckpt = Path(checkpoint_dir) / "best_integrity_model.pth"
    synthetic_archive = Path(checkpoint_dir) / "best_integrity_model_synthetic.pth"
    dlc_orig_archive = Path(checkpoint_dir) / "best_integrity_model_dlc_original_split.pth"
    if target_ckpt.is_file() and not synthetic_archive.is_file():
        shutil.copy2(target_ckpt, synthetic_archive)
        logger.info(f"Archived existing synthetic checkpoint to: {synthetic_archive}")
        print(f"\nArchived existing synthetic checkpoint to: {synthetic_archive}")
    if target_ckpt.is_file() and not dlc_orig_archive.is_file():
        shutil.copy2(target_ckpt, dlc_orig_archive)
        logger.info(f"Archived existing DLC original split checkpoint to: {dlc_orig_archive}")
        print(f"\nArchived existing DLC original split checkpoint to: {dlc_orig_archive}")

    # Step 5: Instantiate Model (EfficientNet-B0 baseline with binary classification head)
    print("\nStep 3: Initializing EfficientNet-B0 transfer learning baseline...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using compute device: {device}")

    classifier = DocumentIntegrityClassifier(
        backbone_name="efficientnet_b0",
        num_classes=2,
        pretrained=True,
        device=device,
    )

    # Step 6: Handle class imbalance using weighted CrossEntropyLoss on training set only
    train_labels = [int(s["label"]) for s in train_samples]
    num_real = sum(1 for l in train_labels if l == 0)
    num_attacked = sum(1 for l in train_labels if l == 1)
    total_train = len(train_labels)

    if num_real > 0 and num_attacked > 0 and num_real != num_attacked:
        w_real = total_train / (2.0 * num_real)
        w_attacked = total_train / (2.0 * num_attacked)
        print(f"\nApplying class-weighted CrossEntropyLoss on training set:")
        print(f"  * REAL weight:     {w_real:.4f} ({total_train} / (2 * {num_real}))")
        print(f"  * ATTACKED weight: {w_attacked:.4f} ({total_train} / (2 * {num_attacked}))")
        class_weights = torch.tensor([w_real, w_attacked], dtype=torch.float, device=device)
        train_criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        train_criterion = nn.CrossEntropyLoss()

    val_criterion = nn.CrossEntropyLoss()

    # Step 7: Train & Validate
    print(f"\nStep 4: Training for {epochs} epochs...")
    trainer = IntegrityTrainer(
        model=classifier,
        criterion=train_criterion,
        val_criterion=val_criterion,
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

    # Step 8: Load Best Model and Evaluate on Test Set
    print("\nStep 5: Evaluating best checkpoint on unseen test set...")
    best_checkpoint = training_summary["best_checkpoint"]
    classifier.load_checkpoint(best_checkpoint)

    test_metrics = trainer.evaluate(test_loader)

    print("\n============================================================")
    print(" TRAINING SUMMARY BY EPOCH")
    print("============================================================")
    history = training_summary.get("history", {})
    train_losses = history.get("train_loss", [])
    val_losses = history.get("val_loss", [])
    train_accs = history.get("train_acc", [])
    val_accs = history.get("val_acc", [])
    val_f1s = history.get("val_f1", [])

    for ep in range(len(train_losses)):
        t_l = train_losses[ep] if ep < len(train_losses) else 0.0
        v_l = val_losses[ep] if ep < len(val_losses) else 0.0
        t_a = train_accs[ep] if ep < len(train_accs) else 0.0
        v_a = val_accs[ep] if ep < len(val_accs) else 0.0
        v_f = val_f1s[ep] if ep < len(val_f1s) else 0.0
        print(f"Epoch {ep+1:02d}/{epochs:02d} | Train Loss: {t_l:.4f}, Train Acc: {t_a*100:.2f}% | Val Loss: {v_l:.4f}, Val Acc: {v_a*100:.2f}%, Val Macro F1: {v_f:.4f}")

    print("\n============================================================")
    print(" TEST EVALUATION RESULTS")
    print("============================================================")
    print(f"Total Test Samples:   {test_metrics.get('total_test_samples')}")
    print(f"Accuracy:             {test_metrics.get('accuracy') * 100:.2f}%")
    print(f"Macro Precision:      {test_metrics.get('macro_precision') * 100:.2f}%")
    print(f"Macro Recall:         {test_metrics.get('macro_recall') * 100:.2f}%")
    print(f"Macro F1-Score:       {test_metrics.get('macro_f1') * 100:.2f}%")
    if test_metrics.get("roc_auc") is not None:
        print(f"ROC-AUC Score:        {test_metrics.get('roc_auc'):.4f}")
    else:
        print(f"ROC-AUC Score:        N/A (single class in test slice)")

    print("\nPer-Class Metrics:")
    print(f"  REAL     -> Precision: {test_metrics.get('precision_real', 0.0)*100:.2f}%, Recall: {test_metrics.get('recall_real', 0.0)*100:.2f}%, F1: {test_metrics.get('f1_real', 0.0)*100:.2f}%")
    print(f"  ATTACKED -> Precision: {test_metrics.get('precision_attacked', 0.0)*100:.2f}%, Recall: {test_metrics.get('recall_attacked', 0.0)*100:.2f}%, F1: {test_metrics.get('f1_attacked', 0.0)*100:.2f}%")

    cm = test_metrics.get("confusion_matrix", {})
    print("\nConfusion Matrix:")
    print(f"  True Negative  (REAL correctly identified):     {cm.get('true_negative_real')}")
    print(f"  False Positive (REAL flagged as ATTACKED):      {cm.get('false_positive_real_as_attacked')}")
    print(f"  False Negative (ATTACKED missed as REAL):       {cm.get('false_negative_attacked_as_real')}")
    print(f"  True Positive  (ATTACKED correctly caught):     {cm.get('true_positive_attacked')}")

    # Step 9: Save Metrics
    metrics_file, cm_file = trainer.save_metrics(
        training_summary=training_summary,
        test_metrics=test_metrics,
    )
    print(f"\nArtifacts saved:")
    print(f"  - Model checkpoint: {best_checkpoint}")
    print(f"  - Metrics summary:  {metrics_file}")
    print(f"  - Confusion matrix: {cm_file}")

    # Step 10: Verify that the final checkpoint can be loaded successfully
    print("\nStep 6: Verifying final checkpoint loadability...")
    verification_classifier = DocumentIntegrityClassifier(
        backbone_name="efficientnet_b0",
        num_classes=2,
        checkpoint_path=best_checkpoint,
        device=device,
    )
    verification_classifier.eval()
    dummy_tensor = torch.zeros((1, 3, image_size[0], image_size[1]), device=device)
    with torch.no_grad():
        _ = verification_classifier(dummy_tensor)
    print("Final checkpoint successfully verified and loaded by DocumentIntegrityClassifier.")
    print("============================================================\n")


if __name__ == "__main__":
    main()
