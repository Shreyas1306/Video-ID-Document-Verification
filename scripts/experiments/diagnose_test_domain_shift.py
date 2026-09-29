"""
Phase 3B: Diagnostic Evaluation of EfficientNet-B0 Test-Domain Behavior
=====================================================================
Analyzes P(REAL) probability distributions on:
1. Untouched Test Set (Azerbaijan Passports: aze_passport_00, aze_passport_01)
2. Validation Set (Albanian IDs: alb_id_04, alb_id_05)

Generates:
- Probability statistics (min, max, mean, median, std) for REAL vs ATTACKED
- Diagnostic threshold analysis (0.05 to 0.95)
- Reconfirmed ROC-AUC
- Validation threshold optimization finding
- Distribution visualization plot
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
from torch.utils.data import DataLoader

from src.visual.dataset import (
    DocumentDataset,
    get_document_transforms,
    load_partitioned_dataset,
)
from src.visual.visual_integrity import DocumentIntegrityClassifier


def evaluate_split(model, dataset, device="cpu", batch_size=16):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()

    all_p_real = []
    all_p_attacked = []
    all_labels = []
    all_doc_ids = []
    all_paths = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            probs = model.predict_proba(images)  # shape (B, 2), [:, 0] is REAL, [:, 1] is ATTACKED
            p_real = probs[:, 0].cpu().numpy()
            p_att = probs[:, 1].cpu().numpy()

            all_p_real.extend(p_real.tolist())
            all_p_attacked.extend(p_att.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_doc_ids.extend(batch["doc_id"])
            all_paths.extend(batch["path"])

    return {
        "p_real": np.array(all_p_real),
        "p_attacked": np.array(all_p_attacked),
        "labels": np.array(all_labels),
        "doc_ids": all_doc_ids,
        "paths": all_paths,
    }


def compute_stats(arr):
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
    }


def threshold_sweep(p_real, y_true, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.05, 1.00, 0.05)

    results = []
    for th in thresholds:
        # Decision rule: if P(REAL) >= th -> predicted 0 (REAL), else 1 (ATTACKED)
        y_pred = np.where(p_real >= th, 0, 1)

        acc = accuracy_score(y_true, y_pred)
        prec_real = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
        rec_real = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        f1_real = f1_score(y_true, y_pred, pos_label=0, zero_division=0)

        prec_att = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        rec_att = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1_att = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        results.append({
            "threshold": round(float(th), 2),
            "accuracy": round(float(acc), 4),
            "prec_real": round(float(prec_real), 4),
            "rec_real": round(float(rec_real), 4),
            "f1_real": round(float(f1_real), 4),
            "prec_att": round(float(prec_att), 4),
            "rec_att": round(float(rec_att), 4),
            "f1_att": round(float(f1_att), 4),
            "macro_f1": round(float(macro_f1), 4),
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        })
    return results


def main():
    dataset_dir = Path("outputs/normalized_public_dataset")
    checkpoint_path = Path("models/weights/best_integrity_model.pth")
    output_plot_path = Path("outputs/metrics/p_real_distributions.png")

    print(f"Loading checkpoint: {checkpoint_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = DocumentIntegrityClassifier(
        backbone_name="efficientnet_b0",
        num_classes=2,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    print(f"Loading partitioned dataset from: {dataset_dir}")
    train_samples, val_samples, test_samples = load_partitioned_dataset(dataset_dir)

    eval_transform = get_document_transforms(image_size=(224, 224), is_training=False)
    val_dataset = DocumentDataset(val_samples, transform=eval_transform)
    test_dataset = DocumentDataset(test_samples, transform=eval_transform)

    print(f"Evaluating validation split ({len(val_dataset)} frames)...")
    val_data = evaluate_split(classifier, val_dataset, device=device)

    print(f"Evaluating test split ({len(test_dataset)} frames)...")
    test_data = evaluate_split(classifier, test_dataset, device=device)

    # 1 & 2: Test Set P(REAL) statistics
    test_labels = test_data["labels"]
    test_p_real = test_data["p_real"]

    real_mask = (test_labels == 0)
    att_mask = (test_labels == 1)

    test_real_stats = compute_stats(test_p_real[real_mask])
    test_att_stats = compute_stats(test_p_real[att_mask])

    print("\n" + "=" * 60)
    print(" 1. TEST SET P(REAL) STATISTICS — REAL IMAGES (N=100)")
    print("=" * 60)
    for k, v in test_real_stats.items():
        print(f"  {k:8s}: {v:.6f}")

    print("\n" + "=" * 60)
    print(" 2. TEST SET P(REAL) STATISTICS — ATTACKED IMAGES (N=260)")
    print("=" * 60)
    for k, v in test_att_stats.items():
        print(f"  {k:8s}: {v:.6f}")

    # Percentiles
    print("\nTest REAL percentiles: p10={:.6f}, p25={:.6f}, p50={:.6f}, p75={:.6f}, p90={:.6f}".format(
        *np.percentile(test_p_real[real_mask], [10, 25, 50, 75, 90])
    ))
    print("Test ATTACKED percentiles: p10={:.6f}, p25={:.6f}, p50={:.6f}, p75={:.6f}, p90={:.6f}".format(
        *np.percentile(test_p_real[att_mask], [10, 25, 50, 75, 90])
    ))

    # 3. Generate Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Test distribution
    axes[0].hist(test_p_real[real_mask], bins=30, alpha=0.6, color="green", label="REAL (aze_passport)", density=True)
    axes[0].hist(test_p_real[att_mask], bins=30, alpha=0.6, color="red", label="ATTACKED (aze_passport)", density=True)
    axes[0].set_title("Test Set P(REAL) Distribution (Azerbaijan Passports)")
    axes[0].set_xlabel("P(REAL)")
    axes[0].set_ylabel("Density")
    axes[0].axvline(0.5, color="black", linestyle="--", label="Default Thresh 0.5")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Val distribution
    val_labels = val_data["labels"]
    val_p_real = val_data["p_real"]
    v_real_mask = (val_labels == 0)
    v_att_mask = (val_labels == 1)

    axes[1].hist(val_p_real[v_real_mask], bins=30, alpha=0.6, color="green", label="REAL (alb_id)", density=True)
    axes[1].hist(val_p_real[v_att_mask], bins=30, alpha=0.6, color="red", label="ATTACKED (alb_id)", density=True)
    axes[1].set_title("Validation Set P(REAL) Distribution (Albanian IDs)")
    axes[1].set_xlabel("P(REAL)")
    axes[1].set_ylabel("Density")
    axes[1].axvline(0.5, color="black", linestyle="--", label="Default Thresh 0.5")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot_path, dpi=150)
    print(f"\nDistribution plot saved to: {output_plot_path}")

    # 4. Diagnostic threshold sweep on TEST set
    test_sweep = threshold_sweep(test_p_real, test_labels)
    print("\n" + "=" * 90)
    print(" 4. TEST-SET DIAGNOSTIC THRESHOLD ANALYSIS ONLY (DO NOT USE FOR PRODUCTION TUNING)")
    print("=" * 90)
    header = (
        f"{'Thresh':>7} | {'Accuracy':>8} | {'REAL Prec':>9} | {'REAL Rec':>8} | {'REAL F1':>8} | "
        f"{'ATT Prec':>8} | {'ATT Rec':>8} | {'ATT F1':>8} | {'Macro F1':>8} | {'(TN, FP, FN, TP)':>16}"
    )
    print(header)
    print("-" * len(header))
    for r in test_sweep:
        cm_str = f"({r['tn']}, {r['fp']}, {r['fn']}, {r['tp']})"
        line = (
            f"{r['threshold']:7.2f} | {r['accuracy']*100:7.2f}% | {r['prec_real']*100:8.2f}% | "
            f"{r['rec_real']*100:7.2f}% | {r['f1_real']*100:7.2f}% | {r['prec_att']*100:7.2f}% | "
            f"{r['rec_att']*100:7.2f}% | {r['f1_att']*100:7.2f}% | {r['macro_f1']*100:7.2f}% | {cm_str:>16}"
        )
        print(line)

    # 5. Reconfirm ROC-AUC from raw probabilities
    # Positive label in standard evaluate is ATTACKED (class 1), so score is P(ATTACKED)
    reconfirmed_auc = roc_auc_score(test_labels, test_data["p_attacked"])
    # If treating REAL as positive class:
    reconfirmed_auc_real = roc_auc_score(1 - test_labels, test_p_real)
    print("\n" + "=" * 60)
    print(" 5. ROC-AUC RECONFIRMATION")
    print("=" * 60)
    print(f"  ROC-AUC (P(ATTACKED) vs ATTACKED): {reconfirmed_auc:.6f}")
    print(f"  ROC-AUC (P(REAL) vs REAL):         {reconfirmed_auc_real:.6f}")

    # 6. Validation Set Analysis
    val_real_stats = compute_stats(val_p_real[v_real_mask])
    val_att_stats = compute_stats(val_p_real[v_att_mask])

    print("\n" + "=" * 60)
    print(" 6. VALIDATION SET P(REAL) STATISTICS (N=400)")
    print("=" * 60)
    print("  Validation REAL (N=100):")
    for k, v in val_real_stats.items():
        print(f"    {k:8s}: {v:.6f}")
    print("  Validation ATTACKED (N=300):")
    for k, v in val_att_stats.items():
        print(f"    {k:8s}: {v:.6f}")

    val_sweep = threshold_sweep(val_p_real, val_labels)
    best_val_row = max(val_sweep, key=lambda x: x["macro_f1"])

    print("\nValidation Threshold Sweep (Macro F1):")
    for r in val_sweep:
        marker = " <--- MAX" if r["threshold"] == best_val_row["threshold"] else ""
        print(f"  Thresh: {r['threshold']:.2f} | Val Acc: {r['accuracy']*100:.2f}% | Val Macro F1: {r['macro_f1']*100:.2f}%{marker}")

    val_auc = roc_auc_score(val_labels, val_data["p_attacked"])
    print(f"\n  Validation ROC-AUC: {val_auc:.6f}")
    print(f"  Optimal Validation Threshold: {best_val_row['threshold']:.2f} (Macro F1 = {best_val_row['macro_f1']*100:.2f}%)")


if __name__ == "__main__":
    main()
