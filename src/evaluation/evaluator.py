"""
Evaluation Module
=================
Empirical evaluation harness for comparative benchmarking of:
  - Experiment A: Single Image -> Visual Model
  - Experiment B: Video -> Visual + Temporal Analysis
  - Experiment C: Video -> Visual + Temporal + OCR -> Evidence Fusion

Computes:
  - Accuracy, Precision, Recall, F1-score
  - ROC-AUC
  - False Positive Rate (FPR)
  - False Negative Rate (FNR)
  - Confusion Matrices and Comparative Visualization
"""

import csv
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from src.fusion.score_fusion import EvidenceFusionEngine
from src.text.ocr_engine import run_text_verification_pipeline
from src.utils.config_loader import load_config
from src.visual.temporal_consistency import TemporalConsistencyAnalyzer
from src.visual.visual_integrity import DocumentIntegrityClassifier

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Standardized verification evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: Optional[float]
    false_positive_rate: float
    false_negative_rate: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    total_samples: int
    confusion_matrix: List[List[int]]


@dataclass
class ExperimentResult:
    """Encapsulates the complete result of an evaluation experiment."""
    experiment_id: str
    experiment_name: str
    description: str
    metrics: EvaluationMetrics
    y_true: List[int]
    y_pred: List[int]
    y_scores: List[float]
    sample_records: List[Dict[str, Any]]
    parameters: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "description": self.description,
            "metrics": asdict(self.metrics),
            "parameters": self.parameters,
            "total_samples": len(self.y_true),
        }


def calculate_evaluation_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_scores: Optional[Sequence[float]] = None,
) -> EvaluationMetrics:
    """
    Calculate full evaluation metrics with ATTACKED (1) as the positive class.

    Definitions:
      - Positive (1): ATTACKED / Forged document
      - Negative (0): REAL / Genuine document
      - TP: Attacked correctly identified as Attacked
      - FP: Real incorrectly flagged as Attacked (False Alarm)
      - TN: Real correctly identified as Real
      - FN: Attacked incorrectly passed as Real (Missed Attack)
      - FPR: FP / (FP + TN)
      - FNR: FN / (FN + TP)
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true ({len(y_true)}) vs y_pred ({len(y_pred)})")

    total = len(y_true)
    if total == 0:
        return EvaluationMetrics(
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            roc_auc=None,
            false_positive_rate=0.0,
            false_negative_rate=0.0,
            true_positives=0,
            false_positives=0,
            true_negatives=0,
            false_negatives=0,
            total_samples=0,
            confusion_matrix=[[0, 0], [0, 0]],
        )

    # Compute 2x2 confusion matrix with labels [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Accuracy
    accuracy = (tp + tn) / max(1, total)

    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Recall (TPR)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1-Score
    f1 = (
        2.0 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # False Positive Rate: FP / (FP + TN)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # False Negative Rate: FN / (FN + TP)
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # ROC-AUC calculation where feasible
    roc_auc: Optional[float] = None
    if y_scores is not None and len(y_scores) == total:
        unique_classes = set(y_true)
        if len(unique_classes) >= 2:
            try:
                roc_auc = float(roc_auc_score(y_true, y_scores))
            except Exception as e:
                logger.warning(f"Could not compute ROC-AUC: {e}")
                roc_auc = None

    return EvaluationMetrics(
        accuracy=round(float(accuracy), 4),
        precision=round(float(precision), 4),
        recall=round(float(recall), 4),
        f1_score=round(float(f1), 4),
        roc_auc=round(roc_auc, 4) if roc_auc is not None else None,
        false_positive_rate=round(float(fpr), 4),
        false_negative_rate=round(float(fnr), 4),
        true_positives=int(tp),
        false_positives=int(fp),
        true_negatives=int(tn),
        false_negatives=int(fn),
        total_samples=total,
        confusion_matrix=cm.tolist(),
    )


def group_samples_by_document(
    samples: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Group individual frame samples into document video sequences."""
    grouped: Dict[str, Dict[str, Any]] = {}
    for sample in samples:
        doc_id = str(sample["doc_id"])
        if doc_id not in grouped:
            grouped[doc_id] = {
                "doc_id": doc_id,
                "label": sample["label"],
                "class_name": sample["class_name"],
                "paths": [],
                "frame_ids": [],
            }
        grouped[doc_id]["paths"].append(str(sample["path"]))
        grouped[doc_id]["frame_ids"].append(Path(sample["path"]).stem)

    # Sort frames within each document group to preserve temporal ordering
    for doc in grouped.values():
        paired = sorted(zip(doc["frame_ids"], doc["paths"]))
        doc["frame_ids"] = [p[0] for p in paired]
        doc["paths"] = [p[1] for p in paired]

    return list(grouped.values())


# =============================================================================
# Experiment A: Single Image -> Visual Model
# =============================================================================

def run_experiment_a(
    test_samples: Sequence[Dict[str, Any]],
    model_checkpoint: Optional[Union[str, Path]] = "models/weights/best_integrity_model.pth",
    threshold: float = 0.50,
) -> ExperimentResult:
    """
    Experiment A: Single Image -> Visual Model.

    Evaluates the visual document classifier on isolated individual frames.
    For each image:
      - Computes visual integrity score S_v (probability document is REAL).
      - Anomaly/Attack score = 1.0 - S_v.
      - Predicted label: 1 (ATTACKED) if (1.0 - S_v) >= threshold, else 0 (REAL).
    """
    logger.info(f"Running Experiment A on {len(test_samples)} isolated test frames...")
    classifier = DocumentIntegrityClassifier(
        checkpoint_path=model_checkpoint if model_checkpoint and Path(model_checkpoint).is_file() else None
    )

    y_true: List[int] = []
    y_pred: List[int] = []
    y_scores: List[float] = []
    records: List[Dict[str, Any]] = []

    for item in test_samples:
        img_path = Path(item["path"])
        label = int(item["label"])  # 0=REAL, 1=ATTACKED
        doc_id = str(item.get("doc_id", img_path.stem))

        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Could not read image: {img_path}")
            continue

        pred = classifier.predict_image(img)
        v_integrity = float(pred["integrity_score"])
        attack_score = float(1.0 - v_integrity)

        predicted_class = 1 if attack_score >= threshold else 0

        y_true.append(label)
        y_pred.append(predicted_class)
        y_scores.append(attack_score)

        records.append({
            "path": str(img_path),
            "doc_id": doc_id,
            "label": label,
            "class_name": item["class_name"],
            "visual_integrity": v_integrity,
            "attack_score": attack_score,
            "predicted_class": predicted_class,
        })

    metrics = calculate_evaluation_metrics(y_true, y_pred, y_scores)

    return ExperimentResult(
        experiment_id="experiment_a",
        experiment_name="Experiment A: Single Image -> Visual Model",
        description="Isolated single-frame visual anomaly classification using EfficientNet-B0.",
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        y_scores=y_scores,
        sample_records=records,
        parameters={"threshold": threshold, "checkpoint": str(model_checkpoint)},
    )


# =============================================================================
# Experiment B: Video -> Visual + Temporal Analysis
# =============================================================================

def run_experiment_b(
    test_sequences: Sequence[Dict[str, Any]],
    model_checkpoint: Optional[Union[str, Path]] = "models/weights/best_integrity_model.pth",
    config_path: Optional[Union[str, Path]] = "config/settings.yaml",
    threshold: float = 0.50,
) -> ExperimentResult:
    """
    Experiment B: Video -> Visual + Temporal Analysis.

    Evaluates combined visual integrity and temporal embedding consistency over
    video frame sequences.
    For each video sequence:
      - Evaluates visual integrity score across sequence frames (mean S_v).
      - Measures consecutive frame embedding cosine similarity (S_t).
      - Re-normalizes visual and temporal weights from configuration:
          w_v_norm = w_v / (w_v + w_t)
          w_t_norm = w_t / (w_v + w_t)
          S_VT = w_v_norm * S_v + w_t_norm * S_t
      - Anomaly/Attack score = 1.0 - S_VT.
      - Predicted label: 1 (ATTACKED) if (1.0 - S_VT) >= threshold, else 0 (REAL).
    """
    logger.info(f"Running Experiment B on {len(test_sequences)} video sequences...")
    cfg = load_config(config_path)
    weights = cfg.get("fusion", {}).get("weights", {"visual_weight": 0.40, "temporal_weight": 0.30})
    w_v = float(weights.get("visual_weight", 0.40))
    w_t = float(weights.get("temporal_weight", 0.30))
    total_w = w_v + w_t
    w_v_norm = w_v / total_w if total_w > 0 else 0.5
    w_t_norm = w_t / total_w if total_w > 0 else 0.5

    classifier = DocumentIntegrityClassifier(
        checkpoint_path=model_checkpoint if model_checkpoint and Path(model_checkpoint).is_file() else None
    )
    temporal_analyzer = TemporalConsistencyAnalyzer(
        checkpoint_path=model_checkpoint if model_checkpoint and Path(model_checkpoint).is_file() else None,
        consistency_threshold=cfg.get("temporal_consistency", {}).get("consistency_threshold", 0.85),
    )

    y_true: List[int] = []
    y_pred: List[int] = []
    y_scores: List[float] = []
    records: List[Dict[str, Any]] = []

    for seq in test_sequences:
        doc_id = str(seq["doc_id"])
        label = int(seq["label"])
        paths = seq["paths"]

        if not paths:
            continue

        # 1. Visual Integrity across sequence
        v_scores: List[float] = []
        for p in paths:
            img = cv2.imread(p)
            if img is not None:
                v_res = classifier.predict_image(img)
                v_scores.append(float(v_res["integrity_score"]))

        mean_v = float(np.mean(v_scores)) if v_scores else 0.0

        # 2. Temporal Consistency across sequence
        temp_res = temporal_analyzer.analyze_sequence(
            frames=paths,
            frame_ids=[Path(p).stem for p in paths],
            video_ids=[doc_id] * len(paths),
        )
        if "temporal_consistency_score" not in temp_res:
            logger.warning(
                "Document %s: temporal consistency result missing canonical key 'temporal_consistency_score'. Available keys: %s",
                doc_id,
                list(temp_res.keys()),
            )
        temp_score = float(temp_res.get("temporal_consistency_score", temp_res.get("aggregate_score", 0.0)))

        # 3. Combined Visual + Temporal Score
        s_vt = float(w_v_norm * mean_v + w_t_norm * temp_score)
        attack_score = float(1.0 - s_vt)
        predicted_class = 1 if attack_score >= threshold else 0

        y_true.append(label)
        y_pred.append(predicted_class)
        y_scores.append(attack_score)

        records.append({
            "doc_id": doc_id,
            "label": label,
            "class_name": seq["class_name"],
            "frame_count": len(paths),
            "visual_score": round(mean_v, 4),
            "temporal_score": round(temp_score, 4),
            "combined_vt_score": round(s_vt, 4),
            "attack_score": round(attack_score, 4),
            "predicted_class": predicted_class,
        })

    metrics = calculate_evaluation_metrics(y_true, y_pred, y_scores)

    return ExperimentResult(
        experiment_id="experiment_b",
        experiment_name="Experiment B: Video -> Visual + Temporal Analysis",
        description="Multi-frame sequence evaluation combining visual integrity and temporal embedding cosine similarity.",
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        y_scores=y_scores,
        sample_records=records,
        parameters={
            "threshold": threshold,
            "weights": {"visual": w_v_norm, "temporal": w_t_norm},
        },
    )


# =============================================================================
# Experiment C: Video -> Visual + Temporal + OCR -> Evidence Fusion
# =============================================================================

def run_experiment_c(
    test_sequences: Sequence[Dict[str, Any]],
    model_checkpoint: Optional[Union[str, Path]] = "models/weights/best_integrity_model.pth",
    config_path: Optional[Union[str, Path]] = "config/settings.yaml",
    threshold: float = 0.50,
) -> ExperimentResult:
    """
    Experiment C: Video -> Visual + Temporal + OCR -> Evidence Fusion.

    Evaluates the complete end-to-end multi-modal evidence fusion pipeline:
      - Visual Document Integrity
      - Temporal Embedding Consistency
      - Character OCR Extraction Confidence
      - Cross-Frame Text Consistency
    Produces a unified fused integrity score S_fused and maps to risk / binary prediction.
    """
    logger.info(f"Running Experiment C on {len(test_sequences)} video sequences with full multi-modal fusion...")
    cfg = load_config(config_path)

    classifier = DocumentIntegrityClassifier(
        checkpoint_path=model_checkpoint if model_checkpoint and Path(model_checkpoint).is_file() else None
    )
    temporal_analyzer = TemporalConsistencyAnalyzer(
        checkpoint_path=model_checkpoint if model_checkpoint and Path(model_checkpoint).is_file() else None,
        consistency_threshold=cfg.get("temporal_consistency", {}).get("consistency_threshold", 0.85),
    )
    fusion_engine = EvidenceFusionEngine(config_path=config_path)

    y_true: List[int] = []
    y_pred: List[int] = []
    y_scores: List[float] = []
    records: List[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        for seq in test_sequences:
            doc_id = str(seq["doc_id"])
            label = int(seq["label"])
            paths = seq["paths"]

            if not paths:
                continue

            # 1. Visual Integrity
            v_scores = []
            for p in paths:
                img = cv2.imread(p)
                if img is not None:
                    v_res = classifier.predict_image(img)
                    v_scores.append(float(v_res["integrity_score"]))
            mean_v = float(np.mean(v_scores)) if v_scores else 0.0

            # 2. Temporal Consistency
            temp_res = temporal_analyzer.analyze_sequence(
                frames=paths,
                frame_ids=[Path(p).stem for p in paths],
                video_ids=[doc_id] * len(paths),
            )
            if "temporal_consistency_score" not in temp_res:
                logger.warning(
                    "Document %s: temporal consistency result missing canonical key 'temporal_consistency_score'. Available keys: %s",
                    doc_id,
                    list(temp_res.keys()),
                )
            temp_score = float(temp_res.get("temporal_consistency_score", temp_res.get("aggregate_score", 0.0)))

            # 3. OCR & Text Verification
            ocr_summary_dir = temp_dir_path / doc_id
            ocr_summary_dir.mkdir(parents=True, exist_ok=True)
            ocr_res = run_text_verification_pipeline(
                normalized_frames_or_paths=paths,
                output_summary_dir=ocr_summary_dir,
                config_path=config_path,
            )
            ocr_conf = float(ocr_res.get("ocr_confidence", 0.0))
            text_cons = float(ocr_res.get("text_consistency", 0.0))

            # 4. Multi-Modal Evidence Fusion
            fusion_res = fusion_engine.fuse(
                visual_integrity=mean_v,
                temporal_consistency=temp_score,
                ocr_confidence=ocr_conf,
                text_consistency=text_cons,
            )
            fused_score = float(fusion_res.get("fused_integrity_score", 0.0))
            attack_score = float(1.0 - fused_score)
            predicted_class = 1 if attack_score >= threshold else 0

            y_true.append(label)
            y_pred.append(predicted_class)
            y_scores.append(attack_score)

            records.append({
                "doc_id": doc_id,
                "label": label,
                "class_name": seq["class_name"],
                "frame_count": len(paths),
                "visual_score": round(mean_v, 4),
                "temporal_score": round(temp_score, 4),
                "ocr_confidence": round(ocr_conf, 4),
                "text_consistency": round(text_cons, 4),
                "fused_integrity_score": round(fused_score, 4),
                "attack_score": round(attack_score, 4),
                "predicted_class": predicted_class,
                "evidence_coverage": fusion_res.get("evidence_coverage", 1.0),
            })

    metrics = calculate_evaluation_metrics(y_true, y_pred, y_scores)

    return ExperimentResult(
        experiment_id="experiment_c",
        experiment_name="Experiment C: Video -> Visual + Temporal + OCR -> Evidence Fusion",
        description="Complete multi-modal evidence fusion integrating visual, temporal, OCR confidence, and cross-frame textual consensus.",
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        y_scores=y_scores,
        sample_records=records,
        parameters={"threshold": threshold, "weights": fusion_engine.base_weights},
    )


# =============================================================================
# Plotting & CSV Export Utilities
# =============================================================================

def plot_confusion_matrix(
    cm: List[List[int]],
    labels: Sequence[str] = ("REAL", "ATTACKED"),
    title: str = "Confusion Matrix",
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Plot and save a clean, annotated confusion matrix."""
    cm_arr = np.array(cm)
    fig = Figure(figsize=(5, 4), dpi=150)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)

    im = ax.imshow(cm_arr, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(cm_arr.shape[1]))
    ax.set_yticks(np.arange(cm_arr.shape[0]))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Ground Truth")
    ax.set_xlabel("Predicted Label")

    # Annotate cells
    thresh = cm_arr.max() / 2.0 if cm_arr.max() > 0 else 1.0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            val = cm_arr[i, j]
            ax.text(
                j, i, format(val, "d"),
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontweight="bold",
            )

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        canvas.print_figure(str(save_path), bbox_inches="tight")
        logger.info(f"Saved confusion matrix plot to: {save_path}")


def plot_roc_comparison(
    experiments: Sequence[ExperimentResult],
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Plot overlaid ROC curves for all experiments."""
    fig = Figure(figsize=(7, 6), dpi=150)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)

    ax.plot([0, 1], [0, 1], linestyle="--", lw=1.5, color="gray", label="Random Chance (AUC = 0.50)")

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for idx, exp in enumerate(experiments):
        y_true = exp.y_true
        y_scores = exp.y_scores
        auc_val = exp.metrics.roc_auc

        if len(set(y_true)) >= 2 and y_scores:
            try:
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                lbl = f"{exp.experiment_id.upper()} (AUC = {auc_val:.3f})" if auc_val is not None else exp.experiment_id.upper()
                ax.plot(fpr, tpr, lw=2.0, color=colors[idx % len(colors)], label=lbl)
            except Exception as e:
                logger.warning(f"Failed to plot ROC curve for {exp.experiment_id}: {e}")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=11)
    ax.set_ylabel("True Positive Rate (TPR / Recall)", fontsize=11)
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        canvas.print_figure(str(save_path), bbox_inches="tight")
        logger.info(f"Saved ROC comparison plot to: {save_path}")


def export_results_csv(
    experiments: Sequence[ExperimentResult],
    csv_path: Union[str, Path],
) -> Path:
    """Export comparative experimental metrics to standard CSV format."""
    csv_file = Path(csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "Experiment",
        "Pipeline Configuration",
        "Samples Evaluated",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC",
        "FPR (False Positive Rate)",
        "FNR (False Negative Rate)",
        "TP",
        "FP",
        "TN",
        "FN",
    ]

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for exp in experiments:
            m = exp.metrics
            writer.writerow([
                exp.experiment_id,
                exp.experiment_name,
                m.total_samples,
                f"{m.accuracy:.4f}",
                f"{m.precision:.4f}",
                f"{m.recall:.4f}",
                f"{m.f1_score:.4f}",
                f"{m.roc_auc:.4f}" if m.roc_auc is not None else "N/A",
                f"{m.false_positive_rate:.4f}",
                f"{m.false_negative_rate:.4f}",
                m.true_positives,
                m.false_positives,
                m.true_negatives,
                m.false_negatives,
            ])

    logger.info(f"Exported experimental results CSV to: {csv_file}")
    return csv_file
