"""
Integrity Trainer & Evaluation Engine
=====================================
Training, validation, checkpointing, and comprehensive metrics evaluation for
binary document integrity classification.
"""

import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

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
import torch.nn as nn
from torch.utils.data import DataLoader

from src.visual.visual_integrity import DocumentIntegrityClassifier

logger = logging.getLogger(__name__)


class IntegrityTrainer:
    """
    Manages the training, validation, checkpointing, and evaluation lifecycle
    for DocumentIntegrityClassifier.
    """

    def __init__(
        self,
        model: DocumentIntegrityClassifier,
        criterion: Optional[nn.Module] = None,
        val_criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 0.0003,
        weight_decay: float = 0.0001,
        checkpoint_dir: Union[str, Path] = "models/weights",
        metrics_dir: Union[str, Path] = "outputs/metrics",
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.model = model
        self.device = model.device if device is None else torch.device(device)
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.val_criterion = val_criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.checkpoint_dir = Path(checkpoint_dir)
        self.metrics_dir = Path(metrics_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
        }
        self.best_val_f1 = -1.0
        self.best_checkpoint_path = self.checkpoint_dir / "best_integrity_model.pth"

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = total_loss / max(1, total)
        epoch_acc = correct / max(1, total)
        return epoch_loss, epoch_acc

    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(images)
                loss = self.val_criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
                total += labels.size(0)

        epoch_loss = total_loss / max(1, total)
        acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) if all_labels else 0.0

        return {
            "loss": float(epoch_loss),
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
    ) -> Dict[str, Any]:
        """Full training loop with validation and best checkpoint tracking."""
        logger.info(f"Starting training for {epochs} epochs on device: {self.device}")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_metrics = self.validate_epoch(val_loader)

            self.history["train_loss"].append(round(train_loss, 4))
            self.history["train_acc"].append(round(train_acc, 4))
            self.history["val_loss"].append(round(val_metrics["loss"], 4))
            self.history["val_acc"].append(round(val_metrics["accuracy"], 4))
            self.history["val_f1"].append(round(val_metrics["macro_f1"], 4))

            logger.info(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} F1: {val_metrics['macro_f1']:.4f}"
            )

            # Checkpoint on best validation F1
            if val_metrics["macro_f1"] >= self.best_val_f1:
                self.best_val_f1 = val_metrics["macro_f1"]
                self.save_checkpoint(
                    path=self.best_checkpoint_path,
                    epoch=epoch,
                    val_metrics=val_metrics,
                )

        elapsed = time.time() - start_time
        logger.info(f"Training completed in {elapsed:.2f} seconds. Best Val F1: {self.best_val_f1:.4f}")

        return {
            "epochs_trained": epochs,
            "best_val_f1": round(self.best_val_f1, 4),
            "best_checkpoint": str(self.best_checkpoint_path),
            "history": self.history,
            "training_duration_seconds": round(elapsed, 2),
        }

    def evaluate(
        self,
        test_loader: DataLoader,
    ) -> Dict[str, Any]:
        """
        Evaluate model on test dataset and calculate comprehensive metrics:
          - Accuracy
          - Precision (macro & binary)
          - Recall (macro & binary)
          - F1-score (macro & binary)
          - ROC-AUC (using positive class softmax probabilities)
          - Confusion Matrix
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs_attacked = []

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                probs = self.model.predict_proba(images)
                preds = torch.argmax(probs, dim=1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
                # Class 1 is ATTACKED
                all_probs_attacked.extend(probs[:, 1].cpu().numpy().tolist())

        if not all_labels:
            logger.warning("Empty test set provided for evaluation.")
            return {}

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs_attacked)

        # Standard metrics
        acc = float(accuracy_score(y_true, y_pred))
        macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        binary_prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
        binary_rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
        binary_f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

        real_prec = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
        real_rec = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
        real_f1 = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))

        # ROC-AUC (requires both classes present in y_true)
        roc_auc: Optional[float] = None
        if len(np.unique(y_true)) > 1:
            try:
                roc_auc = float(roc_auc_score(y_true, y_prob))
            except Exception as e:
                logger.warning(f"Could not compute ROC-AUC: {e}")

        # Confusion Matrix: [[TN, FP], [FN, TP]] where 0=REAL (negative), 1=ATTACKED (positive)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

        results = {
            "total_test_samples": len(y_true),
            "accuracy": round(acc, 4),
            "macro_precision": round(macro_prec, 4),
            "macro_recall": round(macro_rec, 4),
            "macro_f1": round(macro_f1, 4),
            "precision_attacked": round(binary_prec, 4),
            "recall_attacked": round(binary_rec, 4),
            "f1_attacked": round(binary_f1, 4),
            "precision_real": round(real_prec, 4),
            "recall_real": round(real_rec, 4),
            "f1_real": round(real_f1, 4),
            "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
            "confusion_matrix": {
                "raw": cm.tolist(),
                "labels": ["REAL", "ATTACKED"],
                "true_negative_real": tn,
                "false_positive_real_as_attacked": fp,
                "false_negative_attacked_as_real": fn,
                "true_positive_attacked": tp,
            },
        }

        return results

    def save_checkpoint(
        self,
        path: Union[str, Path],
        epoch: int,
        val_metrics: Dict[str, float],
    ) -> None:
        """Save a model checkpoint to disk."""
        ckpt_path = Path(path)
        payload = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_metrics": val_metrics,
            "backbone": self.model.backbone_name,
            "num_classes": self.model.num_classes,
        }
        torch.save(payload, ckpt_path)
        logger.info(f"Saved best model checkpoint to: {ckpt_path}")

    def save_metrics(
        self,
        training_summary: Dict[str, Any],
        test_metrics: Dict[str, Any],
    ) -> Tuple[Path, Path]:
        """Save training history and test evaluation metrics to JSON files."""
        metrics_file = self.metrics_dir / "training_metrics.json"
        cm_file = self.metrics_dir / "confusion_matrix.json"

        full_metrics = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_architecture": self.model.backbone_name,
            "training_summary": training_summary,
            "test_evaluation": test_metrics,
        }

        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(full_metrics, f, indent=2)

        with open(cm_file, "w", encoding="utf-8") as f:
            json.dump(test_metrics.get("confusion_matrix", {}), f, indent=2)

        logger.info(f"Saved training metrics to: {metrics_file}")
        logger.info(f"Saved confusion matrix to: {cm_file}")

        return metrics_file, cm_file
