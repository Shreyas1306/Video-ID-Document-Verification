"""
Tests for Visual Document Integrity Classifier and Trainer
"""

import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.visual.trainer import IntegrityTrainer
from src.visual.visual_integrity import (
    DocumentIntegrityClassifier,
    compute_statistical_proxy_score,
)


def test_classifier_head_replacement():
    """Verify EfficientNet-B0 classification head has 2 binary output classes."""
    model = DocumentIntegrityClassifier(backbone_name="efficientnet_b0", num_classes=2, pretrained=False, device="cpu")
    assert model.num_classes == 2

    # Forward pass with dummy tensor (batch_size=2, channels=3, H=224, W=224)
    dummy_input = torch.randn(2, 3, 224, 224)
    logits = model(dummy_input)

    assert logits.shape == (2, 2), f"Expected shape (2, 2), got {logits.shape}"

    # Verify probability outputs
    probs = model.predict_proba(dummy_input)
    assert probs.shape == (2, 2)
    sums = probs.sum(dim=1).detach().cpu().numpy()
    np.testing.assert_allclose(sums, [1.0, 1.0], atol=1e-5)


def test_feature_extraction():
    """Verify feature extractor extracts 1280-dim embedding vector."""
    model = DocumentIntegrityClassifier(backbone_name="efficientnet_b0", num_classes=2, pretrained=False, device="cpu")
    dummy_input = torch.randn(2, 3, 224, 224)
    features = model.extract_features(dummy_input)

    assert features.shape == (2, 1280), f"Expected shape (2, 1280), got {features.shape}"


def test_predict_image_numpy():
    """Verify single image inference on numpy array."""
    model = DocumentIntegrityClassifier(backbone_name="efficientnet_b0", num_classes=2, pretrained=False, device="cpu")
    mock_frame = np.full((300, 400, 3), 200, dtype=np.uint8)

    res = model.predict_image(mock_frame)
    assert "predicted_class" in res
    assert res["predicted_class"] in {"REAL", "ATTACKED"}
    assert "probability_real" in res
    assert "probability_attacked" in res
    assert 0.0 <= res["integrity_score"] <= 1.0


def test_statistical_proxy_score():
    """Verify statistical fallback proxy calculations."""
    mock_frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    stats = compute_statistical_proxy_score(mock_frame)

    assert "sharpness" in stats
    assert "edge_density" in stats
    assert "color_uniformity" in stats
    assert "proxy_score" in stats
    assert 0.0 <= stats["proxy_score"] <= 1.0


def test_trainer_train_epoch_and_evaluate(tmp_path):
    """Verify IntegrityTrainer training, validation, checkpointing, and evaluation."""
    model = DocumentIntegrityClassifier(backbone_name="efficientnet_b0", num_classes=2, pretrained=False, device="cpu")

    # Construct mock dataset with batch format {"image": tensor, "label": tensor}
    class MockBatchDataset(torch.utils.data.Dataset):
        def __init__(self, size=8):
            self.size = size
            self.images = torch.randn(size, 3, 224, 224)
            # Balanced labels
            self.labels = torch.tensor([0 if i < size // 2 else 1 for i in range(size)], dtype=torch.long)

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {"image": self.images[idx], "label": self.labels[idx]}

    train_loader = DataLoader(MockBatchDataset(size=8), batch_size=4)
    val_loader = DataLoader(MockBatchDataset(size=4), batch_size=2)
    test_loader = DataLoader(MockBatchDataset(size=4), batch_size=2)

    trainer = IntegrityTrainer(
        model=model,
        lr=0.001,
        checkpoint_dir=tmp_path / "checkpoints",
        metrics_dir=tmp_path / "metrics",
        device="cpu",
    )

    # 1. Run 1 training epoch
    loss, acc = trainer.train_epoch(train_loader)
    assert loss > 0.0
    assert 0.0 <= acc <= 1.0

    # 2. Run 1 validation epoch
    val_metrics = trainer.validate_epoch(val_loader)
    assert "loss" in val_metrics
    assert "accuracy" in val_metrics
    assert "macro_f1" in val_metrics

    # 3. Test evaluation
    eval_res = trainer.evaluate(test_loader)
    assert "accuracy" in eval_res
    assert "macro_precision" in eval_res
    assert "macro_recall" in eval_res
    assert "macro_f1" in eval_res
    assert "confusion_matrix" in eval_res
    assert eval_res["confusion_matrix"]["labels"] == ["REAL", "ATTACKED"]

    # 4. Test save metrics
    training_summary = {"epochs_trained": 1, "best_val_f1": 0.8}
    metrics_file, cm_file = trainer.save_metrics(training_summary, eval_res)
    assert metrics_file.exists()
    assert cm_file.exists()

    with open(metrics_file, "r") as f:
        saved_json = json.load(f)
        assert "test_evaluation" in saved_json
