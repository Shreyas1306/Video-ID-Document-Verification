"""
Unit & Integration Tests for Evaluation Subsystem
==================================================
Tests metrics calculation, FPR/FNR boundary conditions, experiment runners,
confusion matrix plotting, and CSV generation.
"""

from pathlib import Path
import tempfile
import cv2
import numpy as np
import pytest

from src.evaluation.evaluator import (
    EvaluationMetrics,
    ExperimentResult,
    calculate_evaluation_metrics,
    export_results_csv,
    group_samples_by_document,
    plot_confusion_matrix,
    plot_roc_comparison,
    run_experiment_a,
    run_experiment_b,
)


def test_calculate_metrics_perfect():
    """Verify metrics for perfect classification."""
    y_true = [0, 0, 0, 1, 1, 1]
    y_pred = [0, 0, 0, 1, 1, 1]
    y_scores = [0.1, 0.05, 0.2, 0.9, 0.85, 0.95]

    metrics = calculate_evaluation_metrics(y_true, y_pred, y_scores)

    assert metrics.accuracy == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1_score == 1.0
    assert metrics.false_positive_rate == 0.0
    assert metrics.false_negative_rate == 0.0
    assert metrics.true_positives == 3
    assert metrics.false_positives == 0
    assert metrics.true_negatives == 3
    assert metrics.false_negatives == 0
    assert metrics.roc_auc == 1.0


def test_calculate_metrics_inverted():
    """Verify metrics when all predictions are completely inverted."""
    y_true = [0, 0, 1, 1]
    y_pred = [1, 1, 0, 0]
    y_scores = [0.9, 0.8, 0.1, 0.2]

    metrics = calculate_evaluation_metrics(y_true, y_pred, y_scores)

    assert metrics.accuracy == 0.0
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1_score == 0.0
    assert metrics.false_positive_rate == 1.0
    assert metrics.false_negative_rate == 1.0
    assert metrics.true_positives == 0
    assert metrics.false_positives == 2
    assert metrics.true_negatives == 0
    assert metrics.false_negatives == 2
    assert metrics.roc_auc == 0.0


def test_calculate_metrics_mixed_and_rates():
    """
    Verify exact FPR and FNR calculation with mixed results:
      - 3 REAL (0): 2 predicted REAL (TN=2), 1 predicted ATTACKED (FP=1)
      - 2 ATTACKED (1): 1 predicted ATTACKED (TP=1), 1 predicted REAL (FN=1)
    """
    y_true = [0, 0, 0, 1, 1]
    y_pred = [0, 0, 1, 1, 0]
    y_scores = [0.2, 0.3, 0.7, 0.8, 0.4]

    metrics = calculate_evaluation_metrics(y_true, y_pred, y_scores)

    assert metrics.total_samples == 5
    assert metrics.true_negatives == 2
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.true_positives == 1

    # FPR = FP / (FP + TN) = 1 / (1 + 2) = 0.3333
    assert pytest.approx(metrics.false_positive_rate, abs=1e-4) == 0.3333
    # FNR = FN / (FN + TP) = 1 / (1 + 1) = 0.5000
    assert pytest.approx(metrics.false_negative_rate, abs=1e-4) == 0.5000
    # Precision = TP / (TP + FP) = 1 / 2 = 0.5
    assert metrics.precision == 0.5
    # Recall = TP / (TP + FN) = 1 / 2 = 0.5
    assert metrics.recall == 0.5
    # F1 = 0.5
    assert metrics.f1_score == 0.5
    # Accuracy = 3 / 5 = 0.6
    assert metrics.accuracy == 0.6


def test_empty_and_mismatch_inputs():
    """Verify empty input returns zeroed metrics and length mismatch raises error."""
    empty_m = calculate_evaluation_metrics([], [])
    assert empty_m.total_samples == 0
    assert empty_m.accuracy == 0.0
    assert empty_m.roc_auc is None

    with pytest.raises(ValueError, match="Length mismatch"):
        calculate_evaluation_metrics([1, 0], [1])


def test_group_samples_by_document():
    """Verify document-level frame aggregation."""
    samples = [
        {"path": "data/docA_f02.png", "label": 0, "class_name": "REAL", "doc_id": "docA"},
        {"path": "data/docA_f01.png", "label": 0, "class_name": "REAL", "doc_id": "docA"},
        {"path": "data/docB_f01.png", "label": 1, "class_name": "ATTACKED", "doc_id": "docB"},
    ]

    grouped = group_samples_by_document(samples)
    assert len(grouped) == 2

    doc_a = next(d for d in grouped if d["doc_id"] == "docA")
    assert doc_a["label"] == 0
    assert len(doc_a["paths"]) == 2
    # Check ordering
    assert "docA_f01" in doc_a["paths"][0]
    assert "docA_f02" in doc_a["paths"][1]


def test_export_results_csv():
    """Verify CSV export format and contents."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "test_results.csv"

        m = EvaluationMetrics(
            accuracy=0.95,
            precision=0.90,
            recall=1.0,
            f1_score=0.9474,
            roc_auc=0.98,
            false_positive_rate=0.05,
            false_negative_rate=0.0,
            true_positives=10,
            false_positives=1,
            true_negatives=19,
            false_negatives=0,
            total_samples=30,
            confusion_matrix=[[19, 1], [0, 10]],
        )
        exp = ExperimentResult(
            experiment_id="exp_test",
            experiment_name="Test Experiment",
            description="Unit test run",
            metrics=m,
            y_true=[0, 1],
            y_pred=[0, 1],
            y_scores=[0.1, 0.9],
            sample_records=[],
            parameters={},
        )

        out = export_results_csv([exp], csv_path)
        assert out.is_file()

        content = out.read_text(encoding="utf-8")
        assert "exp_test" in content
        assert "0.9500" in content
        assert "0.9474" in content
        assert "0.0500" in content


def test_plot_generation():
    """Verify confusion matrix and ROC comparison plotting."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cm_path = Path(tmp_dir) / "cm.png"
        roc_path = Path(tmp_dir) / "roc.png"

        cm = [[15, 0], [1, 14]]
        plot_confusion_matrix(cm, save_path=cm_path)
        assert cm_path.is_file()
        assert cm_path.stat().st_size > 500

        m = EvaluationMetrics(
            accuracy=0.9, precision=0.9, recall=0.9, f1_score=0.9,
            roc_auc=0.95, false_positive_rate=0.1, false_negative_rate=0.1,
            true_positives=9, false_positives=1, true_negatives=9, false_negatives=1,
            total_samples=20, confusion_matrix=cm,
        )
        exp = ExperimentResult(
            experiment_id="exp_a",
            experiment_name="Exp A",
            description="",
            metrics=m,
            y_true=[0] * 10 + [1] * 10,
            y_pred=[0] * 9 + [1] + [0] + [1] * 9,
            y_scores=[0.1] * 9 + [0.8] + [0.3] + [0.9] * 9,
            sample_records=[],
            parameters={},
        )
        plot_roc_comparison([exp], save_path=roc_path)
        assert roc_path.is_file()
        assert roc_path.stat().st_size > 500


def test_experiment_a_and_b_mock():
    """Verify mock execution of Experiment A and Experiment B."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create two tiny dummy images
        img1 = np.full((100, 100, 3), 200, dtype=np.uint8)
        img2 = np.full((100, 100, 3), 50, dtype=np.uint8)

        p1 = tmp_path / "mock_real_f01.png"
        p2 = tmp_path / "mock_real_f02.png"
        cv2.imwrite(str(p1), img1)
        cv2.imwrite(str(p2), img2)

        samples = [
            {"path": str(p1), "label": 0, "class_name": "REAL", "doc_id": "mock_real"},
            {"path": str(p2), "label": 0, "class_name": "REAL", "doc_id": "mock_real"},
        ]

        # Test Experiment A (untrained backbone fallback)
        res_a = run_experiment_a(samples, model_checkpoint="nonexistent.pth")
        assert res_a.metrics.total_samples == 2
        assert len(res_a.y_pred) == 2

        # Test Experiment B
        seqs = group_samples_by_document(samples)
        res_b = run_experiment_b(seqs, model_checkpoint="nonexistent.pth")
        assert res_b.metrics.total_samples == 1
        assert len(res_b.y_pred) == 1
        assert res_b.sample_records[0]["temporal_score"] > 0.0


def test_evaluator_consumes_canonical_temporal_consistency_score():
    """Verify evaluator specifically consumes 'temporal_consistency_score' key without falling back to 0.0."""
    from unittest.mock import MagicMock, patch
    from src.evaluation.evaluator import run_experiment_b

    seqs = [{
        "doc_id": "test_doc_01",
        "label": 0,
        "class_name": "REAL",
        "paths": ["dummy_p1.png", "dummy_p2.png"],
    }]

    mock_classifier = MagicMock()
    mock_classifier.predict_image.return_value = {"integrity_score": 0.95}

    mock_analyzer = MagicMock()
    # Mock analyzer returning ONLY the canonical key 'temporal_consistency_score'
    mock_analyzer.analyze_sequence.return_value = {
        "status": "success",
        "temporal_consistency_score": 0.88,
    }

    with patch("src.evaluation.evaluator.DocumentIntegrityClassifier", return_value=mock_classifier), \
         patch("src.evaluation.evaluator.TemporalConsistencyAnalyzer", return_value=mock_analyzer), \
         patch("cv2.imread", return_value=np.ones((50, 50, 3), dtype=np.uint8)):
        res = run_experiment_b(
            test_sequences=seqs,
            model_checkpoint=None,
            config_path="config/settings.yaml",
            threshold=0.5,
        )

    # Verify temporal consistency score 0.88 was directly consumed, not 0.0 fallback
    assert res.sample_records[0]["temporal_score"] == 0.88
    assert res.sample_records[0]["temporal_score"] > 0.0

