"""
Evaluation Subsystem
====================
Provides evaluation harnesses, metrics calculation, and visualization tools
for empirical benchmarking of Single-Image, Visual+Temporal, and Multi-Modal
Document Verification experiments.
"""

from src.evaluation.evaluator import (
    EvaluationMetrics,
    ExperimentResult,
    calculate_evaluation_metrics,
    export_results_csv,
    plot_confusion_matrix,
    plot_roc_comparison,
    run_experiment_a,
    run_experiment_b,
    run_experiment_c,
)

__all__ = [
    "EvaluationMetrics",
    "ExperimentResult",
    "calculate_evaluation_metrics",
    "export_results_csv",
    "plot_confusion_matrix",
    "plot_roc_comparison",
    "run_experiment_a",
    "run_experiment_b",
    "run_experiment_c",
]
