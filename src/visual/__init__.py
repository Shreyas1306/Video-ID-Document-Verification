"""
Visual Analysis Module
======================
Extract deep visual features and assess document integrity and temporal
consistency across multiple frames.

Submodules:
    - feature_extractor:      Pretrained CNN/ViT embedding extraction
    - visual_integrity:       Per-frame anomaly / tamper signal detection
    - temporal_consistency:   Cross-frame embedding similarity analysis
    - dataset:                Dataset loader, validation, leak-free splits, transforms
    - trainer:                Model training, checkpointing, and evaluation
"""

from src.visual.dataset import (
    CLASS_TO_LABEL,
    LABEL_TO_CLASS,
    DatasetValidationReport,
    DocumentDataset,
    collect_dataset_samples,
    create_leak_free_splits,
    extract_doc_id,
    get_document_transforms,
    validate_dataset_structure,
)
from src.visual.feature_extractor import DocumentFeatureExtractor
from src.visual.temporal_consistency import (
    TemporalConsistencyAnalyzer,
    TransitionAnomaly,
)
from src.visual.trainer import IntegrityTrainer
from src.visual.visual_integrity import (
    DocumentIntegrityClassifier,
    compute_statistical_proxy_score,
)

__all__ = [
    "DocumentFeatureExtractor",
    "TemporalConsistencyAnalyzer",
    "TransitionAnomaly",
    "DocumentIntegrityClassifier",
    "IntegrityTrainer",
    "DocumentDataset",
    "DatasetValidationReport",
    "validate_dataset_structure",
    "collect_dataset_samples",
    "create_leak_free_splits",
    "extract_doc_id",
    "get_document_transforms",
    "compute_statistical_proxy_score",
    "CLASS_TO_LABEL",
    "LABEL_TO_CLASS",
]
