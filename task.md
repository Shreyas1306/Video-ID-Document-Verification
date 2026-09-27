# Task Tracker

## Project Environment & Scaffolding
- `[x]` Create folder structure and all `__init__.py` files
- `[x]` Configure virtual environment and verify all core dependencies
- `[x]` Unified `config.yaml` for all phases

## Phase 1: Video Input & Frame Extraction
- `[x]` `frame_extractor.py` — OpenCV video decoding, configurable sampling rate
- `[x]` Save extracted frames and metadata (FPS, duration, total frames)
- `[x]` Graceful error handling for missing/unreadable videos
- `[x]` Unit tests (`tests/test_frame_extractor.py`)

## Phase 2: Document Detection
- `[x]` `document_detector.py` — YOLO-based document localization
- `[x]` `document_cropper.py` — bounding-box cropping and debug visualizer
- `[x]` Detection summary JSON output
- `[x]` Unit tests (`tests/test_document_detector.py`)

## Phase 3: Document Cropping & Perspective Correction
- `[x]` `perspective_corrector.py` — contour estimation, 4-point transform, homography
- `[x]` Configurable output dimensions (e.g., 640x400)
- `[x]` Debug side-by-side comparison images
- `[x]` Unit tests (`tests/test_perspective_corrector.py`)

## Phase 4: OCR & Text Verification
- `[x]` `ocr_engine.py` — Primary OCR engine with graceful fallback
- `[x]` `field_parser.py` — Extraction for Name, DOB, Document Number, Address
- `[x]` `text_consistency.py` — Cross-frame textual consistency scoring
- `[x]` Unit tests (`tests/test_ocr_engine.py`)

## Phase 5: Visual Document Integrity Classification
- `[x]` `dataset.py` — Document dataset loader with document-level train/val/test splits
- `[x]` `visual_integrity.py` — EfficientNet binary classifier (REAL vs ATTACKED)
- `[x]` `trainer.py` — Training loop, metrics, confusion matrix, checkpoint saving
- `[x]` Unit tests (`tests/test_visual_dataset.py`, `tests/test_visual_integrity.py`)

## Phase 6: Temporal Consistency Analysis
- `[x]` `feature_extractor.py` — Embedding extraction from trained CNN backbone
- `[x]` `temporal_consistency.py` — Cosine similarity tracking across consecutive frames
- `[x]` Debug temporal chart generation
- `[x]` Unit tests (`tests/test_temporal_consistency.py`)

## Phase 7: Evidence Fusion
- `[x]` `score_fusion.py` — Modular weighted multi-modal evidence fusion
- `[x]` Dynamic missing-evidence weight re-normalization
- `[x]` Unit tests (`tests/test_fusion.py`)

## Phase 8: Risk Assessment & Reporting
- `[x]` `risk_classifier.py` — LOW / MEDIUM / HIGH heuristic risk classification
- `[x]` Verification report generator with mandatory non-authentication legal disclaimer
- `[x]` Unit tests for boundary conditions & missing evidence

## Phase 9: End-to-End Verification Pipeline
- `[x]` `src/pipeline.py` — Main 12-stage pipeline orchestrator (`DocumentVerificationPipeline`)
- `[x]` `src/report/report_generator.py` — Detailed human-readable report formatting
- `[x]` `scripts/run_pipeline.py` — Complete CLI entry point
- `[x]` Per-stage timing and non-silent error logging
- `[x]` Pipeline unit tests (`tests/test_pipeline.py`)
- `[x]` Real execution and verification on sample document video

## Phase 10: Experimental Evaluation
- `[x]` `src/evaluation/evaluator.py` — Metric calculation engine (Accuracy, Prec, Rec, F1, ROC-AUC, FPR, FNR)
- `[x]` Document/video level leak-free test split audit (zero train/val overlap)
- `[x]` Experiment A: Single Image $\to$ Visual Model execution
- `[x]` Experiment B: Video $\to$ Visual + Temporal Analysis execution
- `[x]` Experiment C: Video $\to$ Visual + Temporal + OCR $\to$ Evidence Fusion execution
- `[x]` Confusion matrices and ROC comparison curve visualizations
- `[x]` Exported comparative CSV (`outputs/evaluation/experiment_comparison.csv`) and JSON summary
- `[x]` Evaluation unit tests (`tests/test_evaluation.py`)
- `[x]` Academic honesty & dataset limitations analysis

## Phase 11: Streamlit Demonstration Interface
- `[x]` `streamlit_app.py` — Clean, academic B.Tech project presentation interface
- `[x]` Prominent non-authentication legal notice banner
- `[x]` Video upload (.mp4) and built-in sample video selector
- `[x]` Fused integrity score gauge & LOW/MEDIUM/HIGH risk badge
- `[x]` Multi-modal evidence scoreboard (Visual, Temporal, OCR, Text Consistency)
- `[x]` Extracted identity credential cards with format validations
- `[x]` 6 intermediate diagnostic tabs (Frames, Detection, Perspective, Temporal, OCR, Report/JSON)
- `[x]` Automated Streamlit tests (`tests/test_streamlit_app.py`)
- `[x]` Running locally on `http://localhost:8501`
