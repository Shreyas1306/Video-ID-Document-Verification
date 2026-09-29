# Video-Based Identity Document Verification

**Using Visual and Textual Consistency Analysis**

> B.Tech Deep Learning Course Project

## Overview

A course-scale prototype that accepts a short smartphone video of an identity
document (while the user slowly moves or tilts it) and produces a structured
verification report. The system processes multiple frames and analyses the
document through two parallel pipelines:

1. **Visual & Temporal** — deep feature extraction (frozen pretrained backbone),
   per-frame integrity scoring, and cross-frame temporal consistency via cosine
   similarity.
2. **OCR & Textual** — field extraction, format validation, and cross-frame
   text agreement. OCR is an **evidence source only**, never proof of
   authenticity.

The evidence from both branches is fused via a **configurable weighted average**
into a single risk score (Low / Medium / High) with supporting evidence.

> **⚠️ Disclaimer:** This is an academic prototype for educational purposes
> only. It does **not** provide official identity verification and does **not**
> claim guaranteed document authenticity. All experiments use public datasets
> and synthetic/mock documents. No real people's identity documents are used.

## Key Technical Decisions

| Decision | Rationale |
|---|---|
| **OpenCV contour detection** (not YOLO) for document localization | COCO-pretrained YOLO has no identity-document class. Contour-based detection is the baseline; custom YOLO can be added later. |
| **Frozen pretrained backbone** (EfficientNet-B0) | Feature extractor only. No fine-tuning without project-specific REAL vs ATTACKED training data. |
| **Binary classification** (REAL vs ATTACKED) | Simpler initial scope. Multiple attack categories deferred. |
| **Cosine similarity** for temporal analysis | No LSTM or video transformers initially. Frame-level embeddings compared pairwise. |
| **Weighted average** for evidence fusion | No learned classifier until sufficient labelled data exists. |
| **Synthetic/mock document format** | Controlled format for reliable testing. Not attempting arbitrary real-world document types. |
| **OCR as evidence only** | OCR results contribute to scoring but are never presented as proof of authenticity. |

## Pipeline

```
Video Input
  → Frame Extraction (OpenCV)
  → Document Detection (OpenCV contour/quad)
  → Document Cropping
  → Perspective Correction / Homography
  → Normalized Document Frames
  ┌──────────────────────┬──────────────────────┐
  │  Visual & Temporal   │   OCR & Textual      │
  │  ─ Feature Extract   │   ─ PaddleOCR        │
  │    (frozen backbone) │   ─ Field Parsing    │
  │  ─ Visual Integrity  │   ─ Text Consistency  │
  │    (REAL vs ATTACKED)│   (evidence only)     │
  │  ─ Temporal Consist. │                       │
  │    (cosine sim.)     │                       │
  └──────────┬───────────┴──────────┬───────────┘
             └── Evidence Fusion ───┘
                  (weighted avg)
                       │
              Risk Classification
             (Low / Medium / High)
                       │
            Verification Report
```

## Project Structure

```
├── config/              # Global configuration (thresholds, model names)
├── data/                # Raw videos, samples, synthetic docs (gitignored)
├── models/weights/      # Model weights (gitignored)
├── src/                 # Source code
│   ├── preprocessing/   # Video → normalized document crops
│   ├── visual/          # Visual & temporal analysis
│   ├── text/            # OCR & textual analysis
│   ├── fusion/          # Evidence fusion & risk scoring
│   ├── report/          # Verification report generation
│   └── utils/           # Shared helpers
├── notebooks/           # Jupyter exploration notebooks
├── tests/               # Unit & integration tests
├── scripts/             # CLI entry points & verification
└── app/                 # Streamlit demo UI (Phase 4)
```

## Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd Video-ID-Document-Verification

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify the environment
python scripts/verify_environment.py
```

## Usage

### CLI Pipeline (after implementation)
```bash
python scripts/run_pipeline.py --video data/sample/sample_video.mp4
```

### Streamlit Demo
```bash
streamlit run streamlit_app.py
```

### Run Tests
```bash
pytest
```

## Tech Stack

| Component | Library |
|---|---|
| Video / Image Processing | OpenCV |
| Document Detection | OpenCV (contour-based) |
| Visual Features | PyTorch + torchvision (EfficientNet-B0, frozen) |
| OCR | PaddleOCR |
| Fusion / Classification | scikit-learn, NumPy |
| Demo UI | Streamlit |

## License

This project is for academic/educational use only.
