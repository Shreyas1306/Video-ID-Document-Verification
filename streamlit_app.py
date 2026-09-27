"""
Video-Based Identity Document Verification — Streamlit Demonstration Interface
=============================================================================
A clean, professional academic demonstration UI for B.Tech project presentation.
Demonstrates multi-modal visual integrity, temporal consistency, and OCR verification.
"""

import json
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image
import streamlit as st

from src.pipeline import DocumentVerificationPipeline
from src.utils.config_loader import load_config

# =============================================================================
# Streamlit Page Setup & Custom Styling
# =============================================================================
st.set_page_config(
    page_title="Identity Document Verification System",
    page_icon="🪪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional Academic Design Styling
st.markdown(
    """
    <style>
    /* Main Layout & Font */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1280px;
    }
    
    /* Academic Disclaimer Banner */
    .disclaimer-banner {
        background-color: #fff8e6;
        border: 1px solid #ffd591;
        border-left: 5px solid #fa8c16;
        padding: 12px 18px;
        border-radius: 6px;
        font-size: 0.92rem;
        color: #873800;
        margin-bottom: 1.5rem;
        font-weight: 500;
    }

    /* Risk Badges */
    .risk-badge-low {
        background-color: #f6ffed;
        border: 1px solid #b7eb8f;
        color: #389e0d;
        padding: 8px 18px;
        border-radius: 20px;
        font-size: 1.15rem;
        font-weight: 700;
        display: inline-block;
    }
    .risk-badge-med {
        background-color: #fffbe6;
        border: 1px solid #ffe58f;
        color: #d48806;
        padding: 8px 18px;
        border-radius: 20px;
        font-size: 1.15rem;
        font-weight: 700;
        display: inline-block;
    }
    .risk-badge-high {
        background-color: #fff1f0;
        border: 1px solid #ffa39e;
        color: #cf1322;
        padding: 8px 18px;
        border-radius: 20px;
        font-size: 1.15rem;
        font-weight: 700;
        display: inline-block;
    }

    /* Metric Cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e8e8e8;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        text-align: center;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #595959;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #1f1f1f;
    }
    .metric-subtext {
        font-size: 0.78rem;
        color: #8c8c8c;
        margin-top: 4px;
    }

    /* Identity Field Cards */
    .field-card {
        background: #fafafa;
        border: 1px solid #ebebeb;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    .field-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #8c8c8c;
        text-transform: uppercase;
    }
    .field-val {
        font-size: 1.05rem;
        font-weight: 600;
        color: #262626;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# Header & Prominent Disclaimer
# =============================================================================
st.title("🪪 Video-Based Identity Document Verification System")
st.markdown(
    """
    <div class="disclaimer-banner">
        ⚠️ <strong>LEGAL & REGULATORY NOTICE:</strong> This prototype performs document integrity screening 
        based on visual, temporal, and textual consistency heuristics. It is <strong>not official identity authentication</strong> 
        and does not guarantee legal validity.
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Helper Functions
# =============================================================================
@st.cache_resource
def get_pipeline(config_path: str = "config/settings.yaml") -> DocumentVerificationPipeline:
    """Instantiate and cache the verification pipeline."""
    return DocumentVerificationPipeline(config_path=config_path)


def load_cached_results_if_exists(output_dir: Path) -> Optional[Dict[str, Any]]:
    """Check if pipeline results already exist for a given output folder."""
    res_json = output_dir / "pipeline_result.json"
    if res_json.is_file():
        try:
            with open(res_json, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def render_risk_badge(risk_level: str):
    """Render color-coded HTML risk badge."""
    level = str(risk_level).upper()
    if level == "LOW":
        return '<span class="risk-badge-low">🟢 LOW RISK (AUTHENTIC CANDIDATE)</span>'
    elif level == "MEDIUM":
        return '<span class="risk-badge-med">🟡 MEDIUM RISK (MANUAL REVIEW)</span>'
    else:
        return '<span class="risk-badge-high">🔴 HIGH RISK (POTENTIAL FORGERY)</span>'


# =============================================================================
# Sidebar Controls
# =============================================================================
with st.sidebar:
    st.header("⚙️ Verification Controls")
    
    video_source = st.radio(
        "Select Video Input:",
        options=["Pre-Loaded Sample Video", "Upload New Video"],
        index=0,
    )
    
    selected_video_path: Optional[Path] = None
    
    if video_source == "Pre-Loaded Sample Video":
        sample_options = {
            "Genuine Sample Video (Moving Mock ID)": "data/sample/sample_document_video.mp4",
        }
        choice = st.selectbox("Choose built-in test video:", list(sample_options.keys()))
        selected_video_path = Path(sample_options[choice]).resolve()
        if not selected_video_path.is_file():
            st.error(f"Sample video not found at: {selected_video_path}")
            selected_video_path = None
    else:
        uploaded_file = st.file_uploader(
            "Upload Document Video (.mp4, .mov, .avi):",
            type=["mp4", "mov", "avi"],
        )
        if uploaded_file is not None:
            # Save uploaded video to temp directory
            temp_dir = Path("data/uploads")
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_path = temp_dir / uploaded_file.name
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            selected_video_path = temp_path.resolve()
            st.success(f"Uploaded: {uploaded_file.name}")

    st.markdown("---")
    st.subheader("Model & Configuration")
    config_data = load_config("config/settings.yaml")
    st.caption(f"**YOLO Detector:** `{config_data.get('detection', {}).get('model_weights', 'yolov8n.pt')}`")
    st.caption(f"**Visual Backbone:** `{config_data.get('visual_integrity', {}).get('architecture', 'efficientnet_b0')}`")
    st.caption(f"**Sampling Rate:** `{config_data.get('frame_extraction', {}).get('target_fps', 2.5)} FPS`")
    st.caption(f"**OCR Engine:** PaddleOCR / EasyOCR Fallback")
    
    st.markdown("---")
    st.caption("B.Tech Capstone Project • Multi-Modal Consistency Analysis")


# =============================================================================
# Main Layout
# =============================================================================
if selected_video_path is None:
    st.info("👈 Please select or upload a document video in the sidebar to begin.")
    st.stop()

# Layout: Left column for Video Input, Right column for Verification Action & Executive Outcome
col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.subheader("📹 Input Video Source")
    st.video(str(selected_video_path))
    
    # Read video properties
    cap = cv2.VideoCapture(str(selected_video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / fps if fps > 0 else 0.0
    cap.release()
    
    st.caption(
        f"**File:** `{selected_video_path.name}` | "
        f"**Resolution:** {width}×{height} | "
        f"**Duration:** {duration:.1f}s ({frame_count} frames @ {fps:.1f} fps)"
    )

# Execution State & Triggers
video_stem = selected_video_path.stem
default_out_dir = Path("data/pipeline_output") / video_stem

# Check for existing results or session state
if "pipeline_result" not in st.session_state or st.session_state.get("active_video") != str(selected_video_path):
    # Check if results are on disk
    disk_result = load_cached_results_if_exists(default_out_dir)
    if disk_result:
        st.session_state["pipeline_result"] = disk_result
        st.session_state["active_video"] = str(selected_video_path)
    else:
        st.session_state["pipeline_result"] = None

with col_right:
    st.subheader("⚡ Document Verification Action")
    st.write(
        "Execute the 12-stage multi-modal pipeline combining YOLO localization, "
        "perspective rectification, CNN visual integrity, temporal feature cosine tracking, and OCR field verification."
    )
    
    btn_run = st.button("▶ Run Full Verification Pipeline", type="primary", use_container_width=True)

    if btn_run:
        with st.spinner("Executing 12-stage verification pipeline..."):
            pipeline = get_pipeline("config/settings.yaml")
            t_start = time.time()
            res = pipeline.run(
                video_path=selected_video_path,
                output_dir=default_out_dir,
            )
            st.session_state["pipeline_result"] = res
            st.session_state["active_video"] = str(selected_video_path)
            st.success(f"Verification completed in {time.time() - t_start:.2f} seconds!")

    # Display Executive Summary if results available
    pipeline_res = st.session_state.get("pipeline_result")
    if pipeline_res:
        risk_level = pipeline_res.get("risk_level", "UNKNOWN")
        fused_score = pipeline_res.get("integrity_score", 0.0)
        cov = pipeline_res.get("evidence_coverage", 1.0)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(render_risk_badge(risk_level), unsafe_allow_html=True)
        
        st.markdown(f"### Overall Fused Integrity Score: `{fused_score * 100:.1f}%`")
        st.progress(min(1.0, max(0.0, float(fused_score))))
        
        rec = pipeline_res.get("recommendation", "Review supporting evidence below.")
        st.info(f"**Operational Recommendation:** {rec}")


# =============================================================================
# Multi-Modal Evidence Scoreboard & Identity Fields
# =============================================================================
if st.session_state.get("pipeline_result"):
    res = st.session_state["pipeline_result"]
    evidence = res.get("evidence", {})
    
    st.markdown("---")
    st.subheader("📊 Multi-Modal Evidence Breakdown")
    
    s_vis = evidence.get("visual_integrity")
    s_temp = evidence.get("temporal_consistency")
    s_ocr = evidence.get("ocr_confidence")
    s_text = evidence.get("text_consistency")
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    
    with m_col1:
        val_str = f"{s_vis * 100:.1f}%" if s_vis is not None else "N/A"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">1. Visual Document Integrity</div>
                <div class="metric-value" style="color: #1890ff;">{val_str}</div>
                <div class="metric-subtext">EfficientNet-B0 Binary Score</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
    with m_col2:
        val_str = f"{s_temp * 100:.1f}%" if s_temp is not None else "N/A"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">2. Temporal Consistency</div>
                <div class="metric-value" style="color: #52c41a;">{val_str}</div>
                <div class="metric-subtext">Backbone Cosine Similarity</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
    with m_col3:
        val_str = f"{s_ocr * 100:.1f}%" if s_ocr is not None else "N/A"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">3. Character OCR Confidence</div>
                <div class="metric-value" style="color: #fa8c16;">{val_str}</div>
                <div class="metric-subtext">Text Extraction Reliability</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
    with m_col4:
        val_str = f"{s_text * 100:.1f}%" if s_text is not None else "N/A"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">4. Cross-Frame Text Consistency</div>
                <div class="metric-value" style="color: #722ed1;">{val_str}</div>
                <div class="metric-subtext">Multi-Frame Field Consensus</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Document Identity Fields
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📑 Extracted Identity Credentials (OCR Evidence)")
    
    ocr_fields = res.get("ocr_fields", {})
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    
    with f_col1:
        doc_num = ocr_fields.get("document_number") or "Not Available"
        st.markdown(
            f"""
            <div class="field-card">
                <div class="field-label">Document Number</div>
                <div class="field-val">{doc_num}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
    with f_col2:
        name = ocr_fields.get("name") or "Not Available"
        st.markdown(
            f"""
            <div class="field-card">
                <div class="field-label">Full Name</div>
                <div class="field-val">{name}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
    with f_col3:
        dob = ocr_fields.get("dob") or "Not Available"
        st.markdown(
            f"""
            <div class="field-card">
                <div class="field-label">Date of Birth</div>
                <div class="field-val">{dob}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
    with f_col4:
        addr = ocr_fields.get("address") or "Not Available"
        st.markdown(
            f"""
            <div class="field-card">
                <div class="field-label">Address</div>
                <div class="field-val">{addr}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# Intermediate Processing Results & Visualizations (Tabs)
# =============================================================================
if st.session_state.get("pipeline_result"):
    st.markdown("---")
    st.subheader("🔬 Intermediate Processing Artifacts & Diagnostics")
    
    artifacts = res.get("artifacts", {})
    
    tab_frames, tab_detect, tab_norm, tab_temp, tab_ocr, tab_report = st.tabs([
        "🎞️ Extracted Frames",
        "🎯 Document Detection",
        "📐 Perspective Normalization",
        "⏱️ Temporal Analysis",
        "🔍 OCR & Text Verification",
        "📋 Verification Report & JSON",
    ])
    
    # -------------------------------------------------------------------------
    # Tab 1: Extracted Frames
    # -------------------------------------------------------------------------
    with tab_frames:
        st.markdown("#### Sampled Video Frames")
        st.caption("Decoded video frames sampled at the configured frame extraction rate.")
        frames_dir = Path(artifacts.get("frames_directory", default_out_dir / "frames"))
        if frames_dir.is_dir():
            frame_files = sorted(list(frames_dir.glob("*.png")) + list(frames_dir.glob("*.jpg")))
            if frame_files:
                cols = st.columns(min(5, len(frame_files)))
                for idx, f_path in enumerate(frame_files[:10]):
                    with cols[idx % len(cols)]:
                        st.image(str(f_path), caption=f_path.name, use_container_width=True)
            else:
                st.info("No extracted frames found in directory.")
        else:
            st.info("Extracted frames directory not found.")

    # -------------------------------------------------------------------------
    # Tab 2: Document Detection
    # -------------------------------------------------------------------------
    with tab_detect:
        st.markdown("#### Document Detection & Bounding Box Localization (YOLO)")
        st.caption("YOLOv8 identity document localization and crop generation.")
        
        debug_det_dir = Path(artifacts.get("debug_detections_directory", default_out_dir / "debug_detections"))
        crops_dir = Path(artifacts.get("crops_directory", default_out_dir / "crops"))
        
        col_det1, col_det2 = st.columns(2)
        with col_det1:
            st.markdown("**Debug Detections (Bounding Box Overlays):**")
            if debug_det_dir.is_dir():
                det_files = sorted(list(debug_det_dir.glob("*.png")) + list(debug_det_dir.glob("*.jpg")))
                if det_files:
                    for d_path in det_files[:3]:
                        st.image(str(d_path), caption=d_path.name, use_container_width=True)
                else:
                    st.info("No debug detection images found.")
            else:
                st.info("No debug detections available.")

        with col_det2:
            st.markdown("**Cropped Document Regions:**")
            if crops_dir.is_dir():
                crop_files = sorted(list(crops_dir.glob("*.png")) + list(crops_dir.glob("*.jpg")))
                if crop_files:
                    for c_path in crop_files[:3]:
                        st.image(str(c_path), caption=c_path.name, use_container_width=True)
                else:
                    st.info("No document crops found.")
            else:
                st.info("No crops available.")

    # -------------------------------------------------------------------------
    # Tab 3: Perspective Normalization
    # -------------------------------------------------------------------------
    with tab_norm:
        st.markdown("#### Four-Corner Perspective Correction & Rectification")
        st.caption("Homography-based transform mapping tilted document cards into canonical rectangular dimensions (640×400).")
        
        norm_dir = Path(artifacts.get("normalized_directory", default_out_dir / "normalized"))
        debug_persp_dir = Path(artifacts.get("debug_perspective_directory", default_out_dir / "debug_perspective"))
        
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown("**Side-by-Side Perspective Rectification Debug:**")
            if debug_persp_dir.is_dir():
                persp_files = sorted(list(debug_persp_dir.glob("*.png")) + list(debug_persp_dir.glob("*.jpg")))
                if persp_files:
                    for pf in persp_files[:3]:
                        st.image(str(pf), caption=pf.name, use_container_width=True)
                else:
                    st.info("No perspective comparison debug images found.")
            else:
                st.info("Perspective debug directory not found.")

        with col_p2:
            st.markdown("**Normalized Document Frames:**")
            if norm_dir.is_dir():
                norm_files = sorted(list(norm_dir.glob("*.png")) + list(norm_dir.glob("*.jpg")))
                if norm_files:
                    for nf in norm_files[:3]:
                        st.image(str(nf), caption=nf.name, use_container_width=True)
                else:
                    st.info("No normalized frames found.")
            else:
                st.info("Normalized frames directory not found.")

    # -------------------------------------------------------------------------
    # Tab 4: Temporal Analysis
    # -------------------------------------------------------------------------
    with tab_temp:
        st.markdown("#### Temporal Feature Consistency Analysis")
        st.caption("Cosine similarity of visual feature embeddings extracted across consecutive frames via CNN backbone.")
        
        temp_img_path = Path(artifacts.get("debug_temporal_visualization", default_out_dir / "debug_temporal_consistency.png"))
        if temp_img_path.is_file():
            st.image(str(temp_img_path), caption="Consecutive Frame Cosine Similarity Curve", use_container_width=True)
        else:
            st.info("Temporal consistency debug visualization not available for this run.")

    # -------------------------------------------------------------------------
    # Tab 5: OCR & Text Verification
    # -------------------------------------------------------------------------
    with tab_ocr:
        st.markdown("#### OCR Extraction & Consensus Verification")
        st.caption("PaddleOCR / EasyOCR character recognition and regex pattern matching.")
        
        ocr_summary_path = default_out_dir / "ocr_verification_summary.json"
        if ocr_summary_path.is_file():
            try:
                with open(ocr_summary_path, "r", encoding="utf-8") as f:
                    ocr_data = json.load(f)
                
                col_o1, col_o2 = st.columns(2)
                with col_o1:
                    st.markdown("**Consensus Identity Fields:**")
                    st.json(ocr_data.get("fields", {}))
                
                with col_o2:
                    st.markdown("**Field Presence & Format Validation:**")
                    st.json(ocr_data.get("field_presence", {}))
                    
                st.markdown("**Multi-Frame Text Consistency Score:**")
                st.write(f"Consistency: **{ocr_data.get('text_consistency', 0.0):.4f}** across evaluated frames.")
            except Exception as e:
                st.warning(f"Could not load OCR summary JSON: {e}")
        else:
            st.info("OCR summary JSON not found.")

    # -------------------------------------------------------------------------
    # Tab 6: Verification Report & JSON
    # -------------------------------------------------------------------------
    with tab_report:
        st.markdown("#### Comprehensive Verification Report")
        
        report_path = Path(artifacts.get("verification_report_txt", default_out_dir / "verification_report.txt"))
        if report_path.is_file():
            report_text = report_path.read_text(encoding="utf-8")
            st.text_area("Generated Human-Readable Audit Report:", report_text, height=350)
            
            st.download_button(
                "📥 Download Human-Readable Verification Report (.txt)",
                data=report_text,
                file_name=f"verification_report_{video_stem}.txt",
                mime="text/plain",
            )
        else:
            st.info("Verification report file not found.")
            
        st.markdown("#### Structured JSON Output")
        st.json(res)
        
        st.download_button(
            "📥 Download Structured Result (.json)",
            data=json.dumps(res, indent=2),
            file_name=f"pipeline_result_{video_stem}.json",
            mime="application/json",
        )

        # Stage Timing Table
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Stage Processing Durations")
        stage_timings = res.get("stage_timings_seconds", {})
        if stage_timings:
            st.table([{"Pipeline Stage": k, "Duration (Seconds)": f"{v:.4f}s"} for k, v in stage_timings.items()])
