"""
Video-Based Identity Document Verification — Streamlit Demonstration Interface
=============================================================================
A clean, professional academic demonstration UI for B.Tech project presentation.
Analyzes document videos using multi-modal visual integrity, temporal consistency,
and textual cross-frame agreement signals.
"""

import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image
import streamlit as st

from src.pipeline import DocumentVerificationPipeline
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)

# =============================================================================
# 1. Page Configuration & Custom Academic Styling
# =============================================================================
st.set_page_config(
    page_title="Video-Based Identity Document Verification",
    page_icon="🪪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1280px;
    }
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
        font-size: 1.65rem;
        font-weight: 700;
        color: #1f1f1f;
    }
    .metric-subtext {
        font-size: 0.78rem;
        color: #8c8c8c;
        margin-top: 4px;
    }
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
# 2. Header & Disclaimer
# =============================================================================
st.title("🪪 Video-Based Identity Document Verification")
st.markdown("**Analyze a short document video using visual, temporal, and textual consistency signals.**")

st.markdown(
    """
    <div class="disclaimer-banner">
        ⚠️ <strong>LEGAL & REGULATORY NOTICE:</strong> This prototype performs document integrity screening 
        based on visual, temporal, and textual consistency heuristics. It is <strong>not official identity authentication</strong> 
        and does not guarantee legal validity or official citizenship/identity proof.
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# 3. Helper Functions & Pipeline Cache
# =============================================================================
@st.cache_resource
def get_pipeline(config_path: str = "config/settings.yaml") -> DocumentVerificationPipeline:
    """Instantiate and cache the production verification pipeline."""
    return DocumentVerificationPipeline(config_path=config_path)


def find_cached_results(video_stem: str) -> Optional[Dict[str, Any]]:
    """Check known output locations for pre-existing run results."""
    candidate_dirs = [
        Path("outputs") / video_stem,
        Path("outputs/pipeline_runs") / video_stem,
        Path("outputs/pipeline_validation_real") / video_stem,
        Path("outputs/pipeline_validation_attacked") / video_stem,
    ]
    for d in candidate_dirs:
        res_file = d / "pipeline_result.json"
        if res_file.is_file():
            try:
                with open(res_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "verification_result" in data:
                        return data
            except Exception:
                continue
    return None


def render_risk_badge(risk_level: str) -> str:
    """Render color-coded HTML risk badge."""
    level = str(risk_level).upper()
    if level == "LOW":
        return '<span class="risk-badge-low">🟢 LOW RISK (AUTHENTIC CANDIDATE)</span>'
    elif level == "MEDIUM":
        return '<span class="risk-badge-med">🟡 MEDIUM RISK (MANUAL REVIEW REQUIRED)</span>'
    else:
        return '<span class="risk-badge-high">🔴 HIGH RISK (POTENTIAL PRESENTATION ATTACK)</span>'


# =============================================================================
# 4. Sidebar: Video Selection & Configuration
# =============================================================================
with st.sidebar:
    st.header("⚙️ Video Input Selection")

    video_mode = st.radio(
        "Choose Input Source:",
        options=["Pre-Loaded Test Video", "Upload Video File"],
        index=0,
    )

    selected_video_path: Optional[Path] = None

    if video_mode == "Pre-Loaded Test Video":
        sample_options = {}
        sample_video = Path("data/sample/sample_document_video.mp4")
        if sample_video.is_file():
            sample_options["Control Sample Video (Mock ID)"] = str(sample_video)

        attack_video = Path("outputs/validation_videos/alb_id_05_cc0001.mp4")
        if attack_video.is_file():
            sample_options["DLC-2021 Attacked Video (Color Copy Paper)"] = str(attack_video)

        if sample_options:
            choice = st.selectbox("Select pre-loaded video:", list(sample_options.keys()))
            selected_video_path = Path(sample_options[choice]).resolve()
        else:
            st.warning("No pre-loaded videos found in data/sample or outputs/validation_videos.")
    else:
        uploaded_file = st.file_uploader(
            "Upload Document Video:",
            type=["mp4", "avi", "mov", "mkv"],
            help="Accepted formats: MP4, AVI, MOV, MKV",
        )
        if uploaded_file is not None:
            upload_dir = Path("outputs/user_uploads")
            upload_dir.mkdir(parents=True, exist_ok=True)
            save_path = upload_dir / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            selected_video_path = save_path.resolve()
            st.success(f"Uploaded: {uploaded_file.name}")

    st.markdown("---")
    st.subheader("Production ML Configuration")
    cfg = load_config("config/settings.yaml")
    st.caption(f"**Visual Backbone:** `{cfg.get('visual_integrity', {}).get('backbone', 'efficientnet_b0')}`")
    st.caption(f"**Model Checkpoint:** `{cfg.get('visual_integrity', {}).get('model_checkpoint', 'models/weights/best_integrity_model.pth')}`")
    st.caption(f"**Document Detector:** `OpenCV Contour Detector (V2)`")
    st.caption(f"**OCR Engine:** `PaddleOCR / EasyOCR Fallback`")
    st.caption(f"**Sampling Rate:** `{cfg.get('frame_extraction', {}).get('max_frames', 10)} frames (uniform)`")

    st.markdown("---")
    st.caption("B.Tech Capstone Project • Multi-Modal Consistency Analysis")


# =============================================================================
# 5. Main Content Area
# =============================================================================
if selected_video_path is None:
    st.info("👈 Please select or upload a document video in the sidebar to begin verification.")
    st.stop()

col_video, col_action = st.columns([1, 1.2], gap="large")

with col_video:
    st.subheader("📹 Input Video Source")
    st.video(str(selected_video_path))

    # Inspect video metadata via OpenCV
    cap = cv2.VideoCapture(str(selected_video_path))
    v_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    v_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    v_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    v_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    v_dur = v_frames / v_fps if v_fps > 0 else 0.0
    cap.release()

    st.caption(
        f"**Filename:** `{selected_video_path.name}` | "
        f"**Resolution:** {v_w}×{v_h} | "
        f"**Length:** {v_dur:.1f}s ({v_frames} frames @ {v_fps:.1f} fps)"
    )

video_stem = selected_video_path.stem
active_out_dir = Path("outputs") / video_stem

# Load cached run if active video matches or load on demand
if "pipeline_result" not in st.session_state or st.session_state.get("active_video") != str(selected_video_path):
    if video_mode == "Pre-Loaded Test Video":
        cached = find_cached_results(video_stem)
        if cached:
            st.session_state["pipeline_result"] = cached
            st.session_state["result_source"] = "precomputed"
            st.session_state["active_video"] = str(selected_video_path)
        else:
            st.session_state["pipeline_result"] = None
            st.session_state["result_source"] = None
            st.session_state["active_video"] = str(selected_video_path)
    else:
        # Uploaded video: never preload stale cache, require explicit user verification
        st.session_state["pipeline_result"] = None
        st.session_state["result_source"] = None
        st.session_state["active_video"] = str(selected_video_path)

with col_action:
    st.subheader("⚡ Document Verification Action")
    st.write(
        "Execute the 12-stage multi-modal pipeline: frame extraction, contour localization, "
        "perspective rectification, EfficientNet visual integrity scoring, temporal cosine tracking, "
        "OCR field extraction, and evidence fusion."
    )

    btn_verify = st.button("Verify Document", type="primary", use_container_width=True)

    if btn_verify:
        with st.spinner("Executing 12-stage verification pipeline (extracting, detecting, analyzing, OCR)..."):
            t_start = time.time()
            try:
                pipeline = get_pipeline("config/settings.yaml")
                run_result = pipeline.run(
                    video_path=selected_video_path,
                    output_dir=active_out_dir,
                )
                st.session_state["pipeline_result"] = run_result
                st.session_state["result_source"] = "live"
                st.session_state["active_video"] = str(selected_video_path)
                st.success(f"Verification completed successfully in {time.time() - t_start:.2f} seconds!")
            except Exception as e:
                logger.error(f"Pipeline verification failed: {e}", exc_info=True)
                st.error(f"Verification could not be completed: {str(e)}")

    # Executive Outcome Card
    pipeline_res = st.session_state.get("pipeline_result")
    if pipeline_res:
        verif = pipeline_res.get("verification_result", {})
        risk_level = verif.get("risk_level") or pipeline_res.get("risk_level", "UNKNOWN")
        fused_score = verif.get("integrity_score") if verif.get("integrity_score") is not None else pipeline_res.get("integrity_score", 0.0)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(render_risk_badge(risk_level), unsafe_allow_html=True)

        res_source = st.session_state.get("result_source")
        if res_source == "precomputed":
            st.info(
                "ℹ️ **Status: Displaying Precomputed Validation Result** (from prior end-to-end pipeline run). "
                "Click **'Verify Document'** above to execute live 12-stage inference on this video."
            )
        elif res_source == "live":
            st.success(
                "✅ **Status: Displaying Fresh Live Pipeline Result** (executed in this interactive session)."
            )

        st.markdown(f"### Fused Integrity Score: `{float(fused_score) * 100:.1f}%`")
        st.progress(min(1.0, max(0.0, float(fused_score))))

        rec = pipeline_res.get("recommendation")
        if not rec:
            rec = "REVIEW (MANUAL REVIEW REQUIRED): Ambiguities or visual integrity anomalies detected. Manual inspection advised." if risk_level == "MEDIUM" else ("REJECT: High risk of presentation attack." if risk_level == "HIGH" else "ACCEPT: Low risk detected across visual, temporal, and text dimensions.")
        st.info(f"**Operational Decision:** {rec}")


# =============================================================================
# 6. Result Summary & Score Breakdown
# =============================================================================
if st.session_state.get("pipeline_result"):
    res = st.session_state["pipeline_result"]
    verif = res.get("verification_result", {})
    evidence = verif.get("evidence") or res.get("evidence", {})

    st.markdown("---")
    st.subheader("📊 Multi-Modal Evidence Breakdown")

    s_vis = evidence.get("visual_integrity")
    s_temp = evidence.get("temporal_consistency")
    s_ocr = evidence.get("ocr_confidence")
    s_text = evidence.get("text_consistency")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        val_str = f"{s_vis * 100:.1f}%" if s_vis is not None else "N/A"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">1. Visual Document Integrity</div>
                <div class="metric-value" style="color: #1890ff;">{val_str}</div>
                <div class="metric-subtext">EfficientNet-B0 P(REAL)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        val_str = f"{s_temp * 100:.1f}%" if s_temp is not None else "N/A"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">2. Temporal Consistency</div>
                <div class="metric-value" style="color: #52c41a;">{val_str}</div>
                <div class="metric-subtext">Cross-Frame Embedding Sim</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        val_str = f"{s_ocr * 100:.1f}%" if s_ocr is not None else "N/A"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">3. Character OCR Confidence</div>
                <div class="metric-value" style="color: #fa8c16;">{val_str}</div>
                <div class="metric-subtext">Text Extraction Quality</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
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

    # =========================================================================
    # 7. Extracted Document Information
    # =========================================================================
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📑 Extracted Document Information (OCR Evidence)")

    details = res.get("evidence_details", {})
    ocr_details = details.get("ocr_and_text", {})
    extracted_fields = ocr_details.get("extracted_fields") or res.get("ocr_fields", {})

    f1, f2, f3, f4 = st.columns(4)

    name_val = extracted_fields.get("name") or "Not detected"
    dob_val = extracted_fields.get("dob") or "Not detected"
    doc_num_val = extracted_fields.get("document_number") or "Not detected"
    addr_val = extracted_fields.get("address") or "Not detected"

    with f1:
        st.markdown(
            f"""
            <div class="field-card">
                <div class="field-label">Holder Name</div>
                <div class="field-val">{name_val}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with f2:
        st.markdown(
            f"""
            <div class="field-card">
                <div class="field-label">Date of Birth</div>
                <div class="field-val">{dob_val}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with f3:
        st.markdown(
            f"""
            <div class="field-card">
                <div class="field-label">Document Number</div>
                <div class="field-val">{doc_num_val}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with f4:
        st.markdown(
            f"""
            <div class="field-card">
                <div class="field-label">Address</div>
                <div class="field-val">{addr_val}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # =========================================================================
    # 8. Evidence & Interpretation
    # =========================================================================
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🔍 Multimodal Evidence & Technical Interpretation")

    det_info = details.get("detection", {})
    persp_info = details.get("perspective_correction", {})
    vis_info = details.get("visual_integrity", {})
    temp_info = details.get("temporal_consistency", {})

    ev_col1, ev_col2 = st.columns(2)

    with ev_col1:
        st.markdown("**1. Localization & Geometry:**")
        frames_det = det_info.get("frames_detected", "N/A")
        frames_tot = det_info.get("frames_processed", "N/A")
        det_rate = det_info.get("detection_rate", 0.0)
        st.write(f"- **Document Localization Status:** {frames_det} / {frames_tot} frames localized ({det_rate * 100:.1f}%) via OpenCV contour detector.")

        homo_count = persp_info.get("homography_success_count", "N/A")
        homo_rate = persp_info.get("homography_rate", 0.0)
        st.write(f"- **Perspective Rectification Status:** {homo_count} / {frames_tot} frames successfully rectified ({homo_rate * 100:.1f}%) via four-corner homography.")

        st.markdown("**2. Visual Print Integrity:**")
        vis_score = vis_info.get("aggregated_score", s_vis)
        vis_desc = "Genuine physical card print pattern" if (vis_score is not None and vis_score >= 0.50) else "Detected presentation-attack texture / non-standard print pattern"
        st.write(f"- **Visual Integrity Assessment:** Score = `{vis_score:.4f}` ({vis_desc}).")

    with ev_col2:
        st.markdown("**3. Temporal Motion & Embedding Stability:**")
        mean_sim = temp_info.get("mean_consecutive_similarity", "N/A")
        anom_trans = len(temp_info.get("anomalous_transitions", []))
        st.write(f"- **Temporal Consistency Result:** Mean cosine similarity = `{mean_sim}`, Anomalous frame transitions = `{anom_trans}`.")

        st.markdown("**4. Textual Verification:**")
        st.write(f"- **OCR Result:** Aggregate character confidence = `{s_ocr * 100:.1f}%` (Engine: EasyOCR fallback).")
        st.write(f"- **Text Consistency Result:** Cross-frame agreement = `{s_text * 100:.1f}%` across evaluated frame consensus.")

    # =========================================================================
    # 9. Processing Information
    # =========================================================================
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("⏱️ Processing Information")

    meta = res.get("video_metadata", {})
    total_time = res.get("total_processing_time_seconds", "N/A")
    frames_proc = meta.get("number_of_extracted_frames", det_info.get("frames_processed", 10))

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Frames Processed", f"{frames_proc} frames")
    p2.metric("Processing Latency", f"{total_time:.2f}s" if isinstance(total_time, (int, float)) else str(total_time))
    p3.metric("Localization Backend", "OpenCV Contour V2")
    p4.metric("OCR Engine", "EasyOCR (CPU Fallback)")

    # =========================================================================
    # 10. Processed Document Gallery (Visual Output)
    # =========================================================================
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🖼️ Processed Document Frames")

    artifacts = res.get("output_artifacts") or res.get("artifacts", {})
    norm_dir_str = artifacts.get("normalized_directory")
    norm_dir = Path(norm_dir_str) if norm_dir_str else active_out_dir / "normalized"

    if norm_dir.is_dir():
        norm_images = sorted(list(norm_dir.glob("*.png")) + list(norm_dir.glob("*.jpg")))
        if norm_images:
            st.caption(f"Showing representative rectified document crops ($600 \\times 400$) from `{norm_dir.name}`:")
            disp_cols = st.columns(min(4, len(norm_images)))
            for idx, img_p in enumerate(norm_images[:4]):
                with disp_cols[idx]:
                    st.image(str(img_p), caption=f"Rectified Frame {idx+1}", use_container_width=True)
        else:
            st.info("No normalized frames found in output directory.")
    else:
        st.info("Normalized frames directory not located.")

    # Temporal debug plot if available
    temp_vis_path = Path(artifacts.get("debug_temporal_visualization", active_out_dir / "debug_temporal_consistency.png"))
    if temp_vis_path.is_file():
        st.markdown("**Temporal Consistency Curve (Consecutive Frame Similarity):**")
        st.image(str(temp_vis_path), use_container_width=True)

    # =========================================================================
    # 11. Report Download
    # =========================================================================
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📥 Verification Reports & Serialization")

    d_col1, d_col2 = st.columns(2)

    # Download JSON
    with d_col1:
        st.download_button(
            label="📥 Download Structured Result (.json)",
            data=json.dumps(res, indent=2),
            file_name=f"pipeline_result_{video_stem}.json",
            mime="application/json",
            use_container_width=True,
        )

    # Download Text Report
    report_path_str = artifacts.get("verification_report_txt")
    report_file = Path(report_path_str) if report_path_str else active_out_dir / "verification_report.txt"
    report_content = ""
    if report_file.is_file():
        report_content = report_file.read_text(encoding="utf-8")

    with d_col2:
        if report_content:
            st.download_button(
                label="📥 Download Human-Readable Report (.txt)",
                data=report_content,
                file_name=f"verification_report_{video_stem}.txt",
                mime="text/plain",
                use_container_width=True,
            )
        else:
            st.button("Report (.txt) Not Available", disabled=True, use_container_width=True)

    if report_content:
        with st.expander("📄 View Human-Readable Verification Report Text", expanded=False):
            st.text(report_content)
