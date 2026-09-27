"""
OCR Engine Module
=================
Provides text extraction and line-level confidence scoring from document images.

Phase 4 Core Component:
- Supports PaddleOCR as primary OCR engine.
- Includes automatic fallback to EasyOCR if PaddleOCR native libraries are blocked
  by local OS execution policies (e.g. Windows Application Control on C-extensions).
- Returns normalized output structure: (bounding_box, text, confidence).
- Computes aggregate OCR confidence for evidence fusion.

IMPORTANT DISCLAIMER:
OCR results are solely an evidence source and must NEVER be treated or presented
as definitive proof of document authenticity.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.utils.config_loader import load_config


class OCREngine:
    """
    Unified OCR wrapper supporting PaddleOCR with EasyOCR fallback.
    """

    def __init__(
        self,
        engine_name: Optional[str] = None,
        languages: Optional[List[str]] = None,
        use_gpu: Optional[bool] = None,
        min_confidence: Optional[float] = None,
        config_path: Optional[Union[str, Path]] = None,
    ):
        try:
            cfg = load_config(config_path)
            ocr_cfg = cfg.get("ocr", {})
        except Exception:
            ocr_cfg = {}

        self.engine_preference = engine_name or ocr_cfg.get("engine", "paddleocr")
        self.languages = languages or ocr_cfg.get("languages", ["en"])
        self.use_gpu = use_gpu if use_gpu is not None else ocr_cfg.get("use_gpu", False)
        self.min_confidence = (
            min_confidence if min_confidence is not None else ocr_cfg.get("confidence_threshold", 0.5)
        )

        self.active_engine = None
        self.engine_instance = None
        self._init_engine()

    def _init_engine(self):
        """Initialize the requested OCR engine with fallback support."""
        if self.engine_preference.lower() == "paddleocr":
            try:
                # Configure non-interactive backend safely before loading PaddleOCR
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                except Exception:
                    pass

                from paddleocr import PaddleOCR
                self.engine_instance = PaddleOCR(
                    use_angle_cls=True,
                    lang="en",
                    use_gpu=self.use_gpu,
                    show_log=False,
                )
                self.active_engine = "paddleocr"
                return
            except Exception as e:
                # Log PaddleOCR limitation (e.g. Windows AppLocker policy on paddlex/matplotlib DLLs)
                print(
                    f"[WARNING] PaddleOCR initialization failed ({e}). "
                    f"Falling back to EasyOCR as secondary engine."
                )

        # Fallback / Direct EasyOCR
        try:
            import easyocr
            self.engine_instance = easyocr.Reader(
                self.languages,
                gpu=self.use_gpu,
                verbose=False,
            )
            self.active_engine = "easyocr"
        except Exception as e2:
            raise RuntimeError(
                f"Failed to initialize OCR engines (PaddleOCR failed, EasyOCR fallback failed: {e2})"
            ) from e2

    def extract_text(
        self,
        image: np.ndarray,
        frame_id: Optional[Union[str, int]] = None,
    ) -> Dict[str, Any]:
        """
        Run OCR on an image and return structured text lines with bounding boxes and confidences.

        Args:
            image: BGR document image (H, W, C).
            frame_id: Optional frame identifier.

        Returns:
            Dictionary containing:
                - frame_id: str or int
                - engine_used: str ("paddleocr" | "easyocr")
                - lines: List of dicts with {"text": str, "confidence": float, "box": list}
                - full_text: str (concatenated newline-separated text)
                - average_confidence: float [0.0, 1.0]
                - line_count: int

        Raises:
            ValueError: If image is empty or invalid.
        """
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Invalid or empty image supplied to OCR engine.")

        detected_lines: List[Dict[str, Any]] = []

        if self.active_engine == "paddleocr":
            try:
                # PaddleOCR returns [[[box, (text, conf)], ...]]
                results = self.engine_instance.ocr(image, cls=True)
                if results and len(results) > 0 and results[0] is not None:
                    for line_item in results[0]:
                        box = line_item[0]
                        text, conf = line_item[1]
                        conf = float(conf)
                        if conf >= self.min_confidence:
                            detected_lines.append({
                                "text": str(text).strip(),
                                "confidence": round(conf, 4),
                                "box": [[int(p[0]), int(p[1])] for p in box],
                            })
            except Exception as exc:
                print(f"[ERROR] PaddleOCR inference error: {exc}")

        elif self.active_engine == "easyocr":
            try:
                # EasyOCR returns list of (box, text, confidence)
                results = self.engine_instance.readtext(image)
                for item in results:
                    box, text, conf = item
                    conf = float(conf)
                    if conf >= self.min_confidence:
                        detected_lines.append({
                            "text": str(text).strip(),
                            "confidence": round(conf, 4),
                            "box": [[int(p[0]), int(p[1])] for p in box],
                        })
            except Exception as exc:
                print(f"[ERROR] EasyOCR inference error: {exc}")

        # Compute average confidence across valid text lines
        if detected_lines:
            avg_conf = round(float(np.mean([l["confidence"] for l in detected_lines])), 4)
            full_text = "\n".join([l["text"] for l in detected_lines])
        else:
            avg_conf = 0.0
            full_text = ""

        return {
            "frame_id": frame_id,
            "engine_used": self.active_engine,
            "lines": detected_lines,
            "full_text": full_text,
            "average_confidence": avg_conf,
            "line_count": len(detected_lines),
        }


def run_text_verification_pipeline(
    normalized_frames_or_paths: Union[str, Path, List[Union[str, Path]]],
    output_summary_dir: Optional[Union[str, Path]] = None,
    engine_name: Optional[str] = None,
    config_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Run Phase 4: OCR, field extraction, format validation, and multi-frame consistency analysis.

    Args:
        normalized_frames_or_paths: Folder or list of normalized document frame images.
        output_summary_dir: Directory where ocr_verification_summary.json will be saved.
        engine_name: Preferred OCR engine ("paddleocr" | "easyocr").
        config_path: Path to YAML configuration file.

    Returns:
        Structured JSON dictionary containing consensus fields, presence, validity,
        OCR confidence, and multi-frame text consistency score.
    """
    import cv2
    import json
    from src.text.field_parser import FieldParser
    from src.text.text_consistency import TextConsistencyAnalyzer

    # Collect normalized image paths
    frame_files: List[Path] = []
    if isinstance(normalized_frames_or_paths, (str, Path)):
        p = Path(normalized_frames_or_paths).resolve()
        if p.is_dir():
            frame_files = sorted(
                [f for f in p.glob("*.*") if f.suffix.lower() in [".png", ".jpg", ".jpeg"]]
            )
        elif p.is_file():
            frame_files = [p]
    else:
        frame_files = [Path(f).resolve() for f in normalized_frames_or_paths]

    if not frame_files:
        raise ValueError(f"No normalized frame images found at: {normalized_frames_or_paths}")

    ocr_engine = OCREngine(engine_name=engine_name, config_path=config_path)
    field_parser = FieldParser(config_path=config_path)
    consistency_analyzer = TextConsistencyAnalyzer()

    per_frame_extractions: List[Dict[str, Any]] = []
    per_frame_full_reports: List[Dict[str, Any]] = []
    confidences: List[float] = []

    for frame_path in frame_files:
        frame_id = frame_path.stem
        img = cv2.imread(str(frame_path))

        if img is None:
            continue

        ocr_res = ocr_engine.extract_text(image=img, frame_id=frame_id)
        parsed = field_parser.parse_fields(raw_text=ocr_res["full_text"], lines=ocr_res["lines"])

        confidences.append(ocr_res["average_confidence"])
        per_frame_extractions.append(parsed)

        per_frame_full_reports.append({
            "frame_id": frame_id,
            "file": frame_path.name,
            "engine_used": ocr_res["engine_used"],
            "ocr_confidence": ocr_res["average_confidence"],
            "extracted_fields": parsed,
            "raw_text_lines": [l["text"] for l in ocr_res["lines"]],
        })

    mean_ocr_conf = float(np.mean(confidences)) if confidences else 0.0

    # Cross-frame consensus and text consistency analysis
    summary = consistency_analyzer.analyze_cross_frame_consistency(
        frame_extractions=per_frame_extractions,
        overall_ocr_confidence=mean_ocr_conf,
    )

    summary["engine_used"] = ocr_engine.active_engine
    summary["per_frame_reports"] = per_frame_full_reports

    # Save to disk
    out_dir = Path(output_summary_dir or "outputs/frames").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_file = out_dir / "ocr_verification_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
