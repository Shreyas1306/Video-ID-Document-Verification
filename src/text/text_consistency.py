"""
Text Consistency Analyzer Module
================================
Evaluates cross-frame consensus and textual consistency across sequential video frames.

Phase 4 Core Component:
- Compares extracted fields across multiple frames.
- Computes pairwise string similarity metrics (Levenshtein / normalized token similarity).
- Derives consensus field values using majority voting.
- Produces final structured verification output with field presence and format validation.
"""

import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def compute_string_similarity(s1: Optional[str], s2: Optional[str]) -> float:
    """
    Compute normalized similarity ratio between two strings in [0.0, 1.0].
    Handles minor OCR variations (case, whitespace, punctuation) gracefully.
    """
    if s1 is None and s2 is None:
        return 1.0
    if s1 is None or s2 is None:
        return 0.0
    # Clean whitespace and case
    clean1 = " ".join(s1.strip().lower().split())
    clean2 = " ".join(s2.strip().lower().split())
    if clean1 == clean2:
        return 1.0
    # Alphanumeric normalization to absorb minor OCR punctuation/delimiter artifacts
    norm1 = re.sub(r"[^\w]", "", clean1)
    norm2 = re.sub(r"[^\w]", "", clean2)
    if norm1 and norm1 == norm2:
        return 1.0
    return SequenceMatcher(None, clean1, clean2).ratio()


class TextConsistencyAnalyzer:
    """
    Computes cross-frame field agreement and consensus values.
    """

    def __init__(self, key_fields: Optional[List[str]] = None):
        self.key_fields = key_fields or ["name", "dob", "document_number", "address"]

    def analyze_cross_frame_consistency(
        self,
        frame_extractions: List[Dict[str, Any]],
        overall_ocr_confidence: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Evaluate field consistency across multiple frames and select consensus values.

        Args:
            frame_extractions: List of parsed field dictionaries from individual frames.
            overall_ocr_confidence: Mean OCR confidence across frames.

        Returns:
            Structured dictionary matching the Phase 4 specification:
                - name: str or None
                - dob: str or None
                - document_number: str or None
                - address: str or None
                - ocr_confidence: float [0.0, 1.0]
                - field_presence: Dict[str, bool]
                - format_validity: Dict[str, bool]
                - text_consistency: float [0.0, 1.0]
                - per_field_consistency: Dict[str, float]
                - consensus_frequencies: Dict[str, int]
                - disclaimer: str
        """
        if not frame_extractions:
            return {
                "name": None,
                "dob": None,
                "document_number": None,
                "address": None,
                "ocr_confidence": 0.0,
                "field_presence": {k: False for k in self.key_fields},
                "format_validity": {k: False for k in self.key_fields},
                "text_consistency": 0.0,
                "disclaimer": "OCR is an evidence source only and does not prove document authenticity.",
            }

        consensus_fields: Dict[str, Optional[str]] = {}
        per_field_consistency: Dict[str, float] = {}
        consensus_frequencies: Dict[str, int] = {}
        presence_acc: Dict[str, bool] = {}
        validity_acc: Dict[str, bool] = {}

        for field in self.key_fields:
            # Collect all non-empty values for this field across frames
            values = [
                f[field]
                for f in frame_extractions
                if f.get(field) is not None and str(f[field]).strip() != ""
            ]

            if not values:
                consensus_fields[field] = None
                per_field_consistency[field] = 0.0
                consensus_frequencies[field] = 0
                presence_acc[field] = False
                validity_acc[field] = False
                continue

            # Field was present in at least one frame
            presence_acc[field] = True

            # Majority voting for consensus value
            counter = Counter(values)
            best_val, freq = counter.most_common(1)[0]
            consensus_fields[field] = best_val
            consensus_frequencies[field] = freq

            # Format validity from majority value
            # Check if any frame validated it
            validity_acc[field] = any(
                f.get("format_validity", {}).get(field, False) for f in frame_extractions
            )

            # Compute pairwise consistency across frames
            if len(values) == 1:
                per_field_consistency[field] = 1.0
            else:
                pair_scores = []
                for i in range(len(values)):
                    for j in range(i + 1, len(values)):
                        pair_scores.append(compute_string_similarity(values[i], values[j]))
                per_field_consistency[field] = (
                    round(float(sum(pair_scores) / len(pair_scores)), 4) if pair_scores else 1.0
                )

        # Global text consistency is average of present fields
        present_consistencies = [
            per_field_consistency[k] for k in self.key_fields if presence_acc[k]
        ]
        global_consistency = (
            round(float(sum(present_consistencies) / len(present_consistencies)), 4)
            if present_consistencies
            else 0.0
        )

        total_frames = len(frame_extractions)
        field_presence_rate = {
            k: round(consensus_frequencies.get(k, 0) / total_frames, 4) if total_frames > 0 else 0.0
            for k in self.key_fields
        }
        stable_fields = [
            k for k in self.key_fields if presence_acc.get(k, False) and per_field_consistency.get(k, 0.0) >= 0.85
        ]
        problematic_fields = [
            k for k in self.key_fields if (not presence_acc.get(k, False)) or per_field_consistency.get(k, 0.0) < 0.70
        ]

        return {
            "name": consensus_fields.get("name"),
            "dob": consensus_fields.get("dob"),
            "document_number": consensus_fields.get("document_number"),
            "address": consensus_fields.get("address"),
            "ocr_confidence": round(float(overall_ocr_confidence), 4),
            "field_presence": presence_acc,
            "field_presence_rate": field_presence_rate,
            "format_validity": validity_acc,
            "text_consistency": global_consistency,
            "per_field_consistency": per_field_consistency,
            "stable_fields": stable_fields,
            "problematic_fields": problematic_fields,
            "consensus_frequencies": consensus_frequencies,
            "frames_analyzed": total_frames,
            "disclaimer": (
                "IMPORTANT NOTICE: OCR extraction and consistency scoring are solely evidence sources "
                "for visual-text alignment and do NOT verify or guarantee official document authenticity."
            ),
        }

