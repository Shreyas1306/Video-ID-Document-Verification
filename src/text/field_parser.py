"""
Field Parser Module
===================
Extracts structured identity document fields from raw OCR output and performs
format validation.

Phase 4 Core Component:
- Extracts Name, DOB, Document Number, and Address using configurable regex rules.
- Performs field presence checks.
- Validates field formats (date structures, ID number patterns, character sets).
- Reports unextracted fields as unavailable rather than hallucinating values.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.utils.config_loader import load_config


class FieldParser:
    """
    Parses key identity fields from OCR text and validates their formatting.
    """

    def __init__(
        self,
        custom_patterns: Optional[Dict[str, str]] = None,
        config_path: Optional[Union[str, Path]] = None,
    ):
        try:
            cfg = load_config(config_path)
            field_cfg = cfg.get("field_parsing", {})
        except Exception:
            field_cfg = {}

        self.expected_fields = field_cfg.get(
            "expected_fields", ["name", "date_of_birth", "document_number", "address"]
        )

        # Configurable regex patterns for field localization (single-line horizontal space)
        # Handles common OCR separator misreadings: ':', ';', '.', '-', or spaces
        self.patterns = {
            "name": r"\b(?:NAME|Full Name|Holder)[\t :;\.\-]*([A-Za-z\t \.\-]{2,35})",
            "date_of_birth": r"\b(?:DOB|Date of Birth|Birth Date)[\t :;\.\-]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})",
            "document_number": r"\b(?:DOC\s*NO|DOCUMENT\s*NO|ID\s*NO|DOC|ID)[\t :;\.\-]+([A-Z0-9\+\-\t ]{4,30})",
            "address": r"\b(?:ADDR|Address|Add)[\t :;\.\-]*([A-Za-z0-9\t ,\.\-/#]{5,60})",
        }

        # Standalone pattern fallbacks when label prefixes are missing or blurred
        self.standalone_patterns = {
            "date_of_birth": r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
            "document_number": r"\b([A-Z]{1,4}[-\s]?\d{3,6}[-\s]?\d{3,6}|[A-Z]{1,4}\d{6,10})\b",
        }

        if custom_patterns:
            self.patterns.update(custom_patterns)

    def parse_fields(self, raw_text: str, lines: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Extract identity fields from OCR text and validate format.

        Args:
            raw_text: Full OCR concatenated text.
            lines: Optional structured lines list from OCREngine.

        Returns:
            Dictionary matching the standard specification:
                - name: str or None
                - dob: str or None
                - document_number: str or None
                - address: str or None
                - field_presence: Dict[str, bool]
                - format_validity: Dict[str, bool]
                - raw_extracted: Dict[str, Any]
        """
        extracted: Dict[str, Optional[str]] = {
            "name": None,
            "dob": None,
            "document_number": None,
            "address": None,
        }

        text_to_search = raw_text or ""
        if lines:
            # Also search through individual lines
            line_texts = [l["text"] for l in lines]
            combined_lines = "\n".join(line_texts)
            text_to_search = f"{text_to_search}\n{combined_lines}"

        # 1. Primary labeled extraction
        for field, pat in self.patterns.items():
            match = re.search(pat, text_to_search, re.IGNORECASE)
            if match:
                val = match.group(1).strip()
                if val:
                    mapped_key = "dob" if field == "date_of_birth" else field
                    extracted[mapped_key] = val

        # 2. Standalone pattern fallbacks if primary labeled pass missed
        if not extracted["dob"]:
            standalone_dob = re.search(self.standalone_patterns["date_of_birth"], text_to_search)
            if standalone_dob:
                extracted["dob"] = standalone_dob.group(1).strip()

        if not extracted["document_number"]:
            standalone_doc = re.search(self.standalone_patterns["document_number"], text_to_search)
            if standalone_doc:
                extracted["document_number"] = standalone_doc.group(1).strip()

        # 3. Clean and sanitize extracted values
        for k in extracted:
            if extracted[k]:
                # Remove stray OCR noise characters like '+' or leading/trailing symbols
                if k == "document_number":
                    extracted[k] = re.sub(r"[^\w\- ]", "", extracted[k]).strip()
                    # Document numbers must contain at least one digit
                    if not any(c.isdigit() for c in extracted[k]):
                        extracted[k] = None
                else:
                    extracted[k] = re.sub(r"^[^\w]+|[^\w]+$", "", extracted[k]).strip()
                if extracted[k] and len(extracted[k]) == 0:
                    extracted[k] = None

        # 4. Field presence check
        field_presence = {
            "name": extracted["name"] is not None,
            "dob": extracted["dob"] is not None,
            "document_number": extracted["document_number"] is not None,
            "address": extracted["address"] is not None,
        }

        # 5. Format validation
        format_validity = {
            "name": self._validate_name(extracted["name"]),
            "dob": self._validate_dob(extracted["dob"]),
            "document_number": self._validate_doc_no(extracted["document_number"]),
            "address": self._validate_address(extracted["address"]),
        }

        return {
            "name": extracted["name"],
            "dob": extracted["dob"],
            "document_number": extracted["document_number"],
            "address": extracted["address"],
            "field_presence": field_presence,
            "format_validity": format_validity,
        }

    def _validate_name(self, name: Optional[str]) -> bool:
        if not name or len(name) < 2:
            return False
        # Must be predominantly alphabetic and spaces
        alpha_count = sum(c.isalpha() or c.isspace() or c in ".-" for c in name)
        return (alpha_count / len(name)) >= 0.85

    def _validate_dob(self, dob: Optional[str]) -> bool:
        if not dob:
            return False
        # Common date patterns: DD/MM/YYYY or YYYY-MM-DD
        m = re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})$", dob)
        if m:
            d, m_val, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            # Handle 2-digit years
            if y < 100:
                y += 1900 if y > 30 else 2000
            return 1 <= d <= 31 and 1 <= m_val <= 12 and 1900 <= y <= 2026
        return False

    def _validate_doc_no(self, doc_no: Optional[str]) -> bool:
        if not doc_no or len(doc_no) < 5:
            return False
        # Must contain both letters and digits or standard formatted pattern
        has_alpha = any(c.isalpha() for c in doc_no)
        has_digit = any(c.isdigit() for c in doc_no)
        return (has_alpha and has_digit) or (len(doc_no) >= 8 and any(c.isdigit() for c in doc_no))

    def _validate_address(self, addr: Optional[str]) -> bool:
        if not addr or len(addr) < 5:
            return False
        # At least one space and recognizable address length
        return " " in addr and len(addr) >= 6
