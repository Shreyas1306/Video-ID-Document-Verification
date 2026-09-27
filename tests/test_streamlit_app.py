"""
Unit & Integration Tests for Streamlit Demonstration App
=========================================================
Tests Streamlit app loading, disclaimer rendering, element presence,
and execution using streamlit.testing.v1.AppTest.
"""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest


def test_streamlit_app_loads_and_renders_disclaimer():
    """Verify that streamlit_app.py runs without exceptions and displays the legal notice."""
    app_path = Path("streamlit_app.py").resolve()
    assert app_path.is_file(), "streamlit_app.py must exist"

    at = AppTest.from_file(str(app_path))
    at.run(timeout=30)

    # Verify no unhandled exceptions
    assert not at.exception, f"Streamlit app raised an exception: {at.exception}"

    # Verify title
    assert len(at.title) >= 1
    assert "Identity Document Verification" in at.title[0].value

    # Verify prominent disclaimer
    disclaimer_found = any(
        "performs document integrity screening" in m.value and "not official identity authentication" in m.value
        for m in at.markdown
    )
    assert disclaimer_found, "Mandatory academic / non-authentication disclaimer must be rendered."

    # Verify sidebar and subheaders
    assert len(at.subheader) >= 3


def test_streamlit_app_cached_result_display():
    """Verify that cached/existing pipeline results render score cards and identity fields."""
    app_path = Path("streamlit_app.py").resolve()
    at = AppTest.from_file(str(app_path))
    at.run(timeout=30)

    # Check that metric cards or markdown score elements exist
    markdown_texts = " ".join([m.value for m in at.markdown])
    assert "Visual Document Integrity" in markdown_texts
    assert "Temporal Consistency" in markdown_texts
    assert "Character OCR Confidence" in markdown_texts
    assert "Cross-Frame Text Consistency" in markdown_texts
