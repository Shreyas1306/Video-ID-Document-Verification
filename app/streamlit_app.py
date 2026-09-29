"""
Streamlit Compatibility Launcher
================================
Canonical entrypoint: streamlit_app.py (at project root).

Usage:
    streamlit run streamlit_app.py
"""

from pathlib import Path
import runpy

ROOT_APP = Path(__file__).resolve().parent.parent / "streamlit_app.py"
runpy.run_path(str(ROOT_APP), run_name="__main__")
