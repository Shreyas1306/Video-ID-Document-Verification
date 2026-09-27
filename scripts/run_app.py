"""
Streamlit App Launcher
======================
Launches the Phase 11 Streamlit demonstration web application.
"""

from pathlib import Path
import subprocess
import sys

def main():
    repo_root = Path(__file__).resolve().parent.parent
    app_file = repo_root / "streamlit_app.py"
    
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_file),
        "--server.port=8501",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ]
    
    print(f"Starting Streamlit Demonstration App at http://localhost:8501 ...")
    subprocess.run(cmd, cwd=str(repo_root))

if __name__ == "__main__":
    main()
