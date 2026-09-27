"""
Environment Verification Script
================================
Checks that all required dependencies are installed and importable,
the project package structure is valid, and the config file is readable.

Usage:
    python scripts/verify_environment.py
"""

import sys
import importlib
from pathlib import Path


def check_python_version():
    """Verify Python version is 3.9+."""
    major, minor = sys.version_info[:2]
    version_str = f"{major}.{minor}.{sys.version_info.micro}"
    if major < 3 or (major == 3 and minor < 9):
        return False, f"Python {version_str} (NEED 3.9+)"
    return True, f"Python {version_str}"


def check_import(module_name, package_name=None):
    """Try to import a module and return (success, version_or_error)."""
    display_name = package_name or module_name
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", "installed")
        return True, f"{display_name} {version}"
    except ImportError as e:
        return False, f"{display_name} MISSING ({e})"


def check_project_structure():
    """Verify essential project directories and files exist."""
    project_root = Path(__file__).resolve().parent.parent
    required_paths = [
        "config/settings.yaml",
        "src/__init__.py",
        "src/preprocessing/__init__.py",
        "src/preprocessing/frame_extractor.py",
        "src/preprocessing/document_detector.py",
        "src/preprocessing/document_cropper.py",
        "src/preprocessing/perspective_corrector.py",
        "src/visual/__init__.py",
        "src/visual/feature_extractor.py",
        "src/visual/visual_integrity.py",
        "src/visual/temporal_consistency.py",
        "src/text/__init__.py",
        "src/text/ocr_engine.py",
        "src/text/field_parser.py",
        "src/text/text_consistency.py",
        "src/fusion/__init__.py",
        "src/fusion/score_fusion.py",
        "src/fusion/risk_classifier.py",
        "src/report/__init__.py",
        "src/report/report_generator.py",
        "src/utils/__init__.py",
        "src/utils/image_utils.py",
        "src/utils/video_utils.py",
        "src/utils/config_loader.py",
        "tests/__init__.py",
        "requirements.txt",
        "README.md",
    ]
    missing = []
    for p in required_paths:
        if not (project_root / p).exists():
            missing.append(p)
    return missing


def check_config_readable():
    """Try to load the YAML config file."""
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "config" / "settings.yaml"
    try:
        import yaml
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            return False, "Config parsed but is not a dict"
        keys = list(config.keys())
        return True, f"Loaded {len(keys)} top-level keys: {keys}"
    except ImportError:
        return False, "PyYAML not installed"
    except Exception as e:
        return False, f"Config load error: {e}"


def check_src_imports():
    """Verify all src subpackages are importable."""
    # Add project root to sys.path temporarily
    project_root = Path(__file__).resolve().parent.parent
    sys_path_modified = False
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        sys_path_modified = True

    subpackages = [
        "src",
        "src.preprocessing",
        "src.visual",
        "src.text",
        "src.fusion",
        "src.report",
        "src.utils",
    ]
    results = []
    for pkg in subpackages:
        try:
            importlib.import_module(pkg)
            results.append((True, pkg))
        except ImportError as e:
            results.append((False, f"{pkg}: {e}"))

    if sys_path_modified:
        sys.path.remove(str(project_root))

    return results


def main():
    print("=" * 65)
    print("  Environment Verification")
    print("  Video-Based Identity Document Verification")
    print("=" * 65)
    print()

    all_ok = True

    # 1. Python version
    print("1. Python Version")
    print("-" * 40)
    ok, msg = check_python_version()
    status = "OK" if ok else "FAIL"
    print(f"   [{status}] {msg}")
    all_ok = all_ok and ok
    print()

    # 2. Core dependencies
    print("2. Core Dependencies")
    print("-" * 40)
    deps = [
        ("numpy", None),
        ("pandas", None),
        ("cv2", "opencv-python-headless"),
        ("yaml", "PyYAML"),
        ("torch", "PyTorch"),
        ("torchvision", None),
        ("sklearn", "scikit-learn"),
        ("scipy", None),
        ("PIL", "Pillow"),
    ]
    for mod, pkg in deps:
        ok, msg = check_import(mod, pkg)
        status = "OK" if ok else "FAIL"
        print(f"   [{status}] {msg}")
        all_ok = all_ok and ok
    print()

    # 3. Optional / phase-specific dependencies
    print("3. Optional Dependencies (for later phases)")
    print("-" * 40)
    optional_deps = [
        ("ultralytics", None),
        ("paddleocr", "PaddleOCR"),
        ("streamlit", None),
        ("pytest", None),
        ("matplotlib", None),
    ]
    for mod, pkg in optional_deps:
        ok, msg = check_import(mod, pkg)
        status = "OK" if ok else "WARN"
        print(f"   [{status}] {msg}")
        # Optional deps don't fail the overall check
    print()

    # 4. Project structure
    print("4. Project Structure")
    print("-" * 40)
    missing = check_project_structure()
    if missing:
        print(f"   [FAIL] Missing {len(missing)} files:")
        for m in missing:
            print(f"          - {m}")
        all_ok = False
    else:
        print("   [OK] All required files and directories present")
    print()

    # 5. Config file
    print("5. Configuration File")
    print("-" * 40)
    ok, msg = check_config_readable()
    status = "OK" if ok else "FAIL"
    print(f"   [{status}] {msg}")
    all_ok = all_ok and ok
    print()

    # 6. Source package imports
    print("6. Source Package Imports")
    print("-" * 40)
    import_results = check_src_imports()
    for ok, msg in import_results:
        status = "OK" if ok else "FAIL"
        print(f"   [{status}] {msg}")
        all_ok = all_ok and ok
    print()

    # Summary
    print("=" * 65)
    if all_ok:
        print("  RESULT: All checks passed. Environment is ready.")
    else:
        print("  RESULT: Some checks failed. See above for details.")
        print("  Run: pip install -r requirements.txt")
    print("=" * 65)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
