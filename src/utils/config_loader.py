"""
Config Loader Utility
=====================
Loads configuration from YAML file (e.g. config.yaml or config/settings.yaml).
Provides a clean, centralized dictionary interface for all pipeline modules.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml


DEFAULT_CONFIG_LOCATIONS = [
    Path("config/settings.yaml"),
    Path("config.yaml"),
    Path("../config/settings.yaml"),
    Path("../config.yaml"),
]


def find_default_config_path() -> Path:
    """Locate the default configuration file from common locations, prioritizing config/settings.yaml."""
    for loc in DEFAULT_CONFIG_LOCATIONS:
        if loc.is_file():
            return loc.resolve()
    # If not found relative to cwd, try relative to project root
    project_root = Path(__file__).resolve().parent.parent.parent
    for loc in [project_root / "config" / "settings.yaml", project_root / "config.yaml"]:
        if loc.is_file():
            return loc
    raise FileNotFoundError("Could not locate canonical config/settings.yaml in project root.")


def load_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """
    Load YAML configuration file into a dictionary.

    Args:
        config_path: Path to YAML config. If None, default search paths are used.

    Returns:
        Dict containing loaded configuration settings.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If YAML syntax is invalid or file is not a dictionary.
    """
    target_path = Path(config_path).resolve() if config_path is not None else find_default_config_path()

    if not target_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {target_path}")

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML configuration at {target_path}: {e}") from e

    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration file at {target_path} must be a dictionary mapping.")

    return cfg
