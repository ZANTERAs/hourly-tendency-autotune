"""
Centralized path constants for the project.

All scripts under src/ import from this module so save/load destinations stay
consistent and the layout can be reorganized without hunting through code.
"""

from pathlib import Path

# src/paths.py lives one directory below the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC_DIR     = PROJECT_ROOT / "src"
MODELS_DIR  = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Ensure the writable output dirs exist on import — idempotent
for _d in (MODELS_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
