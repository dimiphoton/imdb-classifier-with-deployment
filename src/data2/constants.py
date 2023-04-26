"""
constants.py

This script contains constants, paths, and variables to be used throughout the project.
It includes paths for data, models, and other important directories. By centralizing
these values in one script, it's easier to maintain and update them.

To use these constants in other scripts, simply import them like this:
from constants import DATA_INTERIM_PATH, DATA_FINAL_PATH, MODEL_PATH
"""

from pathlib import Path

# Directory paths
ROOT_DIR = Path(__file__).resolve().parents[2]

# Cleaned but not processed data
DATA_INTERIM_PATH: Path = ROOT_DIR / "data/1.interim"

# Processed, final data
DATA_FINAL_PATH: Path = ROOT_DIR / "data/2.processed"

# Untrained and trained models
MODEL_PATH: Path = ROOT_DIR / "model"
