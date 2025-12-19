
"""
Global configuration for the Movies Success Prediction project.
"""

from pathlib import Path

# Base project directory (can be changed if needed)
BASE_DIR = Path(__file__).resolve().parent.parent

# Path to the raw movies metadata CSV file.
# Put your "movies_metadata.csv" file under the data/ folder.
DATA_PATH = BASE_DIR / "data" / "movies_metadata.csv"

# Random seed for reproducibility
RANDOM_SEED = 42
