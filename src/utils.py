import os
import random
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUTS_TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUTS_PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Aliases
FIGURES_DIR = OUTPUTS_FIGURES_DIR
TABLES_DIR = OUTPUTS_TABLES_DIR
PREDICTIONS_DIR = OUTPUTS_PREDICTIONS_DIR

def set_seed(seed=42):
    """Enforces reproducibility across random generators."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def save_artifact(obj, filepath):
    """Persists any python object or model via joblib."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, filepath)
    print(f"[SUCCESS] Artifact saved to: {filepath}")

def load_artifact(filepath):
    """Loads a persisted artifact via joblib."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Artifact not found: {filepath}")
    return joblib.load(filepath)

def save_figure(fig, filename, dpi=300):
    """Saves a matplotlib figure cleanly."""
    out_path = OUTPUTS_FIGURES_DIR / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"[SUCCESS] Figure saved to: {out_path}")

def save_dataframe(df, filepath):
    """Saves DataFrame to CSV ensuring parent directories exist."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"[SUCCESS] Table saved to: {filepath}")
