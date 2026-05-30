import pandas as pd
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).parent.parent
ARTIFACT_PATH_MLFLOW = BASE_DIR / "artifacts"

def get_artifact_path() -> Path:
    version = (ARTIFACT_PATH_MLFLOW / "active_version.txt").read_text().strip()
    return ARTIFACT_PATH_MLFLOW / f"v{version}"

def load_scores_numerical():
    return joblib.load(get_artifact_path() / "scores_numerical_lookup.pkl")

def load_scores_categorical():
    return joblib.load(get_artifact_path() / "scores_categorical_lookup.pkl")

def load_scorecard_table():
    return pd.read_csv(get_artifact_path() / "scorecard_table.csv")

def load_feature_importance():
    return pd.read_csv(get_artifact_path() / "feature_importance.csv")

def load_woe_numerical():
    return joblib.load(get_artifact_path() / "woe_numerical_lookup.pkl")

def load_woe_categorical():
    return joblib.load(get_artifact_path() / "woe_categorical_lookup.pkl")

def load_model():
    return joblib.load(get_artifact_path() / "model.pkl")

def load_artifacts():
    """Load model and lookup tables from disk."""

    # Same load_model() used in main — no conflict
    bundle        = load_model()
    model         = bundle['calibrated_model']
    feature_order = list(model.feature_names_in_)

    woe_numerical_lookup      = load_woe_numerical()
    woe_categorical_lookup    = load_woe_categorical()
    scores_numerical_lookup   = load_scores_numerical()
    scores_categorical_lookup = load_scores_categorical()

    return (
        model,
        woe_numerical_lookup,
        woe_categorical_lookup,
        scores_numerical_lookup,
        scores_categorical_lookup,
        feature_order
    )
