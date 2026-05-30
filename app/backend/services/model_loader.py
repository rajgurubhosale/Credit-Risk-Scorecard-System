'''
Loads the production model and artifacts from local cache into FastAPI app state.
Called once at FastAPI startup via lifespan context manager.
Requires artifacts to be pre-downloaded by scripts/download_artifacts.py
'''

import joblib
import pandas as pd
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent
ARTIFACT_PATH = BASE_DIR / "artifacts"
def load_model_and_artifacts():
    active_version = (ARTIFACT_PATH / "active_version.txt").read_text().strip()
    versioned_cache = ARTIFACT_PATH / f"v{active_version}"

    if not versioned_cache.exists():
        raise FileNotFoundError(
            f"Artifacts not found at {versioned_cache}. "
            f"Run scripts/download_artifacts.py first."
        )

    print(f"✅ Loading artifacts for model v{active_version}")

    model_bundle     = joblib.load(versioned_cache / "model.pkl")
    calibrated_model = model_bundle['calibrated_model']
    feature_order    = calibrated_model.calibrated_classifiers_[0].estimator.feature_names_in_.tolist()

    woe_numerical_lookup   = joblib.load(versioned_cache / "woe_numerical_lookup.pkl")
    woe_categorical_lookup = joblib.load(versioned_cache / "woe_categorical_lookup.pkl")

    scores_numerical_lookup   = joblib.load(versioned_cache / "scores_numerical_lookup.pkl")
    scores_categorical_lookup = joblib.load(versioned_cache / "scores_categorical_lookup.pkl")

    scorecard_table = pd.read_csv(versioned_cache / "scorecard_table.csv")

    print(f"✅ Model v{active_version} loaded successfully")

    return {
        "calibrated_model":          calibrated_model,
        "feature_order":             feature_order,
        "woe_numerical_lookup":      woe_numerical_lookup,
        "woe_categorical_lookup":    woe_categorical_lookup,
        "scores_numerical_lookup":   scores_numerical_lookup,
        "scores_categorical_lookup": scores_categorical_lookup,
        "scorecard_table":           scorecard_table,
        "version":                   active_version,
    }