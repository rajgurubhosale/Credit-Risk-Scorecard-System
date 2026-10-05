'''
Loads the downloaded model package from artifacts/model/ into a typed ModelBundle.
Loaded once (cached) and shared. Requires scripts/model_loader.py to have run first.
'''
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from mlflow.models import Model

logger = logging.getLogger(__name__)

ARTIFACT_PATH = Path(__file__).resolve().parent.parent / "artifacts"



@dataclass(frozen=True, slots=True, eq=False)
class ModelBundle:
    version: str
    model: Any                      # calibrated sklearn model
    feature_order: list[str]
    woe_numerical: dict
    woe_categorical: dict
    scores_numerical: dict
    scores_categorical: dict
    scaling_params: dict
    scorecard_table: pd.DataFrame
    feature_importance: pd.DataFrame




@lru_cache(maxsize=1)

def load_bundle() -> ModelBundle:
    package = ARTIFACT_PATH / "model"

    if not (package / "MLmodel").exists():
        raise FileNotFoundError(f"No model at {package}. Run scripts/model_loader.py first.")
    

    artifacts = Model.load(str(package / "MLmodel")).flavors["python_function"]["artifacts"]

    raw = joblib.load(package / artifacts["bundle"]["path"])
    

    bundle = ModelBundle(
        version=(ARTIFACT_PATH / "active_version.txt").read_text(encoding="utf-8").strip(),
        model=raw["calibrated_model"],
        feature_order=raw["feature_order"],
        woe_numerical=raw["woe_numerical_lookup"],
        woe_categorical=raw["woe_categorical_lookup"],
        scores_numerical=raw["scores_numerical_lookup"],
        scores_categorical=raw["scores_categorical_lookup"],
        scaling_params=raw["scaling_params"],
        scorecard_table=pd.read_csv(package / artifacts["scorecard"]["path"] / "scorecard_table.csv"),
        feature_importance=pd.read_csv(package / artifacts["training"]["path"] / "feature_importance.csv"),
    )
    logger.info("Loaded serving bundle v%s", bundle.version)
    return bundle
