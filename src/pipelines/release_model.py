"""Check the staging package and promote it to production.

Set SKIP_COMPARE=true for rollbacks; package and prediction checks still run.
Both versions must have evaluation metrics from the same holdout dataset.
"""

import math
import os
import tempfile
from pathlib import Path

import joblib
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.logger import config_logger


logger = config_logger("release_model")

MIN_AUC_GAIN = 0.002
MAX_KS_DROP = 0.01
MAX_BRIER_RISE = 0.002
METRICS = ("AUC_test", "KS_test", "brier_cal_test")


def configure_mlflow():
    """Use the same DagsHub settings as the training pipeline."""
    load_dotenv()
    owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo = os.getenv("DAGSHUB_REPO_NAME") or os.getenv("DAGUSHUB_REPO_NAME")
    token = os.getenv("DAGSHUB_USER_TOKEN")
    if not (owner and repo and token):
        raise ValueError("Set DAGSHUB_REPO_OWNER, DAGSHUB_REPO_NAME and DAGSHUB_USER_TOKEN")
    import dagshub
    dagshub.init(repo_owner=owner, repo_name=repo, mlflow=True)
    logger.info("Connected to DagsHub repository %s/%s", owner, repo)


def get_version(client, model_name, alias):
    """Return None only when the alias is absent; API failures still block release."""
    version = client.get_registered_model(model_name).aliases.get(alias)
    return client.get_model_version(model_name, version) if version else None


def get_metrics(client, version):
    run_id = version.run_id or version.tags.get("run_id")
    if not run_id:
        raise ValueError(f"Model v{version.version} has no source run ID")

    metrics = client.get_run(run_id).data.metrics
    for name in METRICS:
        value = metrics.get(name)
        if value is None or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Model v{version.version} has a missing or invalid {name}")
    return metrics


def check_compatibility(bundle):
    features = bundle["feature_order"]
    model = bundle["calibrated_model"]
    numerical = set(bundle["woe_numerical_lookup"])
    categorical = set(bundle["woe_categorical_lookup"])

    if not features or len(features) != len(set(features)):
        raise ValueError("Model feature order must contain unique features")
    if numerical & categorical or (numerical | categorical) != set(features):
        raise ValueError("WOE lookups do not match the model features")
    if numerical != set(bundle["scores_numerical_lookup"]):
        raise ValueError("Numerical score and WOE lookups do not match")
    if categorical != set(bundle["scores_categorical_lookup"]):
        raise ValueError("Categorical score and WOE lookups do not match")
    if list(model.feature_names_in_) != features:
        raise ValueError("Bundle feature order does not match the calibrated model")
    if list(model.classes_) != [0, 1]:
        raise ValueError("Model probability columns must represent classes 0 and 1")

    for key in ("FACTOR", "INTERCEPT_POINTS", "BASE_SCORE", "BASE_ODDS", "PDO"):
        if not math.isfinite(float(bundle["scaling_params"][key])):
            raise ValueError(f"Scaling parameter {key} is not finite")


def smoke_test(package_path, bundle):
    """Verify the actual serving pyfunc, using neutral WOE values."""
    sample = pd.DataFrame(0.0, index=[0], columns=bundle["feature_order"])
    serving_model = mlflow.pyfunc.load_model(str(package_path))
    probabilities = np.asarray(serving_model.predict(sample), dtype=float)
    if (
        probabilities.shape != (1, 2)
        or not np.isfinite(probabilities).all()
        or not ((probabilities >= 0) & (probabilities <= 1)).all()
        or not np.allclose(probabilities.sum(axis=1), 1.0)
    ):
        raise ValueError("Serving model returned invalid probabilities")
    expected = bundle["calibrated_model"].predict_proba(sample)
    if not np.allclose(probabilities, expected):
        raise ValueError("Serving pyfunc and bundled model predictions do not match")


def compare_with_production(client, candidate, production, skip_compare):
    if skip_compare:
        logger.warning("SKIP_COMPARE=true: skipping metric comparison for rollback")
        return

    candidate_metrics = get_metrics(client, candidate)
    if production is None:
        logger.info("First release: no production version to compare")
        return

    production_metrics = get_metrics(client, production)
    for name in METRICS:
        logger.info("%s: candidate=%.6f production=%.6f",
                    name, candidate_metrics[name], production_metrics[name])

    if candidate_metrics["AUC_test"] < production_metrics["AUC_test"] + MIN_AUC_GAIN:
        raise ValueError(f"AUC_test must improve by at least {MIN_AUC_GAIN}")
    if candidate_metrics["KS_test"] < production_metrics["KS_test"] - MAX_KS_DROP:
        raise ValueError(f"KS_test dropped by more than {MAX_KS_DROP}")
    if candidate_metrics["brier_cal_test"] > production_metrics["brier_cal_test"] + MAX_BRIER_RISE:
        raise ValueError(f"brier_cal_test increased by more than {MAX_BRIER_RISE}")


def main():
    configure_mlflow()
    model_name = os.getenv("MLFLOW_MODEL_NAME", "CreditRisk-LogisticRegression")
    skip_compare = os.getenv("SKIP_COMPARE", "false").strip().lower() == "true"
    client = mlflow.MlflowClient()

    candidate = get_version(client, model_name, "staging")
    if candidate is None:
        raise ValueError("No model version has the staging alias")
    logger.info("Checking staging candidate %s v%s", model_name, candidate.version)

    # Pin the version throughout validation, even if staging changes during release.
    model_uri = f"models:/{model_name}/{candidate.version}"
    with tempfile.TemporaryDirectory() as folder:
        package_path = Path(mlflow.artifacts.download_artifacts(
            artifact_uri=model_uri, dst_path=folder
        ))
        metadata = mlflow.models.Model.load(str(package_path / "MLmodel"))
        artifacts = metadata.flavors.get("python_function", {}).get("artifacts", {})
        bundle_reference = artifacts.get("bundle", {})
        if not bundle_reference.get("path"):
            raise ValueError("Serving package has no bundle artifact")
        bundle_path = package_path / bundle_reference["path"]
        if bundle_path.name != "serving_bundle.joblib" or not bundle_path.is_file():
            raise ValueError("serving_bundle.joblib is missing from the package")

        bundle = joblib.load(bundle_path)
        check_compatibility(bundle)
        smoke_test(package_path, bundle)
        logger.info("Package, compatibility and prediction checks passed")

    production = get_version(client, model_name, "production")
    compare_with_production(client, candidate, production, skip_compare)

    client.set_registered_model_alias(model_name, "production", candidate.version)
    logger.info("Released %s v%s to production", model_name, candidate.version)
    if output_path := os.getenv("GITHUB_OUTPUT"):
        with open(output_path, "a", encoding="utf-8") as file:
            file.write(f"version={candidate.version}\n")
    return candidate.version


if __name__ == "__main__":
    try:
        main()
    except Exception:           
        logger.exception("Model release blocked")
        raise SystemExit(1)
