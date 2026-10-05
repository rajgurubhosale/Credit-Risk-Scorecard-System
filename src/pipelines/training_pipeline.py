"""Train, evaluate, bundle a scorecard, and optionally publish it to staging.

Feature engineering and bin refinement must finish before this pipeline runs.
Production promotion and deployment are separate steps.
"""

import argparse
import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.pyfunc
import pandas as pd
from dotenv import load_dotenv

from src.components.module_07_model_training import ModelTraining
from src.components.module_08_model_evalution import ModelEvaluation
from src.components.module_09_scorecard import Scorecard
from src.logger import config_logger


logger = config_logger("training_pipeline")


def configure_mlflow():
    """Connect MLflow to DagsHub using the project's environment settings."""
    load_dotenv()

    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME") or os.getenv("DAGUSHUB_REPO_NAME")
    token = os.getenv("DAGSHUB_USER_TOKEN")

    if not (repo_owner and repo_name and token):
        raise ValueError(
            "Set DAGSHUB_REPO_OWNER, DAGSHUB_REPO_NAME (or DAGUSHUB_REPO_NAME), "
            "and DAGSHUB_USER_TOKEN"
        )

    # The SDK reads the token at import time, so load .env before importing it.
    import dagshub

    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    mlflow.set_experiment("credit_risk_scorecard_model_rb")
    logger.info("Connected MLflow to DagsHub repository %s/%s", repo_owner, repo_name)


def build_serving_bundle(model_artifact, scorecard_artifact):
    """Save everything needed for score and PD prediction in one local file."""
    trained = joblib.load(model_artifact.model_path)
    features = list(trained["model"].feature_names_in_)
    with open(scorecard_artifact.scorecard_scaling_params_path, encoding="utf-8") as file:
        scaling_params = json.load(file)

    bundle = {
        "schema_version": 1,
        "calibrated_model": trained["calibrated_model"],
        "feature_order": features,
        "woe_numerical_lookup": joblib.load(scorecard_artifact.scorecard_numerical_woe_lookup),
        "woe_categorical_lookup": joblib.load(scorecard_artifact.scorecard_categorical_woe_lookup),
        "scores_numerical_lookup": joblib.load(scorecard_artifact.scorecard_numerical_score_lookup),
        "scores_categorical_lookup": joblib.load(scorecard_artifact.scorecard_categorical_score_lookup),
        "scaling_params": scaling_params,
    }

    bundle_path = Path(scorecard_artifact.scorecard_dir) / "serving_bundle.joblib"
    joblib.dump(bundle, bundle_path)
    logger.info("Serving bundle saved at %s", bundle_path)
    return bundle_path


def publish_candidate(trainer, model_artifact, evaluation_artifact, scorecard_artifact, bundle_path):
    """Log the model and artifacts together, register it, and assign staging."""
    logger.info("Preparing model and metrics for publication")

    bundle = joblib.load(bundle_path)
    calibrated_model = bundle["calibrated_model"]
    features = bundle["feature_order"]
    # Backend preprocessing produces these WOE features before prediction.
    input_example = pd.read_csv(
        trainer.bin_merge_artifact.X_train_final_path,
        usecols=features,
        nrows=5,
    ).loc[:, features].astype(float)
    with open(evaluation_artifact.metrics_path, encoding="utf-8") as file:
        evaluation_metrics = json.load(file)
    with open(scorecard_artifact.scorecard_metrics_path, encoding="utf-8") as file:
        scorecard_metrics = json.load(file)

    configure_mlflow()
    model_name = os.getenv("MLFLOW_MODEL_NAME", "CreditRisk-LogisticRegression")

    with mlflow.start_run(run_name="scorecard_candidate") as run:
        logger.info("Publishing candidate to MLflow run %s", run.info.run_id)

        mlflow.log_params(trainer.params)
        mlflow.log_metrics(evaluation_metrics)
        mlflow.log_metrics({
            f"scorecard_{name}": value
            for name, value in scorecard_metrics.items()
        })

        model_info = mlflow.pyfunc.log_model(
            name="scorecard",
            python_model=calibrated_model.predict_proba,
            artifacts={
                "bundle": str(bundle_path),
                "training": str(model_artifact.artifact_model_dir),
                "evaluation": str(evaluation_artifact.model_eval_dir),
                "scorecard": str(scorecard_artifact.scorecard_dir),
            },
            input_example=input_example,
            registered_model_name=model_name,
        )

        publication = {
            "run_id": run.info.run_id,
            "model_uri": model_info.model_uri,
            "model_name": model_name,
            "model_version": model_info.registered_model_version,
            "alias": "staging",
        }

    mlflow.MlflowClient().set_registered_model_alias(
        model_name, "staging", model_info.registered_model_version
    )
    logger.info("Candidate published to staging: %s", publication)
    return publication


def run_training_pipeline(publish=False):
    """Run local computation first, then publish only when requested."""
    try:
        logger.info("Starting model training and calibration")
        trainer = ModelTraining()
        model_artifact = trainer.orchestrate()

        logger.info("Starting model evaluation")
        evaluation_artifact = ModelEvaluation().evaluate_model()

        logger.info("Starting scorecard construction")
        scorecard_artifact = Scorecard().orchestrate()

        bundle_path = build_serving_bundle(model_artifact, scorecard_artifact)
        result = {
            "model": model_artifact,
            "evaluation": evaluation_artifact,
            "scorecard": scorecard_artifact,
            "serving_bundle": bundle_path,
        }
        logger.info("All local outputs saved successfully")

        if publish:
            result["publication"] = publish_candidate(
                trainer,
                model_artifact,
                evaluation_artifact,
                scorecard_artifact,
                bundle_path,
            )

        logger.info("Training pipeline completed successfully")
        return result

    except Exception:
        logger.exception("Training pipeline failed")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish the calibrated model and serving bundle to staging",
    )
    args = parser.parse_args()
    run_training_pipeline(publish=args.publish)
