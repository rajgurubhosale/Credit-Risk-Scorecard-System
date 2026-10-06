'''
Connects to DagsHub MLflow and downloads a registered model version into
app/backend/artifacts/model/, so the Docker image / HF Space needs no MLflow at prediction time.

- Only artifacts/model/ is replaced; other files in artifacts/ (e.g. portfolio CSV) are untouched.
- Writes artifacts/active_version.txt with the downloaded version.
'''
import argparse
import os
import shutil
import sys
from pathlib import Path

import dagshub
import mlflow
from mlflow import MlflowClient

from scripts.mlflow_config import DAGSHUB_REPO_NAME, DAGSHUB_REPO_OWNER, MLFLOW_MODEL_ALIAS, MLFLOW_MODEL_NAME

# repo root = parent of scripts/, so this works from any working directory
ARTIFACT_PATH_MLFLOW = Path(__file__).resolve().parents[1] / "app" / "backend" / "artifacts"
MODEL_DIR = ARTIFACT_PATH_MLFLOW / "model"
DOWNLOAD_DIR = ARTIFACT_PATH_MLFLOW / "_download"


def setup_mlflow():
    try:
        os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_REPO_OWNER
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_USER_TOKEN")

        # dagshub.init(mlflow=True) already sets the tracking URI
        dagshub.init(
            repo_owner=DAGSHUB_REPO_OWNER,
            repo_name=DAGSHUB_REPO_NAME,
            mlflow=True,
        )
    except Exception as e:
        raise Exception(f"MLflow setup failed: {e}")


def download_model_and_artifacts(version=None):
    if version is None:
        client = MlflowClient()
        version = client.get_model_version_by_alias(MLFLOW_MODEL_NAME, MLFLOW_MODEL_ALIAS).version

    ARTIFACT_PATH_MLFLOW.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(DOWNLOAD_DIR, ignore_errors=True)  # scratch folder only
    DOWNLOAD_DIR.mkdir()

    print(f"Downloading production model v{version}...")
    try:
        # MLflow returns the real path of the downloaded package
        downloaded = Path(mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{MLFLOW_MODEL_NAME}/{version}",
            dst_path=str(DOWNLOAD_DIR),
        ))

        # fail before touching the old model if the download is bad
        if not list(downloaded.rglob("serving_bundle.joblib")):
            raise RuntimeError("serving_bundle.joblib not found in the downloaded package")

        # no ignore_errors here: if the old model can't be deleted, fail loudly
        if MODEL_DIR.exists():
            shutil.rmtree(MODEL_DIR)
        shutil.move(str(downloaded), str(MODEL_DIR))
    finally:
        shutil.rmtree(DOWNLOAD_DIR, ignore_errors=True)

    (ARTIFACT_PATH_MLFLOW / "active_version.txt").write_text(str(version))
    print(f"Active version set to v{version}")
    return version


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a registered model and its artifacts")
    parser.add_argument("--version", type=int, help="Exact model version; defaults to the configured alias")
    args = parser.parse_args()
    if args.version is not None and args.version <= 0:
        parser.error("--version must be a positive integer")
    try:
        setup_mlflow()
        version = download_model_and_artifacts(args.version)
        print(f"\n🎉 Production model v{version} is ready.")
    except Exception as e:
        print(f"\n❌ Model loader failed: {e}")
        sys.exit(1)