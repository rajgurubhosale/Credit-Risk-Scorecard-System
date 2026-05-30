'''
Connects to DagsHub MLflow tracking server and loads the best production model
along with its artifacts (WoE lookups, score tables, scorecard) into local cache.
Artifacts are cached under artifacts/v{version}/  inside the app/backend  file.

Run as a standalone script during CI/CD pipeline before building the Docker image.
- download production model from mlflow
- saves into app/backend/artifact with verion
- then it can be dockerized and then there will be no need to load the model from mlflow at time of prediction

'''
import os
import mlflow
from pathlib import Path
from mlflow import MlflowClient
import dagshub
from scripts.mlflow_config import DAGSHUB_REPO_NAME,DAGSHUB_REPO_OWNER,MLFLOW_MODEL_ALIAS,MLFLOW_MODEL_NAME
import sys


def setup_mlflow():
    try:
        os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_REPO_OWNER
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_USER_TOKEN")

        dagshub.init(
            repo_owner=DAGSHUB_REPO_OWNER,
            repo_name=DAGSHUB_REPO_NAME,
            mlflow=True
        )
        mlflow.set_tracking_uri(os.getenv("DAGSHUB_TRACKING_URI"))

    except Exception as e:
        raise Exception(f"MLflow setup failed: {e}")



ARTIFACT_PATH_MLFLOW = Path("app/backend/artifacts")


def download_model_and_artifacts():
    client = MlflowClient()

    model_name  = MLFLOW_MODEL_NAME
    model_alias = MLFLOW_MODEL_ALIAS

    model_version = client.get_model_version_by_alias(model_name, model_alias)
    run_id        = model_version.run_id
    version       = model_version.version

    versioned_cache = ARTIFACT_PATH_MLFLOW / f"v{version}"

    if versioned_cache.exists() and any(versioned_cache.iterdir()):
        print(f"✅ Artifacts already exist for model v{version}, skipping download")
    else:
        print(f"⬇️  Downloading artifacts for model v{version}...")
        versioned_cache.mkdir(parents=True, exist_ok=True)
        mlflow.artifacts.download_artifacts(
            run_id=run_id, dst_path=str(versioned_cache)
        )
        print(f"✅ Artifacts saved to {versioned_cache}")

    (ARTIFACT_PATH_MLFLOW / "active_version.txt").write_text(str(version))
    print(f"✅ Active version set to v{version}")
    return version

if __name__ == "__main__":
    try:
        setup_mlflow()        
        version = download_model_and_artifacts()
        print(f"\n🎉 Production model v{version} is ready.")

    except (RuntimeError, Exception) as e:
        print(f"\n❌ Model loader failed: {e}")
        sys.exit(1)
