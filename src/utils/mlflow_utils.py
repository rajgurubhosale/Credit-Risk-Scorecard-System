import os
import mlflow
import dagshub
from dotenv import load_dotenv

load_dotenv()

def setup_mlflow():
    try:
        os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_REPO_OWNER")
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_USER_TOKEN")
        
        dagshub.init(
            repo_owner=os.getenv("DAGSHUB_REPO_OWNER"),
            repo_name=os.getenv("DAGUSHUB_REPO_NAME"),
            mlflow=True
        )
        mlflow.set_tracking_uri(os.getenv("DAGUSHUB_TRACKING_URI"))
        
    except Exception as e:
        raise Exception(f"MLflow setup failed: {e}")
    
load_dotenv()
print(os.getenv("DAGSHUB_REPO_OWNER"))
print(os.getenv("DAGSHUB_USER_TOKEN"))
print(os.getenv("DAGUSHUB_REPO_NAME"))
print(os.getenv("DAGUSHUB_TRACKING_URI"))