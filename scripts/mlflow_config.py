# scripts/mlflow_config.py
from dotenv import load_dotenv

load_dotenv()
MLFLOW_MODEL_NAME='CreditRisk-LogisticRegression'
MLFLOW_MODEL_ALIAS='production_model'
DAGSHUB_REPO_OWNER = 'rajgurubhosale'
DAGSHUB_REPO_NAME = 'Credit-Risk-Scorecard-System'