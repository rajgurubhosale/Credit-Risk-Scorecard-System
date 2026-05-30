from pathlib import Path

BASE_FASTAPI_URL             = "http://localhost:8000"
BASE_APP_PATH                = Path('app')
BASE_BACKEND_ARTIFACT_PATH = Path(__file__).resolve().parent.parent / 'artifacts'

# Always resolves correctly regardless of where Python is run from
BATCH_RESULT_PATH = BASE_BACKEND_ARTIFACT_PATH / 'test_prediction_portfolio' / 'batch_results.csv'