"""Check a running backend before deployment; do not assert a particular model score."""

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app" / "backend"))

from scripts.check_deployment import check_deployment
from services.portfoilio_analysis import validate_report


def smoke_test(url, version, fixture=ROOT / "tests" / "fixtures" / "prediction_request.json"):
    url = url.rstrip("/")
    check_deployment(f"{url}/health", version)

    with open(fixture, encoding="utf-8") as file:
        applicant = json.load(file)
    response = requests.post(f"{url}/predict", json=applicant, timeout=30)
    response.raise_for_status()
    prediction = response.json()
    if not math.isfinite(prediction["score"]) or not (
        math.isfinite(prediction["pd_value"]) and 0 <= prediction["pd_value"] <= 1
    ):
        raise ValueError("Prediction returned an invalid score or probability")

    response = requests.get(f"{url}/portfolio_analysis", timeout=30)
    response.raise_for_status()
    validate_report(pd.DataFrame(response.json()["results_df"]), version)
    print(f"Health, prediction and portfolio checks passed for model v{version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Backend base URL, without /health")
    parser.add_argument("--version", required=True, help="Approved model version")
    args = parser.parse_args()
    smoke_test(args.url, args.version)
