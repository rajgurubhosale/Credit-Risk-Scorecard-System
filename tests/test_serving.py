import json
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app" / "backend"))

from schemas.requests import CreditApplicationRequest
from services.predict import single_get_numerical_column_score
from services.portfoilio_analysis import get_portfolio_summary, validate_report


def report(version="7", decision="REJECT"):
    return pd.DataFrame([dict(
        score=650.0, pd_value=0.2, EAD=100.0, LGD=0.45, ECL=9.0,
        approval_decision=decision, risk_label="Very High Risk", model_version=version,
    )])


class ReleaseTests(unittest.TestCase):
    def test_lowest_boundary(self):
        bins = pd.IntervalIndex.from_tuples([(0, 10), (10, 20)], closed="right")
        lookup = {"f": {"interval": pd.Series([0.25, 0.75], index=bins),
                        "discrete": pd.Series(dtype=float), "special": {-99999: -1.0}}}
        for value, expected in ((0, 0.25), (5, 0.25), (10, 0.25), (11, 0.75), (-99999, -1.0)):
            with self.subTest(value=value):
                self.assertEqual(single_get_numerical_column_score(lookup, "f", value), expected)

    def test_portfolio_summary(self):
        summary, *_ = get_portfolio_summary(report(decision="REJECT"))
        self.assertIsNone(summary["avg_pd_pct"])           # nobody approved
        json.dumps(summary, allow_nan=False)               # must be valid JSON
        summary, *_ = get_portfolio_summary(report(decision="APPROVE"))
        self.assertEqual(summary["avg_pd_pct"], 20.0)      # normal case still works

    def test_report_validation(self):
        validate_report(report(), "7")
        bad_cases = (("score", float("nan")), ("ECL", float("inf")), ("pd_value", 1.1),
                     ("approval_decision", "ERROR"), ("model_version", "8"))
        for column, value in bad_cases:
            bad = report()
            bad.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                validate_report(bad, "7")
        with self.assertRaises(ValueError):
            validate_report(pd.DataFrame(), "7")

    def test_fixture_matches_schema(self):
        payload = json.loads((ROOT / "tests/fixtures/prediction_request.json").read_text())
        CreditApplicationRequest.model_validate(payload)


if __name__ == "__main__":
    unittest.main()
