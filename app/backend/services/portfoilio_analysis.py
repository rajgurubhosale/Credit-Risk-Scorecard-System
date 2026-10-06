import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # adds app/backend/ to path

from services.model_loader import load_bundle, ModelBundle
from services.predict import get_woe_array, get_risk_label, single_score_applicant, get_approval_decision
import pandas as pd
import numpy as np
from config.paths import BATCH_RESULT_PATH
#-----------------------------
#       PORTFOLIO RISK ECL
#----------------------------
LGD = 0.45


def validate_report(results_df, expected_version):
    """Reject incomplete or invalid results before saving or deploying them."""
    if results_df.empty or results_df["approval_decision"].eq("ERROR").any():
        raise ValueError("Portfolio report is empty or contains scoring errors")
    values = results_df[["score", "pd_value", "EAD", "LGD", "ECL"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Portfolio report contains non-finite values")
    if not results_df["pd_value"].between(0, 1).all():
        raise ValueError("Portfolio probabilities must be between 0 and 1")
    if not results_df["model_version"].astype(str).eq(str(expected_version)).all():
        raise ValueError("Portfolio report does not match the approved model version")


def score_one_applicant(row, bundle: ModelBundle):
    """Score a single applicant row. Returns a result dict."""
    user_info = row.to_dict()
    # Step 1: Convert raw features to WOE values
    woe_df = get_woe_array(bundle, user_info)

    # Step 2: Get probability of default from model
    pd_value = float(bundle.model.predict_proba(woe_df)[:, 1][0])

    # Step 3: Get scorecard score
    score_result = single_score_applicant(bundle, user_info)
    score        = float(score_result["total_score"])

    # Step 4: Get risk label and approval decision from score
    risk_label        = get_risk_label(score)
    approval_decision = get_approval_decision(score)

    # Step 5: Calculate ECL = PD x LGD x EAD
    ead = float(user_info.get("AMT_CREDIT", 0))
    ecl = round(pd_value * LGD * ead, 2)

    return {
        "score":             round(score, 4),
        "pd_value":          round(pd_value, 6),
        "risk_label":        risk_label,
        "approval_decision": approval_decision,
        "EAD":               ead,
        "LGD":               LGD,
        "ECL":               ecl,
    }


def batch_predict(test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Score all applicants in test_df.
    Returns a DataFrame with score, PD, risk label, decision, and ECL for each row.
    """
    # Load everything once — not inside the loop
    bundle = load_bundle()

    results = []
    total   = len(test_df)

    for i, (_, row) in enumerate(test_df.iterrows()):

        # Simple progress print every 1000 rows
        if i % 1000 == 0:
            print(f"  Scoring row {i} / {total}...")

        try:
            result = score_one_applicant(row, bundle)
        except Exception as e:
            print(f"  ERROR row {i}: {e}")  # ← add this
            result = {
            "score":             np.nan,
            "pd_value":          np.nan,
            "risk_label":        "ERROR",
            "approval_decision": "ERROR",
            "EAD":               np.nan,
            "LGD":               LGD,
            "ECL":               np.nan,
            "error":             str(e),
          }

        results.append(result)

    return pd.DataFrame(results)


def get_portfolio_summary(results_df: pd.DataFrame | None = None):
    """
    Summarize the portfolio after batch scoring.
    Only approved loans are included in ECL/value calculations.
    Pass results_df to use scored results in memory; otherwise reads the saved CSV.
    """
    import numpy as np
    if results_df is None:
        results_df = pd.read_csv(BATCH_RESULT_PATH, dtype={"model_version": str})
        if (
            "model_version" not in results_df.columns
            or not results_df["model_version"].eq(load_bundle().version).all()
        ):
            raise ValueError("Portfolio report is outdated; regenerate it")
    approved = results_df[results_df["approval_decision"] == "APPROVE"]

    total_portfolio_value = approved["EAD"].sum()
    total_ecl             = approved["ECL"].sum()
    ecl_pct               = (total_ecl / total_portfolio_value * 100) if total_portfolio_value > 0 else 0
    portfolio_summary = {
        "total_applicants":      len(results_df),
        "total_approved":        len(approved),
        "approval_rate_pct":     round(len(approved) / len(results_df) * 100, 2),
        "total_portfolio_value": round(total_portfolio_value, 2),
        "total_ecl":             round(total_ecl, 2),
        "ecl_as_pct_portfolio":  round(ecl_pct, 4),
        "avg_pd_pct":            round(approved["pd_value"].mean() * 100, 4) if not approved.empty else None,
        "avg_score":             round(approved["score"].mean(), 4) if not approved.empty else None,
        "lgd_assumption":        LGD,
    }
    risk_decision = results_df["risk_label"].value_counts().to_dict()
    decision_breakdown = results_df["approval_decision"].value_counts().to_dict()

    return portfolio_summary,risk_decision,decision_breakdown,results_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate a portfolio report for the loaded model")
    parser.add_argument("--version", type=int, help="Expected model version for release validation")
    args = parser.parse_args()
    if args.version is not None and args.version <= 0:
        parser.error("--version must be a positive integer")

    TEST_DATA_PATH = (
        Path(__file__).resolve().parents[1]
        / "data" / "portfolio_input.csv"
    )
    OUTPUT_PATH = BATCH_RESULT_PATH

    bundle = load_bundle()
    if args.version is not None and bundle.version != str(args.version):
        raise ValueError("Loaded model does not match the approved version")

    print(f"Loading test data...")
    test_df = pd.read_csv(TEST_DATA_PATH)
    
    print(f"  {len(test_df):,} applicants found")

    # Save SK_ID_CURR separately then drop before scoring
    customer_ids = test_df["SK_ID_CURR"]
    test_df      = test_df.drop(columns=["SK_ID_CURR"])

    print(f"\nRunning batch scoring...")
    results_df = batch_predict(test_df)
    results_df["model_version"] = bundle.version
    validate_report(results_df, args.version if args.version is not None else bundle.version)

    # Add customer ID to results
    results_df.insert(0, "SK_ID_CURR", customer_ids.values)
    results_df = results_df.sort_values(by='SK_ID_CURR')

    print(f"\nPortfolio Summary:")
    portfolio_summary, risk_breakdown, decision_breakdown, _ = get_portfolio_summary(results_df)
    for key, value in portfolio_summary.items():
        if key not in ("risk_breakdown", "decision_breakdown"):
            print(f"  {key:<30} {value}")

    print(f"\n  Risk Breakdown:     {risk_breakdown}")
    print(f"  Decision Breakdown: {decision_breakdown}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")
