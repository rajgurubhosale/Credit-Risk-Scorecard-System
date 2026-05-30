import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # adds app/backend/ to path

from services.loaders import load_artifacts, load_model
from services.predict import get_woe_array, get_risk_label, single_score_applicant, get_approval_decision
import pandas as pd
import numpy as np
from config.paths import BATCH_RESULT_PATH
#-----------------------------
#       PORTFOLIO RISK ECL
#----------------------------
model,woe_numerical_lookup,  woe_categorical_lookup,scores_numerical_lookup,scores_categorical_lookup,feature_order = load_artifacts()
LGD = 0.45


def score_one_applicant(row, model, woe_num, woe_cat, score_num, score_cat,feature_order):
    """Score a single applicant row. Returns a result dict."""
    user_info = row.to_dict()
    # Step 1: Convert raw features to WOE values
    woe_df = get_woe_array(woe_num, woe_cat, user_info, feature_order)

    woe_df = woe_df[feature_order]

    # Step 2: Get probability of default from model
    pd_value = float(model.predict_proba(woe_df)[:, 1][0])

    # Step 3: Get scorecard score
    score_result = single_score_applicant(score_num, score_cat, user_info)
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
    model, woe_num, woe_cat, score_num, score_cat, feature_order = load_artifacts()

    results = []
    total   = len(test_df)

    for i, (_, row) in enumerate(test_df.iterrows()):

        # Simple progress print every 1000 rows
        if i % 1000 == 0:
            print(f"  Scoring row {i} / {total}...")

        try:
            result = score_one_applicant(
                row, model, woe_num, woe_cat, score_num, score_cat, feature_order
            )
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


def get_portfolio_summary() :
    """
    Summarize the portfolio after batch scoring.
    Only approved loans are included in ECL/value calculations.
    """
    import numpy as np
    results_df = pd.read_csv(BATCH_RESULT_PATH)
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
        "avg_pd_pct":            round(approved["pd_value"].mean() * 100, 4),
        "avg_score":             round(approved["score"].mean(), 4),
        "lgd_assumption":        LGD,
    }
    risk_decision = results_df["risk_label"].value_counts().to_dict()
    decision_breakdown = results_df["approval_decision"].value_counts().to_dict()

    return portfolio_summary,risk_decision,decision_breakdown,results_df


if __name__ == "__main__":

    TEST_DATA_PATH = "artifact/data/data_splits/X_test.csv"
    OUTPUT_PATH    = "app/backend/artifacts/test_prediction_portfolio/batch_results.csv"

    bundle   = load_model()
    features = bundle['features']

    # Load features + SK_ID_CURR
    cols_to_load = features + ["SK_ID_CURR"]

    print(f"Loading test data...")
    test_df = pd.read_csv(TEST_DATA_PATH, usecols=cols_to_load)
    test_df = test_df.sample(500, random_state=42)
    
    print(f"  {len(test_df):,} applicants found")

    # Save SK_ID_CURR separately then drop before scoring
    customer_ids = test_df["SK_ID_CURR"]
    test_df      = test_df.drop(columns=["SK_ID_CURR"])

    print(f"\nRunning batch scoring...")
    results_df = batch_predict(test_df)

    # Add customer ID to results
    results_df.insert(0, "SK_ID_CURR", customer_ids.values)

    print(f"\nPortfolio Summary:")
    portfolio_summary, risk_breakdown, decision_breakdown = get_portfolio_summary(results_df)
    for key, value in portfolio_summary.items():
        if key not in ("risk_breakdown", "decision_breakdown"):
            print(f"  {key:<30} {value}")

    print(f"\n  Risk Breakdown:     {risk_breakdown}")
    print(f"  Decision Breakdown: {decision_breakdown}")
    resutlt_df = results_df.sort_values(by='SK_ID_CURR')
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")