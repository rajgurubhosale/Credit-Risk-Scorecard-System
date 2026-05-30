import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  

from ui_utils import hide_sidebar,display_top_nav_bar,BASE_FASTAPI_URL
import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="ECL & Risk Analysis", page_icon="💳", layout="wide",initial_sidebar_state="collapsed",
)
hide_sidebar()
display_top_nav_bar()

st.title("ECL & Portfolio Risk Analytics")
st.caption("Portfolio-level credit risk monitoring — 500 applicant sample from test data")
st.markdown("""
> Simplified IFRS 9 Stage 1 Expected Credit Loss framework using application-level exposure approximation.

> Probability of Default (PD) is generated using the trained application scorecard model.

> Loss Given Default (LGD) is modeled using a simplified 45% regulatory-style assumption for demonstration purposes.

> Exposure at Default (EAD) is approximated using sanctioned loan exposure.
""")

st.divider()

try:
    response = requests.get(f'{BASE_FASTAPI_URL}/portfolio_analysis',timeout=10)
    
    
    if response.status_code == 200:
        response.raise_for_status()

        result = response.json()
    else:
        raise st.error(f'API error:{response.status_code}: {response.text}')
except requests.exceptions.ConnectionError:
    st.error('Could not connect to fastapi server')    
except requests.exceptions.Timeout:
    st.error("Prediction request timed out.")

portfolio_summary = result['portfolio_summary']
risk_decision = result['risk_decision']
decision_breakdown = result['decision_breakdown']
df = pd.DataFrame(result['results_df'])

# ── Row 1: Key metrics ──
st.markdown("#### 📊 Portfolio Overview")
col1, col2, col3, col4, col5, col6,col7 = st.columns(7)
col1.metric("Total Applicants", f"{portfolio_summary['total_applicants']:,}")
col2.metric("Approved", f"{portfolio_summary['total_approved']:,}")
col3.metric("Approved Rate", f"{portfolio_summary['approval_rate_pct']}%")
col4.metric("Rejected / Review", f"{portfolio_summary['total_applicants'] - portfolio_summary['total_approved']:,}")
col5.metric("Portfolio Value", f"₹{portfolio_summary['total_portfolio_value']/1e7:.2f}Cr")
col6.metric("Total ECL", f"₹{portfolio_summary['total_ecl']/1e5:.1f}L")
col7.metric("ECL / Portfolio", f"{portfolio_summary['ecl_as_pct_portfolio']}%",
            "✅ Healthy" if portfolio_summary['ecl_as_pct_portfolio'] < 2 else "⚠️ Elevated")

st.divider()

# ── Row 2: Model metrics ──
st.markdown("#### 🧮 Model Metrics (Approved Loans)")
col7, col8, col9 = st.columns(3)
col7.metric("Avg Probability of Default", f"{portfolio_summary['avg_pd_pct']}%",
            help="Average PD across approved applicants only")
col8.metric("Avg Scorecard Score", f"{portfolio_summary['avg_score']:.1f}",
            help="Scores above 725 = Approve, 705–725 = Manual Review, <705 = Reject")
col9.metric("LGD Assumption", f"{portfolio_summary['lgd_assumption']:.0%}",
            help="Loss Given Default — % of EAD lost if borrower defaults")

st.divider()

# ── Row 3: Charts ──
st.markdown("#### 📈 Risk & Decision Distribution")
col10, col11 = st.columns(2)

with col10:
    st.markdown("**Risk tier breakdown** — all applicants")
    risk_order = ["Very Low Risk", "Low Risk", "Medium Risk", "Elevated Risk", "Very High Risk"]
    risk_series = pd.Series(risk_decision).reindex(risk_order).dropna()
    st.bar_chart(risk_series, horizontal=True)
    st.caption("Risk tiers based on scorecard score thresholds")

with col11:
    st.markdown("**Approval decision breakdown** — all applicants")
    decision_order = ["APPROVE", "MANUAL REVIEW", "REJECT"]
    decision_series = pd.Series(decision_breakdown).reindex(decision_order).dropna()
    st.bar_chart(decision_series, horizontal=True)
    st.caption("Score ≥ 725 → Approve · 705–725 → Manual Review · < 705 → Reject")

st.divider()
# ── Modelling Assumptions ──
st.markdown("#### 📌 Modelling Assumptions")
st.markdown("""
- PD generated using supervised application scorecard model  
- LGD fixed at 45% as simplified regulatory assumption  
- EAD approximated using sanctioned loan amount  
- Simplified Stage 1 ECL framework only  
- No macroeconomic forward-looking adjustments applied  
""")

st.divider()

# ── Segment Table ──
st.markdown("#### 📊 ECL by Risk Segment")

approved = df[df["approval_decision"] == "APPROVE"].copy()

segment_table = (
    approved.groupby("risk_label")
    .agg(
        Applicants   = ("pd_value", "count"),
        Avg_PD       = ("pd_value", "mean"),
        Portfolio_ECL = ("ECL", "sum")
    )
    .reset_index()
)

segment_table["Avg PD"]        = segment_table["Avg_PD"].apply(lambda x: f"{x*100:.1f}%")
segment_table["Portfolio ECL"] = segment_table["Portfolio_ECL"].apply(lambda x: f"₹{x/1e5:.1f}L")
segment_table = segment_table.rename(columns={"risk_label": "Segment"})[["Segment", "Applicants", "Avg PD", "Portfolio ECL"]]
segment_table = segment_table.sort_values(by='Applicants')
st.table(segment_table)

st.divider()
# ── Row 4: Raw data table ──
with st.expander("📋 View applicant-level data"):
    st.dataframe(
        df[["SK_ID_CURR", "score", "pd_value", "risk_label", "approval_decision", "EAD", "ECL"]]
        .sort_values("score", ascending=False)
        .reset_index(drop=True),
        use_container_width=True
    )