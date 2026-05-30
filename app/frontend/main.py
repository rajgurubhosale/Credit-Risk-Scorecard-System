import streamlit as st
import sys
from pathlib import Path
from ui_utils import hide_sidebar,display_top_nav_bar
sys.path.append(str(Path(__file__).parent))

st.set_page_config(
    page_title="Credit Risk Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)
hide_sidebar()
## -Title and top nav bar
display_top_nav_bar()
# ── Nav Bar ──────────────────────────────────────────────────────────


st.header("🏦 Credit Risk Application Scorecard")
st.write(
    "An end-to-end credit risk application scorecard that generates credit scores based on "
    "a borrower's credit history and behaviour — returning a **credit score**, **loan approval decision**, "
    "**probability of default**, and **risk level** in real time."
)

st.divider()


# ──────────────────────────────────────────────────────────────────────────────
# ABOUT
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("About the Scorecard")
st.write("Built on industry-standard methods used by banks and financial institutions worldwide:")

c1, c2, c3 = st.columns(3)
with c1:
    st.info("⚡ **Faster & Consistent Decisions**\n\nAutomated scoring eliminates subjectivity — every applicant evaluated on the same objective criteria.")
with c2:
    st.info("🏛️ **Regulatory-Compliant**\n\nLogistic Regression provides interpretable, auditable outputs accepted by financial regulators.")
with c3:
    st.info("📉 **Reduces Default Losses**\n\nBottom score segments capture ~51% of total defaults — filtered out automatically.")



st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# TECHNICAL APPROACH
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("🎯 What This Platform Demonstrates")
st.write(
    "A Probability of Default (PD) Application Scorecard System built on the Home Credit dataset"
    " assessing borrower credit worthiness through statistical modeling and delivering an automated loan decision system. "
)
st.subheader("⚙️ Technical Approach: End-to-End Pipeline")
with st.expander("Show Details", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        st.write(
            "**Feature Engineering**\n"
            "- 7 relational datasets → 557 aggregated features\n"
            "- Payment behaviour, DPD patterns, credit trends & time-window aggregations\n\n"
            "**WOE & Information Value (IV)**\n"
            "- Features converted to WOE bins; IV used to rank predictive strength\n\n"
            "**Feature Selection**\n"
            "- Correlation, VIF, PSI & monotonicity checks → 557 → 29 final features\n\n"
            "**Model — Logistic Regression**\n"
            "- Interpretable, regulatory-accepted; coefficients map directly to score points\n\n"
            "**Scorecard Scaling**\n"
            "- PDO: 20 · Base Score: 600 · Base Odds: 50:1 — scores segmented into risk deciles"
        )

st.divider()

st.subheader("🔁 How It Works")
st.write(
    "Each applicant's feature values are mapped to pre-assigned score bins. "
    "Bin points are summed with a base intercept to produce a final credit score, "
    "which is then bucketed into a risk decile — driving the approve/reject decision."
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.info("🎯 **Credit Score**\nTotal points from all feature bins")
with col2:
    st.success("✅ **Loan Decision**\nApprove, Review, or Reject")
with col3:
    st.warning("⚠️ **Risk Level**\nApplicant risk at their score band")
with col4:
    st.info("🔬 **Feature Breakdown**\nContribution of each feature bin")

st.divider()

st.subheader("**📈 Model Performance**")
st.write("The model shows stable performance across train and test sets, indicating good generalization\n")
import pandas as pd

import pandas as pd

df = pd.DataFrame(
    {
        "Metric": ["AUC", "Gini", "KS", "Brier Score (Calibrated)"],
        "Train":  [0.7659, 0.5318, 0.3981, 0.0676],
        "Test":   [0.7663, 0.5327, 0.4032, 0.0676],
    }
)

st.table(df)

st.divider()


# ──────────────────────────────────────────────────────────────────────────────
# EV FRAMEWORK
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("Cutoff Decision — Expected Value Framework")
st.write("Cutoffs are derived using an Expected Value Framework to ensure only profitable loans are approved.")

st.latex(r"EV = (1 - PD) \times Gain - PD \times Loss")

c1, c2, c3 = st.columns(3)
c1.metric("Cost of Approving a Bad Loan",    "0.70")
c2.metric("Profit of Approving a Good Loan", "0.06")
c3.metric("Optimal Cutoff Score",            "725")

st.write("**Score Decision Bands:**")
ev_df = pd.DataFrame(
    {
        "Score Band": ["748+", "733 – 747", "725 – 732", "705 – 724", "Below 705"],
        "Decision":   ["✅ Approve", "✅ Approve", "✅ Approve", "⚠️ Manual Review", "❌ Reject"],
        "Risk Level": ["Very Low Risk", "Low Risk", "Medium Risk", "Borderline", "High Risk"],
        "Notes": [
            "Strong positive expected value",
            "Profitable lending band",
            "Marginally positive expected value",
            "Routed to underwriter for human judgment",
            "Bottom 2 bands capture ~51% of total defaults",
        ],
    })
st.table(ev_df)


# ──────────────────────────────────────────────────────────────────────────────
# SYSTEM ARCHITECTURE
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("🏗️ System Architecture")
st.image('images/credit_risk_architecture.png', width=600)


st.caption("📌 System architecture diagram will be added here")

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# BUSINESS IMPACT
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("Business Impact")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.success("⚡ **Faster Decisions**\n\nAuto approve/reject — only edge cases go for manual review.")

with c2:
    st.success("💰 **Profitable Loans Only**\n\nCutoff score 725 set via Expected Value framework — only loans with positive expected profit are approved.")
    
with c3:
    st.success("🛡️ **Reduces Default Losses**\n\nBottom 2 score bands catch 51% of all defaults.")
    
with c4:
    st.success("📋 **Regulatory Compliant**\n\nLogistic regression scorecard — fully auditable and explainable.")

st.subheader("🚀 Start Your Risk Analysis")
st.write("Use the navigation buttons above to get started.")

c1, c2, c3, c4 = st.columns(4)

clicked_bottom = None

if c1.button("🏠 Home",                   use_container_width=True, key="nav_b0"):
    clicked_bottom = "main.py"
if c2.button("📊 Prediction",             use_container_width=True, key="nav_b1"):
    clicked_bottom = "pages/2_Prediction.py"
if c3.button("💳 Credit Score Simulator", use_container_width=True, key="nav_b2"):
    clicked_bottom = "pages/3_Credit_Score_Simulator.py"
if c4.button("⚠️ Risk Analysis",          use_container_width=True, key="nav_b3"):
    clicked_bottom = "pages/4_Ecl_risk_analysis.py"

if clicked_bottom:
    st.switch_page(clicked_bottom)