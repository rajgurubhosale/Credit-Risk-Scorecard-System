import streamlit as st
import os

BASE_FASTAPI_URL = os.getenv("BACKEND_URL", "http://backend:8000")

def hide_sidebar():
    st.markdown("""
        <style>

            [data-testid="collapsedControl"] {display: none;}
        </style>
    """, unsafe_allow_html=True)
    

def display_top_nav_bar():
    st.title("🏦 Credit Risk Application Scorecard")

    c1, c2, c3, c4= st.columns(4)

    clicked = None

    if c1.button("🏠 Home",                   use_container_width=True, key="nav_0"):
        clicked = "main.py"
    if c2.button("📊 Prediction",             use_container_width=True, key="nav_1"):
        clicked = "pages/2_Prediction.py"
    if c3.button("💳 Credit Score Simulator", use_container_width=True, key="nav_2"):
        clicked = "pages/3_Credit_Score_Simulator.py"
    if c4.button("⚠️ Ecl Risk Analysis",          use_container_width=True, key="nav_3"):
        clicked = "pages/4_Ecl_risk_analysis.py"

    st.divider()

    if clicked:
        st.switch_page(clicked)