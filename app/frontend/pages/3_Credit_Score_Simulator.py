import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  

from ui_utils import hide_sidebar,display_top_nav_bar,BASE_FASTAPI_URL
import streamlit as st
import pandas as pd
import requests


st.set_page_config(page_title=" Credit Score Simulator", page_icon="💳", layout="wide",initial_sidebar_state="collapsed",
)
hide_sidebar()
display_top_nav_bar()

@st.cache_data(ttl=600) 
def fetch_feature_importance():
    response = requests.get(f"{BASE_FASTAPI_URL}/feature_importance")
    response.raise_for_status()
    return pd.DataFrame(response.json())


st.title("🔍 Credit Score Simulator")
st.markdown("Select a feature, explore its score bins, and see how each value impacts your credit score.")
with st.spinner('Loading Feature Names'):
    try:
        feature_importance = fetch_feature_importance()
        feature_names = feature_importance['feature']
        
    except Exception as e:
        st.error(f'failed to laod to feature importance{e}')
    
selected_feature = st.selectbox(
        "Select Feature",
        options=feature_names,
        index=1
    )

check_scores_analysis = st.button(
    "🚀 Run Score simulator",
    type="primary",
    use_container_width=True,
    key="run_prediction_main"
)
st.divider()
# fixed Streamlit block
if check_scores_analysis:
    try:
        response = requests.get(f'{BASE_FASTAPI_URL}/run_score_simulator/{selected_feature}')
        if response.status_code == 200:
            df = pd.DataFrame(response.json())
            st.table(df)
        else:
            st.error(f'API error {response.status_code}: {response.text}')
    except requests.exceptions.ConnectionError:
        st.error('Could not connect to fastapi server') 
    except Exception as e:
        st.error(f'Request failed: {e}')
    