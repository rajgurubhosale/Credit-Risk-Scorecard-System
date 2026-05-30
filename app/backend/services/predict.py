import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # adds app/backend/ to path

import numpy as np
import pandas as pd
from services.loaders import load_scores_numerical, load_scores_categorical

def single_get_numerical_column_score(numerical_lookup, feature, value):
    feature_lookup = numerical_lookup[feature]
    if value in feature_lookup['special']:
        return feature_lookup['special'][value]
    if not feature_lookup['interval'].empty:
        try:
            feature_low  = feature_lookup['interval'].index[0].left
            feature_high = feature_lookup['interval'].index[-1].right
            if value < feature_low:
                return feature_lookup['interval'].iloc[0]
            elif value > feature_high:
                return feature_lookup['interval'].iloc[-1]
            else:
                return feature_lookup['interval'][value]
        except KeyError:
            pass
    if not feature_lookup['discrete'].empty:
        try:
            if value < feature_lookup['discrete'].index.min():
                return feature_lookup['discrete'].iloc[0]
            elif value > feature_lookup['discrete'].index.max():
                return feature_lookup['discrete'].iloc[-1]
            else:
                return feature_lookup['discrete'][value]
        except KeyError:
            pass
    return np.nan


def single_get_cat_score(categorical_lookup, feature, value):
    feature_lookup = categorical_lookup[feature]
    return feature_lookup.get(value, feature_lookup.get('RARE', np.nan))


def single_score_applicant(scores_numerical_lookup, scores_categorical_lookup, user_info: dict):
    total_score = 0
    breakdown   = {}

    for feature in scores_numerical_lookup.keys():
        if feature in user_info:
            value = user_info[feature]
            if pd.isna(value):
                value = -99999.0
            score = single_get_numerical_column_score(scores_numerical_lookup, feature, value)
            if not pd.isna(score):
                total_score        += score
                breakdown[feature]  = score

    for feature in scores_categorical_lookup.keys():
        if feature in user_info:
            value = user_info[feature]
            if pd.isna(value):
                value = 'MISSING'
            score = single_get_cat_score(scores_categorical_lookup, feature, value)
            if not pd.isna(score):
                total_score        += score
                breakdown[feature]  = score

    BASE_SCORE = 719.8443041917163
    total_score = BASE_SCORE + total_score

    return {
        'total_score': round(total_score, 4),
        'breakdown':   breakdown,
    }

# get woe values to feed to model
def get_woe_array(woe_numerical_lookup, woe_categorical_lookup, user_info: dict, feature_order: list):
    woe_dict = {}

    for feature in woe_numerical_lookup.keys():
        if feature in user_info:
            value = user_info[feature]
            if pd.isna(value):
                value = -99999.0
            woe = single_get_numerical_column_score(woe_numerical_lookup, feature, value)
            woe_dict[feature] = woe if not pd.isna(woe) else 0.0

    for feature in woe_categorical_lookup.keys():
        if feature in user_info:
            value = user_info[feature]
            if pd.isna(value):
                value = 'MISSING'
            woe = single_get_cat_score(woe_categorical_lookup, feature, value)
            woe_dict[feature] = woe if not pd.isna(woe) else 0.0

    return pd.DataFrame([woe_dict])[feature_order]


def get_risk_label(score:float):
    risk_category = ''
    if score >= 748:
        risk_category = "Very Low Risk"
    elif score >= 733:
        risk_category = "Low Risk"
    elif score >= 725:
        risk_category = "Medium Risk"
    elif score >= 705:
        risk_category = "Elevated Risk"
    else:
        risk_category = "Very High Risk"
        
    return risk_category

def get_approval_decision(score):
    decision = ''
    if score >= 725:
        decision ="APPROVE"
    elif score >= 705:
        decision = "MANUAL REVIEW"
    else:
        decision = "REJECT"
    return decision


def run_prediction(calibrated_model,
                   woe_numerical_lookup,
                   woe_categorical_lookup,
                   user_info,
                   feature_order,
                   scores_numerical_lookup,
                   scores_categorical_lookup
                   ):
    ''' this function return the prediction for the user input:
    returns:
        credit score:
        probability of default:
        approval decision:
        risk label:
    '''
    
    woe_df   = get_woe_array(woe_numerical_lookup, woe_categorical_lookup, user_info, feature_order)

    pd_value = calibrated_model.predict_proba(woe_df)[:, 1][0]
    
    result = single_score_applicant(scores_numerical_lookup, scores_categorical_lookup, user_info)
    score  = result['total_score']
    risk_label = get_risk_label(score)
    approval_decision = get_approval_decision(score)
    
    return {
        
        'score':float(score), #numpy brokes in json
        'pd_value':float(pd_value),
        'approval_decision':approval_decision,
        'risk_label':risk_label,
        'feature_score_breakdown':result['breakdown'],
        'user_info':user_info
        }
    
def get_feature_data(selected_feature):
    '''
    used for the 3rd page 
    credit score simulator:
    Select a feature, explore its score bins, and see how each value impacts your credit score.
    '''
    numerical_lookup   = load_scores_numerical()
    categorical_lookup = load_scores_categorical()
    
    rows = []
    
    if selected_feature in numerical_lookup:
        score_analysis = numerical_lookup[selected_feature]
        if not score_analysis['interval'].empty:
            for bin_, score in score_analysis['interval'].items():
                rows.append({'Bin': str(bin_), 'Score': round(score, 4)})
        if not score_analysis['discrete'].empty:
            for bin_, score in score_analysis['discrete'].items():
                rows.append({'Bin': str(bin_), 'Score': round(score, 4)})
        for bin_, score in score_analysis['special'].items():
            rows.append({'Bin': f'Special ({bin_})', 'Score': round(score, 4)})

    elif selected_feature in categorical_lookup:
        score_analysis = categorical_lookup[selected_feature]
        for bin_, score in score_analysis.items():
            rows.append({'Bin': str(bin_), 'Score': round(score, 4)})

    return pd.DataFrame(rows)
