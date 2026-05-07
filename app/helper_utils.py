import numpy as np
import pandas as pd

def safe_ratio(numerator, denominator):
    DN_MISSING_PLACEHOLDER = -99999.0
    D_MISSING_PLACEHOLDER  = -88888.0
    N_MISSING_PLACEHOLDER  = -77777.0


    n_missing = (numerator is None) or np.isnan(numerator)
    d_missing = (denominator is None) or np.isnan(denominator)
    if n_missing and d_missing:
        return DN_MISSING_PLACEHOLDER
    if d_missing:
        return D_MISSING_PLACEHOLDER
    if n_missing:
        return N_MISSING_PLACEHOLDER
    if denominator == 0:
        return np.nan
    return numerator / denominator

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


def single_score_applicant(numerical_lookup, categorical_lookup, user_info: dict):
    total_score = 0
    breakdown   = {}

    for feature in numerical_lookup.keys():
        if feature in user_info:
            value = user_info[feature]
            if pd.isna(value):
                value = -99999.0
            score = single_get_numerical_column_score(numerical_lookup, feature, value)
            if not pd.isna(score):
                total_score        += score
                breakdown[feature]  = score

    for feature in categorical_lookup.keys():
        if feature in user_info:
            value = user_info[feature]
            if pd.isna(value):
                value = 'MISSING'
            score = single_get_cat_score(categorical_lookup, feature, value)
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
def get_woe_array(numerical_woe_lookup, categorical_woe_lookup, user_info: dict, feature_order: list):
    woe_dict = {}

    for feature in numerical_woe_lookup.keys():
        if feature in user_info:
            value = user_info[feature]
            if pd.isna(value):
                value = -99999.0
            woe = single_get_numerical_column_score(numerical_woe_lookup, feature, value)
            woe_dict[feature] = woe if not pd.isna(woe) else 0.0

    for feature in categorical_woe_lookup.keys():
        if feature in user_info:
            value = user_info[feature]
            if pd.isna(value):
                value = 'MISSING'
            woe = single_get_cat_score(categorical_woe_lookup, feature, value)
            woe_dict[feature] = woe if not pd.isna(woe) else 0.0

    return pd.DataFrame([woe_dict])[feature_order]

