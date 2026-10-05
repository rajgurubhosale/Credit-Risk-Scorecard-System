from services.model_loader import load_model_and_artifacts as load_model

def load_scores_numerical():
    return load_model()["scores_numerical_lookup"]


def load_scores_categorical():
    return load_model()["scores_categorical_lookup"]


def load_woe_numerical():
    return load_model()["woe_numerical_lookup"]


def load_woe_categorical():
    return load_model()["woe_categorical_lookup"]


def load_scorecard_table():
    return load_model()["scorecard_table"]


def load_feature_importance():
    return load_model()["feature_importance"]


def load_artifacts():
    bundle = load_model()
    return (
        bundle["calibrated_model"],
        bundle["woe_numerical_lookup"],
        bundle["woe_categorical_lookup"],
        bundle["scores_numerical_lookup"],
        bundle["scores_categorical_lookup"],
        bundle["feature_order"],
    )