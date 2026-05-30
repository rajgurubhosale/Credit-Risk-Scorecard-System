# app/schemas/response.py

from pydantic import BaseModel, Field
from typing import Literal

class PredictionResponse(BaseModel):
    
    # core results
    scorecard_score: float = Field(description='Final credit scorecard score')
    probability_of_default: float = Field(ge=0, le=1, description='Probability of default of borrower for 12 months between 0 and 1')
    
    # decision
    decision: Literal["APPROVE", "MANUAL REVIEW", "REJECT"]
    risk_category: Literal["Very Low Risk", "Low Risk", "Medium Risk", "Elevated Risk", "Very High Risk"]
    
    # breakdowns
    # stremlit will take this and return in df
    feature_score_breakdown: dict[str, float] = Field(description='Scorecard contribution per feature')
    user_input_breakdown: dict[str, float] = Field(description='Raw input values provided by user')