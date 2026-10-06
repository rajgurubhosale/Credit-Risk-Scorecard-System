import sys
from pathlib import Path

# Add app/backend to path - works both locally and in Docker
sys.path.append(str(Path(__file__).parent))

from schemas.requests import CreditApplicationRequest
from fastapi import FastAPI
from fastapi import Path as FastApiPath
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from services.predict import run_prediction, get_feature_data
from services.portfoilio_analysis import get_portfolio_summary
from services.model_loader import load_bundle


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bundle = load_bundle()   # load once at startup
    yield
    app.state.bundle = None            # cleanup on shutdown
    
app = FastAPI(lifespan=lifespan)


@app.post('/predict')
def prediction(request: CreditApplicationRequest):
    bundle = app.state.bundle
    try:

        user_info = {}
        for field_value in request.model_dump().values():
            user_info.update(field_value)

        result = run_prediction(bundle, user_info)
        return JSONResponse(status_code=200, content=result)
    except Exception as e:
            return JSONResponse(status_code=500,content=str(e))


@app.get('/run_score_simulator/{selected_feature}')
def get_feature_score_bins(selected_feature: str = FastApiPath(..., description="Feature name to get score bins for")):
    df = get_feature_data(app.state.bundle, selected_feature)
    if not df.empty:
        return JSONResponse(status_code=200, content=df.to_dict(orient='records'))
    else:
        raise HTTPException(status_code=404,detail='feature not found in lookups')

@app.get('/feature_importance')
def get_feature_importance():
    feature_importance = app.state.bundle.feature_importance
    if not feature_importance.empty:
        return JSONResponse(status_code=200,content=feature_importance.to_dict(orient='records'))
    else:
        raise HTTPException(status_code=404,detail='feature importace file is empty')
    
    
@app.get('/portfolio_analysis')
def portfolio_ecl_analysis():
    try:
        portfolio_summary,risk_decision,decision_breakdown,results_df = get_portfolio_summary()
        
        result = {
            'portfolio_summary':portfolio_summary,
            'risk_decision':risk_decision,
            'decision_breakdown':decision_breakdown,
            'results_df': results_df.to_dict(orient='records')
            }
        return JSONResponse(status_code=200, content=result)
    except Exception as e:
        return JSONResponse(status_code=500,content=str(e))


@app.get('/')
def home_page():
    return {'message':'Credit Risk Application System API'}

@app.get('/health')
def health_check():
    bundle = app.state.bundle
    return {
        'status':       'OK',
        'model_loaded': bundle.model is not None,
        'version':      bundle.version,
    }