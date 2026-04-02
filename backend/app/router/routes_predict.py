from fastapi import APIRouter, Depends

from app.schemas.schemas import PredictionInput
from app.services.prediction_service import create_prediction
from app.core.security import enforce_api_key, rate_limiter
from app.ml.country_features import list_available_countries
from app.ml.predictor import get_model_summary

router = APIRouter(
    dependencies=[Depends(enforce_api_key), Depends(rate_limiter)],
)


@router.post("/predict")
async def predict(data: PredictionInput):
    return await create_prediction(data)


@router.get("/countries")
async def countries():
    return {"countries": list_available_countries()}


@router.get("/model/summary")
async def model_summary():
    return get_model_summary()