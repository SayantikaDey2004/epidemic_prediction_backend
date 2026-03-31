from fastapi import APIRouter, Depends

from app.schemas.schemas import PredictionInput
from app.services.prediction_service import create_prediction
from app.core.security import enforce_api_key, rate_limiter

router = APIRouter(
    dependencies=[Depends(enforce_api_key), Depends(rate_limiter)],
)


@router.post("/predict")
async def predict(data: PredictionInput):
    return await create_prediction(data)