from fastapi import APIRouter, Depends

from app.services.prediction_service import get_prediction_history
from app.core.security import enforce_api_key, rate_limiter

router = APIRouter(
    dependencies=[Depends(enforce_api_key), Depends(rate_limiter)],
)


@router.get("/history")
async def history():
    return await get_prediction_history()