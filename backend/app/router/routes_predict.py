from fastapi import APIRouter
from datetime import datetime

from app.schemas.schemas import PredictionInput
from app.ml.predictor import make_prediction
from db.mongodb import prediction_collection

router = APIRouter()

@router.post("/predict")
async def predict(data: PredictionInput):
    result = make_prediction(data.dict())

    record = {
        "input": data.dict(),
        "output": result,
        "timestamp": datetime.utcnow()
    }

    await prediction_collection.insert_one(record)

    return result