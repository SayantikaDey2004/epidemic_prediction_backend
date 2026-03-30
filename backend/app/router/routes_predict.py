from fastapi import APIRouter
from app.models.schemas import PredictionInput
from app.ml.predictor import make_prediction
from app.db.mongodb import prediction_collection
from datetime import datetime

router = APIRouter()

@router.post("/predict")
def predict(data: PredictionInput):
    result = make_prediction(data.dict())

    record = {
        "input": data.dict(),
        "output": result,
        "timestamp": datetime.utcnow()
    }

    prediction_collection.insert_one(record)

    return result