from fastapi import APIRouter
from app.db.mongodb import prediction_collection

router = APIRouter()

@router.get("/history")
def get_history():
    data = list(prediction_collection.find({}, {"_id": 0}))
    return data