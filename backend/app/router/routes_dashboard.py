from fastapi import APIRouter

from db.mongodb import prediction_collection

router = APIRouter()

@router.get("/history")
async def get_history():
    cursor = prediction_collection.find({}, {"_id": 0})
    data = await cursor.to_list(length=None)
    return data