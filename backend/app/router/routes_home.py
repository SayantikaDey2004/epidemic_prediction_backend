from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def home():
    return {
        "message": "COVID Prediction API is running"
    }