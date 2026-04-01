from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import MONGO_URL, DB_NAME

client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]

# Prediction-related collections
prediction_collection = db["hotspot"]
nextdaycases_collection = db["Nextdaycases"]
risks_collection = db["risks"]
user_collection = db["users"]