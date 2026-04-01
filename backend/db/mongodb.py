from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import MONGO_URL, DB_NAME

client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]

# Use the existing 'hotspot' collection in the epidemic_spread_prediction database
prediction_collection = db["hotspot"]
user_collection = db["users"]