from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import MONGO_URL, DB_NAME

client = AsyncIOMotorClient(
	MONGO_URL,
	serverSelectionTimeoutMS=5000,
	connectTimeoutMS=5000,
	socketTimeoutMS=10000,
)
db = client[DB_NAME]

# Unified prediction collection (one document per region)
prediction_collection = db["predictions"]
user_collection = db["users"]

LEGACY_PREDICTION_COLLECTIONS = ("hotspot", "Nextdaycases", "risks")


async def ensure_prediction_indexes() -> None:
	await prediction_collection.create_index("region", unique=True)


async def ensure_user_indexes() -> None:
	await user_collection.create_index("email", unique=True)


async def cleanup_legacy_prediction_collections() -> None:
	for collection_name in LEGACY_PREDICTION_COLLECTIONS:
		await db.drop_collection(collection_name)