from datetime import datetime
from typing import Any, Dict, List

from app.core.exceptions import MLModelError, DatabaseError
from app.ml.predictor import make_prediction
from app.schemas.schemas import PredictionInput
from db.mongodb import prediction_collection


async def create_prediction(input_data: PredictionInput) -> Dict[str, Any]:
	"""Run ML prediction and persist the result to the database."""

	payload = input_data.dict()

	try:
		result = make_prediction(payload)
	except Exception as exc:  # noqa: BLE001
		raise MLModelError("Failed to generate prediction") from exc

	record = {
		"input": payload,
		"output": result,
		"timestamp": datetime.utcnow(),
	}

	try:
		await prediction_collection.insert_one(record)
	except Exception as exc:  # noqa: BLE001
		raise DatabaseError("Failed to store prediction in database") from exc

	return result


async def get_prediction_history() -> List[Dict[str, Any]]:
	"""Return all past prediction records from the database."""

	try:
		cursor = prediction_collection.find({}, {"_id": 0})
		return await cursor.to_list(length=None)
	except Exception as exc:  # noqa: BLE001
		raise DatabaseError("Failed to read prediction history from database") from exc

