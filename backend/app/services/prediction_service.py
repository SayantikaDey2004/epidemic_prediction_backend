from datetime import datetime
from typing import Any, Dict, List

from app.core.exceptions import MLModelError, DatabaseError
from app.ml.predictor import make_prediction
from app.schemas.schemas import PredictionInput
from db.mongodb import nextdaycases_collection, prediction_collection, risks_collection


async def create_prediction(input_data: PredictionInput) -> Dict[str, Any]:
	"""Run ML prediction and persist the result to the database."""

	payload = input_data.model_dump(exclude_none=True) if hasattr(input_data, "model_dump") else input_data.dict(exclude_none=True)

	try:
		result = make_prediction(payload)
	except Exception as exc:  # noqa: BLE001
		raise MLModelError("Failed to generate prediction") from exc

	timestamp = datetime.utcnow()
	region = result.get("region") or payload.get("region") or "Global"
	risk_score = None
	if isinstance(result.get("regions"), list) and result["regions"]:
		first_region = result["regions"][0]
		if isinstance(first_region, dict):
			risk_score = first_region.get("risk")

	record = {
		"input": payload,
		"output": result,
		"timestamp": timestamp,
	}

	nextdaycases_record = {
		"region": region,
		"predicted_cases": result.get("predicted_cases"),
		"timestamp": timestamp,
	}

	risks_record = {
		"region": region,
		"risk": result.get("risk"),
		"predicted_risk": result.get("predicted_risk"),
		"risk_score": risk_score,
		"timestamp": timestamp,
	}

	try:
		await prediction_collection.insert_one(record)
		await nextdaycases_collection.insert_one(nextdaycases_record)
		await risks_collection.insert_one(risks_record)
	except Exception as exc:  # noqa: BLE001
		raise DatabaseError("Failed to store prediction in prediction collections") from exc

	return result


async def get_prediction_history() -> List[Dict[str, Any]]:
	"""Return all past prediction records from the database."""

	try:
		cursor = prediction_collection.find({}, {"_id": 0})
		return await cursor.to_list(length=None)
	except Exception as exc:  # noqa: BLE001
		raise DatabaseError("Failed to read prediction history from database") from exc

