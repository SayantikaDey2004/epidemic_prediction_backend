from pathlib import Path
from typing import List
import joblib

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
PROJECT_DIR = Path(__file__).resolve().parents[3]

def _load_first_existing(candidates: List[Path]):
	for path in candidates:
		if not path.exists():
			continue

		try:
			return joblib.load(path)
		except Exception:
			# Continue scanning candidates so local development can still run
			# when one artifact is incompatible with the active environment.
			continue

	return None


regressor = _load_first_existing(
	[
		MODEL_DIR / "regressor.joblib",
		MODEL_DIR / "covid_regressor_v1.joblib",
		PROJECT_DIR / "covid_regressor_v1.joblib",
	]
)

classifier = _load_first_existing(
	[
		MODEL_DIR / "classifier.joblib",
		MODEL_DIR / "covid_classifier_v1.joblib",
		PROJECT_DIR / "covid_classifier_v1.joblib",
	]
)