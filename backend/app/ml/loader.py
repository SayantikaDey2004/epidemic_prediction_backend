from pathlib import Path
import joblib

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
PROJECT_DIR = Path(__file__).resolve().parents[3]

def _load_first_existing(candidates: list[Path]):
	for path in candidates:
		if path.exists():
			return joblib.load(path)
	names = ", ".join(str(p) for p in candidates)
	raise FileNotFoundError(f"No model file found. Tried: {names}")


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