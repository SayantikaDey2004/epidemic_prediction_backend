from pathlib import Path
import joblib

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"

regressor = joblib.load(MODEL_DIR / "regressor.joblib")
classifier = joblib.load(MODEL_DIR / "classifier.joblib")