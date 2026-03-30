import pandas as pd
from app.ml.loader import regressor, classifier

def make_prediction(data: dict):
    df = pd.DataFrame([data])

    cases = regressor.predict(df)[0]
    hotspot = classifier.predict(df)[0]

    return {
        "predicted_cases": float(cases),
        "hotspot": int(hotspot)
    }