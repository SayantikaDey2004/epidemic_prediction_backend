import pandas as pd
from app.ml.loader import regressor, classifier

def make_prediction(data: dict):
    df = pd.DataFrame([data])

    reg_pred = regressor.predict(df)[0]
    hotspot = classifier.predict(df)[0]

    # Support both single-output and multi-output regression models.
    if hasattr(reg_pred, "__len__") and not isinstance(reg_pred, (str, bytes)):
        values = list(reg_pred)
        cases = float(values[0])
        risk = float(values[1]) if len(values) > 1 else None
    else:
        cases = float(reg_pred)
        risk = None

    result = {
        "predicted_cases": cases,
        "hotspot_level": int(hotspot)
    }

    if risk is not None:
        result["predicted_risk"] = risk

    return result