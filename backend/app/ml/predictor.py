from app.ml.loader import regressor, classifier


def _build_feature_vector(data: dict) -> list[float]:
    """Convert API payload into the 22-feature vector expected by the models.

    Currently the API exposes only two numeric features (``feature1`` and
    ``feature2``). The trained models, however, were fitted on 22 features.
    To keep the API simple and still satisfy the model's expected input
    shape, we map the two provided features to the first two positions and
    pad the remaining 20 with zeros.

    If you later expose more features via the API, update this function to
    build the full feature vector accordingly.
    """

    feature1 = float(data.get("feature1", 0.0))
    feature2 = float(data.get("feature2", 0.0))

    # First two elements are the provided features, remaining 20 are zeros
    # to match the 22-feature expectation of the models.
    vector = [feature1, feature2] + [0.0] * 20
    return vector


def make_prediction(data: dict):
    features = _build_feature_vector(data)
    X = [features]

    reg_pred = regressor.predict(X)[0]
    hotspot = classifier.predict(X)[0]

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
        "hotspot_level": int(hotspot),
    }

    if risk is not None:
        result["predicted_risk"] = risk

    return result