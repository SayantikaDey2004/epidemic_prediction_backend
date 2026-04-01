import hashlib

from app.ml.loader import regressor, classifier


def _derive_features_from_region(region: str) -> tuple[float, float]:
    """Build deterministic numeric features from a region string."""

    digest = hashlib.sha256(region.strip().lower().encode("utf-8")).digest()
    first = int.from_bytes(digest[:4], byteorder="big") / (2**32)
    second = int.from_bytes(digest[4:8], byteorder="big") / (2**32)
    return round(first * 10, 6), round(second * 10, 6)


def _score_from_hotspot_level(hotspot_level: int) -> int:
    if hotspot_level >= 2:
        return 85
    if hotspot_level == 1:
        return 55
    return 25


def _risk_label(score: float) -> str:
    if score >= 67:
        return "High"
    if score >= 34:
        return "Medium"
    return "Low"


def _hotspot_as_int(hotspot: object) -> int:
    if isinstance(hotspot, (int, float)):
        return int(round(float(hotspot)))

    if isinstance(hotspot, str):
        normalized = hotspot.strip().lower()
        if normalized in {"high", "severe", "critical", "hotspot"}:
            return 2
        if normalized in {"medium", "moderate"}:
            return 1
        if normalized in {"low", "mild", "safe"}:
            return 0

    return 0


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

    region = str(data.get("region", "")).strip()
    feature1_raw = data.get("feature1")
    feature2_raw = data.get("feature2")

    if feature1_raw is not None and feature2_raw is not None:
        feature1 = float(feature1_raw)
        feature2 = float(feature2_raw)
    elif region:
        feature1, feature2 = _derive_features_from_region(region)
    else:
        feature1 = 0.0
        feature2 = 0.0

    # First two elements are the provided features, remaining 20 are zeros
    # to match the 22-feature expectation of the models.
    vector = [feature1, feature2] + [0.0] * 20
    return vector


def _fallback_prediction(features: list[float]) -> tuple[float, int, None]:
    feature1 = float(features[0])
    feature2 = float(features[1])
    hotspot_level = 2 if feature1 + feature2 >= 12 else 1 if feature1 + feature2 >= 7 else 0
    cases = round((feature1 * 1200) + (feature2 * 900), 2)
    return cases, hotspot_level, None


def make_prediction(data: dict):
    features = _build_feature_vector(data)
    X = [features]

    if regressor is not None and classifier is not None:
        try:
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

            hotspot_level = _hotspot_as_int(hotspot)
        except Exception:
            cases, hotspot_level, risk = _fallback_prediction(features)
    else:
        cases, hotspot_level, risk = _fallback_prediction(features)

    predicted_risk_value = float(risk) if risk is not None else float(_score_from_hotspot_level(hotspot_level))
    risk_score = max(0.0, min(100.0, predicted_risk_value * 100.0 if predicted_risk_value <= 1 else predicted_risk_value))

    requested_region = str(data.get("region", "")).strip() or "Global"

    result = {
        "region": requested_region,
        "risk": _risk_label(risk_score),
        "regions": [{"name": requested_region, "risk": round(risk_score, 2)}],
        "predicted_cases": cases,
        "hotspot_level": hotspot_level,
    }

    if risk is not None:
        result["predicted_risk"] = risk

    return result