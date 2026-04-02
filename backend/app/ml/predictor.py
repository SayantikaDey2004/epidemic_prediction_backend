from app.ml.loader import regressor, classifier
from app.ml.country_features import (
    build_model_feature_vector,
    is_country_in_training_scope,
    list_available_countries,
)


_HIGH_RISK_PROBABILITY_THRESHOLD = 0.14


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


def _risk_label_from_models(hotspot_level: int, hotspot_probabilities: dict[str, float]) -> str:
    p_high = max(0.0, float(hotspot_probabilities.get("high", 0.0)))

    if hotspot_level >= 2 or p_high >= _HIGH_RISK_PROBABILITY_THRESHOLD:
        return "High"
    if hotspot_level == 1:
        return "Medium"
    return "Low"


def _hotspot_as_int(hotspot: object) -> int:
    try:
        return int(round(float(hotspot)))
    except (TypeError, ValueError):
        pass

    if isinstance(hotspot, str):
        normalized = hotspot.strip().lower()
        if normalized in {"high", "severe", "critical", "hotspot"}:
            return 2
        if normalized in {"medium", "moderate"}:
            return 1
        if normalized in {"low", "mild", "safe"}:
            return 0

    return 0


def _hotspot_probabilities(model: object, X: list[list[float]], hotspot_level: int) -> dict[str, float]:
    fallback = {"low": 0.0, "medium": 0.0, "high": 0.0}
    if hotspot_level <= 0:
        fallback["low"] = 1.0
    elif hotspot_level == 1:
        fallback["medium"] = 1.0
    else:
        fallback["high"] = 1.0

    predict_proba = getattr(model, "predict_proba", None)
    classes = getattr(model, "classes_", None)

    if not callable(predict_proba) or classes is None:
        return fallback

    try:
        probabilities = predict_proba(X)[0]
        class_probs: dict[int, float] = {}

        for idx, label in enumerate(classes):
            label_as_int = int(round(float(label)))
            class_probs[label_as_int] = float(probabilities[idx])

        low = max(0.0, class_probs.get(0, 0.0))
        medium = max(0.0, class_probs.get(1, 0.0))
        high = max(0.0, class_probs.get(2, 0.0))
        total = low + medium + high

        if total <= 0:
            return fallback

        return {
            "low": low / total,
            "medium": medium / total,
            "high": high / total,
        }
    except Exception:
        return fallback


def _hotspot_signal(hotspot_level: int, hotspot_probabilities: dict[str, float]) -> float:
    base_signal = max(0.0, min(1.0, hotspot_level / 2.0))
    p_medium = max(0.0, float(hotspot_probabilities.get("medium", 0.0)))
    p_high = max(0.0, float(hotspot_probabilities.get("high", 0.0)))
    blended_signal = (p_medium * 0.55) + (p_high * 1.0)
    return max(base_signal, min(1.0, blended_signal))


def _normalize_model_risk(risk: float | None) -> float | None:
    if risk is None:
        return None

    normalized = float(risk)
    if normalized > 1.0:
        normalized = normalized / 100.0

    return max(0.0, min(1.0, normalized))


def _final_risk_score(
    risk: float | None,
    hotspot_level: int,
    hotspot_probabilities: dict[str, float],
    risk_label: str,
) -> float:
    normalized_risk = _normalize_model_risk(risk)
    base_score = (
        normalized_risk * 100.0
        if normalized_risk is not None
        else float(_score_from_hotspot_level(hotspot_level))
    )

    p_medium = max(0.0, float(hotspot_probabilities.get("medium", 0.0)))
    p_high = max(0.0, float(hotspot_probabilities.get("high", 0.0)))

    if risk_label == "High":
        high_confidence = (p_high - _HIGH_RISK_PROBABILITY_THRESHOLD) / (1.0 - _HIGH_RISK_PROBABILITY_THRESHOLD)
        high_confidence = max(0.0, min(1.0, high_confidence))
        return max(67.0, min(100.0, 67.0 + (high_confidence * 33.0)))

    if risk_label == "Medium":
        medium_support = (p_medium * 0.6) + (p_high * 0.4)
        medium_support = max(0.0, min(1.0, medium_support))
        return max(34.0, min(66.0, 34.0 + (medium_support * 32.0)))

    low_attenuation = max(0.2, min(1.0, 1.0 - p_high))
    return max(0.0, min(33.0, base_score * low_attenuation))


def _predict_model_outputs(country: str) -> dict[str, object]:
    features, feature_meta = build_model_feature_vector(country)
    X = [features]

    reg_pred = regressor.predict(X)[0]
    hotspot_pred = classifier.predict(X)[0]

    if hasattr(reg_pred, "__len__") and not isinstance(reg_pred, (str, bytes)):
        values = list(reg_pred)
        cases = float(values[0])
        risk = float(values[1]) if len(values) > 1 else None
    else:
        cases = float(reg_pred)
        risk = None

    hotspot_level = _hotspot_as_int(hotspot_pred)
    hotspot_probabilities = _hotspot_probabilities(classifier, X, hotspot_level)
    normalized_risk = _normalize_model_risk(risk)
    base_score = normalized_risk * 100.0 if normalized_risk is not None else float(_score_from_hotspot_level(hotspot_level))
    risk_label = _risk_label_from_models(hotspot_level, hotspot_probabilities)
    raw_score = _final_risk_score(risk, hotspot_level, hotspot_probabilities, risk_label)

    return {
        "cases": cases,
        "risk": risk,
        "hotspot_level": hotspot_level,
        "hotspot_probabilities": hotspot_probabilities,
        "risk_label": risk_label,
        "base_score": base_score,
        "raw_score": raw_score,
        "feature_meta": feature_meta,
    }


def get_model_summary() -> dict[str, object]:
    scores: list[float] = []
    counts = {"Low": 0, "Medium": 0, "High": 0}

    for country in list_available_countries():
        try:
            outputs = _predict_model_outputs(country)
            score = float(outputs["raw_score"])
            label = str(outputs.get("risk_label") or _risk_label(score))
            scores.append(score)
            counts[label] += 1
        except Exception:
            continue

    if not scores:
        return {
            "countries_considered": 0,
            "country_scope": "strict_dataset_countries_only",
            "risk_label_counts": counts,
            "thresholds": {"low_max": 33, "medium_max": 66, "high_min": 67},
            "scoring": "classifier_category_with_regressor_support",
            "raw_score_range": {"min": 0.0, "max": 0.0},
        }

    return {
        "countries_considered": len(scores),
        "country_scope": "strict_dataset_countries_only",
        "risk_label_counts": counts,
        "thresholds": {"low_max": 33, "medium_max": 66, "high_min": 67},
        "scoring": "classifier_category_with_regressor_support",
        "raw_score_range": {
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
        },
    }


def make_prediction(data: dict):
    requested_country = str(data.get("country") or data.get("region") or "").strip()
    if not requested_country:
        raise ValueError("Country is required to run prediction")

    if not is_country_in_training_scope(requested_country):
        raise ValueError(
            f"Country '{requested_country}' is outside the model training dataset scope. Please choose a listed country."
        )

    if regressor is None or classifier is None:
        raise RuntimeError("Model artifacts could not be loaded. Check Python package versions and model files.")

    outputs = _predict_model_outputs(requested_country)
    cases = float(outputs["cases"])
    risk = outputs["risk"]
    hotspot_level = int(outputs["hotspot_level"])
    hotspot_probabilities = outputs["hotspot_probabilities"]
    risk_label = str(outputs.get("risk_label") or "Low")
    base_score = float(outputs.get("base_score") or 0.0)
    raw_score = float(outputs["raw_score"])
    feature_meta = outputs["feature_meta"]
    risk_score = max(0.0, min(100.0, raw_score))

    resolved_country = str(feature_meta.get("country") or requested_country)

    result = {
        "country": resolved_country,
        "region": resolved_country,
        "risk": risk_label,
        "regions": [{"name": resolved_country, "risk": round(risk_score, 2)}],
        "predicted_cases": cases,
        "hotspot_level": hotspot_level,
        "feature_snapshot_date": feature_meta.get("date"),
        "model_evidence": {
            "raw_model_score": round(base_score, 4),
            "final_risk_score": round(risk_score, 4),
            "risk_label": risk_label,
            "percentile_score": round(risk_score, 4),
            "feature_snapshot_date": feature_meta.get("date"),
            "hotspot_probabilities": {
                "low": round(float(hotspot_probabilities.get("low", 0.0)), 4),
                "medium": round(float(hotspot_probabilities.get("medium", 0.0)), 4),
                "high": round(float(hotspot_probabilities.get("high", 0.0)), 4),
            },
            "scoring": "classifier_category_with_regressor_support",
        },
    }

    if risk is not None:
        result["predicted_risk"] = risk

    return result