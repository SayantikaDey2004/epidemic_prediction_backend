import bisect
import math
import time

from app.ml.loader import regressor, classifier
from app.ml.country_features import build_model_feature_vector, list_available_countries


_RISK_DISTRIBUTION_TTL_SECONDS = 60 * 30
_risk_distribution_cache: tuple[float, list[float]] | None = None


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


def _compose_risk_score(
    cases: float,
    risk: float | None,
    hotspot_level: int,
    hotspot_signal: float,
) -> float:
    if risk is None:
        base_risk = _score_from_hotspot_level(hotspot_level) / 100.0
    else:
        raw_risk = float(risk)
        base_risk = raw_risk if raw_risk <= 1 else raw_risk / 100.0

    base_risk = max(0.0, min(1.0, base_risk))
    safe_cases = max(0.0, float(cases))
    case_signal = math.log1p(safe_cases) / math.log1p(50000.0)
    case_signal = max(0.0, min(1.0, case_signal))

    combined_risk = (
        (base_risk * 0.55)
        + (hotspot_signal * 0.30)
        + (case_signal * 0.15)
    )

    # Sigmoid calibration widens 0-100 spread while preserving model ranking.
    calibrated = 1.0 / (1.0 + math.exp(-8.0 * (combined_risk - 0.45)))
    return max(0.0, min(100.0, calibrated * 100.0))


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
    hotspot_confidence = _hotspot_signal(hotspot_level, hotspot_probabilities)
    raw_score = _compose_risk_score(cases, risk, hotspot_level, hotspot_confidence)

    return {
        "cases": cases,
        "risk": risk,
        "hotspot_level": hotspot_level,
        "hotspot_probabilities": hotspot_probabilities,
        "raw_score": raw_score,
        "feature_meta": feature_meta,
    }


def _get_risk_score_distribution() -> list[float]:
    global _risk_distribution_cache

    now = time.time()
    if _risk_distribution_cache and now - _risk_distribution_cache[0] < _RISK_DISTRIBUTION_TTL_SECONDS:
        return _risk_distribution_cache[1]

    distribution: list[float] = []
    for country in list_available_countries():
        try:
            outputs = _predict_model_outputs(country)
            distribution.append(float(outputs["raw_score"]))
        except Exception:
            continue

    if not distribution:
        raise RuntimeError("Unable to build model score distribution")

    distribution.sort()
    _risk_distribution_cache = (now, distribution)
    return distribution


def _to_percentile_score(raw_score: float, distribution: list[float]) -> float:
    if not distribution:
        return max(0.0, min(100.0, raw_score))

    if len(distribution) == 1:
        return 50.0

    rank_index = bisect.bisect_right(distribution, raw_score) - 1
    rank_index = max(0, min(len(distribution) - 1, rank_index))
    percentile = (rank_index / (len(distribution) - 1)) * 100.0
    return max(0.0, min(100.0, percentile))


def get_model_summary() -> dict[str, object]:
    distribution = _get_risk_score_distribution()
    counts = {"Low": 0, "Medium": 0, "High": 0}

    if len(distribution) == 1:
        counts[_risk_label(50.0)] = 1
    else:
        for idx in range(len(distribution)):
            percentile = (idx / (len(distribution) - 1)) * 100.0
            counts[_risk_label(percentile)] += 1

    return {
        "countries_considered": len(distribution),
        "risk_label_counts": counts,
        "thresholds": {"low_max": 33, "medium_max": 66, "high_min": 67},
        "scoring": "model_only_percentile_ranking",
        "raw_score_range": {
            "min": round(distribution[0], 4),
            "max": round(distribution[-1], 4),
        },
    }


def make_prediction(data: dict):
    requested_country = str(data.get("country") or data.get("region") or "").strip()
    if not requested_country:
        raise ValueError("Country is required to run prediction")

    if regressor is None or classifier is None:
        raise RuntimeError("Model artifacts could not be loaded. Check Python package versions and model files.")

    outputs = _predict_model_outputs(requested_country)
    cases = float(outputs["cases"])
    risk = outputs["risk"]
    hotspot_level = int(outputs["hotspot_level"])
    hotspot_probabilities = outputs["hotspot_probabilities"]
    raw_score = float(outputs["raw_score"])
    feature_meta = outputs["feature_meta"]

    try:
        risk_distribution = _get_risk_score_distribution()
        risk_score = _to_percentile_score(raw_score, risk_distribution)
    except Exception:
        risk_score = raw_score

    resolved_country = str(feature_meta.get("country") or requested_country)

    result = {
        "country": resolved_country,
        "region": resolved_country,
        "risk": _risk_label(risk_score),
        "regions": [{"name": resolved_country, "risk": round(risk_score, 2)}],
        "predicted_cases": cases,
        "hotspot_level": hotspot_level,
        "feature_snapshot_date": feature_meta.get("date"),
        "model_evidence": {
            "raw_model_score": round(raw_score, 4),
            "percentile_score": round(risk_score, 4),
            "feature_snapshot_date": feature_meta.get("date"),
            "hotspot_probabilities": {
                "low": round(float(hotspot_probabilities.get("low", 0.0)), 4),
                "medium": round(float(hotspot_probabilities.get("medium", 0.0)), 4),
                "high": round(float(hotspot_probabilities.get("high", 0.0)), 4),
            },
            "scoring": "model_only_percentile_ranking",
        },
    }

    if risk is not None:
        result["predicted_risk"] = risk

    return result