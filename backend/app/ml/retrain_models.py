import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from app.ml.country_features import _load_history_dataset, _mobility_proxy, _to_timeline_points


@dataclass
class SampleRow:
    country: str
    features: list[float]
    next_cases: float
    growth: float


def _daily_delta(values: list[float]) -> list[float]:
    if not values:
        return []

    deltas = [0.0]
    for idx in range(1, len(values)):
        deltas.append(max(0.0, values[idx] - values[idx - 1]))
    return deltas


def _build_rows_for_country(country: str, points: list[dict[str, float]]) -> list[SampleRow]:
    confirmed = [float(point["confirmed"]) for point in points]
    deaths = [float(point["deaths"]) for point in points]
    recovered = [float(point["recovered"]) for point in points]

    cases_daily = _daily_delta(confirmed)
    deaths_daily = _daily_delta(deaths)
    recovered_daily = _daily_delta(recovered)
    active = [max(0.0, confirmed[i] - deaths[i] - recovered[i]) for i in range(len(points))]

    rows: list[SampleRow] = []
    for idx in range(3, len(points) - 1):
        latest_cases = cases_daily[idx]
        rolling_window = cases_daily[max(0, idx - 6) : idx + 1]
        rolling_mean_cases = sum(rolling_window) / max(1, len(rolling_window))
        mobility = _mobility_proxy(latest_cases, rolling_mean_cases)

        latest_date = points[idx]["date"]
        iso_calendar = latest_date.isocalendar()

        features = [
            cases_daily[idx],
            deaths_daily[idx],
            recovered_daily[idx],
            active[idx],
            mobility,
            cases_daily[idx - 1],
            cases_daily[idx - 2],
            cases_daily[idx - 3],
            deaths_daily[idx - 1],
            deaths_daily[idx - 2],
            deaths_daily[idx - 3],
            recovered_daily[idx - 1],
            recovered_daily[idx - 2],
            recovered_daily[idx - 3],
            active[idx - 1],
            active[idx - 2],
            active[idx - 3],
            float(latest_date.year),
            float(latest_date.month),
            float(latest_date.day),
            float(latest_date.weekday()),
            float(iso_calendar.week),
        ]

        next_cases = max(0.0, float(cases_daily[idx + 1]))
        prev_cases = max(0.0, float(cases_daily[idx - 1]))
        growth = (latest_cases - prev_cases) / max(1.0, prev_cases)
        growth = max(-1.0, min(5.0, growth))

        rows.append(
            SampleRow(
                country=country,
                features=features,
                next_cases=next_cases,
                growth=growth,
            )
        )

    return rows


def _build_dataset() -> tuple[np.ndarray, np.ndarray, dict[str, list[int]], float, float, float]:
    entries = _load_history_dataset()
    rows: list[SampleRow] = []

    for item in entries.values():
        country = str(item.get("country", "")).strip()
        if not country:
            continue

        try:
            points = _to_timeline_points(country, item)
        except Exception:
            continue

        if len(points) < 12:
            continue

        rows.extend(_build_rows_for_country(country, points))

    if not rows:
        raise RuntimeError("No training rows could be built from the historical dataset")

    all_next_cases = np.array([row.next_cases for row in rows], dtype=float)
    case_scale = float(np.percentile(all_next_cases, 95))
    case_scale = max(1.0, case_scale)

    X: list[list[float]] = []
    y_reg: list[list[float]] = []
    country_to_indices: dict[str, list[int]] = {}

    for row in rows:
        case_component = min(1.0, math.log1p(row.next_cases) / math.log1p(case_scale))
        growth_component = min(1.0, max(0.0, row.growth / 2.0))
        risk_target = max(0.0, min(1.0, (0.75 * case_component) + (0.25 * growth_component)))

        sample_idx = len(X)
        X.append(row.features)
        y_reg.append([row.next_cases, risk_target])
        country_to_indices.setdefault(row.country, []).append(sample_idx)

    X_np = np.array(X, dtype=float)
    y_np = np.array(y_reg, dtype=float)

    q1, q2 = np.quantile(y_np[:, 1], [0.33, 0.66])
    return X_np, y_np, country_to_indices, case_scale, float(q1), float(q2)


def _temporal_country_split(country_to_indices: dict[str, list[int]]) -> tuple[list[int], list[int]]:
    train_indices: list[int] = []
    test_indices: list[int] = []

    for indices in country_to_indices.values():
        if len(indices) < 8:
            train_indices.extend(indices)
            continue

        cut = int(len(indices) * 0.75)
        cut = min(max(cut, 4), len(indices) - 2)
        train_indices.extend(indices[:cut])
        test_indices.extend(indices[cut:])

    if not train_indices or not test_indices:
        raise RuntimeError("Failed to build temporal country train/test split")

    return train_indices, test_indices


def _status_from_smape(smape_pct: float) -> str:
    if smape_pct <= 20.0:
        return "good"
    if smape_pct <= 40.0:
        return "ok"
    return "poor"


def retrain_and_save() -> dict[str, object]:
    X, y_reg, country_to_indices, case_scale, q1, q2 = _build_dataset()
    train_idx, test_idx = _temporal_country_split(country_to_indices)

    X_train = X[train_idx]
    y_train = y_reg[train_idx]
    X_test = X[test_idx]
    y_test = y_reg[test_idx]

    y_cls_train = np.where(y_train[:, 1] >= q2, 2, np.where(y_train[:, 1] >= q1, 1, 0))
    y_cls_test = np.where(y_test[:, 1] >= q2, 2, np.where(y_test[:, 1] >= q1, 1, 0))

    regressor = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=350,
            max_depth=16,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
    )
    regressor.fit(X_train, y_train)

    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    classifier.fit(X_train, y_cls_train)

    reg_pred_test = regressor.predict(X_test)
    pred_cases_test = np.maximum(0.0, reg_pred_test[:, 0])
    true_cases_test = y_test[:, 0]

    mae = float(np.mean(np.abs(pred_cases_test - true_cases_test)))
    rmse = float(np.sqrt(np.mean((pred_cases_test - true_cases_test) ** 2)))
    smape = float(
        np.mean(
            (2.0 * np.abs(pred_cases_test - true_cases_test)
             / np.maximum(1.0, np.abs(pred_cases_test) + np.abs(true_cases_test))) * 100.0
        )
    )

    cls_pred_test = classifier.predict(X_test)
    cls_accuracy = float(np.mean(cls_pred_test == y_cls_test))

    per_country: dict[str, dict[str, object]] = {}
    for local_idx, global_idx in enumerate(test_idx):
        country = ""
        for name, indices in country_to_indices.items():
            if global_idx in indices:
                country = name
                break

        if not country:
            continue

        bucket = per_country.setdefault(country, {"true": [], "pred": []})
        bucket["true"].append(float(true_cases_test[local_idx]))
        bucket["pred"].append(float(pred_cases_test[local_idx]))

    report_rows: list[dict[str, object]] = []
    status_counts = {"good": 0, "ok": 0, "poor": 0}

    for country, data in per_country.items():
        true_values = data["true"]
        pred_values = data["pred"]

        abs_errors = [abs(p - t) for p, t in zip(pred_values, true_values)]
        ape_values = [abs(p - t) / max(1.0, t) * 100.0 for p, t in zip(pred_values, true_values)]
        smape_values = [
            (2.0 * abs(p - t) / max(1.0, abs(p) + abs(t))) * 100.0
            for p, t in zip(pred_values, true_values)
        ]

        country_mae = mean(abs_errors)
        country_rmse = math.sqrt(mean([(p - t) ** 2 for p, t in zip(pred_values, true_values)]))
        country_mape = mean(ape_values)
        country_smape = mean(smape_values)
        status = _status_from_smape(country_smape)
        status_counts[status] += 1

        report_rows.append(
            {
                "country": country,
                "samples": len(true_values),
                "mae": round(country_mae, 4),
                "rmse": round(country_rmse, 4),
                "mape_pct": round(country_mape, 4),
                "median_ape_pct": round(median(ape_values), 4),
                "smape_pct": round(country_smape, 4),
                "status": status,
            }
        )

    report_rows.sort(key=lambda row: (row["status"], row["smape_pct"], row["country"]))

    model_dir = Path(__file__).resolve().parent.parent / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    regressor_path = model_dir / "regressor.joblib"
    classifier_path = model_dir / "classifier.joblib"
    joblib.dump(regressor, regressor_path)
    joblib.dump(classifier, classifier_path)

    report_path = model_dir.parent / "country_accuracy_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "country",
                "samples",
                "mae",
                "rmse",
                "mape_pct",
                "median_ape_pct",
                "smape_pct",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(report_rows)

    summary = {
        "samples_total": int(X.shape[0]),
        "train_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
        "countries_total": int(len(country_to_indices)),
        "countries_evaluated": int(len(report_rows)),
        "target_case_scale_p95": round(case_scale, 4),
        "risk_quantiles": {"q33": round(q1, 4), "q66": round(q2, 4)},
        "test_metrics": {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "smape_pct": round(smape, 4),
            "classifier_accuracy": round(cls_accuracy, 4),
        },
        "country_status_counts": status_counts,
        "artifacts": {
            "regressor": str(regressor_path),
            "classifier": str(classifier_path),
            "country_report": str(report_path),
        },
    }

    summary_path = model_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["artifacts"]["training_summary"] = str(summary_path)

    return summary


if __name__ == "__main__":
    result = retrain_and_save()
    print(json.dumps(result, indent=2))