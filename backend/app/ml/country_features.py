import json
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Dict, List, Tuple


COUNTRIES_API_URL = "https://disease.sh/v3/covid-19/countries"
COUNTRY_HISTORY_URL_TEMPLATE = "https://disease.sh/v3/covid-19/historical/{country}?lastdays=40"
ALL_HISTORY_URL = "https://disease.sh/v3/covid-19/historical?lastdays=40"
CACHE_TTL_SECONDS = 60 * 30


# Fallback list so UI remains usable if the upstream API is unavailable.
FALLBACK_COUNTRIES = [
    "Argentina",
    "Australia",
    "Bangladesh",
    "Brazil",
    "Canada",
    "China",
    "Egypt",
    "France",
    "Germany",
    "India",
    "Indonesia",
    "Iran",
    "Italy",
    "Japan",
    "Kenya",
    "Mexico",
    "Morocco",
    "Nepal",
    "Nigeria",
    "Pakistan",
    "Peru",
    "Philippines",
    "Russia",
    "Saudi Arabia",
    "South Africa",
    "South Korea",
    "Spain",
    "Sri Lanka",
    "Turkey",
    "United Kingdom",
    "United States",
]


_country_list_cache: Tuple[float, List[str]] | None = None
_country_history_cache: Dict[str, Tuple[float, List[Dict[str, float]]]] = {}
_country_alias_map_cache: Tuple[float, Dict[str, str]] | None = None
_history_dataset_cache: Tuple[float, Dict[str, Dict[str, object]]] | None = None
_strict_country_allowlist_cache: Tuple[float, set[str]] | None = None

COUNTRY_ALIASES = {
    "united states": "USA",
    "united states of america": "USA",
    "us": "USA",
    "u s a": "USA",
    "united kingdom": "UK",
    "great britain": "UK",
    "south korea": "S. Korea",
    "republic of korea": "S. Korea",
    "north korea": "N. Korea",
    "democratic peoples republic of korea": "N. Korea",
    "dr congo": "DRC",
    "democratic republic of the congo": "DRC",
    "congo kinshasa": "DRC",
    "cote divoire": "Côte d'Ivoire",
    "ivory coast": "Côte d'Ivoire",
    "curacao": "Curaçao",
    "reunion": "Réunion",
    "laos": "Lao People's Democratic Republic",
    "vatican": "Holy See (Vatican City State)",
}


def _fetch_json(url: str) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": "EpidemicPredictor/1.0"})
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def _normalize_country_name(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    ascii_value = ascii_value.lower().replace("&", " and ")
    ascii_value = re.sub(r"[^a-z0-9]+", " ", ascii_value)
    return " ".join(ascii_value.split())


def _load_history_dataset() -> Dict[str, Dict[str, object]]:
    global _history_dataset_cache

    now = time.time()
    if _history_dataset_cache and now - _history_dataset_cache[0] < CACHE_TTL_SECONDS:
        return _history_dataset_cache[1]

    payload = _fetch_json(ALL_HISTORY_URL)
    if not isinstance(payload, list):
        raise ValueError("Historical payload is not a list")

    entries_by_key: Dict[str, Dict[str, object]] = {}

    for item in payload:
        if not isinstance(item, dict):
            continue

        country_name = str(item.get("country", "")).strip()
        if not country_name:
            continue

        key = _normalize_country_name(country_name)
        if not key:
            continue

        entries_by_key[key] = item

    _history_dataset_cache = (now, entries_by_key)
    return entries_by_key


def _build_country_alias_map() -> Dict[str, str]:
    global _country_alias_map_cache

    now = time.time()
    if _country_alias_map_cache and now - _country_alias_map_cache[0] < CACHE_TTL_SECONDS:
        return _country_alias_map_cache[1]

    alias_map: Dict[str, str] = {}
    entries_by_key = _load_history_dataset()

    for item in entries_by_key.values():
        name = str(item.get("country", "")).strip()
        if name:
            alias_map[_normalize_country_name(name)] = name

    for alias_key, canonical_name in COUNTRY_ALIASES.items():
        canonical_key = _normalize_country_name(canonical_name)
        resolved_name = alias_map.get(canonical_key)
        if resolved_name:
            alias_map[_normalize_country_name(alias_key)] = resolved_name

    _country_alias_map_cache = (now, alias_map)
    return alias_map


def _resolve_country_name(country: str) -> str:
    normalized_input = _normalize_country_name(country)

    if not normalized_input:
        return country

    try:
        alias_map = _build_country_alias_map()
    except Exception:
        return country

    direct_match = alias_map.get(normalized_input)
    if direct_match:
        return direct_match

    # Fallback fuzzy containment for minor naming differences.
    for key, value in alias_map.items():
        if normalized_input in key or key in normalized_input:
            return value

    return country


def _parse_date(value: str) -> datetime:
    for fmt in ("%m/%d/%y", "%m/%d/%Y"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue

    raise ValueError(f"Unsupported date format from upstream API: {value}")


def _to_timeline_points(country: str, payload: object) -> List[Dict[str, float]]:
    timeline = payload.get("timeline") if isinstance(payload, dict) else None
    if not isinstance(timeline, dict):
        raise ValueError(f"No timeline found for country: {country}")

    cases_raw = timeline.get("cases")
    deaths_raw = timeline.get("deaths")
    recovered_raw = timeline.get("recovered")

    if not isinstance(cases_raw, dict) or not isinstance(deaths_raw, dict):
        raise ValueError(f"Incomplete timeline data for country: {country}")

    if not isinstance(recovered_raw, dict):
        recovered_raw = {}

    points: List[Dict[str, float]] = []
    for date_key in sorted(cases_raw.keys(), key=_parse_date):
        confirmed = float(cases_raw.get(date_key, 0) or 0)
        deaths = float(deaths_raw.get(date_key, 0) or 0)
        recovered = float(recovered_raw.get(date_key, 0) or 0)

        points.append(
            {
                "date": _parse_date(date_key),
                "confirmed": max(0.0, confirmed),
                "deaths": max(0.0, deaths),
                "recovered": max(0.0, recovered),
            }
        )

    if len(points) < 8:
        raise ValueError(f"Not enough timeline points for country: {country}")

    return points


def _daily_delta(values: List[float]) -> List[float]:
    if not values:
        return []

    deltas = [0.0]
    for idx in range(1, len(values)):
        deltas.append(max(0.0, values[idx] - values[idx - 1]))
    return deltas


def _mobility_proxy(latest_cases: float, rolling_mean_cases: float) -> float:
    # Training used random mobility between 0.3 and 1.0.
    # We infer a stable proxy in the same range from recent case intensity.
    if rolling_mean_cases <= 0:
        return 0.3

    ratio = latest_cases / rolling_mean_cases
    scaled = 0.55 + (0.15 * min(max(ratio, 0.0), 2.5))
    return round(min(1.0, max(0.3, scaled)), 6)


def list_available_countries() -> List[str]:
    global _country_list_cache

    now = time.time()
    if _country_list_cache and now - _country_list_cache[0] < CACHE_TTL_SECONDS:
        return _country_list_cache[1]

    try:
        entries_by_key = _load_history_dataset()
        countries: List[str] = []

        for item in entries_by_key.values():
            country_name = str(item.get("country", "")).strip()
            if not country_name:
                continue

            try:
                points = _to_timeline_points(country_name, item)
            except Exception:
                continue

            if len(points) >= 8:
                countries.append(country_name)

        countries = sorted(set(countries))

        if countries:
            _country_list_cache = (now, countries)
            return countries
    except Exception:
        try:
            payload = _fetch_json(COUNTRIES_API_URL)
            if not isinstance(payload, list):
                raise ValueError("Countries payload is not a list")

            countries = sorted(
                {
                    str(item.get("country", "")).strip()
                    for item in payload
                    if isinstance(item, dict) and str(item.get("country", "")).strip()
                }
            )

            if countries:
                _country_list_cache = (now, countries)
                return countries
        except Exception:
            pass

    return FALLBACK_COUNTRIES


def get_strict_country_allowlist() -> set[str]:
    global _strict_country_allowlist_cache

    now = time.time()
    if _strict_country_allowlist_cache and now - _strict_country_allowlist_cache[0] < CACHE_TTL_SECONDS:
        return _strict_country_allowlist_cache[1]

    allowlist: set[str] = set()
    for country_name in list_available_countries():
        normalized = _normalize_country_name(country_name)
        if normalized:
            allowlist.add(normalized)

    _strict_country_allowlist_cache = (now, allowlist)
    return allowlist


def is_country_in_training_scope(country: str) -> bool:
    normalized = _normalize_country_name(country)
    if not normalized:
        return False

    return normalized in get_strict_country_allowlist()


def _fetch_country_timeline(country: str) -> List[Dict[str, float]]:
    requested_country = country.strip()
    resolved_country = _resolve_country_name(requested_country)
    country_key = _normalize_country_name(resolved_country)
    now = time.time()

    cached = _country_history_cache.get(country_key)
    if cached and now - cached[0] < CACHE_TTL_SECONDS:
        return cached[1]

    try:
        entries_by_key = _load_history_dataset()
    except Exception:
        entries_by_key = {}

    dataset_entry = entries_by_key.get(_normalize_country_name(resolved_country))
    if dataset_entry:
        points = _to_timeline_points(resolved_country, dataset_entry)
        _country_history_cache[country_key] = (now, points)
        return points

    candidates: List[str] = []
    for item in (requested_country, resolved_country):
        value = item.strip()
        if value and value not in candidates:
            candidates.append(value)

    payload = None
    last_error: Exception | None = None
    resolved_name = resolved_country

    for candidate in candidates:
        encoded_country = urllib.parse.quote(candidate)
        url = COUNTRY_HISTORY_URL_TEMPLATE.format(country=encoded_country)
        try:
            payload = _fetch_json(url)
            resolved_name = candidate
            break
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 404:
                continue
            raise

    if payload is None:
        if isinstance(last_error, urllib.error.HTTPError) and last_error.code == 404:
            raise ValueError(
                f"Historical data unavailable for '{requested_country}'. Please choose another country."
            ) from last_error
        raise ValueError(f"Could not fetch historical data for '{requested_country}'.")

    points = _to_timeline_points(resolved_name, payload)
    _country_history_cache[country_key] = (now, points)
    return points


def build_model_feature_vector(country: str) -> Tuple[List[float], Dict[str, object]]:
    normalized_country = country.strip()
    if not normalized_country:
        raise ValueError("Country is required")

    points = _fetch_country_timeline(normalized_country)

    confirmed = [float(point["confirmed"]) for point in points]
    deaths = [float(point["deaths"]) for point in points]
    recovered = [float(point["recovered"]) for point in points]

    cases_daily = _daily_delta(confirmed)
    deaths_daily = _daily_delta(deaths)
    recovered_daily = _daily_delta(recovered)
    active = [max(0.0, confirmed[i] - deaths[i] - recovered[i]) for i in range(len(points))]

    latest_idx = len(points) - 1
    if latest_idx < 3:
        raise ValueError(f"Not enough data to build lag features for country: {normalized_country}")

    latest_cases = cases_daily[latest_idx]
    rolling_cases_window = cases_daily[max(0, latest_idx - 6) : latest_idx + 1]
    rolling_mean_cases = sum(rolling_cases_window) / max(1, len(rolling_cases_window))
    mobility = _mobility_proxy(latest_cases, rolling_mean_cases)

    latest_date = points[latest_idx]["date"]
    iso_calendar = latest_date.isocalendar()

    feature_vector = [
        cases_daily[latest_idx],
        deaths_daily[latest_idx],
        recovered_daily[latest_idx],
        active[latest_idx],
        mobility,
        cases_daily[latest_idx - 1],
        cases_daily[latest_idx - 2],
        cases_daily[latest_idx - 3],
        deaths_daily[latest_idx - 1],
        deaths_daily[latest_idx - 2],
        deaths_daily[latest_idx - 3],
        recovered_daily[latest_idx - 1],
        recovered_daily[latest_idx - 2],
        recovered_daily[latest_idx - 3],
        active[latest_idx - 1],
        active[latest_idx - 2],
        active[latest_idx - 3],
        float(latest_date.year),
        float(latest_date.month),
        float(latest_date.day),
        float(latest_date.weekday()),
        float(iso_calendar.week),
    ]

    metadata = {
        "country": normalized_country,
        "date": latest_date.strftime("%Y-%m-%d"),
        "rolling_mean_cases": round(rolling_mean_cases, 4),
        "latest_cases": round(latest_cases, 4),
    }

    return feature_vector, metadata
