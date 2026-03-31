import os
from typing import List, Optional

MONGO_URL: str = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME: str = os.getenv("DB_NAME", "epidemic_spread_prediction")

# Comma-separated list of allowed origins for CORS, e.g. "http://localhost:5173,https://myapp.com"
_raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS: List[str] = [origin.strip() for origin in _raw_origins.split(",")] if _raw_origins else ["*"]

# Optional API key; if not set, auth is effectively disabled.
API_KEY: Optional[str] = os.getenv("API_KEY") or None
