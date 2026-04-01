import os
from pathlib import Path
from typing import List, Optional

try:
	from dotenv import load_dotenv
except ImportError:  # pragma: no cover
	load_dotenv = None


_THIS_FILE = Path(__file__).resolve()
_BACKEND_ROOT = _THIS_FILE.parents[2]
_WORKSPACE_ROOT = _THIS_FILE.parents[3]

# Load env vars from workspace root or backend root when available.
for _env_path in (_WORKSPACE_ROOT / ".env", _BACKEND_ROOT / ".env"):
	if _env_path.exists() and load_dotenv is not None:
		load_dotenv(_env_path, override=False)

MONGO_URL: str = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME: str = os.getenv("DB_NAME", "epidemic_spread_prediction")

# Comma-separated list of allowed origins for CORS, e.g. "http://localhost:5173,https://myapp.com"
_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173")
_configured_origins = [origin.strip().rstrip("/") for origin in _raw_origins.split(",") if origin.strip()] if _raw_origins else ["http://localhost:5173"]
_local_dev_origins = [
	"http://localhost:5173",
	"http://127.0.0.1:5173",
	"http://localhost:5174",
	"http://127.0.0.1:5174",
	"http://localhost:3000",
	"http://127.0.0.1:3000",
]
ALLOWED_ORIGINS: List[str] = list(dict.fromkeys([*_configured_origins, *_local_dev_origins]))

# Optional API key; if not set, auth is effectively disabled.
API_KEY: Optional[str] = os.getenv("API_KEY") or None
