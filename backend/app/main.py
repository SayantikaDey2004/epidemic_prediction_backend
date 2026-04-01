from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.config import ALLOWED_ORIGINS
from app.core.exceptions import MLModelError, DatabaseError
from app.router import routes_predict, routes_dashboard, routes_home, routes_users
from db.mongodb import ensure_prediction_indexes, cleanup_legacy_prediction_collections

app = FastAPI(title="COVID Prediction API")


@app.on_event("startup")
async def startup_event():
	await cleanup_legacy_prediction_collections()
	await ensure_prediction_indexes()


app.add_middleware(
	CORSMiddleware,
	allow_origins=ALLOWED_ORIGINS,
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
	return JSONResponse(
		status_code=422,
		content={
			"error": {
				"type": "ValidationError",
				"message": "Invalid request payload",
				"details": exc.errors(),
			},
		},
	)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
	return JSONResponse(
		status_code=exc.status_code,
		content={
			"error": {
				"type": "HTTPError",
				"message": exc.detail,
			},
		},
	)


@app.exception_handler(MLModelError)
async def ml_exception_handler(request: Request, exc: MLModelError):
	return JSONResponse(
		status_code=500,
		content={
			"error": {
				"type": "MLModelError",
				"message": str(exc),
			},
		},
	)


@app.exception_handler(DatabaseError)
async def database_exception_handler(request: Request, exc: DatabaseError):
	return JSONResponse(
		status_code=500,
		content={
			"error": {
				"type": "DatabaseError",
				"message": str(exc),
			},
		},
	)


# Include routes
app.include_router(routes_home.router)
app.include_router(routes_predict.router)
app.include_router(routes_dashboard.router)
app.include_router(routes_users.router)