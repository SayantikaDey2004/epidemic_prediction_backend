from fastapi import FastAPI
from app.router import routes_predict, routes_dashboard, routes_home

app = FastAPI(title="COVID Prediction API")

# Include routes
app.include_router(routes_home.router)
app.include_router(routes_predict.router)
app.include_router(routes_dashboard.router)