from fastapi import APIRouter, HTTPException

from app.schemas.schemas import (
    OkResponse,
    UserLoginRequest,
    UserPasswordLookupRequest,
    UserPasswordResetRequest,
    UserPredictionUpdateRequest,
    UserProfileResponse,
    UserProfileUpdateRequest,
    UserSignupRequest,
)
from app.services.user_service import (
    UserOperationError,
    get_user_profile,
    login_user,
    register_prediction,
    request_password_reset,
    reset_user_password,
    signup_user,
    update_user_profile,
)

router = APIRouter(prefix="/users", tags=["users"])


@router.post("/signup", response_model=UserProfileResponse, status_code=201)
async def signup(payload: UserSignupRequest):
    try:
        return await signup_user(payload)
    except UserOperationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@router.post("/login", response_model=UserProfileResponse)
async def login(payload: UserLoginRequest):
    try:
        return await login_user(payload)
    except UserOperationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@router.get("/profile", response_model=UserProfileResponse)
async def profile(email: str):
    try:
        return await get_user_profile(email)
    except UserOperationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@router.put("/profile", response_model=UserProfileResponse)
async def update_profile(payload: UserProfileUpdateRequest):
    try:
        return await update_user_profile(payload)
    except UserOperationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@router.post("/password/request", response_model=OkResponse)
async def request_reset(payload: UserPasswordLookupRequest):
    try:
        return await request_password_reset(payload)
    except UserOperationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@router.post("/password/reset", response_model=OkResponse)
async def reset_password(payload: UserPasswordResetRequest):
    try:
        return await reset_user_password(payload)
    except UserOperationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@router.post("/prediction", response_model=UserProfileResponse)
async def prediction(payload: UserPredictionUpdateRequest):
    try:
        return await register_prediction(payload)
    except UserOperationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
