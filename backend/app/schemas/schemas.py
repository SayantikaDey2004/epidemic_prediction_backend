from pydantic import BaseModel, Field
from typing import Optional

class PredictionInput(BaseModel):
    region: Optional[str] = Field(default=None, min_length=1, max_length=120)
    feature1: Optional[float] = None
    feature2: Optional[float] = None


class UserSignupRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=80)
    email: str = Field(..., min_length=5, max_length=254)
    password: str = Field(..., min_length=8, max_length=128)


class UserLoginRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=254)
    password: str = Field(..., min_length=8, max_length=128)


class UserProfileUpdateRequest(BaseModel):
    current_email: str = Field(..., min_length=5, max_length=254)
    name: str = Field(..., min_length=2, max_length=80)
    email: str = Field(..., min_length=5, max_length=254)


class UserPasswordLookupRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=254)


class UserPasswordResetRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=254)
    password: str = Field(..., min_length=8, max_length=128)


class UserPredictionUpdateRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=254)
    region: str = Field(..., min_length=1, max_length=120)
    risk: str = Field(..., min_length=1, max_length=32)


class UserProfileResponse(BaseModel):
    name: str
    email: str
    region: str
    lastPrediction: str
    accountStatus: str
    predictionsCount: int
    lastLogin: str


class OkResponse(BaseModel):
    ok: bool