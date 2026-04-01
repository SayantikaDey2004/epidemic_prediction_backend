from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Dict

from app.core.exceptions import DatabaseError
from app.schemas.schemas import (
    UserLoginRequest,
    UserPasswordLookupRequest,
    UserPasswordResetRequest,
    UserPredictionUpdateRequest,
    UserProfileUpdateRequest,
    UserSignupRequest,
)
from db.mongodb import user_collection


class UserOperationError(Exception):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_password(password: str) -> str:
    return sha256(password.encode("utf-8")).hexdigest()


def _public_user(user_doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": user_doc.get("name", ""),
        "email": user_doc.get("email", ""),
        "region": user_doc.get("region", "Not selected"),
        "lastPrediction": user_doc.get("lastPrediction", "No predictions yet"),
        "accountStatus": user_doc.get("accountStatus", "Active"),
        "predictionsCount": int(user_doc.get("predictionsCount", 0) or 0),
        "lastLogin": user_doc.get("lastLogin", ""),
    }


async def signup_user(payload: UserSignupRequest) -> Dict[str, Any]:
    normalized_email = _normalize_email(payload.email)

    try:
        existing_user = await user_collection.find_one({"email": normalized_email})
    except Exception as exc:  # noqa: BLE001
        raise DatabaseError("Failed to check existing user account") from exc

    if existing_user:
        raise UserOperationError("An account already exists with this email.", 409)

    now_value = _utc_now_iso()
    new_user = {
        "name": payload.name.strip(),
        "email": normalized_email,
        "password": _hash_password(payload.password),
        "region": "Not selected",
        "lastPrediction": "No predictions yet",
        "accountStatus": "Active",
        "predictionsCount": 0,
        "lastLogin": now_value,
        "createdAt": now_value,
        "updatedAt": now_value,
    }

    try:
        await user_collection.insert_one(new_user)
    except Exception as exc:  # noqa: BLE001
        raise DatabaseError("Failed to create user account") from exc

    return _public_user(new_user)


async def login_user(payload: UserLoginRequest) -> Dict[str, Any]:
    normalized_email = _normalize_email(payload.email)

    try:
        existing_user = await user_collection.find_one({"email": normalized_email})
    except Exception as exc:  # noqa: BLE001
        raise DatabaseError("Failed to read user account") from exc

    if not existing_user:
        raise UserOperationError("No account found for this email. Please sign up first.", 404)

    if existing_user.get("password") != _hash_password(payload.password):
        raise UserOperationError("Incorrect password.", 401)

    now_value = _utc_now_iso()

    try:
        await user_collection.update_one(
            {"email": normalized_email},
            {
                "$set": {
                    "accountStatus": "Active",
                    "lastLogin": now_value,
                    "updatedAt": now_value,
                }
            },
        )
    except Exception as exc:  # noqa: BLE001
        raise DatabaseError("Failed to update login status") from exc

    existing_user["accountStatus"] = "Active"
    existing_user["lastLogin"] = now_value
    existing_user["updatedAt"] = now_value
    return _public_user(existing_user)


async def get_user_profile(email: str) -> Dict[str, Any]:
    normalized_email = _normalize_email(email)

    try:
        user_doc = await user_collection.find_one({"email": normalized_email})
    except Exception as exc:  # noqa: BLE001
        raise DatabaseError("Failed to read user profile") from exc

    if not user_doc:
        raise UserOperationError("No profile found for this email.", 404)

    return _public_user(user_doc)


async def update_user_profile(payload: UserProfileUpdateRequest) -> Dict[str, Any]:
    current_email = _normalize_email(payload.current_email)
    next_email = _normalize_email(payload.email)

    try:
        existing_user = await user_collection.find_one({"email": current_email})
    except Exception as exc:  # noqa: BLE001
        raise DatabaseError("Failed to read user profile") from exc

    if not existing_user:
        raise UserOperationError("No active session found.", 404)

    if next_email != current_email:
        try:
            duplicate_user = await user_collection.find_one({"email": next_email})
        except Exception as exc:  # noqa: BLE001
            raise DatabaseError("Failed to validate email uniqueness") from exc

        if duplicate_user:
            raise UserOperationError("An account already exists with this email.", 409)

    now_value = _utc_now_iso()

    try:
        await user_collection.update_one(
            {"email": current_email},
            {
                "$set": {
                    "name": payload.name.strip(),
                    "email": next_email,
                    "updatedAt": now_value,
                }
            },
        )
    except Exception as exc:  # noqa: BLE001
        raise DatabaseError("Failed to update user profile") from exc

    existing_user["name"] = payload.name.strip()
    existing_user["email"] = next_email
    existing_user["updatedAt"] = now_value
    return _public_user(existing_user)


async def request_password_reset(payload: UserPasswordLookupRequest) -> Dict[str, Any]:
    normalized_email = _normalize_email(payload.email)

    try:
        existing_user = await user_collection.find_one({"email": normalized_email})
    except Exception as exc:  # noqa: BLE001
        raise DatabaseError("Failed to check reset request") from exc

    if not existing_user:
        raise UserOperationError("No account found for this email.", 404)

    return {"ok": True}


async def reset_user_password(payload: UserPasswordResetRequest) -> Dict[str, Any]:
    normalized_email = _normalize_email(payload.email)

    try:
        update_result = await user_collection.update_one(
            {"email": normalized_email},
            {
                "$set": {
                    "password": _hash_password(payload.password),
                    "updatedAt": _utc_now_iso(),
                }
            },
        )
    except Exception as exc:  # noqa: BLE001
        raise DatabaseError("Failed to reset user password") from exc

    if update_result.matched_count == 0:
        raise UserOperationError("No account found for this email.", 404)

    return {"ok": True}


async def register_prediction(payload: UserPredictionUpdateRequest) -> Dict[str, Any]:
    normalized_email = _normalize_email(payload.email)

    try:
        user_doc = await user_collection.find_one({"email": normalized_email})
    except Exception as exc:  # noqa: BLE001
        raise DatabaseError("Failed to read user profile") from exc

    if not user_doc:
        raise UserOperationError("No account found for this email.", 404)

    next_count = int(user_doc.get("predictionsCount", 0) or 0) + 1
    now_value = _utc_now_iso()

    try:
        await user_collection.update_one(
            {"email": normalized_email},
            {
                "$set": {
                    "region": payload.region.strip(),
                    "lastPrediction": payload.risk.strip(),
                    "predictionsCount": next_count,
                    "updatedAt": now_value,
                }
            },
        )
    except Exception as exc:  # noqa: BLE001
        raise DatabaseError("Failed to update user prediction stats") from exc

    user_doc["region"] = payload.region.strip()
    user_doc["lastPrediction"] = payload.risk.strip()
    user_doc["predictionsCount"] = next_count
    user_doc["updatedAt"] = now_value
    return _public_user(user_doc)
