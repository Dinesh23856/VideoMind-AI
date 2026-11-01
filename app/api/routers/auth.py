"""Authentication and identity router."""

from __future__ import annotations

import uuid
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.schemas import LoginRequest, RegisterRequest
from app.database.models import User
from app.database.session import SessionLocal
from app.security.auth import (
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
)

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(body: RegisterRequest):
    with SessionLocal() as db:
        existing = db.query(User).filter(User.email == body.email).first()
        if existing:
            raise HTTPException(status_code=400, detail="User with this email already exists")

        user = User(
            id=f"usr_{uuid.uuid4().hex[:12]}",
            email=body.email,
            hashed_password=hash_password(body.password),
            full_name=body.full_name,
            role=body.role,
        )
        db.add(user)
        db.commit()
        token = create_access_token(user.id, role=user.role)
        return {"user_id": user.id, "email": user.email, "role": user.role, "token": token}


@router.post("/login")
async def login(body: LoginRequest):
    with SessionLocal() as db:
        user = db.query(User).filter(User.email == body.email).first()
        if not user or not verify_password(body.password, user.hashed_password):
            raise HTTPException(status_code=401, detail="Invalid email or password")

        token = create_access_token(user.id, role=user.role)
        return {"user_id": user.id, "email": user.email, "role": user.role, "token": token}


@router.get("/me")
async def get_me(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Returns the authenticated user's profile."""
    return current_user
