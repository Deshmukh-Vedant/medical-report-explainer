"""
routes/profile.py
------------------
Handles authenticated user profile viewing and updating:
  GET  /profile       — returns current user info (already in auth.py, this is redundant — skip)
  PUT  /profile       — update display name
  POST /change-password — change password (requires current password verification)
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from models.database import get_db
from models.user import User
from routes.auth import get_current_user, hash_password, verify_password
from schemas.user import UserUpdate, ChangePasswordRequest, UserPublic, MessageResponse
from utils.email import send_password_changed_email

logger = logging.getLogger("profile")

router = APIRouter(tags=["Profile"])


@router.put("/profile", response_model=UserPublic)
def update_profile(
    payload: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Updates the authenticated user's display name."""
    current_user.name = payload.name.strip()
    db.commit()
    db.refresh(current_user)
    return current_user


@router.post("/change-password", response_model=MessageResponse)
async def change_password(
    payload: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Changes the authenticated user's password after verifying the current password.
    Sends a notification email on success.
    """
    if not verify_password(payload.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect.",
        )

    current_user.hashed_password = hash_password(payload.new_password)
    db.commit()

    await send_password_changed_email(current_user.email, current_user.name)

    return MessageResponse(message="Password changed successfully.")
