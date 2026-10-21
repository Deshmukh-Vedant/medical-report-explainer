"""
routes/auth.py
----------------
Handles user registration, login, and profile retrieval.

- Passwords are hashed with passlib (bcrypt) — never stored in plaintext.
- Successful login/register issues a JWT access token.
- get_current_user() is a reusable dependency other routers (upload,
  analysis, history) import to protect their endpoints.
"""

import hashlib
import os
import secrets
from datetime import datetime, timedelta

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from bcrypt import checkpw, gensalt, hashpw
from sqlalchemy.orm import Session

from models.database import get_db
from models.user import User
from schemas.user import (
    UserRegister,
    UserLogin,
    UserPublic,
    ForgotPasswordRequest,
    ResetPasswordRequest,
    MessageResponse,
)
from schemas.token import Token, TokenPayload
from utils.email import send_password_reset_email

load_dotenv()

router = APIRouter(tags=["Authentication"])

# --- Config ---
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-secret-change-me")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
RESET_TOKEN_EXPIRE_MINUTES = int(os.getenv("RESET_TOKEN_EXPIRE_MINUTES", "30"))

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


# --- Password helpers ---
def hash_password(plain_password: str) -> str:
    password_bytes = plain_password.encode("utf-8")
    if len(password_bytes) > 72:
        password_bytes = password_bytes[:72]
    return hashpw(password_bytes, gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    password_bytes = plain_password.encode("utf-8")
    if len(password_bytes) > 72:
        password_bytes = password_bytes[:72]
    return checkpw(password_bytes, hashed_password.encode("utf-8"))


# --- JWT helpers ---
def create_access_token(subject: str, expires_minutes: int = ACCESS_TOKEN_EXPIRE_MINUTES) -> str:
    expire = datetime.utcnow() + timedelta(minutes=expires_minutes)
    to_encode = {"sub": subject, "exp": expire}
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
) -> User:
    """
    FastAPI dependency that decodes the JWT bearer token and returns
    the corresponding User ORM object. Raises 401 if invalid/expired
    or the user no longer exists.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenPayload(sub=email, exp=payload.get("exp"))
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.email == token_data.sub).first()
    if user is None:
        raise credentials_exception
    return user


# --- Routes ---
@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
def register(payload: UserRegister, db: Session = Depends(get_db)):
    """Creates a new user account and returns a JWT access token."""
    existing_user = db.query(User).filter(User.email == payload.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email already exists.",
        )

    new_user = User(
        name=payload.name.strip(),
        email=payload.email.lower().strip(),
        hashed_password=hash_password(payload.password),
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    access_token = create_access_token(subject=new_user.email)
    return Token(access_token=access_token)


@router.post("/login", response_model=Token)
def login(payload: UserLogin, db: Session = Depends(get_db)):
    """Authenticates a user and returns a JWT access token."""
    user = db.query(User).filter(User.email == payload.email.lower().strip()).first()

    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
        )

    access_token = create_access_token(subject=user.email)
    return Token(access_token=access_token)


@router.get("/profile", response_model=UserPublic)
def get_profile(current_user: User = Depends(get_current_user)):
    """Returns the logged-in user's basic profile info."""
    return current_user


def _hash_token(raw_token: str) -> str:
    """Returns HMAC-SHA256 hex digest of the raw token string."""
    return hashlib.sha256(raw_token.encode()).hexdigest()


@router.post("/forgot-password", response_model=MessageResponse)
async def forgot_password(payload: ForgotPasswordRequest, db: Session = Depends(get_db)):
    """
    Generates a single-use, time-limited password-reset token and emails it
    to the user. Always returns the same generic message to prevent account enumeration.
    """
    # Generic message regardless of whether email exists
    GENERIC_MSG = "If an account with that email exists, a password reset link has been sent."

    user = db.query(User).filter(User.email == payload.email.lower().strip()).first()
    if not user:
        return MessageResponse(message=GENERIC_MSG)

    raw_token = secrets.token_urlsafe(48)
    user.reset_token_hash = _hash_token(raw_token)
    user.reset_token_expires = datetime.utcnow() + timedelta(minutes=RESET_TOKEN_EXPIRE_MINUTES)
    db.commit()

    reset_link = f"{os.getenv('APP_BASE_URL', 'http://localhost:5500')}/reset_password.html?token={raw_token}"
    await send_password_reset_email(user.email, reset_link)

    return MessageResponse(message=GENERIC_MSG)


@router.post("/reset-password", response_model=MessageResponse)
def reset_password(payload: ResetPasswordRequest, db: Session = Depends(get_db)):
    """
    Validates the reset token and updates the user's password.
    The token is consumed (cleared) immediately after successful use.
    """
    token_hash = _hash_token(payload.token)
    user = db.query(User).filter(User.reset_token_hash == token_hash).first()

    if not user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired reset token.")
    if user.reset_token_expires is None or datetime.utcnow() > user.reset_token_expires:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Reset token has expired. Please request a new one.")

    user.hashed_password = hash_password(payload.new_password)
    user.reset_token_hash = None
    user.reset_token_expires = None
    db.commit()

    return MessageResponse(message="Your password has been reset successfully. Please log in with your new password.")
