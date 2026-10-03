import os
import sys
import secrets
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import jwt, JWTError
import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from gateway.db.database import get_db
from gateway.db.models import User, RefreshToken

SECRET_KEY = os.getenv("JWT_SECRET", "5224")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login", auto_error=False)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        password_bytes = plain_password.encode('utf-8')[:72]
        hashed_bytes = hashed_password.encode('utf-8')
        return bcrypt.checkpw(password_bytes, hashed_bytes)
    except Exception:
        return False

def get_password_hash(password: str) -> str:
    password_bytes = password.encode('utf-8')[:72]
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
def hash_token(raw_token: str) -> str:
    # Refresh tokens are high-entropy random strings, not human passwords, so a
    # fast, unsalted SHA-256 digest is the right tool here (unlike bcrypt for
    # passwords): there's no weak-password/brute-force risk to defend against,
    # we just need to avoid storing the bearer token itself in the database.
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def create_refresh_token(db: Session, user_id: int) -> str:
    """Issue a new refresh token for user_id, store only its hash, return the raw token."""
    raw_token = secrets.token_urlsafe(48)
    expires_at = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    db.add(RefreshToken(user_id=user_id, token_hash=hash_token(raw_token), expires_at=expires_at))
    db.commit()
    return raw_token
def get_current_user(token: Optional[str] = Depends(oauth2_scheme), db: Session = Depends(get_db)):
   
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    if not token:
        raise credentials_exception

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = db.query(User).filter(User.id == int(user_id)).first()
    if user is None:
        raise credentials_exception
    return user
