import os
import jwt
import datetime as dt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from app.models.user import User, Role
from app.db import get_db

pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2 = OAuth2PasswordBearer(tokenUrl="/auth/login")
SECRET = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")

def hash_pw(p: str) -> str:
    return pwd.hash(p)

def verify_pw(p: str, h: str) -> bool:
    return pwd.verify(p, h)

def create_token(sub: str, role: str) -> str:
    payload = {
        "sub": sub,
        "role": role,
        "exp": dt.datetime.utcnow() + dt.timedelta(hours=12)
    }
    return jwt.encode(payload, SECRET, algorithm="HS256")

def get_current_user(db: Session = Depends(get_db), token: str = Depends(oauth2)) -> User:
    try:
        payload = jwt.decode(token, SECRET, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user = db.query(User).filter(User.email == payload["sub"]).first()
    if not user:
        raise HTTPException(401, "User not found")
    return user

def require_roles(*roles: Role):
    def dep(user: User = Depends(get_current_user)):
        if user.role not in roles:
            raise HTTPException(status_code=403, detail="Forbidden")
        return user
    return dep
