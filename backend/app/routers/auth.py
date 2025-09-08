from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db import get_db
from app.models.user import User, Role
from app.schemas.auth import LoginIn, RegisterIn, TokenOut, MeOut
from app.utils.security import hash_pw, verify_pw, create_token, get_current_user, require_roles

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/register", response_model=MeOut, dependencies=[Depends(require_roles(Role.admin))])
def register(body: RegisterIn, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(400, "Email exists")
    u = User(email=body.email, password_hash=hash_pw(body.password), role=Role(body.role))
    db.add(u)
    db.commit()
    db.refresh(u)
    return MeOut(email=u.email, role=u.role.value)

@router.post("/login", response_model=TokenOut)
def login(body: LoginIn, db: Session = Depends(get_db)):
    u = db.query(User).filter(User.email == body.email).first()
    if not u or not verify_pw(body.password, u.password_hash):
        raise HTTPException(401, "Bad credentials")
    tok = create_token(u.email, u.role.value)
    return TokenOut(access_token=tok)

@router.get("/me", response_model=MeOut)
def me(u: User = Depends(get_current_user)):
    return MeOut(email=u.email, role=u.role.value)
