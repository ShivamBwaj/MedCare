from pydantic import BaseModel, EmailStr
from enum import Enum

class Role(str, Enum):
    admin = "admin"
    manager = "manager"
    doctor = "doctor"
    patient = "patient"

class LoginIn(BaseModel):
    email: EmailStr
    password: str

class RegisterIn(BaseModel):
    email: EmailStr
    password: str
    role: Role

class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"

class MeOut(BaseModel):
    email: EmailStr
    role: Role
