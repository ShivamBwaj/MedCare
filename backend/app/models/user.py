import enum
import datetime as dt
from sqlalchemy import Column, String, DateTime, Enum, Boolean
from sqlalchemy import String as UUID
from uuid import uuid4
from app.models.base import Base

class Role(str, enum.Enum):
    admin = "admin"
    manager = "manager"
    doctor = "doctor"
    patient = "patient"

class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(Enum(Role), nullable=False, default=Role.patient)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
