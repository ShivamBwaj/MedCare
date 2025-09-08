import datetime as dt
from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey
from sqlalchemy import String as UUID
from uuid import uuid4
from app.models.base import Base

class AdherenceLog(Base):
    __tablename__ = "adherence_logs"
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    patient_id = Column(String, ForeignKey("users.id"))
    medication = Column(String, nullable=False)
    due_time = Column(DateTime, nullable=False)
    taken = Column(Boolean, default=False)
    logged_at = Column(DateTime, default=dt.datetime.utcnow)
