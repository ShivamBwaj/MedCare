import datetime as dt
from uuid import uuid4
from sqlalchemy import Column, String, DateTime, ForeignKey, Text
from sqlalchemy import String as UUID
from app.models.base import Base

class Document(Base):
    __tablename__ = "documents"
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    title = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    mime = Column(String, nullable=False)
    sha256 = Column(String, nullable=False, index=True)
    tx_hash = Column(String, nullable=True)
    status = Column(String, default="pending")
    owner_id = Column(String, ForeignKey("users.id"))
    created_at = Column(DateTime, default=dt.datetime.utcnow)
