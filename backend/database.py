from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import databases

DATABASE_URL = "sqlite:///./medcare.db"

database = databases.Database(DATABASE_URL)
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String)  # manager, doctor, patient
    full_name = Column(String)
    employee_id = Column(String, unique=True, nullable=True)  # For doctors/managers
    patient_id = Column(String, unique=True, nullable=True)   # For patients
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    sender_id = Column(Integer, ForeignKey("users.id"))
    receiver_id = Column(Integer, ForeignKey("users.id"))
    message = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    is_read = Column(Boolean, default=False)
    
    sender = relationship("User", foreign_keys=[sender_id])
    receiver = relationship("User", foreign_keys=[receiver_id])

class Prescription(Base):
    __tablename__ = "prescriptions"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("users.id"))
    doctor_id = Column(Integer, ForeignKey("users.id"))
    medication_name = Column(String)
    dosage = Column(String)
    frequency = Column(String)
    duration = Column(String)
    instructions = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    patient = relationship("User", foreign_keys=[patient_id])
    doctor = relationship("User", foreign_keys=[doctor_id])

class FileRecord(Base):
    __tablename__ = "file_records"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    file_hash = Column(String)
    verification_hash = Column(String, nullable=True)
    uploaded_by = Column(Integer, ForeignKey("users.id"))
    file_type = Column(String)
    file_size = Column(Integer)
    upload_timestamp = Column(DateTime, default=datetime.utcnow)
    
    uploader = relationship("User")

class DiseaseSymptom(Base):
    __tablename__ = "disease_symptoms"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("users.id"))
    symptoms = Column(Text)  # JSON string of symptoms
    predicted_disease = Column(String)
    confidence_score = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    patient = relationship("User")

# Create tables
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
