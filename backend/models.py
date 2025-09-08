from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

# Authentication Models
class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    role: str  # manager, doctor, patient
    full_name: str
    employee_id: Optional[str] = None
    patient_id: Optional[str] = None

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user_role: str
    user_id: int
    full_name: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    role: str
    full_name: str
    employee_id: Optional[str]
    patient_id: Optional[str]
    is_active: bool

# Chat Models
class ChatMessageCreate(BaseModel):
    receiver_id: int
    message: str

class ChatMessageResponse(BaseModel):
    id: int
    sender_id: int
    receiver_id: int
    message: str
    timestamp: datetime
    is_read: bool
    sender_name: str
    receiver_name: str

# Disease Prediction Models
class SymptomPredictionRequest(BaseModel):
    symptoms: List[str]
    patient_age: Optional[int] = None
    patient_gender: Optional[str] = None

class SymptomPredictionResponse(BaseModel):
    predicted_diseases: List[dict]
    confidence_scores: List[float]
    recommendations: List[str]

class SkinDiseaseRequest(BaseModel):
    image_data: str  # Base64 encoded
    patient_id: Optional[int] = None

class SkinDiseaseResponse(BaseModel):
    predicted_condition: str
    confidence_score: float
    severity: str
    recommendations: List[str]

# File Management Models
class FileUploadResponse(BaseModel):
    file_id: int
    filename: str
    file_hash: str
    verification_hash: Optional[str]
    upload_timestamp: datetime

# Prescription Models
class PrescriptionCreate(BaseModel):
    patient_id: int
    medication_name: str
    dosage: str
    frequency: str
    duration: str
    instructions: str

class PrescriptionResponse(BaseModel):
    id: int
    patient_id: int
    doctor_id: int
    medication_name: str
    dosage: str
    frequency: str
    duration: str
    instructions: str
    created_at: datetime
    doctor_name: str
    patient_name: str
