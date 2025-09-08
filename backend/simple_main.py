from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional
import json
from datetime import datetime, timedelta
import hashlib
from passlib.context import CryptContext
from jose import JWTError, jwt

# Simple in-memory databases
users_db = [
    {
        "id": 1,
        "email": "manager@medcare.com",
        "username": "manager1",
        "full_name": "Hospital Manager",
        "role": "manager",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # manager123
        "employee_id": "MGR001",
        "patient_id": None
    },
    {
        "id": 2,
        "email": "doctor@medcare.com", 
        "username": "doctor1",
        "full_name": "Dr. Sarah Johnson",
        "role": "doctor",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # doctor123
        "employee_id": "DOC001",
        "patient_id": None
    },
    {
        "id": 3,
        "email": "patient@medcare.com",
        "username": "patient1", 
        "full_name": "John Smith",
        "role": "patient",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # patient123
        "employee_id": None,
        "patient_id": "PAT001"
    }
]

messages_db = []
prescriptions_db = [
    {
        "id": 1,
        "patient_id": 3,
        "doctor_id": 2,
        "medication_name": "Amoxicillin",
        "dosage": "500mg",
        "frequency": "Twice daily",
        "doctor_name": "Dr. Sarah Johnson",
        "created_at": "2024-01-15T10:00:00Z"
    }
]

# Security
SECRET_KEY = "your-secret-key-here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Models
class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user_role: str
    user_id: int
    full_name: str

class ChatMessageCreate(BaseModel):
    receiver_id: int
    message: str

class SymptomPredictionRequest(BaseModel):
    symptoms: List[str]

# App setup
app = FastAPI(title="MedCare Hospital Management API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user = next((u for u in users_db if u["username"] == username), None)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

def get_manager(current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "manager":
        raise HTTPException(status_code=403, detail="Manager access required")
    return current_user

def get_doctor_or_manager(current_user: dict = Depends(get_current_user)):
    if current_user["role"] not in ["doctor", "manager"]:
        raise HTTPException(status_code=403, detail="Doctor or Manager access required")
    return current_user

# Routes
@app.post("/auth/login", response_model=Token)
async def login_user(user_data: UserLogin):
    user = next((u for u in users_db if u["username"] == user_data.username), None)
    if not user or not verify_password(user_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]},
        expires_delta=access_token_expires
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        user_role=user["role"],
        user_id=user["id"],
        full_name=user["full_name"]
    )

@app.get("/auth/me")
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    return {
        "id": current_user["id"],
        "username": current_user["username"],
        "email": current_user["email"],
        "role": current_user["role"],
        "full_name": current_user["full_name"],
        "employee_id": current_user["employee_id"],
        "patient_id": current_user["patient_id"]
    }

@app.get("/users/patients")
async def get_patients(current_user: dict = Depends(get_doctor_or_manager)):
    patients = [u for u in users_db if u["role"] == "patient"]
    return {"patients": [{"id": p["id"], "name": p["full_name"], "patient_id": p["patient_id"]} for p in patients]}

@app.get("/users/doctors")
async def get_doctors(current_user: dict = Depends(get_current_user)):
    doctors = [u for u in users_db if u["role"] == "doctor"]
    return {"doctors": [{"id": d["id"], "name": d["full_name"], "employee_id": d["employee_id"]} for d in doctors]}

@app.post("/chat/send")
async def send_message(message: ChatMessageCreate, current_user: dict = Depends(get_current_user)):
    receiver = next((u for u in users_db if u["id"] == message.receiver_id), None)
    if not receiver:
        raise HTTPException(status_code=404, detail="Receiver not found")
    
    new_message = {
        "id": len(messages_db) + 1,
        "sender_id": current_user["id"],
        "receiver_id": message.receiver_id,
        "message": message.message,
        "timestamp": datetime.now().isoformat(),
        "is_read": False
    }
    messages_db.append(new_message)
    return {"message": "Message sent successfully"}

@app.get("/chat/conversations")
async def get_conversations(current_user: dict = Depends(get_current_user)):
    user_messages = [m for m in messages_db if m["sender_id"] == current_user["id"] or m["receiver_id"] == current_user["id"]]
    partners = {}
    
    for msg in user_messages:
        partner_id = msg["receiver_id"] if msg["sender_id"] == current_user["id"] else msg["sender_id"]
        if partner_id not in partners:
            partner = next((u for u in users_db if u["id"] == partner_id), None)
            if partner:
                partners[partner_id] = {
                    "user_id": partner_id,
                    "name": partner["full_name"],
                    "role": partner["role"],
                    "unread_count": 0
                }
    
    return {"conversations": list(partners.values())}

@app.get("/chat/messages/{partner_id}")
async def get_chat_messages(partner_id: int, current_user: dict = Depends(get_current_user)):
    messages = [
        m for m in messages_db 
        if (m["sender_id"] == current_user["id"] and m["receiver_id"] == partner_id) or
           (m["sender_id"] == partner_id and m["receiver_id"] == current_user["id"])
    ]
    
    partner = next((u for u in users_db if u["id"] == partner_id), None)
    
    message_list = []
    for msg in messages:
        message_list.append({
            "id": msg["id"],
            "sender_id": msg["sender_id"],
            "message": msg["message"],
            "timestamp": msg["timestamp"],
            "is_own_message": msg["sender_id"] == current_user["id"]
        })
    
    return {
        "partner": {"id": partner["id"], "name": partner["full_name"], "role": partner["role"]},
        "messages": message_list
    }

@app.get("/prescriptions/patient/{patient_id}")
async def get_patient_prescriptions(patient_id: int, current_user: dict = Depends(get_current_user)):
    patient_prescriptions = [p for p in prescriptions_db if p["patient_id"] == patient_id]
    return {"prescriptions": patient_prescriptions}

@app.post("/predict/symptoms")
async def predict_disease_from_symptoms(request: SymptomPredictionRequest, current_user: dict = Depends(get_current_user)):
    # Simple symptom prediction simulation
    diseases = [
        {"disease": "Common Cold", "confidence": 85},
        {"disease": "Flu", "confidence": 70},
        {"disease": "Allergies", "confidence": 45}
    ]
    
    recommendations = [
        "Get plenty of rest",
        "Stay hydrated",
        "Consider seeing a doctor if symptoms persist"
    ]
    
    return {
        "predicted_diseases": diseases,
        "recommendations": recommendations
    }

@app.post("/predict/skin")
async def predict_skin_disease(request: dict, current_user: dict = Depends(get_current_user)):
    # Simple skin prediction simulation
    return {
        "predicted_condition": "Eczema",
        "confidence_score": 78,
        "severity": "Mild",
        "recommendations": [
            "Apply moisturizer regularly",
            "Avoid harsh soaps",
            "Consult dermatologist if condition worsens"
        ]
    }

@app.get("/dashboard/stats")
async def get_dashboard_stats(current_user: dict = Depends(get_current_user)):
    return {
        "total_patients": 150,
        "pending_consultations": 8,
        "health_score": 85,
        "upcoming_appointments": 3
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
