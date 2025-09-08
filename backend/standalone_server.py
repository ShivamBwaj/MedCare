from fastapi import FastAPI, HTTPException, Depends, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional
import json
from datetime import datetime, timedelta
import hashlib
import secrets
import asyncio
import random

# Simple in-memory databases
users_db = [
    {
        "id": 1,
        "email": "manager@medcare.com",
        "username": "manager1",
        "full_name": "Hospital Manager",
        "role": "manager",
        "password": "manager123",  # Plain text for demo
        "employee_id": "MGR001",
        "patient_id": None
    },
    {
        "id": 2,
        "email": "doctor@medcare.com", 
        "username": "doctor1",
        "full_name": "Dr. Sarah Johnson",
        "role": "doctor",
        "password": "doctor123",  # Plain text for demo
        "employee_id": "DOC001",
        "patient_id": None
    },
    {
        "id": 3,
        "email": "patient@medcare.com",
        "username": "patient1", 
        "full_name": "John Smith",
        "role": "patient",
        "password": "patient123",  # Plain text for demo
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

# Simple token storage
active_tokens = {}

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

class SensorData(BaseModel):
    batchID: str
    temperature: float
    humidity: float
    timestamp: str
    location: str = "Storage Unit A"
    sensorID: str = "SENSOR001"

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                disconnected.append(connection)
        
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()
coldchain_db = []

# App setup
app = FastAPI(title="MedCare Hospital Management API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# Auth functions
def create_access_token(user_id: int, username: str, role: str):
    token = secrets.token_urlsafe(32)
    active_tokens[token] = {
        "user_id": user_id,
        "username": username,
        "role": role,
        "expires": datetime.now() + timedelta(hours=24)
    }
    return token

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    if token not in active_tokens:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    token_data = active_tokens[token]
    if datetime.now() > token_data["expires"]:
        del active_tokens[token]
        raise HTTPException(status_code=401, detail="Token expired")
    
    user = next((u for u in users_db if u["id"] == token_data["user_id"]), None)
    if not user:
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
    if not user or user["password"] != user_data.password:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    access_token = create_access_token(user["id"], user["username"], user["role"])
    
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

# Original inventory and cold chain endpoints for manager
@app.get("/storage")
async def get_storage():
    return [
        {"id": 1, "name": "Amoxicillin 500mg", "stock": 45, "status": "low"},
        {"id": 2, "name": "Surgical Gloves", "stock": 156, "status": "good"},
        {"id": 3, "name": "Insulin Pens", "stock": 12, "status": "critical"}
    ]

@app.get("/alerts")
async def get_alerts():
    return [
        {"id": 1, "message": "Amoxicillin running low", "severity": "warning"},
        {"id": 2, "message": "Insulin Pens critical", "severity": "critical"}
    ]

@app.get("/trials")
async def get_trials():
    return [
        {"batchID": "BATCH001", "drugName": "COVID-19 Vaccine", "status": "approved"},
        {"batchID": "BATCH002", "drugName": "Cancer Treatment", "status": "pending"}
    ]

@app.get("/coldchain/data/{batch_id}")
async def get_coldchain_data(batch_id: str):
    batch_data = [data for data in coldchain_db if data["batchID"] == batch_id]
    if not batch_data:
        # Return empty data instead of 404
        return {"data": [], "batch_id": batch_id}
    return {"data": batch_data[-50:], "batch_id": batch_id}  # Last 50 points

@app.get("/coldchain/risk")
async def get_risk_analysis():
    if not coldchain_db:
        return {
            "risk_level": "SAFE",
            "risk_score": 0,
            "confidence": 95,
            "recommendations": ["No immediate action required"]
        }
    
    latest_data = coldchain_db[-1]
    temp = latest_data["temperature"]
    humidity = latest_data["humidity"]
    
    # Risk analysis logic
    risk_score = 0
    risk_level = "SAFE"
    recommendations = []
    
    if temp < 2 or temp > 8:
        risk_score += 40
        recommendations.append("Temperature out of safe range (2-8°C)")
    if humidity < 30 or humidity > 70:
        risk_score += 30
        recommendations.append("Humidity out of optimal range (30-70%)")
    
    if risk_score >= 50:
        risk_level = "CRITICAL"
    elif risk_score >= 25:
        risk_level = "WARNING"
    
    if not recommendations:
        recommendations = ["All parameters within safe ranges"]
    
    return {
        "risk_level": risk_level,
        "risk_score": min(risk_score, 100),
        "confidence": 95,
        "recommendations": recommendations
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    manager.active_connections.append(websocket)
    print(f"WebSocket connected. Total connections: {len(manager.active_connections)}")
    
    try:
        while True:
            # Keep connection alive - just wait for messages
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in manager.active_connections:
            manager.active_connections.remove(websocket)
        print(f"WebSocket disconnected. Total connections: {len(manager.active_connections)}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket in manager.active_connections:
            manager.active_connections.remove(websocket)

async def generate_fake_sensor_data():
    """Generate realistic sensor data with occasional anomalies"""
    print("Starting sensor data generation...")
    
    batch_configs = {
        "BATCH001": {"base_temp": 4.5, "temp_variance": 1.0, "base_humidity": 45},
        "BATCH002": {"base_temp": 3.8, "temp_variance": 0.8, "base_humidity": 47},
        "BATCH003": {"base_temp": 12.0, "temp_variance": 4.0, "base_humidity": 80}  # Critical batch for demo
    }
    
    # Initialize with some historical data
    for batch_id, config in batch_configs.items():
        for i in range(10):
            timestamp = (datetime.now() - timedelta(minutes=i*3)).isoformat()
            temp = config["base_temp"] + random.uniform(-0.5, 0.5)
            humidity = config["base_humidity"] + random.uniform(-5, 5)
            
            sensor_data = SensorData(
                batchID=batch_id,
                temperature=round(temp, 1),
                humidity=round(humidity, 1),
                timestamp=timestamp
            )
            coldchain_db.append(sensor_data.dict())
    
    while True:
        try:
            for batch_id, config in batch_configs.items():
                # BATCH003 has randomized warning temperature
                if batch_id == "BATCH003":
                    temperature = random.uniform(8.2, 9.5)  # Warning range with variation
                    humidity = random.uniform(75, 85)       # Some humidity variation
                else:
                    # Generate temperature with occasional anomalies for other batches
                    if random.random() < 0.2:  # 20% chance of anomaly
                        temperature = config["base_temp"] + random.uniform(-3, 4)
                    else:
                        temperature = config["base_temp"] + random.uniform(-config["temp_variance"], config["temp_variance"])
                    
                    # Generate humidity
                    humidity = config["base_humidity"] + random.uniform(-10, 10)
                
                sensor_data = SensorData(
                    batchID=batch_id,
                    temperature=round(temperature, 1),
                    humidity=round(max(0, min(100, humidity)), 1),
                    timestamp=datetime.now().isoformat()
                )
                
                coldchain_db.append(sensor_data.dict())
                
                # Keep only last 1000 records per batch
                batch_data = [d for d in coldchain_db if d["batchID"] == batch_id]
                if len(batch_data) > 1000:
                    # Remove oldest records for this batch
                    oldest_records = sorted(batch_data, key=lambda x: x["timestamp"])[:len(batch_data)-1000]
                    for record in oldest_records:
                        coldchain_db.remove(record)
                
                # Broadcast to WebSocket clients
                if manager.active_connections:
                    await manager.broadcast(json.dumps({
                        "type": "sensor_data",
                        "data": sensor_data.dict()
                    }))
            
            await asyncio.sleep(3)  # Generate data every 3 seconds
        except Exception as e:
            print(f"Error in sensor data generation: {e}")
            await asyncio.sleep(3)

@app.on_event("startup")
async def startup_event():
    print("Starting background sensor data generation...")
    asyncio.create_task(generate_fake_sensor_data())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
