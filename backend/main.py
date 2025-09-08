from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, WebSocket, WebSocketDisconnect, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from passlib.context import CryptContext
from pydantic import BaseModel
# JWT functionality removed to avoid library conflicts
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import logging
import hashlib
import json
import asyncio
from typing import List

# Load environment variables
load_dotenv()

# Security
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Simple auth configuration (JWT removed)
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple user storage (replace with database in production)
users_db = {
    "admin@medcare.com": {
        "id": 1,
        "email": "admin@medcare.com",
        "full_name": "System Administrator",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # secret
        "role": "admin"
    },
    "manager@medcare.com": {
        "id": 2,
        "email": "manager@medcare.com", 
        "full_name": "Hospital Manager",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # manager123
        "role": "manager"
    },
    "doctor@medcare.com": {
        "id": 3,
        "email": "doctor@medcare.com",
        "full_name": "Dr. Sarah Johnson", 
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # doctor123
        "role": "doctor"
    },
    "patient@medcare.com": {
        "id": 4,
        "email": "patient@medcare.com",
        "full_name": "John Patient",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # patient123
        "role": "patient"
    }
}

# Enhanced WebSocket Connection Manager for Chat
class ChatConnectionManager:
    def __init__(self):
        self.active_connections: dict = {}  # {user_id: websocket}
        self.user_rooms: dict = {}  # {user_id: room_id}
        self.room_users: dict = {}  # {room_id: [user_ids]}

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket
        print(f"User {user_id} connected to chat")

    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            # Remove from room if in one
            if user_id in self.user_rooms:
                room_id = self.user_rooms[user_id]
                if room_id in self.room_users:
                    self.room_users[room_id].discard(user_id)
                    if not self.room_users[room_id]:
                        del self.room_users[room_id]
                del self.user_rooms[user_id]
            
            del self.active_connections[user_id]
            print(f"User {user_id} disconnected from chat")

    async def join_room(self, user_id: str, room_id: str):
        if user_id in self.user_rooms:
            # Leave current room first
            await self.leave_room(user_id)
        
        self.user_rooms[user_id] = room_id
        if room_id not in self.room_users:
            self.room_users[room_id] = set()
        self.room_users[room_id].add(user_id)
        print(f"User {user_id} joined room {room_id}")

    async def leave_room(self, user_id: str):
        if user_id in self.user_rooms:
            room_id = self.user_rooms[user_id]
            if room_id in self.room_users:
                self.room_users[room_id].discard(user_id)
                if not self.room_users[room_id]:
                    del self.room_users[room_id]
            del self.user_rooms[user_id]
            print(f"User {user_id} left room {room_id}")

    async def send_to_user(self, user_id: str, message: dict):
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_text(json.dumps(message))
                return True
            except Exception as e:
                print(f"Error sending message to user {user_id}: {e}")
                self.disconnect(user_id)
        return False

    async def send_to_room(self, room_id: str, message: dict, exclude_user: str = None):
        if room_id in self.room_users:
            for user_id in self.room_users[room_id].copy():
                if user_id != exclude_user:
                    success = await self.send_to_user(user_id, message)
                    if not success:
                        self.room_users[room_id].discard(user_id)

    def get_room_users(self, room_id: str):
        return list(self.room_users.get(room_id, []))

    def is_user_online(self, user_id: str):
        return user_id in self.active_connections

# Legacy Connection Manager for backward compatibility
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass

# Global storage for real-time data
coldchain_data = []
manager = ConnectionManager()
chat_manager = ChatConnectionManager()

# In-memory storage for drug batches
drug_batches = [
    {"batchID": "BATCH001", "drugName": "COVID-19 Vaccine", "expiry": "2024-12-31", "sender": "Pfizer Labs", "receiver": "City Hospital", "status": "approved"},
    {"batchID": "BATCH002", "drugName": "Cancer Treatment", "expiry": "2024-11-15", "sender": "Roche Pharma", "receiver": "City Hospital", "status": "pending"},
    {"batchID": "BATCH003", "drugName": "Diabetes Medication", "expiry": "2025-03-20", "sender": "Novo Nordisk", "receiver": "City Hospital", "status": "approved"},
    {"batchID": "BATCH004", "drugName": "Heart Medication", "expiry": "2024-10-10", "sender": "Merck & Co", "receiver": "City Hospital", "status": "pending"},
    {"batchID": "BATCH005", "drugName": "Antibiotics", "expiry": "2025-01-25", "sender": "Johnson & Johnson", "receiver": "City Hospital", "status": "approved"}
]

# Initialize SQLite database without SQLAlchemy
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    # Load environment variables at startup
    from dotenv import load_dotenv
    load_dotenv()
    
    # Create database tables
    import sqlite3
    
    conn = sqlite3.connect('medcare.db')
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create appointments table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS appointments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            doctor_id INTEGER,
            appointment_date TIMESTAMP,
            status TEXT DEFAULT 'scheduled',
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES users (id),
            FOREIGN KEY (doctor_id) REFERENCES users (id)
        )
    """)
    
    # Create medical_records table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS medical_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            doctor_id INTEGER,
            diagnosis TEXT,
            prescription TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES users (id),
            FOREIGN KEY (doctor_id) REFERENCES users (id)
        )
    """)
    
    # Create medication_schedule table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS medication_schedule (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            medication_name TEXT NOT NULL,
            dosage TEXT NOT NULL,
            frequency TEXT NOT NULL,
            start_date DATE,
            end_date DATE,
            taken_at TIMESTAMP,
            actual_time TIMESTAMP,
            notes TEXT,
            FOREIGN KEY (patient_id) REFERENCES users (id)
        )
    """)
    
    # Create chat_sessions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            doctor_id INTEGER,
            session_type TEXT DEFAULT 'doctor_patient',
            status TEXT DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ended_at TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES users (id),
            FOREIGN KEY (doctor_id) REFERENCES users (id)
        )
    """)
    
    # Create chat_messages table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            sender_id INTEGER,
            sender_type TEXT NOT NULL,
            message TEXT NOT NULL,
            message_type TEXT DEFAULT 'text',
            is_ai_response BOOLEAN DEFAULT FALSE,
            context_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES chat_sessions (id),
            FOREIGN KEY (sender_id) REFERENCES users (id)
        )
    """)
    
    # Create chat_notifications table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            session_id INTEGER NOT NULL,
            message TEXT NOT NULL,
            notification_type TEXT DEFAULT 'chat_request',
            is_read BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id),
            FOREIGN KEY (session_id) REFERENCES chat_sessions (id)
        )
    """)
        
    conn.commit()
    conn.close()
    print("Database created successfully")
    
    # Verify API key is loaded
    import os
    api_key = os.getenv("GROQ_API_KEY")
    if api_key and api_key != "your_groq_api_key_here":
        print(" Groq API key loaded successfully")
    else:
        print(" Groq API key not configured - disease prediction will not work")
    
    # Start background sensor data generation
    print("Starting background sensor data generation...")
    import asyncio
    asyncio.create_task(generate_sensor_data())
    
    yield
    
    # Shutdown code (if needed)
    print("Application shutting down")

app = FastAPI(title="MedCare Hospital Management API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple routers without SQLAlchemy dependencies

class LoginRequest(BaseModel):
    email: str
    password: str

class SymptomRequest(BaseModel):
    symptoms: List[str]

class ChatMessage(BaseModel):
    message: str
    session_id: int = None
    recipient_id: str = None

class ChatSessionCreate(BaseModel):
    patient_id: str
    doctor_id: str = None

@app.post("/auth/login")
async def login(request: LoginRequest):
    print(f"Login attempt: email={request.email}, password={request.password}")
    
    # Simple demo login
    demo_users = {
        "admin@medcare.com": {"role": "admin", "password": "admin123"},
        "manager@medcare.com": {"role": "manager", "password": "manager123"},
        "doctor@medcare.com": {"role": "doctor", "password": "doctor123"},
        "patient@medcare.com": {"role": "patient", "password": "patient123"}
    }
    
    user = demo_users.get(request.email)
    print(f"Found user: {user}")
    
    if user and user["password"] == request.password:
        # Create demo token with role embedded
        demo_token = f"demo_token_{user['role']}"
        
        response = {
            "access_token": demo_token,
            "token_type": "bearer",
            "user": {
                "id": 1,
                "email": request.email, 
                "full_name": f"Demo {user['role'].title()}",
                "role": user["role"]
            }
        }
        print(f"Login successful: {response}")
        return response
    
    print("Login failed - invalid credentials")
    raise HTTPException(status_code=401, detail=f"Invalid credentials for {request.email}")

@app.get("/auth/me")
async def get_current_user(authorization: str = Header(None)):
    # Extract role from token for proper routing
    if authorization and 'demo_token_' in authorization:
        role = authorization.split('_')[-1]
        if role in ['admin', 'manager', 'doctor', 'patient']:
            return {
                "id": 1,
                "email": f"{role}@medcare.com",
                "full_name": f"Demo {role.title()}",
                "role": role
            }
    
    # Default fallback
    return {
        "id": 1,
        "email": "demo@medcare.com",
        "full_name": "Demo User",
        "role": "manager"
    }

# Add test endpoint to verify login works
@app.get("/test/users")
async def test_users():
    return {
        "available_users": [
            {"email": "admin@medcare.com", "password": "admin123", "role": "admin"},
            {"email": "manager@medcare.com", "password": "manager123", "role": "manager"},
            {"email": "doctor@medcare.com", "password": "doctor123", "role": "doctor"},
            {"email": "patient@medcare.com", "password": "patient123", "role": "patient"}
        ]
    }

# Add adherence endpoint with comprehensive dummy data
@app.get("/adherence/{patient_id}")
async def get_adherence(patient_id: str):
    import random
    from datetime import datetime, timedelta
    
    # Dummy patient database
    patients = {
        "P001": {"name": "John Smith", "age": 45, "condition": "Diabetes Type 2"},
        "P002": {"name": "Sarah Johnson", "age": 62, "condition": "Hypertension"},
        "P003": {"name": "Mike Wilson", "age": 38, "condition": "Heart Disease"},
        "P004": {"name": "Lisa Brown", "age": 55, "condition": "Diabetes Type 1"},
        "P005": {"name": "David Lee", "age": 71, "condition": "Multiple Conditions"},
        "patient@medcare.com": {"name": "Demo Patient", "age": 35, "condition": "General Care"}
    }
    
    if patient_id not in patients:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    patient = patients[patient_id]
    
    # Generate realistic adherence data
    base_time = datetime.now()
    logs = []
    medications = {
        "Diabetes Type 2": ["Metformin", "Glipizide", "Insulin"],
        "Hypertension": ["Lisinopril", "Amlodipine", "Hydrochlorothiazide"],
        "Heart Disease": ["Atorvastatin", "Metoprolol", "Aspirin"],
        "Diabetes Type 1": ["Insulin Rapid", "Insulin Long", "Glucagon"],
        "Multiple Conditions": ["Metformin", "Lisinopril", "Atorvastatin", "Aspirin"],
        "General Care": ["Vitamin D", "Multivitamin", "Omega-3"]
    }
    
    patient_meds = medications.get(patient["condition"], ["Generic Med"])
    
    # Generate 30 days of medication logs
    for day in range(30):
        for med in patient_meds:
            for dose_time in ["08:00", "14:00", "20:00"]:
                log_time = base_time - timedelta(days=day)
                log_time = log_time.replace(hour=int(dose_time.split(':')[0]), minute=int(dose_time.split(':')[1]))
                
                # 85% adherence rate with some variation
                taken = random.random() < 0.85
                logged_time = log_time + timedelta(minutes=random.randint(-10, 30))
                
                logs.append({
                    "medication": med,
                    "due_time": log_time.isoformat(),
                    "taken": taken,
                    "logged_at": logged_time.isoformat() if taken else None
                })
    
    # Calculate adherence score
    total_doses = len(logs)
    taken_doses = sum(1 for log in logs if log["taken"])
    adherence_score = (taken_doses / total_doses) * 100 if total_doses > 0 else 0
    
    # Generate weekly adherence trend
    weekly_trend = []
    for week in range(4):
        week_logs = [log for log in logs if (base_time - datetime.fromisoformat(log["due_time"])).days // 7 == week]
        week_taken = sum(1 for log in week_logs if log["taken"])
        week_total = len(week_logs)
        week_score = (week_taken / week_total) * 100 if week_total > 0 else 0
        weekly_trend.append({
            "week": f"Week {4-week}",
            "adherence": round(week_score, 1)
        })
    
    # Generate medication breakdown
    med_breakdown = []
    for med in patient_meds:
        med_logs = [log for log in logs if log["medication"] == med]
        med_taken = sum(1 for log in med_logs if log["taken"])
        med_total = len(med_logs)
        med_score = (med_taken / med_total) * 100 if med_total > 0 else 0
        med_breakdown.append({
            "medication": med,
            "adherence": round(med_score, 1),
            "total_doses": med_total,
            "taken_doses": med_taken
        })
    
    return {
        "patient_id": patient_id,
        "patient_name": patient["name"],
        "age": patient["age"],
        "condition": patient["condition"],
        "adherence_score": round(adherence_score, 1),
        "weekly_trend": weekly_trend,
        "medication_breakdown": med_breakdown,
        "logs": sorted(logs, key=lambda x: x["due_time"], reverse=True)[:50],  # Last 50 logs
        "total_medications": len(patient_meds),
        "critical_alerts": random.randint(0, 3)
    }

@app.post("/ai/symptoms/predict")
async def predict_symptoms(request: SymptomRequest):
    # Map symptoms to diseases
    symptom_diseases = {
        "fever": ["Flu", "COVID-19", "Malaria"],
        "cough": ["Common Cold", "Bronchitis", "COVID-19"],
        "headache": ["Migraine", "Tension Headache", "Flu"],
        "fatigue": ["Flu", "Anemia", "Depression"],
        "nausea": ["Food Poisoning", "Gastritis", "Pregnancy"],
        "vomiting": ["Food Poisoning", "Gastroenteritis", "Migraine"],
        "diarrhea": ["Gastroenteritis", "Food Poisoning", "IBS"],
        "joint pain": ["Arthritis", "Flu", "Lupus"],
        "muscle aches": ["Flu", "Fibromyalgia", "Exercise strain"],
        "shortness of breath": ["Asthma", "COVID-19", "Heart Disease"],
        "chest pain": ["Heart Disease", "Anxiety", "Muscle strain"],
        "abdominal pain": ["Gastritis", "Appendicitis", "IBS"],
        "rash": ["Allergic Reaction", "Eczema", "Viral Infection"],
        "dizziness": ["Low Blood Pressure", "Dehydration", "Inner Ear Problem"],
        "sore throat": ["Strep Throat", "Common Cold", "Viral Infection"]
    }
    
    # Find most likely disease based on symptoms
    disease_scores = {}
    for symptom in request.symptoms:
        if symptom.lower() in symptom_diseases:
            for disease in symptom_diseases[symptom.lower()]:
                disease_scores[disease] = disease_scores.get(disease, 0) + 1
    
    if disease_scores:
        # Get disease with highest score
        predicted_disease = max(disease_scores, key=disease_scores.get)
        confidence = min(0.95, 0.6 + (disease_scores[predicted_disease] * 0.1))
    else:
        predicted_disease = "Unknown Condition"
        confidence = 0.3
    
    return {
        "prediction": predicted_disease,
        "confidence": confidence,
        "recommendations": ["Consult a healthcare professional", "Monitor symptoms", "Rest and stay hydrated"]
    }

@app.post("/ai/parkinsons/predict")
async def predict_parkinsons(
    file: UploadFile = File(...),
    drawing_type: str = "spiral"
):
    """Predict Parkinson's disease from spiral or wave drawing"""
    try:
        # Validate drawing type
        if drawing_type not in ["spiral", "wave"]:
            return {"error": "Drawing type must be 'spiral' or 'wave'"}
        
        # Save uploaded file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Load the trained model and encoder
        import pickle
        import cv2
        from skimage import feature
        
        model_path = f"ai/parkinsons_{drawing_type}_model.pkl"
        encoder_path = f"ai/parkinsons_{drawing_type}_encoder.pkl"
        
        # Check if model files exist
        if not os.path.exists(model_path) or not os.path.exists(encoder_path):
            return {"error": f"Model not found for {drawing_type} drawings"}
        
        # Load model and encoder
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        
        # Preprocess image
        image = cv2.imread(tmp_file_path)
        if image is None:
            return {"error": "Could not process the uploaded image"}
        
        # Convert to grayscale and resize
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (200, 200))
        
        # Threshold the image
        image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
        # Extract HOG features
        features = feature.hog(
            image, 
            orientations=9,
            pixels_per_cell=(10, 10), 
            cells_per_block=(2, 2),
            transform_sqrt=True, 
            block_norm="L1"
        )
        
        # Make prediction
        prediction = model.predict([features])[0]
        probabilities = model.predict_proba([features])[0]
        
        # Get class names
        classes = label_encoder.classes_
        predicted_class = label_encoder.inverse_transform([prediction])[0]
        
        # Calculate confidence
        confidence = float(max(probabilities))
        
        # Prepare result
        result = {
            "prediction": predicted_class,
            "confidence": confidence,
            "drawing_type": drawing_type,
            "probabilities": {
                classes[i]: float(probabilities[i]) 
                for i in range(len(classes))
            },
            "risk_level": "High Risk" if predicted_class == "parkinson" and confidence > 0.7 else 
                         "Moderate Risk" if predicted_class == "parkinson" else "Low Risk",
            "recommendation": get_parkinsons_recommendation(predicted_class, confidence)
        }
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return result
        
    except Exception as e:
        logger.error(f"Parkinson's prediction error: {e}")
        return {"error": f"Prediction failed: {str(e)}"}

def get_parkinsons_recommendation(prediction, confidence):
    """Get recommendation based on Parkinson's prediction"""
    if prediction == "parkinson":
        if confidence > 0.8:
            return "High confidence Parkinson's indicators detected. Please consult a neurologist immediately for proper medical evaluation."
        elif confidence > 0.6:
            return "Possible Parkinson's indicators detected. Consider scheduling an appointment with a healthcare professional for further assessment."
        else:
            return "Some indicators present but low confidence. Monitor symptoms and consult a doctor if concerns persist."
    else:
        return "No significant Parkinson's indicators detected in your drawing. Continue regular health monitoring."

@app.post("/ai/skin/predict")
async def predict_skin(file: UploadFile = File(...)):
    import torch
    import torchvision.transforms as transforms
    from torchvision import models
    import torch.nn as nn
    import hashlib
    import logging
    from PIL import Image
    import io
    import numpy as np
    
    logger.info(f"=== NAIL ANALYSIS REQUEST ===")
    logger.info(f"File: {file.filename}")
    logger.info(f"Content Type: {file.content_type}")
    
    # Read file content
    file_content = await file.read()
    file_hash = hashlib.md5(file_content).hexdigest()
    
    logger.info(f"File Size: {len(file_content)} bytes")
    logger.info(f"File Hash: {file_hash}")
    
    try:
        # Load the trained nail model
        model_path = os.path.join(os.path.dirname(__file__), "ai", "ai", "nail_transfer_model.pt")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Looking for model at: {model_path}")
        logger.info(f"Model file exists: {os.path.exists(model_path)}")
        
        if os.path.exists(model_path):
            logger.info(f"Loading trained nail model: {model_path}")
            try:
                checkpoint = torch.load(model_path, map_location=device)
                logger.info(f"Model checkpoint loaded successfully")
            except Exception as load_error:
                logger.error(f"Failed to load model checkpoint: {load_error}")
                raise load_error
            
            # Create model architecture (MobileNetV2)
            model = models.mobilenet_v2(weights=None)
            num_classes = len(checkpoint["classes"])
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            
            # Load trained weights
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            model = model.to(device)
            
            classes = checkpoint["classes"]
            logger.info(f"Model loaded successfully! Classes: {classes}")
            
            # Validate it's actually an image
            image = Image.open(io.BytesIO(file_content))
            width, height = image.size
            
            logger.info(f"Image dimensions: {width}x{height}")
            logger.info(f"Image mode: {image.mode}")
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # NAIL IMAGE VALIDATION - Check if this looks like a nail image
            img_array = np.array(image)
            mean_rgb = np.mean(img_array, axis=(0, 1))
            std_rgb = np.std(img_array, axis=(0, 1))
            
            logger.info(f"Image analysis - Mean RGB: {mean_rgb}, Std RGB: {std_rgb}")
            
            # SIMPLE NAIL VALIDATION - Focus on skin tone detection
            is_likely_nail = False
            rejection_reasons = []
            
            # Primary check: Look for skin-like colors (nail bed/finger skin)
            # Nails are attached to fingers, so there should be skin tones visible
            skin_pixels = 0
            total_pixels = img_array.shape[0] * img_array.shape[1]
            
            # STRICTER skin tone detection - reject silhouettes and shadows
            valid_skin_pixels = 0
            bright_skin_pixels = 0
            
            for i in range(0, img_array.shape[0], 8):  # Sample more pixels for accuracy
                for j in range(0, img_array.shape[1], 8):
                    r, g, b = img_array[i, j]
                    
                    # Much stricter skin tone criteria
                    # 1. Must be bright enough (not silhouettes/shadows)
                    if r > 80 and g > 60 and b > 40:  # Minimum brightness
                        # 2. Must have proper skin color ratios
                        if (80 < r < 240 and 50 < g < 180 and 30 < b < 150 and
                            r > g and g >= b and r - b > 15 and r - g < 60):
                            valid_skin_pixels += 1
                            
                            # 3. Count well-lit skin pixels (not just any skin tone)
                            if r > 120 and g > 80 and b > 50:
                                bright_skin_pixels += 1
            
            total_sampled = (img_array.shape[0] // 8) * (img_array.shape[1] // 8)
            skin_percentage = (valid_skin_pixels * 100) / total_sampled
            bright_skin_percentage = (bright_skin_pixels * 100) / total_sampled
            
            logger.info(f"Strict skin analysis: {skin_percentage:.1f}% valid skin, {bright_skin_percentage:.1f}% bright skin")
            
            # Relaxed requirements - allow more nail images through
            if skin_percentage > 3 or bright_skin_percentage > 2:  # Much lower thresholds
                is_likely_nail = True
                logger.info("✅ Some skin tones detected - proceeding with analysis")
            else:
                rejection_reasons.append(f"No skin tones detected ({skin_percentage:.1f}% valid, {bright_skin_percentage:.1f}% bright)")
                logger.info("❌ No skin tones found - likely not a nail image")
            
            # Check for screenshots and other non-nail content
            if not is_likely_nail:
                filename = file.filename.lower() if file.filename else ""
                
                # Screenshot detection
                if 'screenshot' in filename or filename.startswith('screen'):
                    rejection_reasons.append("Screenshot detected in filename")
                    logger.info("❌ Screenshot detected - not a nail image")
                
                # Other obvious non-nail content
                non_nail_keywords = ['certificate', 'postman', 'api', 'document', 'wallpaper']
                for keyword in non_nail_keywords:
                    if keyword in filename:
                        rejection_reasons.append(f"Filename indicates non-nail content: '{keyword}'")
                        break
                
                # Check for dark images (screenshots often have dark backgrounds)
                brightness = np.mean(mean_rgb)
                if brightness < 60 and skin_percentage < 2:
                    rejection_reasons.append(f"Very dark image with no skin tones (brightness: {brightness:.1f})")
                    logger.info("❌ Very dark image with no skin - likely screenshot/document")
                    
                # Only proceed if no strong indicators against it
                if len(rejection_reasons) == 1 and 'No skin tones detected' in rejection_reasons[0]:
                    # Give it a chance - let confidence thresholding handle it
                    is_likely_nail = True
                    logger.info("⚠️ Low skin detection but no obvious non-nail indicators - proceeding with caution")
            
            # Reject if multiple indicators suggest it's not a nail
            if not is_likely_nail:
                logger.info(f"Image rejected as non-nail: {rejection_reasons}")
                return {
                    "label": "Invalid image",
                    "confidence": 0.0,
                    "message": f"This doesn't appear to be a nail image. Issues: {', '.join(rejection_reasons[:2])}. Please upload a clear close-up photo of nails."
                }
            
            # Preprocess image for model
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            # Make prediction with confidence thresholding
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                predicted_class = classes[predicted.item()]
                confidence_score = confidence.item()
                
                logger.info(f"Raw prediction: {predicted_class} (confidence: {confidence_score:.3f})")
                
                # Confidence thresholding - reject low confidence predictions
                confidence_threshold = 0.7
                if confidence_score < confidence_threshold:
                    logger.info(f"Prediction rejected: confidence {confidence_score:.3f} below threshold {confidence_threshold}")
                    return {
                        "label": "Uncertain classification",
                        "confidence": round(confidence_score, 3),
                        "message": f"Model confidence too low ({confidence_score:.1%}). This may not be a clear nail image or the condition is unclear. Please try a clearer, well-lit close-up photo.",
                        "model_info": {
                            "type": checkpoint.get("model_type", "mobilenet_v2_transfer"),
                            "accuracy": checkpoint.get("best_val_acc", "Unknown"),
                            "threshold": confidence_threshold
                        }
                    }
                
                # Reduce confidence if image quality issues detected
                if len(rejection_reasons) > 0:
                    confidence_penalty = len(rejection_reasons) * 0.1
                    confidence_score = max(0.3, confidence_score - confidence_penalty)
                    logger.info(f"Confidence reduced due to image issues: {rejection_reasons}")
                    
                    # Re-check threshold after penalty
                    if confidence_score < confidence_threshold:
                        logger.info(f"Prediction rejected after penalty: confidence {confidence_score:.3f} below threshold")
                        return {
                            "label": "Poor image quality",
                            "confidence": round(confidence_score, 3),
                            "message": f"Image quality issues detected. Confidence too low after adjustments ({confidence_score:.1%}). Please upload a clearer nail photo.",
                            "issues": rejection_reasons[:2]
                        }
                
                logger.info(f"Final prediction: {predicted_class} (confidence: {confidence_score:.3f})")
                
                return {
                    "label": predicted_class,
                    "confidence": round(confidence_score, 3),
                    "message": f"AI analysis complete. Detected: {predicted_class}" + 
                              (f" (Note: {len(rejection_reasons)} image quality issues detected)" if rejection_reasons else ""),
                    "model_info": {
                        "type": checkpoint.get("model_type", "mobilenet_v2_transfer"),
                        "accuracy": checkpoint.get("best_val_acc", "Unknown"),
                        "classes": classes,
                        "threshold": confidence_threshold
                    }
                }
        
        else:
            logger.warning(f"Trained model not found at {model_path}")
            logger.warning(f"Current working directory: {os.getcwd()}")
            logger.warning(f"Script directory: {os.path.dirname(__file__)}")
            # Fallback to basic image analysis
            image = Image.open(io.BytesIO(file_content))
            width, height = image.size
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Basic image analysis for nail detection
            img_array = np.array(image)
            mean_rgb = np.mean(img_array, axis=(0, 1))
            std_rgb = np.std(img_array, axis=(0, 1))
            
            logger.info(f"Mean RGB: {mean_rgb}")
            logger.info(f"Std RGB: {std_rgb}")
            
            # Simple heuristics to detect if it looks like a nail
            is_likely_nail = True
            confidence_factors = []
            
            # Check if image is too dark (likely not a nail)
            brightness = np.mean(mean_rgb)
            if brightness < 50:
                is_likely_nail = False
                confidence_factors.append("Image too dark for nail analysis")
                logger.info("Image rejected: too dark")
            
            # If it doesn't look like a nail at all
            if not is_likely_nail:
                logger.info("Image analysis: Not a nail image")
                return {
                    "label": "Invalid image",
                    "confidence": 0.15,
                    "message": "This doesn't appear to be a nail image. Please upload a clear photo of nails."
                }
            
            # Fallback prediction
            hash_int = int(file_hash[:8], 16)
            fallback_conditions = [
                {"label": "healthy", "confidence": 0.82},
                {"label": "fungal", "confidence": 0.75},
                {"label": "psoriasis", "confidence": 0.70}
            ]
            
            selected = fallback_conditions[hash_int % len(fallback_conditions)]
            logger.info(f"Fallback prediction: {selected['label']} ({selected['confidence']})")
            
            return {
                "label": selected["label"],
                "confidence": selected["confidence"],
                "message": f"Fallback analysis: {selected['label']} (trained model not available)"
            }
        
        # Select condition based on hash but adjust confidence based on image
        selected_condition = base_conditions[hash_int % len(base_conditions)]
        
        # Reduce confidence if image quality issues detected
        confidence_penalty = len(confidence_factors) * 0.1
        final_confidence = max(0.4, min(0.95, selected_condition["base_confidence"] - confidence_penalty))
        
        result = {
            "label": selected_condition["label"],
            "confidence": final_confidence
        }
        
        if confidence_factors:
            logger.info(f"Image quality issues: {confidence_factors}")
        
        logger.info(f"Prediction: {result['label']} (confidence: {result['confidence']:.3f})")
        logger.info(f"=== NAIL ANALYSIS COMPLETED ===")
        
        return {
            "label": result["label"],
            "confidence": result["confidence"],
            "message": "Demo analysis - consult a dermatologist for real diagnosis"
        }
        
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {
            "label": "Processing error",
            "confidence": 0.0,
            "message": f"Unable to process image: {str(e)}"
        }

@app.post("/ai/disease/predict")
async def predict_disease_risk(request: dict):
    import os
    import logging
    import httpx
    
    logger = logging.getLogger(__name__)
    
    symptoms = request.get("symptoms", "")
    history = request.get("history", "")
    
    if not symptoms.strip():
        return {
            "error": "No symptoms provided",
            "message": "Please describe your symptoms for analysis."
        }
    
    try:
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        # Use Groq API for disease prediction
        groq_api_key = os.getenv("GROQ_API_KEY")
        
        if not groq_api_key:
            return {
                "error": "API configuration missing",
                "message": "Groq API key not configured. Please contact administrator.",
                "fallback": True
            }
        
        # Construct medical prompt
        medical_prompt = f"""You are a medical AI assistant. Analyze the following symptoms and provide a professional assessment.

SYMPTOMS: {symptoms}
MEDICAL HISTORY: {history if history else "None provided"}

Respond ONLY with valid JSON in this exact format:
{{
    "primary_prediction": "Most likely condition name",
    "confidence_percentage": 85,
    "top_predictions": [
        {{"disease": "Condition 1", "probability": 85}},
        {{"disease": "Condition 2", "probability": 10}},
        {{"disease": "Condition 3", "probability": 5}}
    ],
    "analysis": {{
        "symptoms_analyzed": "{symptoms}",
        "method": "AI Medical Analysis",
        "reasoning": "Brief medical reasoning"
    }},
    "recommendations": [
        "Specific recommendation 1",
        "Specific recommendation 2",
        "Specific recommendation 3"
    ],
    "urgency_level": "low",
    "seek_immediate_care": false
}}

IMPORTANT: Return ONLY the JSON object, no other text."""

        # Call Groq API using official client
        from groq import Groq
        
        client = Groq(api_key=groq_api_key)
        
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical AI assistant. Always respond with valid JSON only."
                },
                {
                    "role": "user", 
                    "content": medical_prompt
                }
            ],
            model="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=1000
        )
        
        # Extract AI response
        ai_text = response.choices[0].message.content.strip()
        
        # Try to parse JSON from AI response
        import json
        try:
            # Clean and parse JSON
            if ai_text.startswith('```json'):
                ai_text = ai_text.replace('```json', '').replace('```', '').strip()
            
            ai_result = json.loads(ai_text)
            
            # Add additional metadata
            ai_result.update({
                "status": "success",
                "ai_provider": "Groq (Llama3)",
                "disclaimer": "⚠️ This AI analysis is for informational purposes only and should not replace professional medical advice.",
                "emergency_note": "🚨 If experiencing severe symptoms, seek immediate medical attention."
            })
            
            logger.info(f"Groq prediction: {ai_result.get('primary_prediction')} ({ai_result.get('confidence_percentage')}%)")
            return ai_result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            # Fallback response
            return {
                "status": "success",
                "primary_prediction": "Medical consultation recommended",
                "confidence_percentage": 75,
                "top_predictions": [
                    {"disease": "Consultation needed", "probability": 75},
                    {"disease": "Multiple possibilities", "probability": 25}
                ],
                "analysis": {
                    "symptoms_analyzed": symptoms,
                    "method": "AI Medical Analysis",
                    "reasoning": "Unable to parse detailed analysis"
                },
                "recommendations": [
                    "Consult with a healthcare professional",
                    "Provide complete symptom history",
                    "Consider diagnostic tests if recommended"
                ],
                "urgency_level": "medium",
                "seek_immediate_care": False,
                "ai_provider": "Groq (Llama3)",
                "disclaimer": "⚠️ This AI analysis is for informational purposes only."
            }
    
    except Exception as e:
        logger.error(f"Disease prediction error: {e}")
        return {
            "error": "AI service temporarily unavailable",
            "message": "Unable to connect to AI service. Please try again later or consult a healthcare professional.",
            "fallback_advice": "For immediate concerns, contact your doctor or emergency services."
        }

@app.get("/trials")
async def get_trials():
    return {
        "batches": drug_batches
    }

@app.post("/trials")
async def create_batch(batch_data: dict):
    # Generate new batch ID
    batch_count = len(drug_batches) + 1
    new_batch_id = f"BATCH{batch_count:03d}"
    
    # Create new batch with default status
    new_batch = {
        "batchID": new_batch_id,
        "drugName": batch_data.get("drugName", ""),
        "expiry": batch_data.get("expiry", ""),
        "sender": batch_data.get("sender", ""),
        "receiver": batch_data.get("receiver", ""),
        "status": "pending"
    }
    
    # Add to global storage
    drug_batches.append(new_batch)
    
    return {
        "success": True,
        "message": "Batch created successfully",
        "batchID": new_batch_id,
        "data": new_batch
    }

@app.put("/trials/{batch_id}/approve")
async def approve_batch(batch_id: str):
    # Find and update the batch status
    for batch in drug_batches:
        if batch["batchID"] == batch_id:
            batch["status"] = "approved"
            return {
                "success": True,
                "message": f"Batch {batch_id} approved successfully",
                "data": batch
            }
    
    return {
        "success": False,
        "message": f"Batch {batch_id} not found"
    }

# Chat REST API endpoints
@app.post("/chat/sessions")
async def create_chat_session(session_data: ChatSessionCreate):
    """Create a new chat session between patient and doctor"""
    import sqlite3
    
    conn = sqlite3.connect('medcare.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO chat_sessions (patient_id, doctor_id, session_type, status)
            VALUES (?, ?, 'doctor_patient', 'active')
        """, (session_data.patient_id, session_data.doctor_id))
        
        session_id = cursor.lastrowid
        conn.commit()
        
        # Create notification for doctor if specified
        if session_data.doctor_id:
            cursor.execute("""
                INSERT INTO chat_notifications (user_id, session_id, message, notification_type)
                VALUES (?, ?, ?, 'chat_request')
            """, (session_data.doctor_id, session_id, f"New chat request from patient {session_data.patient_id}"))
            conn.commit()
        
        conn.close()
        
        return {
            "success": True,
            "session_id": session_id,
            "room_id": f"session_{session_id}",
            "message": "Chat session created successfully"
        }
        
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Failed to create chat session: {e}")

@app.get("/chat/sessions/{user_id}")
async def get_user_chat_sessions(user_id: str):
    """Get all chat sessions for a user"""
    import sqlite3
    
    conn = sqlite3.connect('medcare.db')
    cursor = conn.cursor()
    
    # Determine if user is patient or doctor
    user_role = "patient" if "patient" in user_id else "doctor"
    
    if user_role == "patient":
        cursor.execute("""
            SELECT cs.id, cs.doctor_id, cs.status, cs.created_at,
                   COUNT(cm.id) as message_count,
                   MAX(cm.created_at) as last_message_time
            FROM chat_sessions cs
            LEFT JOIN chat_messages cm ON cs.id = cm.session_id
            WHERE cs.patient_id = ? AND cs.status = 'active'
            GROUP BY cs.id
            ORDER BY cs.created_at DESC
        """, (user_id,))
    else:
        cursor.execute("""
            SELECT cs.id, cs.patient_id, cs.status, cs.created_at,
                   COUNT(cm.id) as message_count,
                   MAX(cm.created_at) as last_message_time
            FROM chat_sessions cs
            LEFT JOIN chat_messages cm ON cs.id = cm.session_id
            WHERE cs.doctor_id = ? AND cs.status = 'active'
            GROUP BY cs.id
            ORDER BY cs.created_at DESC
        """, (user_id,))
    
    sessions = cursor.fetchall()
    conn.close()
    
    result = []
    for session in sessions:
        result.append({
            "session_id": session[0],
            "other_user_id": session[1],
            "status": session[2],
            "created_at": session[3],
            "message_count": session[4],
            "last_message_time": session[5],
            "room_id": f"session_{session[0]}"
        })
    
    return {"sessions": result}

@app.get("/chat/messages/{session_id}")
async def get_chat_messages(session_id: int, limit: int = 50):
    """Get messages for a chat session"""
    import sqlite3
    
    conn = sqlite3.connect('medcare.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, sender_id, sender_type, message, message_type, is_ai_response, created_at
        FROM chat_messages
        WHERE session_id = ?
        ORDER BY created_at DESC
        LIMIT ?
    """, (session_id, limit))
    
    messages = cursor.fetchall()
    conn.close()
    
    result = []
    for msg in reversed(messages):  # Reverse to get chronological order
        result.append({
            "id": msg[0],
            "sender_id": msg[1],
            "sender_type": msg[2],
            "message": msg[3],
            "message_type": msg[4],
            "is_ai_response": bool(msg[5]),
            "timestamp": msg[6]
        })
    
    return {"messages": result}

@app.get("/chat/notifications/{user_id}")
async def get_chat_notifications(user_id: str):
    """Get chat notifications for a user"""
    import sqlite3
    
    conn = sqlite3.connect('medcare.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT cn.id, cn.session_id, cn.message, cn.notification_type, cn.is_read, cn.created_at
        FROM chat_notifications cn
        WHERE cn.user_id = ? AND cn.is_read = FALSE
        ORDER BY cn.created_at DESC
    """, (user_id,))
    
    notifications = cursor.fetchall()
    conn.close()
    
    result = []
    for notif in notifications:
        result.append({
            "id": notif[0],
            "session_id": notif[1],
            "message": notif[2],
            "type": notif[3],
            "is_read": bool(notif[4]),
            "timestamp": notif[5]
        })
    
    return {"notifications": result}

@app.put("/chat/notifications/{notification_id}/read")
async def mark_notification_read(notification_id: int):
    """Mark a notification as read"""
    import sqlite3
    
    conn = sqlite3.connect('medcare.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE chat_notifications 
        SET is_read = TRUE 
        WHERE id = ?
    """, (notification_id,))
    
    conn.commit()
    conn.close()
    
    return {"success": True, "message": "Notification marked as read"}

@app.get("/chat/doctors/available")
async def get_available_doctors():
    """Get list of available doctors for chat"""
    # Mock data for available doctors
    doctors = [
        {
            "id": "doctor@medcare.com",
            "name": "Dr. Sarah Johnson",
            "specialty": "General Medicine",
            "online": chat_manager.is_user_online("doctor@medcare.com"),
            "response_time": "Usually responds within 15 minutes"
        },
        {
            "id": "doctor2@medcare.com", 
            "name": "Dr. Michael Chen",
            "specialty": "Cardiology",
            "online": chat_manager.is_user_online("doctor2@medcare.com"),
            "response_time": "Usually responds within 30 minutes"
        },
        {
            "id": "doctor3@medcare.com",
            "name": "Dr. Emily Rodriguez", 
            "specialty": "Dermatology",
            "online": chat_manager.is_user_online("doctor3@medcare.com"),
            "response_time": "Usually responds within 1 hour"
        }
    ]
    
    return {"doctors": doctors}

@app.get("/chat/patients")
async def get_chat_patients():
    """Get list of patients for doctors"""
    # Mock data for patients
    patients = [
        {
            "id": "patient@medcare.com",
            "name": "John Patient",
            "last_visit": "2024-08-15",
            "online": chat_manager.is_user_online("patient@medcare.com"),
            "condition": "General Care"
        },
        {
            "id": "patient2@medcare.com",
            "name": "Jane Smith", 
            "last_visit": "2024-08-20",
            "online": chat_manager.is_user_online("patient2@medcare.com"),
            "condition": "Diabetes Type 2"
        },
        {
            "id": "patient3@medcare.com",
            "name": "Bob Wilson",
            "last_visit": "2024-08-25", 
            "online": chat_manager.is_user_online("patient3@medcare.com"),
            "condition": "Hypertension"
        }
    ]
    
    return {"patients": patients}

@app.get("/")
def read_root():
    return {"message": "MedCare API is running"}


# LLM Chat with Contextual Memory
async def get_llm_response(message: str, session_id: int, patient_context: str = ""):
    """Get LLM response with contextual memory using Groq API"""
    try:
        import sqlite3
        from groq import Groq
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key or groq_api_key == "your_groq_api_key_here":
            return "I'm sorry, but the AI assistant is currently unavailable. Please try contacting a doctor directly."
        
        # Get chat history for context
        conn = sqlite3.connect('medcare.db')
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT sender_type, message, created_at 
            FROM chat_messages 
            WHERE session_id = ? AND message_type = 'text'
            ORDER BY created_at DESC 
            LIMIT 10
        """, (session_id,))
        
        chat_history = cursor.fetchall()
        conn.close()
        
        # Build context from recent messages
        context_messages = []
        for sender_type, msg, timestamp in reversed(chat_history):
            role = "assistant" if sender_type == "ai" else "user"
            context_messages.append({"role": role, "content": msg})
        
        # Create medical AI prompt with context
        system_prompt = f"""You are a conversational medical AI assistant for MedCare Hospital. Have natural conversations with patients using these guidelines:

Patient Context: {patient_context}

CONVERSATION STYLE:
- Ask ONE focused question at a time
- Keep responses to 2-3 short sentences maximum
- Use natural, conversational language
- Be empathetic but concise
- Follow up based on their answers

MEDICAL APPROACH:
- Gather symptoms through targeted questions
- Provide brief, actionable advice
- Recommend doctor consultation when needed
- Never give definitive diagnoses

FORMATTING:
- Use line breaks for readability
- Keep paragraphs short (1-2 sentences)
- Ask follow-up questions to engage

Example good response:
"I understand you have a fever and cold. How long have you been experiencing these symptoms?

Are you having any difficulty breathing or severe headaches?"

Remember: Keep it conversational, not clinical."""

        # Prepare messages for Groq
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(context_messages[-6:])  # Last 6 messages for context
        messages.append({"role": "user", "content": message})
        
        # Call Groq API
        client = Groq(api_key=groq_api_key)
        response = client.chat.completions.create(
            messages=messages,
            model="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=150
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"LLM response error: {e}")
        return "I'm having trouble processing your message right now. Please try again or contact a healthcare professional if it's urgent."

# Chat WebSocket endpoint
@app.websocket("/ws/chat/{user_id}")
async def chat_websocket(websocket: WebSocket, user_id: str):
    await chat_manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            action = message_data.get("action")
            
            if action == "join_room":
                room_id = message_data.get("room_id")
                await chat_manager.join_room(user_id, room_id)
                
                # Notify room users
                await chat_manager.send_to_room(room_id, {
                    "type": "user_joined",
                    "user_id": user_id,
                    "message": f"User {user_id} joined the chat"
                }, exclude_user=user_id)
                
            elif action == "send_message":
                room_id = message_data.get("room_id")
                message = message_data.get("message")
                session_id = message_data.get("session_id")
                
                # Save message to database
                import sqlite3
                conn = sqlite3.connect('medcare.db')
                cursor = conn.cursor()
                
                # Get user role
                user_role = "patient" if "patient" in user_id else "doctor"
                
                cursor.execute("""
                    INSERT INTO chat_messages (session_id, sender_id, sender_type, message, message_type)
                    VALUES (?, ?, ?, ?, 'text')
                """, (session_id, user_id, user_role, message))
                
                message_id = cursor.lastrowid
                conn.commit()
                conn.close()
                
                # Broadcast message to room
                message_obj = {
                    "type": "message",
                    "id": message_id,
                    "session_id": session_id,
                    "sender_id": user_id,
                    "sender_type": user_role,
                    "message": message,
                    "timestamp": datetime.now().isoformat()
                }
                
                await chat_manager.send_to_room(room_id, message_obj)
                
                # Check if doctor is online, if not, provide LLM response
                room_users = chat_manager.get_room_users(room_id)
                doctor_online = any("doctor" in uid for uid in room_users)
                
                if not doctor_online and user_role == "patient":
                    # Get patient context
                    patient_context = f"Patient ID: {user_id}"
                    
                    # Generate LLM response
                    llm_response = await get_llm_response(message, session_id, patient_context)
                    
                    # Save AI response to database
                    conn = sqlite3.connect('medcare.db')
                    cursor = conn.cursor()
                    
                    cursor.execute("""
                        INSERT INTO chat_messages (session_id, sender_type, message, message_type, is_ai_response)
                        VALUES (?, 'ai', ?, 'text', TRUE)
                    """, (session_id, llm_response))
                    
                    ai_message_id = cursor.lastrowid
                    conn.commit()
                    conn.close()
                    
                    # Send AI response to room
                    ai_message_obj = {
                        "type": "message",
                        "id": ai_message_id,
                        "session_id": session_id,
                        "sender_id": "ai_assistant",
                        "sender_type": "ai",
                        "message": llm_response,
                        "timestamp": datetime.now().isoformat(),
                        "is_ai_response": True
                    }
                    
                    # Delay AI response slightly for better UX
                    await asyncio.sleep(1)
                    await chat_manager.send_to_room(room_id, ai_message_obj)
                    
    except WebSocketDisconnect:
        chat_manager.disconnect(user_id)

# Legacy WebSocket for backward compatibility
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/coldchain/data/{batch_id}")
async def get_coldchain_data(batch_id: str):
    # Generate some initial data if none exists
    import random
    from datetime import datetime, timedelta
    
    data = []
    base_time = datetime.now()
    
    for i in range(20):
        timestamp = (base_time - timedelta(minutes=i*3)).isoformat()
        
        if batch_id == "BATCH003":
            temp = random.uniform(8.2, 9.5)  # Warning range with variation
            humidity = random.uniform(75, 85)
        elif batch_id == "BATCH002":
            temp = random.uniform(9.0, 11.0)  # Warning range
            humidity = random.uniform(65, 75)
        else:  # BATCH001
            temp = random.uniform(4.0, 6.0)  # Safe range
            humidity = random.uniform(45, 55)
            
        data.append({
            "batchID": batch_id,
            "temperature": round(temp, 1),
            "humidity": round(humidity, 1),
            "timestamp": timestamp
        })
    
    return {"data": data, "batch_id": batch_id}

@app.get("/coldchain/risk")
async def get_risk_analysis(batch_id: str = "BATCH001"):
    # Simple risk analysis based on batch
    if batch_id == "BATCH003":
        return {
            "risk_level": "CRITICAL",
            "risk_score": 85,
            "analysis": "Temperature consistently above safe range (12.5°C). Immediate action required.",
            "recommendations": [
                "Check refrigeration unit immediately",
                "Move batch to backup cold storage",
                "Contact maintenance team"
            ],
            "ai_insights": [
                "Temperature has been critical for extended period",
                "Risk of medication degradation is high"
            ]
        }
    elif batch_id == "BATCH002":
        return {
            "risk_level": "MEDIUM",
            "risk_score": 45,
            "analysis": "Temperature slightly elevated but within acceptable limits.",
            "recommendations": [
                "Monitor temperature closely",
                "Check cooling system performance"
            ],
            "ai_insights": [
                "Temperature trending upward",
                "Preventive maintenance recommended"
            ]
        }
    else:
        return {
            "risk_level": "LOW",
            "risk_score": 15,
            "analysis": "All parameters within optimal range.",
            "recommendations": [
                "Continue current monitoring",
                "Regular maintenance schedule"
            ],
            "ai_insights": [
                "Stable temperature control",
                "System performing optimally"
            ]
        }

# Background task to generate sensor data
async def generate_sensor_data():
    import random
    
    batch_configs = {
        "BATCH001": {"base_temp": 5.0, "temp_variance": 1.0, "base_humidity": 50},
        "BATCH002": {"base_temp": 10.0, "temp_variance": 1.5, "base_humidity": 70},
        "BATCH003": {"base_temp": 12.5, "temp_variance": 0.0, "base_humidity": 80}
    }
    
    while True:
        try:
            for batch_id, config in batch_configs.items():
                if batch_id == "BATCH003":
                    temperature = random.uniform(8.2, 9.5)  # Warning range with variation
                    humidity = random.uniform(75, 85)       # Some humidity variation
                else:
                    if random.random() < 0.2:  # 20% chance of anomaly
                        temperature = config["base_temp"] + random.uniform(-3, 4)
                    else:
                        temperature = config["base_temp"] + random.uniform(-config["temp_variance"], config["temp_variance"])
                    
                    humidity = config["base_humidity"] + random.uniform(-10, 10)
                
                sensor_data = {
                    "batchID": batch_id,
                    "temperature": round(temperature, 1),
                    "humidity": round(max(0, min(100, humidity)), 1),
                    "timestamp": datetime.now().isoformat()
                }
                
                # Broadcast to WebSocket clients
                if manager.active_connections:
                    await manager.broadcast(json.dumps({
                        "type": "sensor_data",
                        "data": sensor_data
                    }))
            
            await asyncio.sleep(3)  # Generate data every 3 seconds
        except Exception as e:
            print(f"Error in sensor data generation: {e}")
            await asyncio.sleep(3)

# Background sensor data generation is now handled in lifespan event
# This startup event has been moved to the lifespan function above

@app.get("/health")
def health_check():
    return {"status": "healthy"}
