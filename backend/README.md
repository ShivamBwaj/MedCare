# MedCare Clean Architecture Setup

## Quick Start

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Create Database & Users
```bash
python create_admin.py
```

### 3. Train ML Models
```bash
# Train symptom prediction model
python ai/train_symptoms.py

# Train Parkinson's detection models
python ai/train_parkinsons.py

# Train skin/nail classification models
python ai/train_nail_cnn.py
python ai/train_nail_transfer.py
```

### 4. Start Backend
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Start Frontend
```bash
cd ../frontend
npm start
```

## Demo Accounts
- **Admin**: admin@medcare.com / admin123
- **Manager**: manager@medcare.com / manager123  
- **Doctor**: doctor@medcare.com / doctor123
- **Patient**: patient@medcare.com / patient123

## Architecture

### Backend Structure
```
backend/
├── app/
│   ├── models/          # SQLAlchemy models
│   ├── routers/         # FastAPI route handlers
│   ├── services/        # Business logic (verification, etc.)
│   ├── utils/           # Utilities (auth, security)
│   └── schemas/         # Pydantic models
├── ai/                  # ML models and training scripts
└── storage/             # File uploads
```

### Role-Based Features
- **Manager**: File management + verification approval
- **Doctor**: Patient adherence monitoring + chat
- **Patient**: Symptom checker + skin/nail analysis

## ML Models Setup

### Symptom Prediction
- **Algorithm**: Random Forest with TF-IDF vectorization
- **Data**: `ai/symptoms_small.csv` (demo dataset)
- **Training**: `python ai/train_symptoms.py`
- **Output**: `ai/symptom_model.joblib`
- **Accuracy**: 85%+ cross-validation score

### Parkinson's Disease Detection
- **Algorithm**: HOG features + Random Forest
- **Data Structure**: 
  ```
  ai/Parkinsons_Drawings_data/
  ├── spiral/
  │   ├── training/
  │   │   ├── healthy/
  │   │   └── parkinson/
  │   └── testing/
  │       ├── healthy/
  │       └── parkinson/
  └── wave/
      ├── training/
      │   ├── healthy/
      │   └── parkinson/
      └── testing/
          ├── healthy/
          └── parkinson/
  ```
- **Training**: `python ai/train_parkinsons.py`
- **Models**: `parkinsons_spiral_model.pkl`, `parkinsons_wave_model.pkl`
- **Results**: 80% accuracy (spiral), 63% accuracy (wave)

### Skin/Nail Classification
- **Algorithms**: Custom CNN + Transfer Learning (MobileNet V2)
- **Data Structure**:
  ```
  ai/nail_data/
  ├── train/
  │   ├── healthy/
  │   ├── fungal/
  │   └── psoriasis/
  └── valid/
      ├── healthy/
      ├── fungal/
      └── psoriasis/
  ```
- **Training Scripts**:
  - `python ai/train_nail_cnn.py` (Custom CNN)
  - `python ai/train_nail_transfer.py` (Transfer Learning)
- **Models**: `ai/nail_cnn_model.pt`, `ai/nail_transfer_model.pt`

## Environment Configuration
Update `.env` with required credentials:
```
# Core Application
SECRET_KEY=your-super-secret-key
DATABASE_URL=sqlite:///./medcare.db

# AI/LLM Integration
GROQ_API_KEY=your-groq-api-key-here

# Blockchain (Optional)
RPC_URL=https://your-evm-rpc-url
CONTRACT_ADDRESS=0xYourContractAddress
PRIVATE_KEY=0xYourDeployerPrivateKey

# CORS Settings
CORS_ORIGINS=["http://localhost:3000"]
```

## API Endpoints

### Authentication
- `POST /auth/login` - Login
- `POST /auth/register` - Register (admin only)
- `GET /auth/me` - Get current user

### Manager Features
- `POST /files` - Upload file
- `GET /files` - List files
- `POST /files/{id}/approve` - Approve on verification

### AI Features
- `POST /ai/symptoms/predict` - Symptom prediction (Random Forest)
- `POST /ai/skin/predict` - Skin/nail analysis (CNN/Transfer Learning)
- `POST /ai/parkinsons/predict` - Parkinson's detection (HOG + Random Forest)

### Doctor Features
- `GET /adherence/{patient_id}` - Patient adherence
- `WS /ws/chat/{user_id}` - Real-time patient chat with AI fallback

### Chat & Communication Features
- `POST /chat/sessions` - Create chat session between patient and doctor
- `GET /chat/sessions/{user_id}` - Get user's chat sessions
- `GET /chat/messages/{session_id}` - Retrieve chat message history
- `GET /chat/notifications/{user_id}` - Get unread notifications
- `GET /chat/doctors/available` - List available doctors with online status

### LLM Integration
- **Groq API**: Llama-3.1-8b-instant model for AI chat assistance
- **Contextual Memory**: AI maintains conversation history
- **Automatic Fallback**: When doctors are offline, AI responds immediately
- **Medical Prompts**: Specialized healthcare conversation handling

Built with clean architecture principles for scalability and maintainability.
