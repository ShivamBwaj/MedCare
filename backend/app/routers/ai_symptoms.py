from fastapi import APIRouter
from pydantic import BaseModel
import joblib
import os

router = APIRouter(prefix="/ai/symptoms", tags=["ai-symptoms"])

# Load model on startup
MODEL_PATH = "ai/symptom_model.joblib"
if os.path.exists(MODEL_PATH):
    MODEL = joblib.load(MODEL_PATH)
else:
    MODEL = None

class SymptomIn(BaseModel):
    symptoms: list[str]  # ["fever","cough"]

@router.post("/predict")
def predict(body: SymptomIn):
    if not MODEL:
        return {"error": "Model not trained yet"}
    
    text = ",".join([s.strip().lower() for s in body.symptoms])
    pred = MODEL.predict([text])[0]
    proba = max(MODEL.predict_proba([text])[0]) if hasattr(MODEL, "predict_proba") else None
    return {"disease": pred, "confidence": round(float(proba), 3) if proba else None}
