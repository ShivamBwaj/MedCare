import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
from typing import List, Dict
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64

# Disease Symptom Prediction Model
class SymptomDiseasePredictor:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.symptoms_diseases_data = self._load_symptoms_data()
        
    def _load_symptoms_data(self):
        """Load or create symptom-disease mapping data"""
        # Sample data - in production, use real medical dataset
        return [
            {"symptoms": "fever headache cough", "disease": "Common Cold"},
            {"symptoms": "fever cough shortness_of_breath", "disease": "Pneumonia"},
            {"symptoms": "chest_pain shortness_of_breath", "disease": "Heart Disease"},
            {"symptoms": "frequent_urination excessive_thirst", "disease": "Diabetes"},
            {"symptoms": "joint_pain swelling stiffness", "disease": "Arthritis"},
            {"symptoms": "nausea vomiting abdominal_pain", "disease": "Gastroenteritis"},
            {"symptoms": "headache sensitivity_to_light nausea", "disease": "Migraine"},
            {"symptoms": "rash itching fever", "disease": "Allergic Reaction"},
            {"symptoms": "fatigue weight_loss night_sweats", "disease": "Tuberculosis"},
            {"symptoms": "sore_throat fever swollen_glands", "disease": "Strep Throat"}
        ]
    
    def train_model(self):
        """Train the symptom-disease prediction model"""
        df = pd.DataFrame(self.symptoms_diseases_data)
        
        # Vectorize symptoms
        self.vectorizer = TfidfVectorizer(max_features=100)
        X = self.vectorizer.fit_transform(df['symptoms'])
        
        # Encode diseases
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df['disease'])
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        
        # Save model
        joblib.dump(self.model, 'symptom_disease_model.pkl')
        joblib.dump(self.vectorizer, 'symptom_vectorizer.pkl')
        joblib.dump(self.label_encoder, 'disease_label_encoder.pkl')
        
        return True
    
    def load_model(self):
        """Load trained model"""
        try:
            self.model = joblib.load('symptom_disease_model.pkl')
            self.vectorizer = joblib.load('symptom_vectorizer.pkl')
            self.label_encoder = joblib.load('disease_label_encoder.pkl')
            return True
        except:
            return False
    
    def predict(self, symptoms: List[str]) -> Dict:
        """Predict disease from symptoms"""
        if not self.model:
            if not self.load_model():
                self.train_model()
        
        # Prepare symptoms text
        symptoms_text = " ".join(symptoms).lower()
        
        # Vectorize
        X = self.vectorizer.transform([symptoms_text])
        
        # Predict
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        # Get top 3 predictions
        top_indices = np.argsort(probabilities)[-3:][::-1]
        
        predicted_diseases = []
        confidence_scores = []
        
        for idx in top_indices:
            disease = self.label_encoder.inverse_transform([idx])[0]
            confidence = probabilities[idx]
            predicted_diseases.append({
                "disease": disease,
                "confidence": round(confidence * 100, 2)
            })
            confidence_scores.append(confidence)
        
        return {
            "predicted_diseases": predicted_diseases,
            "confidence_scores": confidence_scores,
            "recommendations": self._get_recommendations(predicted_diseases[0]["disease"])
        }
    
    def _get_recommendations(self, disease: str) -> List[str]:
        """Get recommendations based on predicted disease"""
        recommendations_map = {
            "Common Cold": [
                "Rest and stay hydrated",
                "Consider over-the-counter cold medications",
                "Consult doctor if symptoms worsen"
            ],
            "Pneumonia": [
                "Seek immediate medical attention",
                "Complete prescribed antibiotic course",
                "Monitor breathing and fever"
            ],
            "Heart Disease": [
                "Urgent cardiology consultation required",
                "Monitor blood pressure regularly",
                "Avoid strenuous activities"
            ],
            "Diabetes": [
                "Blood sugar monitoring essential",
                "Dietary modifications recommended",
                "Regular endocrinologist follow-up"
            ],
            "Arthritis": [
                "Anti-inflammatory medications may help",
                "Physical therapy consultation",
                "Joint protection strategies"
            ]
        }
        
        return recommendations_map.get(disease, [
            "Consult with healthcare provider",
            "Monitor symptoms closely",
            "Follow up if symptoms persist"
        ])

# Skin Disease Detection Model (Simplified)
class SkinDiseaseDetector:
    def __init__(self):
        self.conditions = [
            "Eczema", "Psoriasis", "Acne", "Melanoma", "Basal Cell Carcinoma",
            "Dermatitis", "Fungal Infection", "Normal Skin"
        ]
    
    def predict_from_image(self, image_data: str) -> Dict:
        """Predict skin condition from image (simplified simulation)"""
        try:
            # Decode image
            image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Simulate AI prediction (replace with actual CNN model)
            predicted_condition = np.random.choice(self.conditions)
            confidence = np.random.uniform(0.7, 0.95)
            
            severity = "Mild" if confidence < 0.8 else "Moderate" if confidence < 0.9 else "Severe"
            
            recommendations = self._get_skin_recommendations(predicted_condition, severity)
            
            return {
                "predicted_condition": predicted_condition,
                "confidence_score": round(confidence * 100, 2),
                "severity": severity,
                "recommendations": recommendations
            }
        except Exception as e:
            raise Exception(f"Image processing failed: {str(e)}")
    
    def _get_skin_recommendations(self, condition: str, severity: str) -> List[str]:
        """Get recommendations for skin conditions"""
        base_recommendations = {
            "Eczema": [
                "Use gentle, fragrance-free moisturizers",
                "Avoid known triggers",
                "Consider topical corticosteroids"
            ],
            "Psoriasis": [
                "Moisturize regularly",
                "Consider phototherapy",
                "Dermatologist consultation recommended"
            ],
            "Acne": [
                "Gentle cleansing routine",
                "Avoid picking or squeezing",
                "Consider topical treatments"
            ],
            "Melanoma": [
                "URGENT: Immediate dermatologist consultation",
                "Avoid sun exposure",
                "Monitor for changes"
            ],
            "Normal Skin": [
                "Continue current skincare routine",
                "Regular skin checks recommended",
                "Maintain sun protection"
            ]
        }
        
        recommendations = base_recommendations.get(condition, [
            "Consult dermatologist for proper diagnosis",
            "Monitor condition closely",
            "Maintain good skin hygiene"
        ])
        
        if severity == "Severe":
            recommendations.insert(0, "Seek immediate medical attention")
        
        return recommendations

# Initialize models
symptom_predictor = SymptomDiseasePredictor()
skin_detector = SkinDiseaseDetector()
