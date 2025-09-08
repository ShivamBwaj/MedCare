import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from skimage import feature

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('parkinsons_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ParkinsonsDetector:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.models = {}
        self.label_encoders = {}
        
    def quantify_image(self, image):
        """Extract HOG features from image"""
        features = feature.hog(
            image, 
            orientations=9,
            pixels_per_cell=(10, 10), 
            cells_per_block=(2, 2),
            transform_sqrt=True, 
            block_norm="L1"
        )
        return features
    
    def preprocess_image(self, image_path):
        """Load and preprocess image for feature extraction"""
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Could not load image: {image_path}")
                return None
                
            # Convert to grayscale and resize
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (200, 200))
            
            # Threshold the image (drawing appears white on black background)
            image = cv2.threshold(image, 0, 255,
                                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            
            return image
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            return None
    
    def load_dataset(self, drawing_type, split):
        """Load and process images for specified drawing type and split"""
        logger.info(f"Loading {drawing_type} {split} data...")
        
        data_dir = self.data_path / drawing_type / split
        if not data_dir.exists():
            logger.error(f"Directory not found: {data_dir}")
            return np.array([]), np.array([])
        
        data = []
        labels = []
        
        # Process healthy images
        healthy_dir = data_dir / "healthy"
        if healthy_dir.exists():
            for img_path in healthy_dir.glob("*.png"):
                image = self.preprocess_image(img_path)
                if image is not None:
                    features = self.quantify_image(image)
                    data.append(features)
                    labels.append("healthy")
        
        # Process parkinson images
        parkinson_dir = data_dir / "parkinson"
        if parkinson_dir.exists():
            for img_path in parkinson_dir.glob("*.png"):
                image = self.preprocess_image(img_path)
                if image is not None:
                    features = self.quantify_image(image)
                    data.append(features)
                    labels.append("parkinson")
        
        logger.info(f"Loaded {len(data)} samples for {drawing_type} {split}")
        return np.array(data), np.array(labels)
    
    def train_model(self, drawing_type):
        """Train Random Forest model for specified drawing type"""
        logger.info(f"Training model for {drawing_type} drawings...")
        
        # Load training and testing data
        train_X, train_y = self.load_dataset(drawing_type, "training")
        test_X, test_y = self.load_dataset(drawing_type, "testing")
        
        if len(train_X) == 0 or len(test_X) == 0:
            logger.error(f"No data found for {drawing_type}")
            return None
        
        # Encode labels
        le = LabelEncoder()
        train_y_encoded = le.fit_transform(train_y)
        test_y_encoded = le.transform(test_y)
        
        # Store label encoder
        self.label_encoders[drawing_type] = le
        
        # Train Random Forest model
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5
        )
        
        model.fit(train_X, train_y_encoded)
        
        # Make predictions
        train_pred = model.predict(train_X)
        test_pred = model.predict(test_X)
        
        # Calculate metrics
        train_accuracy = accuracy_score(train_y_encoded, train_pred)
        test_accuracy = accuracy_score(test_y_encoded, test_pred)
        
        # Confusion matrix
        cm = confusion_matrix(test_y_encoded, test_pred)
        if len(cm.ravel()) == 4:
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            sensitivity = specificity = 0
        
        # Log results
        logger.info(f"{drawing_type.capitalize()} Model Results:")
        logger.info(f"Training Accuracy: {train_accuracy:.4f}")
        logger.info(f"Testing Accuracy: {test_accuracy:.4f}")
        logger.info(f"Sensitivity: {sensitivity:.4f}")
        logger.info(f"Specificity: {specificity:.4f}")
        logger.info(f"Classification Report:\n{classification_report(test_y_encoded, test_pred, target_names=le.classes_)}")
        
        # Store model
        self.models[drawing_type] = {
            'model': model,
            'accuracy': test_accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity
        }
        
        return model
    
    def save_models(self, save_dir):
        """Save trained models and label encoders"""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        for drawing_type in self.models:
            # Save model
            model_path = save_path / f"parkinsons_{drawing_type}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(self.models[drawing_type]['model'], f)
            logger.info(f"Saved {drawing_type} model to {model_path}")
            
            # Save label encoder
            le_path = save_path / f"parkinsons_{drawing_type}_encoder.pkl"
            with open(le_path, 'wb') as f:
                pickle.dump(self.label_encoders[drawing_type], f)
            logger.info(f"Saved {drawing_type} label encoder to {le_path}")
    
    def predict_image(self, image_path, drawing_type):
        """Predict Parkinson's from image"""
        if drawing_type not in self.models:
            raise ValueError(f"No model trained for {drawing_type}")
        
        # Preprocess image
        image = self.preprocess_image(image_path)
        if image is None:
            return None
        
        # Extract features
        features = self.quantify_image(image)
        
        # Make prediction
        model = self.models[drawing_type]['model']
        le = self.label_encoders[drawing_type]
        
        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0]
        
        # Get class names and probabilities
        classes = le.classes_
        result = {
            'prediction': le.inverse_transform([prediction])[0],
            'confidence': float(max(probability)),
            'probabilities': {
                classes[i]: float(probability[i]) 
                for i in range(len(classes))
            }
        }
        
        return result

def main():
    """Main training function"""
    logger.info("Starting Parkinson's Detection Model Training...")
    
    # Set data path
    data_path = Path("ai/Parkinsons_Drawings_data")
    
    if not data_path.exists():
        logger.error(f"Data directory not found: {data_path}")
        return
    
    # Initialize detector
    detector = ParkinsonsDetector(data_path)
    
    # Train models for both drawing types
    drawing_types = ["spiral", "wave"]
    
    for drawing_type in drawing_types:
        try:
            model = detector.train_model(drawing_type)
            if model is None:
                logger.error(f"Failed to train {drawing_type} model")
        except Exception as e:
            logger.error(f"Error training {drawing_type} model: {e}")
    
    # Save models
    if detector.models:
        detector.save_models(".")
        logger.info("Model training completed successfully!")
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("TRAINING SUMMARY")
        logger.info("="*50)
        for drawing_type, metrics in detector.models.items():
            logger.info(f"{drawing_type.upper()} MODEL:")
            logger.info(f"  Accuracy: {metrics['accuracy']:.2%}")
            logger.info(f"  Sensitivity: {metrics['sensitivity']:.2%}")
            logger.info(f"  Specificity: {metrics['specificity']:.2%}")
            logger.info("-" * 30)
    else:
        logger.error("No models were successfully trained!")

if __name__ == "__main__":
    main()
