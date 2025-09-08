import pandas as pd
import joblib
import logging
import datetime
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# Setup logging
log_dir = "training_logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"{log_dir}/symptom_training_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logging.info("=== SYMPTOM PREDICTION MODEL TRAINING STARTED ===")
logging.info(f"Training script: train_symptoms.py")
logging.info(f"Log file: {log_file}")
logging.info(f"Timestamp: {timestamp}")

# Load data
logging.info("Loading training data...")
df = pd.read_csv("ai/symptoms_small.csv")
X, y = df["text"], df["label"]

logging.info(f"Dataset loaded successfully!")
logging.info(f"Total samples: {len(df)}")
logging.info(f"Features: {df.columns.tolist()}")
logging.info(f"Unique diseases: {y.nunique()}")
logging.info(f"Disease distribution:\n{y.value_counts()}")

# Split for proper evaluation
logging.info("Splitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logging.info(f"Training samples: {len(X_train)}")
logging.info(f"Test samples: {len(X_test)}")

# Random Forest Pipeline
logging.info("Creating Random Forest pipeline...")
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
    ("clf", RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'  # Handle imbalanced classes
    ))
])

logging.info("Pipeline configuration:")
logging.info(f"  - TF-IDF: ngram_range=(1,2), max_features=5000")
logging.info(f"  - Random Forest: n_estimators=100, max_depth=10, class_weight='balanced'")

# Train and evaluate
logging.info("Training model...")
import time
start_time = time.time()
pipe.fit(X_train, y_train)
training_time = time.time() - start_time
logging.info(f"Training completed in {training_time:.2f} seconds")

logging.info("Evaluating model...")
y_pred = pipe.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

logging.info("=== TRAINING RESULTS ===")
logging.info(f"Test Accuracy: {accuracy:.3f}")
logging.info(f"Training samples: {len(X_train)}")
logging.info(f"Test samples: {len(X_test)}")

classification_rep = classification_report(y_test, y_pred)
logging.info(f"Classification Report:\n{classification_rep}")

# Cross-validation
logging.info("Performing cross-validation...")
cv_scores = cross_val_score(pipe, X, y, cv=5)
logging.info(f"Cross-validation scores: {cv_scores}")
logging.info(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Feature importance (top symptoms)
logging.info("Analyzing feature importance...")
feature_names = pipe.named_steps['tfidf'].get_feature_names_out()
importances = pipe.named_steps['clf'].feature_importances_
top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]

logging.info("Top 10 Important Features:")
for i, (feature, importance) in enumerate(top_features, 1):
    logging.info(f"  {i}. {feature}: {importance:.4f}")

# Save model
model_path = "ai/symptom_model.joblib"
logging.info(f"Saving model to {model_path}...")
joblib.dump(pipe, model_path)
logging.info(f"Model saved successfully!")

# Final summary
logging.info("=== TRAINING SUMMARY ===")
logging.info(f"Algorithm: Random Forest Classifier")
logging.info(f"Dataset: symptoms_small.csv ({len(df)} samples)")
logging.info(f"Final Accuracy: {accuracy:.3f}")
logging.info(f"CV Score: {cv_scores.mean():.3f} ± {cv_scores.std() * 2:.3f}")
logging.info(f"Training Time: {training_time:.2f} seconds")
logging.info(f"Model File: {model_path}")
logging.info(f"Log File: {log_file}")
logging.info("=== TRAINING COMPLETED SUCCESSFULLY ===")

print(f"\n✅ Training completed! Check detailed logs at: {log_file}")
