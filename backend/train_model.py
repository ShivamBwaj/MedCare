import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

def train_model():
    # Load the data
    df = pd.read_csv('batch_data.csv')
    
    # Prepare features and target
    X = df[['temp_c', 'humidity']]
    y = df['target']
    
    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Train the model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Training accuracy: {train_score:.3f}")
    print(f"Testing accuracy: {test_score:.3f}")
    
    # Save the model and label encoder
    joblib.dump(model, 'model.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    
    print("Model saved to model.pkl")
    print("Label encoder saved to label_encoder.pkl")

if __name__ == "__main__":
    train_model()