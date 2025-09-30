import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from preprocess import load_and_preprocess

def train_and_save_model():
    (
        X_train, X_test, y_train, y_test,
        scaler, feature_names
    ) = load_and_preprocess("../data/Realistic_E-Commerce_Dataset.csv")

    # Train model
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # ✅ Ensure models folder exists
    os.makedirs("../models", exist_ok=True)

    # ✅ Save model artifacts in correct folder
    joblib.dump(model, "../models/ecommerce_model.pkl")
    joblib.dump(scaler, "../models/scaler.pkl")
    joblib.dump(feature_names, "../models/feature_names.pkl")

    print("✅ Model, scaler, and feature names saved successfully!")

if __name__ == "__main__":
    train_and_save_model()
