# src/api/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# --- 1. Model Loading ---
MODEL_PATH = os.environ.get("MODEL_PATH", "src/model/model.pkl") 

try:
    model = joblib.load(MODEL_PATH)
    print(f"Successfully loaded model from {MODEL_PATH}")
except FileNotFoundError:
    print(f"ERROR: Model file not found at {MODEL_PATH}")
    model = None # Set to None to handle the error gracefully

# --- 2. FastAPI Setup ---
app = FastAPI(
    title="MLOps Churn Prediction API",
    description="A FastAPI service for predicting customer churn."
)

# --- 3. Pydantic Schema (Defines the input payload structure) ---
# NOTE: The feature names MUST match the features the model was trained on
class CustomerFeatures(BaseModel):
    tenure: float
    MonthlyCharges: float
    Contract: str # e.g., "Month-to-month", "One year", "Two year"
    OnlineSecurity: str # e.g., "Yes" or "No"

# --- 4. Endpoints ---

@app.get("/health")
def health_check():
    """Simple health check endpoint for Kubernetes Liveness/Readiness probes."""
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict_churn(features: CustomerFeatures):
    """Predict the probability of a customer churning."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or available.")

    # Convert Pydantic model to a Pandas DataFrame
    input_data = pd.DataFrame([features.dict()])
    
    try:
        # Get the prediction probabilities (0 and 1)
        # prediction_proba returns a 2D array: [[Prob_NoChurn, Prob_Churn]]
        proba = model.predict_proba(input_data)[0]
        
        # Predict the final class (0 or 1)
        prediction = int(model.predict(input_data)[0])
        
        return {
            "prediction": prediction,
            "probability_churn": round(proba[1], 4),
            "probability_no_churn": round(proba[0], 4),
            "input_features": features.dict()
        }
    except Exception as e:
        # Catch any exceptions during prediction (e.g., missing features)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# --- 5. Required Dependencies (src/api/requirements.txt) ---
# fastapi
# uvicorn
# pandas
# scikit-learn
# joblib