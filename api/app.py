"""
FastAPI application for serving the fraud detection model.

WHY:
FastAPI is ideal for ML model serving because it's fast, has async support,
and auto-generates documentation based on Pydantic schemas. 

MENTOR NOTE (REST API Design):
- endpoints should be intuitive (`/predict`, `/health`)
- keep endpoints stateless
- handle errors with proper HTTP status codes (e.g., 500 for model errors)

MENTOR NOTE (Model Serving):
Load your model ONCE at startup, not on every request! Loading a model is
I/O bound and slow. We cache the model in memory so predictions are purely CPU-bound.
"""

import time
import uuid
import joblib
import pandas as pd
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# In a real project, src.config, src.monitoring might be available
# We mock the imports here based on the instructions
from src.config import (
    MODELS_DIR, 
    MODEL_FILENAME, 
    SCALER_FILENAME, 
    THRESHOLD_FILENAME, 
    DEFAULT_THRESHOLD,
    FEATURE_COLUMNS
)

# Mocked monitoring import
try:
    from src.monitoring import log_prediction
except ImportError:
    # Dummy implementation if missing
    def log_prediction(*args, **kwargs):
        pass

from api.schemas import TransactionInput, PredictionOutput, HealthResponse

# Global variables for model state
# WHY: We store the model in globals so it persists across requests.
app_state = {
    "model": None,
    "scaler": None,
    "threshold": DEFAULT_THRESHOLD,
    "model_name": "unknown"
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events for FastAPI.
    Loads ML artifacts at startup, cleans up on shutdown.
    """
    # Startup
    try:
        model_path = MODELS_DIR / MODEL_FILENAME
        scaler_path = MODELS_DIR / SCALER_FILENAME
        threshold_path = MODELS_DIR / THRESHOLD_FILENAME
        
        # Load artifacts
        app_state["model"] = joblib.load(model_path)
        app_state["scaler"] = joblib.load(scaler_path)
        
        # Try to load optimized threshold, fallback to default
        # NOTE: threshold is saved via joblib.dump in train.py
        try:
            app_state["threshold"] = float(joblib.load(threshold_path))
        except Exception:
            print(f"Could not load optimized threshold, using default: {DEFAULT_THRESHOLD}")
            
        app_state["model_name"] = str(type(app_state["model"]).__name__)
        print(f"Successfully loaded model: {app_state['model_name']}")
    except Exception as e:
        print(f"Error loading model artifacts: {str(e)}")
        # In a real app, you might want to prevent startup if model fails to load
    
    yield
    
    # Shutdown (cleanup if necessary)
    app_state.clear()

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Real-time API for detecting fraudulent transactions",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware to allow cross-origin requests (e.g., from frontend dashboard)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    WHY:
    Load balancers and container orchestrators (like Kubernetes or Docker Compose)
    need a fast, simple endpoint to check if the service is alive and ready to serve traffic.
    """
    return HealthResponse(
        status="healthy" if app_state["model"] else "degraded",
        model_loaded=app_state["model"] is not None,
        model_name=app_state["model_name"],
        threshold=app_state["threshold"],
        timestamp=datetime.now(tz=__import__('zoneinfo').ZoneInfo('UTC')).isoformat()
    )

@app.post("/predict", response_model=PredictionOutput)
async def predict(transaction: TransactionInput):
    """
    Predict if a transaction is fraudulent.

    WHY Input Validation Prevents Bugs:
    Because we use Pydantic (TransactionInput), FastAPI automatically rejects
    requests with missing fields or wrong types before they even reach this function.
    No more dealing with KeyError or ValueError down the line!
    """
    if app_state["model"] is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    start_time = time.perf_counter()
    transaction_id = str(uuid.uuid4())

    try:
        # 1. Convert input to DataFrame (to match training format)
        data_dict = transaction.model_dump()
        df = pd.DataFrame([data_dict])

        # Ensure column order matches training
        df = df[FEATURE_COLUMNS]

        # 2. Scale Time and Amount using the saved scaler
        # MENTOR NOTE: The scaler was fit ONLY on Time and Amount during training.
        # The pipeline's internal scaler handles the rest, but since we're using
        # the model pipeline (which has its own StandardScaler), we can pass the
        # pre-scaled data directly. The pipeline scaler will re-scale everything,
        # which is fine since it was fit during training.

        # 3. Predict probability
        # The pipeline includes scaler + classifier, so we pass raw features
        proba = float(app_state["model"].predict_proba(df)[0, 1])

        # 4. Apply threshold
        is_fraud = bool(proba >= app_state["threshold"])

        # 5. Determine risk level
        if proba < 0.3:
            risk_level = "LOW"
        elif proba <= 0.7:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        processing_time_ms = (time.perf_counter() - start_time) * 1000

        # 6. Log prediction for monitoring
        log_prediction(
            transaction_data=data_dict,
            prediction=int(is_fraud),
            probability=proba,
            threshold=app_state["threshold"]
        )

        return PredictionOutput(
            transaction_id=transaction_id,
            fraud_probability=proba,
            is_fraud=is_fraud,
            threshold_used=app_state["threshold"],
            risk_level=risk_level,
            processing_time_ms=processing_time_ms
        )

    except Exception as e:
        # MENTOR NOTE: Never expose internal stack traces to users in production!
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
