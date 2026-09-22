"""
Module for API schemas and data validation.

WHY:
In production, we cannot trust user input. An API might receive missing fields,
wrong data types, or out-of-range values. If we pass invalid data directly to our ML
model, it will crash or silently produce wrong predictions. Pydantic models act as
a gatekeeper, ensuring that our model only sees exactly what it expects.

MENTOR NOTE:
Strong typing and validation are the difference between a brittle notebook and
a robust production service. Always validate at the boundaries of your system!
"""

from pydantic import BaseModel, Field
from typing import Optional

class TransactionInput(BaseModel):
    """
    Schema for incoming transaction data.
    
    WHY:
    Each field is explicitly typed and documented. We use Field() to provide
    metadata, which FastAPI automatically turns into interactive API documentation
    (Swagger UI).
    """
    Time: float = Field(..., description="Seconds elapsed between this transaction and the first transaction in the dataset")
    V1: float = Field(..., description="PCA-transformed feature V1")
    V2: float = Field(..., description="PCA-transformed feature V2")
    V3: float = Field(..., description="PCA-transformed feature V3")
    V4: float = Field(..., description="PCA-transformed feature V4")
    V5: float = Field(..., description="PCA-transformed feature V5")
    V6: float = Field(..., description="PCA-transformed feature V6")
    V7: float = Field(..., description="PCA-transformed feature V7")
    V8: float = Field(..., description="PCA-transformed feature V8")
    V9: float = Field(..., description="PCA-transformed feature V9")
    V10: float = Field(..., description="PCA-transformed feature V10")
    V11: float = Field(..., description="PCA-transformed feature V11")
    V12: float = Field(..., description="PCA-transformed feature V12")
    V13: float = Field(..., description="PCA-transformed feature V13")
    V14: float = Field(..., description="PCA-transformed feature V14")
    V15: float = Field(..., description="PCA-transformed feature V15")
    V16: float = Field(..., description="PCA-transformed feature V16")
    V17: float = Field(..., description="PCA-transformed feature V17")
    V18: float = Field(..., description="PCA-transformed feature V18")
    V19: float = Field(..., description="PCA-transformed feature V19")
    V20: float = Field(..., description="PCA-transformed feature V20")
    V21: float = Field(..., description="PCA-transformed feature V21")
    V22: float = Field(..., description="PCA-transformed feature V22")
    V23: float = Field(..., description="PCA-transformed feature V23")
    V24: float = Field(..., description="PCA-transformed feature V24")
    V25: float = Field(..., description="PCA-transformed feature V25")
    V26: float = Field(..., description="PCA-transformed feature V26")
    V27: float = Field(..., description="PCA-transformed feature V27")
    V28: float = Field(..., description="PCA-transformed feature V28")
    Amount: float = Field(..., ge=0, description="Transaction amount (must be non-negative)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "Time": 0.0, "V1": -1.3598071336738, "V2": -0.0727811733098497, 
                "V3": 2.53634673796914, "V4": 1.37815522427443, "V5": -0.338320769942518, 
                "V6": 0.462387777762292, "V7": 0.239598554061257, "V8": 0.0986979012610507, 
                "V9": 0.363786969611213, "V10": 0.0907941719789316, "V11": -0.551599533260813, 
                "V12": -0.617800855762348, "V13": -0.991389847235408, "V14": -0.311169353699879, 
                "V15": 1.46817697209427, "V16": -0.470400525259478, "V17": 0.207971241929242, 
                "V18": 0.0257905801985591, "V19": 0.403992960255733, "V20": 0.251412098239705, 
                "V21": -0.018306777944153, "V22": 0.277837575558899, "V23": -0.110473910188767, 
                "V24": 0.0669280749146731, "V25": 0.128539358273528, "V26": -0.189114843888824, 
                "V27": 0.133558376740387, "V28": -0.0210530534538215, "Amount": 149.62
            }
        }
    }


class PredictionOutput(BaseModel):
    """
    Schema for the API response.
    
    WHY:
    Standardizing the output ensures the consumer (frontend, another service) 
    always gets a predictable JSON structure. Including the threshold and risk level
    adds context, rather than just returning a boolean.
    """
    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    threshold_used: float
    risk_level: str
    processing_time_ms: float


class HealthResponse(BaseModel):
    """
    Schema for health check endpoint.
    """
    status: str
    model_loaded: bool
    model_name: str
    threshold: float
    timestamp: str
