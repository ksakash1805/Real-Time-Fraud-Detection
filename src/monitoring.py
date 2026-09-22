"""
Module for production monitoring and data drift detection.

MENTOR NOTE:
What is Data Drift vs Concept Drift?
- Data Drift (Feature Drift): The statistical properties of the input features (X) change over time.
  For example, average transaction amounts increase due to inflation.
- Concept Drift: The relationship between features (X) and the target (y) changes.
  For example, fraudsters adopt a new tactic, so what previously looked like a normal transaction 
  is now fraudulent.

Why monitoring is critical in production:
- ML models decay silently. Unlike traditional software that crashes when broken, ML models 
  will happily keep making bad predictions if the world changes around them.

Retraining Strategies:
- Scheduled: Retrain every week/month. (Simple, but might be too frequent or too late).
- Triggered: Retrain when performance drops or drift exceeds a threshold. (Efficient, requires monitoring).
- Continuous: Online learning where the model updates with every new batch of data. (Complex, risky).
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime

from src.config import LOGS_DIR, PSI_THRESHOLD, PERFORMANCE_DECAY_THRESHOLD

def calculate_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """
    Calculate the Population Stability Index (PSI) between two distributions.
    
    Formula: PSI = sum((actual% - expected%) * ln(actual%/expected%))
    
    Args:
        reference (np.ndarray): Baseline/training data array.
        current (np.ndarray): New/production data array.
        bins (int): Number of bins to use for the distributions.
        
    Returns:
        float: The calculated PSI value.
    """
    # Define bin edges based on reference distribution
    # We add slight noise to min/max to ensure all values fall into bins
    min_val = min(np.min(reference), np.min(current)) - 1e-5
    max_val = max(np.max(reference), np.max(current)) + 1e-5
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    
    # Calculate frequencies in bins
    ref_freq, _ = np.histogram(reference, bins=bin_edges)
    curr_freq, _ = np.histogram(current, bins=bin_edges)
    
    # Convert to percentages and add small epsilon to avoid divide-by-zero or log(0)
    eps = 1e-4
    ref_pct = (ref_freq / len(reference)) + eps
    curr_pct = (curr_freq / len(current)) + eps
    
    # MENTOR NOTE: The PSI formula evaluates the divergence. 
    # Usually: < 0.1 means no change, 0.1-0.2 means minor change, > 0.2 means significant drift.
    psi_values = (curr_pct - ref_pct) * np.log(curr_pct / ref_pct)
    return float(np.sum(psi_values))

def detect_data_drift(reference_data: pd.DataFrame, new_data: pd.DataFrame, feature_columns: List[str], threshold: float = None) -> Dict[str, Any]:
    """
    Calculate PSI for each feature to detect data drift.
    
    Args:
        reference_data (pd.DataFrame): Training data baseline.
        new_data (pd.DataFrame): Recent production data.
        feature_columns (List[str]): List of features to check.
        threshold (float): Threshold above which a feature is considered drifted.
        
    Returns:
        Dict[str, Any]: Drift analysis containing per-feature PSI, overall boolean, and list of drifted features.
    """
    threshold = threshold or PSI_THRESHOLD
    psi_dict = {}
    drifted_features = []
    
    for feature in feature_columns:
        # Extract 1D arrays, dropping NaNs if any exist
        ref = reference_data[feature].dropna().values
        curr = new_data[feature].dropna().values
        
        # Only calculate if we have data
        if len(ref) > 0 and len(curr) > 0:
            psi_val = calculate_psi(ref, curr)
            psi_dict[feature] = psi_val
            
            if psi_val > threshold:
                drifted_features.append(feature)
                
    overall_drift = len(drifted_features) > 0
    
    return {
        'per_feature_psi': psi_dict,
        'overall_drift': overall_drift,
        'drifted_features': drifted_features
    }

def log_prediction(transaction_data: Dict[str, Any], prediction: int, probability: float, threshold: float, log_file: str = None) -> None:
    """
    Append a prediction record to a CSV log file for monitoring.
    
    Args:
        transaction_data (Dict[str, Any]): Dictionary of input features.
        prediction (int): Predicted class (0 or 1).
        probability (float): Probability score of the positive class.
        threshold (float): The threshold used to make the prediction.
        log_file (str): Path to the log CSV file.
    """
    if log_file is None:
        os.makedirs(LOGS_DIR, exist_ok=True)
        log_file = os.path.join(LOGS_DIR, "prediction_logs.csv")
        
    timestamp = datetime.now().isoformat()
    # Simple hash of features to avoid logging sensitive PII directly, 
    # though in practice you'd log a unique transaction ID.
    features_hash = hash(json.dumps(transaction_data, sort_keys=True))
    
    record = pd.DataFrame([{
        'timestamp': timestamp,
        'features_hash': features_hash,
        'prediction': prediction,
        'probability': round(probability, 4),
        'threshold': threshold
    }])
    
    # Append to CSV. If file doesn't exist, write headers.
    file_exists = os.path.isfile(log_file)
    record.to_csv(log_file, mode='a', header=not file_exists, index=False)

def check_performance_decay(recent_metrics: Dict[str, float], baseline_metrics: Dict[str, float], threshold: float = None) -> Dict[str, Any]:
    """
    Compare key metrics to detect model decay.
    
    Args:
        recent_metrics (Dict[str, float]): Metrics from recent production data (e.g., ground truth labels gathered later).
        baseline_metrics (Dict[str, float]): Metrics from training/validation phase.
        threshold (float): Acceptable percentage drop before decay is declared.
        
    Returns:
        Dict[str, Any]: Dictionary detailing decay status, metric deltas, and recommendations.
    """
    threshold = threshold or PERFORMANCE_DECAY_THRESHOLD
    deltas = {}
    decayed = False
    
    for metric, baseline_val in baseline_metrics.items():
        if metric in recent_metrics:
            recent_val = recent_metrics[metric]
            delta = baseline_val - recent_val # Positive delta means performance dropped
            deltas[metric] = delta
            
            # If the drop is larger than the threshold, flag it
            if delta > threshold:
                decayed = True
                
    recommendation = "Model performance is stable."
    if decayed:
        recommendation = "Performance decay detected across key metrics. Model investigation required."
        
    return {
        'decayed': decayed,
        'metric_deltas': deltas,
        'recommendation': recommendation
    }

def get_retraining_recommendation(drift_report: Dict[str, Any], performance_report: Dict[str, Any]) -> str:
    """
    Business-logic decision: determine when to retrain based on drift and performance reports.
    
    Args:
        drift_report (Dict[str, Any]): Output from detect_data_drift.
        performance_report (Dict[str, Any]): Output from check_performance_decay.
        
    Returns:
        str: Human-readable recommendation string.
    """
    has_drift = drift_report.get('overall_drift', False)
    has_decay = performance_report.get('decayed', False)
    
    # MENTOR NOTE: This encapsulates our triggering strategy.
    # Drift without performance decay might just be a shift in benign patterns.
    # Performance decay without drift implies concept drift (the world changed, not the feature distributions).
    
    if has_decay and has_drift:
        return "URGENT ACTION: Both data drift and performance decay detected. Immediate retraining is required."
    elif has_decay:
        return "ACTION REQUIRED: Performance decay detected without feature drift (Concept Drift likely). Retrain model on recent data."
    elif has_drift:
        drifted_feats = drift_report.get('drifted_features', [])
        return f"WARNING: Data drift detected in features {drifted_feats}. Performance is stable, but monitor closely. Consider proactive retraining."
    else:
        return "ALL CLEAR: No significant drift or decay. Continue standard operations."
