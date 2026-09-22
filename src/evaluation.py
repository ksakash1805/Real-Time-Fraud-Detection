"""
Evaluation Module.

This module handles metric calculation, plotting, and threshold tuning 
based on a business cost matrix.

MENTOR NOTE: In Fraud Detection, ACCURACY IS A LIE. 
If 99.83% of transactions are legitimate, a completely useless model that predicts 
"Legit" 100% of the time will have 99.83% accuracy. We must look at Precision, Recall, 
and PR-AUC instead.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from typing import Tuple, Dict, Any

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, auc
)

from src.config import PLOTS_DIR, COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE

logger = logging.getLogger(__name__)

def _ensure_dir(save_dir: str):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

def get_predictions(model: Any, X_test: pd.DataFrame, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gets binary predictions and probabilities from the model.
    """
    # MENTOR NOTE: We predict_proba because we might want to shift the threshold 
    # away from the default 0.5 later.
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return y_pred, y_prob

def calculate_metrics(y_test: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Calculates various classification metrics.
    """
    # Precision-Recall curve AUC
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'pr_auc': pr_auc
    }
    return metrics

def plot_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray, model_name: str, save_dir: str):
    """
    Plots and saves the confusion matrix.
    """
    _ensure_dir(save_dir)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legit (0)', 'Fraud (1)'],
                yticklabels=['Legit (0)', 'Fraud (1)'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_confusion_matrix.png'))
    plt.close()

def plot_roc_curve(y_test: np.ndarray, y_prob: np.ndarray, model_name: str, save_dir: str):
    """Plots and saves the ROC curve."""
    _ensure_dir(save_dir)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_roc_curve.png'))
    plt.close()

def plot_precision_recall_curve(y_test: np.ndarray, y_prob: np.ndarray, model_name: str, save_dir: str):
    """
    Plots and saves the Precision-Recall curve.
    
    MENTOR NOTE: For highly imbalanced datasets, PR Curve is vastly superior to ROC. 
    ROC includes True Negatives in its False Positive Rate calculation. Because TNs 
    are massive (99.8%), FPR stays tiny even if we have many False Positives. 
    PR focuses strictly on the minority class (Fraud) and its false alarms.
    """
    _ensure_dir(save_dir)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.4f}')
    plt.xlabel('Recall (True Positive Rate)')
    plt.ylabel('Precision (Positive Predictive Value)')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_pr_curve.png'))
    plt.close()

def find_optimal_threshold(y_test: np.ndarray, y_prob: np.ndarray, cost_fn: float, cost_fp: float, save_dir: str = None, model_name: str = "") -> Tuple[float, Dict[float, float]]:
    """
    Finds the probability threshold that minimizes the total business cost.
    
    MENTOR NOTE: In the real world, ML models aren't evaluated on abstract math metrics, 
    they are evaluated on money. 
    A False Negative (missed fraud) might cost the bank $1000. 
    A False Positive (declining a legit transaction) upsets a customer, maybe costing $10.
    We sweep through probability thresholds (0.01 to 0.99) to find the exact point 
    that minimizes the total dollar loss to the business.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = {}
    
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_test, preds)
        
        # Guard against completely predicting one class
        if cm.shape == (2, 2):
            fp = cm[0, 1]
            fn = cm[1, 0]
        else:
            # If everything is predicted as 0
            if len(np.unique(preds)) == 1 and preds[0] == 0:
                fp = 0
                fn = sum(y_test == 1)
            # If everything is predicted as 1
            else:
                fp = sum(y_test == 0)
                fn = 0
                
        total_cost = (fn * cost_fn) + (fp * cost_fp)
        costs[t] = total_cost
        
    optimal_threshold = min(costs, key=costs.get)
    
    if save_dir:
        _ensure_dir(save_dir)
        plt.figure(figsize=(8, 6))
        plt.plot(list(costs.keys()), list(costs.values()))
        plt.axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Optimal T={optimal_threshold:.2f}')
        plt.xlabel('Probability Threshold')
        plt.ylabel('Total Business Cost')
        plt.title(f'Business Cost vs Threshold - {model_name}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{model_name}_cost_curve.png'))
        plt.close()
        
    return optimal_threshold, costs

def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: str, threshold: float = 0.5, save_dir: str = None) -> Dict[str, Any]:
    """
    Evaluates a model comprehensively and saves plots.
    """
    save_dir = save_dir or str(PLOTS_DIR)
    
    y_pred, y_prob = get_predictions(model, X_test, threshold)
    metrics = calculate_metrics(y_test, y_pred, y_prob)
    
    plot_confusion_matrix(y_test, y_pred, model_name, save_dir)
    plot_roc_curve(y_test, y_prob, model_name, save_dir)
    plot_precision_recall_curve(y_test, y_prob, model_name, save_dir)
    
    opt_thresh, _ = find_optimal_threshold(
        y_test, y_prob, COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE, 
        save_dir=save_dir, model_name=model_name
    )
    
    metrics['optimal_threshold'] = opt_thresh
    
    # Print report
    print(f"--- Evaluation Report for {model_name} (Threshold: {threshold}) ---")
    print(f"PR AUC (Most important!): {metrics['pr_auc']:.4f}")
    print(f"Recall (Fraud detection rate): {metrics['recall']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Business Cost Optimal Threshold: {opt_thresh:.2f}")
    print("-" * 50)
    
    return metrics

def compare_models(results_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Takes a dictionary mapping model_name -> metrics dict, and returns a DataFrame.
    """
    df = pd.DataFrame.from_dict(results_dict, orient='index')
    # Sort by PR AUC as it's the most important metric for imbalanced data
    df = df.sort_values(by='pr_auc', ascending=False)
    return df
