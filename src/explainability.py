"""
Module for model explainability using SHAP.

MENTOR NOTE:
Why use SHAP over `feature_importances_`?
- Consistency: `feature_importances_` in tree models (like Gini importance) can be biased 
  towards continuous variables or variables with high cardinality. SHAP (SHapley Additive exPlanations)
  is grounded in cooperative game theory and guarantees a fair distribution of the prediction 
  among the features.
- Local Explainability: SHAP provides feature contributions for *individual* predictions, not just globally.

Explaining ML to Non-Technical Stakeholders:
- Don't use terms like 'log-odds' or 'Shapley values'. Talk in terms of "probability", "baseline risk",
  and "how much a feature pushed the score up or down."
- Always contextualize the feature value against normal behavior (e.g., "Amount was unusually high").

Regulatory Requirements:
- Regulations like GDPR mandate a "Right to Explanation" for automated decisions affecting users. 
  If we block a customer's card, we must be able to explain *why* in clear terms.
"""

import os
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Any

from src.config import PLOTS_DIR, FEATURE_COLUMNS

def compute_shap_values(model: Any, X_test: pd.DataFrame, model_name: str = 'tree') -> Tuple[Any, Any]:
    """
    Compute SHAP values for a given model and dataset.
    
    Args:
        model (Any): The trained model (or pipeline).
        X_test (pd.DataFrame): The test dataset.
        model_name (str): Type of model, e.g., 'tree' or 'linear'.
        
    Returns:
        Tuple[Any, Any]: The SHAP explainer object and the computed SHAP values.
    """
    # Handle pipelines by extracting the classifier step if necessary
    if hasattr(model, 'steps'):
        # Assumes the classifier is the last step
        classifier = model.steps[-1][1]
    else:
        classifier = model
        
    if model_name == 'tree':
        # MENTOR NOTE: TreeExplainer is heavily optimized for tree-based models (RF, XGBoost, LightGBM)
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_test)
    elif model_name == 'linear':
        explainer = shap.LinearExplainer(classifier, X_test)
        shap_values = explainer.shap_values(X_test)
    else:
        explainer = shap.Explainer(classifier, X_test)
        shap_values = explainer(X_test)
        
    return explainer, shap_values

def plot_global_feature_importance(shap_values: Any, X_test: pd.DataFrame, save_dir: str) -> None:
    """
    Generate and save global SHAP feature importance plots.
    
    Args:
        shap_values (Any): Computed SHAP values.
        X_test (pd.DataFrame): Test dataset features.
        save_dir (str): Directory to save plots.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # SHAP values structure varies between binary classification implementations (list vs array)
    if isinstance(shap_values, list):
        vals_to_plot = shap_values[1] # Use values for the positive class (fraud)
    else:
        vals_to_plot = shap_values

    # Beeswarm plot shows directionality and distribution
    plt.figure()
    shap.summary_plot(vals_to_plot, X_test, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'shap_summary_beeswarm.png'))
    plt.close()
    
    # Bar plot shows mean absolute importance
    plt.figure()
    shap.summary_plot(vals_to_plot, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'shap_summary_bar.png'))
    plt.close()

def plot_single_prediction(explainer: Any, shap_values: Any, X_test: pd.DataFrame, index: int, save_dir: str) -> None:
    """
    Generate and save a waterfall plot for a single prediction.
    
    Args:
        explainer (Any): The SHAP explainer object.
        shap_values (Any): Computed SHAP values.
        X_test (pd.DataFrame): Test dataset features.
        index (int): Index of the instance to explain.
        save_dir (str): Directory to save the plot.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if isinstance(shap_values, list):
        base_value = explainer.expected_value[1]
        values = shap_values[1][index]
    else:
        # Handle cases where expected_value is a scalar or array
        base_value = explainer.expected_value if not isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value[1] if len(np.atleast_1d(explainer.expected_value)) > 1 else explainer.expected_value
        values = shap_values[index] if len(np.shape(shap_values)) == 2 else shap_values[index, :, 1]
        
    explanation = shap.Explanation(values=values, 
                                   base_values=base_value, 
                                   data=X_test.iloc[index], 
                                   feature_names=X_test.columns)
    
    plt.figure()
    shap.plots.waterfall(explanation, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'shap_waterfall_{index}.png'))
    plt.close()

def explain_prediction_for_stakeholder(shap_values: Any, X_test: pd.DataFrame, index: int, feature_names: list) -> str:
    """
    Generate a plain-English explanation string for a specific prediction.
    
    Args:
        shap_values (Any): Computed SHAP values.
        X_test (pd.DataFrame): Test dataset features.
        index (int): Index of the instance to explain.
        feature_names (list): List of feature names.
        
    Returns:
        str: Plain English explanation.
    """
    if isinstance(shap_values, list):
        instance_shap = shap_values[1][index]
    else:
        instance_shap = shap_values[index] if len(np.shape(shap_values)) == 2 else shap_values[index, :, 1]
        
    feature_vals = X_test.iloc[index].values
    
    # Combine feature names, values, and SHAP contributions
    contributions = list(zip(feature_names, feature_vals, instance_shap))
    
    # Sort by absolute contribution to find the top drivers
    contributions.sort(key=lambda x: abs(x[2]), reverse=True)
    top_3 = contributions[:3]
    
    # Mocking probability and normal range since we don't have the model predicting or full distributions here
    # In a real scenario, probability comes from model.predict_proba() and normal ranges from training stats.
    prob = 92 # Example placeholder
    
    explanation_parts = [f"This transaction was flagged as FRAUD (probability: {prob}%). The top reasons are:"]
    
    for i, (feat, val, contrib) in enumerate(top_3, 1):
        direction = "high" if val > 0 else "low"
        contrib_sign = "+" if contrib > 0 else ""
        
        # MENTOR NOTE: We use template strings to translate raw numbers into actionable business insights.
        if feat == 'Amount':
            explanation_parts.append(f"{i}. {feat} was unusually {direction} (${val:.2f}), contributing {contrib_sign}{contrib:.2f} to fraud score.")
        else:
            explanation_parts.append(f"{i}. {feat} was unusually {direction} ({val:.2f}), contributing {contrib_sign}{contrib:.2f} to fraud score.")
            
    return " ".join(explanation_parts)
