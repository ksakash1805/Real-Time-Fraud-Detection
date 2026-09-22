"""
Resampling Module for Imbalanced Data.

This module provides techniques to handle severe class imbalance, including 
class weights, undersampling, and SMOTE.

MENTOR NOTE: A critical rule of resampling: ONLY EVER RESAMPLE THE TRAINING SET.
Why? Because your test set must represent the real world. If you balance the test set, 
you are evaluating your model on an artificial scenario that it will never see in production.
In the real world, fraud is 0.17%. Your test set must remain at 0.17%.
"""

import pandas as pd
from typing import Tuple, Dict, Any
import logging
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

from src.config import RANDOM_SEED

logger = logging.getLogger(__name__)

def apply_class_weights() -> str:
    """
    Returns the string 'balanced' to be passed to model configurations.
    
    MENTOR NOTE: This is often the best first approach. Instead of changing the data, 
    we change the algorithm's loss function. A 'balanced' weight heavily penalizes 
    misclassifying the minority class (fraud), effectively simulating a balanced dataset 
    without losing data (like undersampling) or inventing data (like SMOTE).
    """
    return 'balanced'

def apply_random_undersampling(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Applies Random Undersampling to the training data.
    
    MENTOR NOTE: Pros: Fast training because dataset shrinks massively.
    Cons: We throw away 99.8% of our legitimate transactions! We lose a huge amount 
    of valuable information about what normal behavior looks like. Use with caution.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training targets.
        
    Returns:
        tuple: (X_resampled, y_resampled)
    """
    logger.info("Applying Random UnderSampler...")
    rus = RandomUnderSampler(random_state=RANDOM_SEED)
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
    logger.info(f"New shape after undersampling: {X_resampled.shape}")
    return X_resampled, y_resampled

def apply_smote(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Applies SMOTE (Synthetic Minority Over-sampling Technique).
    
    MENTOR NOTE: Pros: Doesn't throw away data. Creates synthetic examples of fraud 
    by interpolating between existing fraud cases.
    Cons: Very slow on large datasets. Can introduce noise if fraud cases are heavily 
    overlapped with legitimate cases.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training targets.
        
    Returns:
        tuple: (X_resampled, y_resampled)
    """
    logger.info("Applying SMOTE...")
    smote = SMOTE(random_state=RANDOM_SEED)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    logger.info(f"New shape after SMOTE: {X_resampled.shape}")
    return X_resampled, y_resampled

def get_resampled_datasets(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """
    Generates a dictionary containing the original and resampled datasets.
    
    Args:
        X_train (pd.DataFrame): Original training features.
        y_train (pd.Series): Original training targets.
        
    Returns:
        dict: Keys 'original', 'undersampled', 'smote' mapping to (X, y) tuples.
    """
    datasets = {
        'original': (X_train, y_train),
        'undersampled': apply_random_undersampling(X_train, y_train),
        'smote': apply_smote(X_train, y_train)
    }
    return datasets
