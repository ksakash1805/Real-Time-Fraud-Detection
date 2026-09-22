"""
Preprocessing Module.

This module handles feature scaling and train-test splitting.
Crucially, it is designed to prevent data leakage.
"""

import pandas as pd
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

from src.config import (
    FEATURE_COLUMNS, 
    TARGET_COLUMN, 
    SCALE_COLUMNS, 
    TEST_SIZE, 
    RANDOM_SEED
)

logger = logging.getLogger(__name__)

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame, columns_to_scale: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scales specific features using StandardScaler.
    
    WHY WE DO THIS:
    MENTOR NOTE: Data leakage is the #1 mistake junior ML engineers make.
    If you fit your scaler on the ENTIRE dataset before splitting, information from the 
    test set "leaks" into the scaler's mean and variance calculations. 
    When we evaluate the model, it will perform artificially well because it has 
    indirectly "seen" the test set statistics.
    ALWAYS fit on train, transform on train and test.
    
    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        columns_to_scale (List[str]): Columns to apply scaling to.
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, fitted_scaler)
    """
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    scaler = StandardScaler()
    
    # Fit ONLY on training data
    scaler.fit(X_train[columns_to_scale])
    
    # Transform both
    X_train_scaled[columns_to_scale] = scaler.transform(X_train[columns_to_scale])
    X_test_scaled[columns_to_scale] = scaler.transform(X_test[columns_to_scale])
    
    return X_train_scaled, X_test_scaled, scaler

def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """
    Executes the full preprocessing pipeline.
    
    1. Separates features and target.
    2. Performs a stratified train-test split.
    3. Scales specified features.
    
    Args:
        df (pd.DataFrame): The raw loaded dataset.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    logger.info("Starting data preparation pipeline...")
    
    # 1. Separate features and target
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    
    # 2. Stratified train-test split
    # MENTOR NOTE: 'stratify=y' is absolutely critical here. With only 0.17% fraud cases, 
    # a random split might result in 0 fraud cases in the test set just by bad luck.
    # Stratification ensures the train and test sets have the same proportion of classes.
    logger.info(f"Splitting data (test_size={TEST_SIZE}, stratify=True)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_SEED, 
        stratify=y
    )
    
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # 3. Scale features (Amount, Time - as V1-V28 are already PCA transformed)
    logger.info(f"Scaling columns: {SCALE_COLUMNS}")
    X_train, X_test, scaler = scale_features(X_train, X_test, SCALE_COLUMNS)
    
    return X_train, X_test, y_train, y_test, scaler
