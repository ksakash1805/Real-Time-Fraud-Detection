"""
Data Loader Module for Credit Card Fraud Detection.

This module provides functions to load the dataset and perform basic validation
and summary statistics.

MENTOR NOTE: Always validate your data immediately after loading. Don't assume the CSV 
is formatted correctly or hasn't changed. Fail early if something is wrong!
"""

import pandas as pd
import logging
import os
from typing import Dict, Any

from src.config import DATA_PATH

logger = logging.getLogger(__name__)

def load_data(path: str = None) -> pd.DataFrame:
    """
    Loads the credit card fraud dataset from CSV and validates it.
    
    Args:
        path (str, optional): Override the default DATA_PATH. Defaults to None.
        
    Returns:
        pd.DataFrame: The validated dataset.
        
    Raises:
        FileNotFoundError: If the data file is not found.
        ValueError: If validation fails.
    """
    data_file = path if path is not None else DATA_PATH
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found at {data_file}")
    
    logger.info(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)
    
    # Validation step 1: Check shape
    if df.empty:
        raise ValueError("Loaded dataset is empty!")
        
    logger.info(f"Successfully loaded dataset with shape: {df.shape}")
    
    # Validation step 2: Check target column exists
    # MENTOR NOTE: We hardcode 'Class' here because it's specific to this dataset,
    # but normally we'd import TARGET_COLUMN from config.
    if 'Class' not in df.columns:
        raise ValueError("Target column 'Class' is missing from the dataset!")
        
    # Validation step 3: Check for nulls
    null_counts = df.isnull().sum().sum()
    if null_counts > 0:
        logger.warning(f"Found {null_counts} missing values in the dataset. Further preprocessing required.")
    
    return df

def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates a summary of the dataset.
    
    Args:
        df (pd.DataFrame): The dataset to summarize.
        
    Returns:
        dict: A dictionary containing shape, class_distribution, null_counts, and dtypes.
    """
    # MENTOR NOTE: Always check class distribution early in a classification problem. 
    # For fraud detection, we expect an extreme imbalance. This will guide our choices 
    # for resampling and evaluation metrics.
    
    summary = {
        'shape': df.shape,
        'class_distribution': df['Class'].value_counts(normalize=True).to_dict(),
        'class_counts': df['Class'].value_counts().to_dict(),
        'null_counts': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.astype(str).to_dict()
    }
    
    logger.info(f"Data summary: {summary['shape']} rows/cols, {summary['class_counts']} class counts.")
    
    return summary
