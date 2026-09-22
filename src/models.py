"""
Models Module.

This module defines model configurations and training pipelines.

MENTOR NOTE: We use sklearn Pipelines here. While we already scaled our data in the 
preprocessing step for EDA purposes, putting the model inside a Pipeline (with any 
additional transformers if needed later) is a best practice for deployment. It ensures 
that the exact same sequence of transformations is applied to incoming production data.
"""

import time
import logging
from typing import Dict, Any, Optional

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

from src.config import MODEL_CONFIGS

logger = logging.getLogger(__name__)

def get_model_pipeline(model_name: str, custom_params: Optional[Dict[str, Any]] = None) -> Pipeline:
    """
    Creates an sklearn Pipeline for the specified model.
    
    Args:
        model_name (str): The name of the model ('logistic_regression', 'random_forest', 
                          'xgboost', 'lightgbm').
        custom_params (dict, optional): Override default params from config.
        
    Returns:
        Pipeline: An sklearn Pipeline object.
        
    Raises:
        ValueError: If model_name is not supported.
    """
    # Base params from config, updated with any custom params provided
    params = MODEL_CONFIGS.get(model_name, {}).copy()
    if custom_params:
        params.update(custom_params)
        
    if model_name == 'logistic_regression':
        # MENTOR NOTE: Logistic Regression is a great baseline. It's fast, interpretable, 
        # and often works surprisingly well if features are linearly separable.
        model = LogisticRegression(**params)
        
    elif model_name == 'random_forest':
        # MENTOR NOTE: RF handles non-linearities well and is robust to outliers, 
        # but it can be very slow to train on large datasets and might overfit.
        model = RandomForestClassifier(**params)
        
    elif model_name == 'xgboost':
        # MENTOR NOTE: XGBoost is often the winning algorithm for tabular data. 
        # For severe imbalance, the 'scale_pos_weight' parameter (which is basically 
        # count(negative cases) / count(positive cases)) is magical.
        model = xgb.XGBClassifier(**params)
        
    elif model_name == 'lightgbm':
        # MENTOR NOTE: LightGBM is typically faster than XGBoost and uses less memory, 
        # making it fantastic for huge datasets like this one.
        model = lgb.LGBMClassifier(**params)
        
    else:
        raise ValueError(f"Model {model_name} is not supported.")
        
    # Build pipeline. We add a passthrough scaler just in case raw data is passed
    # directly to the pipeline in the future.
    pipeline = Pipeline([
        ('scaler', StandardScaler()), # Ensures data is scaled if pipeline used standalone
        ('classifier', model)
    ])
    
    return pipeline

def get_all_model_pipelines() -> Dict[str, Pipeline]:
    """
    Returns a dictionary of all initialized model pipelines.
    """
    return {
        'logistic_regression': get_model_pipeline('logistic_regression'),
        'random_forest': get_model_pipeline('random_forest'),
        'xgboost': get_model_pipeline('xgboost'),
        'lightgbm': get_model_pipeline('lightgbm')
    }

def train_model(pipeline: Pipeline, X_train: Any, y_train: Any, model_name: str) -> Pipeline:
    """
    Trains the model pipeline and logs the time taken.
    
    Args:
        pipeline (Pipeline): The initialized pipeline.
        X_train: Training features.
        y_train: Training target.
        model_name (str): Identifier for logging.
        
    Returns:
        Pipeline: The fitted pipeline.
    """
    logger.info(f"Starting training for {model_name}...")
    start_time = time.time()
    
    pipeline.fit(X_train, y_train)
    
    elapsed = time.time() - start_time
    logger.info(f"Finished training {model_name} in {elapsed:.2f} seconds.")
    
    return pipeline
