"""
Module for hyperparameter tuning.

MENTOR NOTE:
Hyperparameter tuning is essential for maximizing model performance. In this module,
we implement RandomizedSearchCV and Optuna. 
- Random Search is generally more efficient than Grid Search because it samples a given 
  number of candidates from a parameter space. Grid search evaluates every combination, 
  which scales exponentially with the number of parameters.
- Optuna uses Bayesian optimization (specifically the Tree-structured Parzen Estimator, TPE sampler).
  Instead of guessing randomly, it learns from previous trials to suggest better hyperparameters, 
  and its pruning feature stops unpromising trials early, saving significant time.
- Cross-validation provides reliable estimates of model performance on unseen data by training
  on multiple folds of the training set.
- Choosing the right scoring metric is critical. For highly imbalanced datasets like fraud detection,
  accuracy is misleading. We use PR-AUC (average_precision) because it focuses on the positive (fraud) class.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
import optuna

from src.config import RANDOM_SEED

def tune_with_randomized_search(model_name: str, X_train: pd.DataFrame, y_train: pd.Series, n_iter: int = 50, cv: int = 5) -> Dict[str, Any]:
    """
    Tune hyperparameters using RandomizedSearchCV.
    
    Args:
        model_name (str): Name of the model ('random_forest' or 'xgboost').
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training targets.
        n_iter (int): Number of parameter settings that are sampled.
        cv (int): Number of cross-validation folds.
        
    Returns:
        Dict[str, Any]: Dictionary containing best_params, best_score, best_estimator, and cv_results.
        
    WHY: RandomizedSearchCV is a robust baseline for hyperparameter tuning that is easy to set up 
    and parallelize. Using StratifiedKFold ensures each fold has the same proportion of fraud cases.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
    
    if model_name == 'random_forest':
        estimator = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1, class_weight='balanced')
        param_distributions = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
    elif model_name == 'xgboost':
        estimator = XGBClassifier(random_state=RANDOM_SEED, n_jobs=-1, scale_pos_weight=100) # scale_pos_weight for imbalance
        param_distributions = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }
    else:
        raise ValueError(f"Unsupported model_name for RandomizedSearchCV: {model_name}")

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring='average_precision',
        cv=skf,
        verbose=1,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    
    search.fit(X_train, y_train)
    
    return {
        'best_params': search.best_params_,
        'best_score': search.best_score_,
        'best_estimator': search.best_estimator_,
        'cv_results': search.cv_results_
    }

def tune_with_optuna(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, n_trials: int = 50) -> Dict[str, Any]:
    """
    Tune LightGBM hyperparameters using Optuna.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training targets.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation targets.
        n_trials (int): Number of Optuna trials.
        
    Returns:
        Dict[str, Any]: Dictionary containing best_params, best_score, and the study object.
        
    WHY: Optuna's TPE sampler is highly efficient at finding optimal parameters for gradient boosting 
    frameworks like LightGBM. It learns from past trials rather than searching blindly.
    """
    from sklearn.metrics import average_precision_score

    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'custom',
            'boosting_type': 'gbdt',
            'random_state': RANDOM_SEED,
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'verbose': -1
        }
        
        # MENTOR NOTE: We pass validation sets to LightGBM to allow early stopping, 
        # which synergizes well with Optuna's pruning capabilities.
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)]
        )
        
        preds_proba = model.predict_proba(X_val)[:, 1]
        score = average_precision_score(y_val, preds_proba)
        return score

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=n_trials)
    
    return {
        'best_params': study.best_params,
        'best_score': study.best_value,
        'study': study
    }

def run_hyperparameter_tuning(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Dict[str, Any]]:
    """
    Runs tuning for all models and returns a dictionary of results.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training targets.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation targets.
        
    Returns:
        Dict[str, Dict[str, Any]]: Tuning results keyed by model name.
    """
    results = {}
    
    print("Tuning Random Forest...")
    results['random_forest'] = tune_with_randomized_search('random_forest', X_train, y_train)
    
    print("Tuning XGBoost...")
    results['xgboost'] = tune_with_randomized_search('xgboost', X_train, y_train)
    
    print("Tuning LightGBM...")
    results['lightgbm'] = tune_with_optuna(X_train, y_train, X_val, y_val)
    
    return results
