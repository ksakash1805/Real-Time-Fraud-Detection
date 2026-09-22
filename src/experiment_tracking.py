"""
Module for MLflow experiment tracking.

MENTOR NOTE:
Why Experiment Tracking Matters:
- Reproducibility: If you run an experiment today, you want to be able to recreate it exactly 
  six months from now. Tracking saves code versions, hyperparameters, and seeds.
- Collaboration: In a team, everyone logs to a central server. You can see what others tried 
  and avoid duplicating work.

MLflow Concepts:
- Experiments: High-level groupings of runs (e.g., 'credit_card_fraud_detection').
- Runs: Individual model training events.
- Artifacts: Files produced by a run (models, plots, CSVs, logs).

MLOps Lifecycle:
- Tracking is the foundation of MLOps. Once we have a reliable registry of models and their metrics,
  we can automate the promotion of the best model to a staging/production environment.
"""

import os
import mlflow
import mlflow.sklearn
import pandas as pd
from typing import Dict, Any, Optional

from src.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI

def setup_mlflow() -> None:
    """
    Set up MLflow tracking URI and experiment name.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    print(f"MLflow setup complete. Tracking URI: {MLFLOW_TRACKING_URI}, Experiment: {MLFLOW_EXPERIMENT_NAME}")

def log_experiment(model_name: str, model: Any, params: Dict[str, Any], metrics: Dict[str, float], artifacts_dir: Optional[str] = None) -> str:
    """
    Log a model training experiment to MLflow.
    
    Args:
        model_name (str): Name of the run/model.
        model (Any): The trained model object.
        params (Dict[str, Any]): Hyperparameters.
        metrics (Dict[str, float]): Evaluation metrics.
        artifacts_dir (Optional[str]): Path to a directory containing artifacts (plots, etc.) to log.
        
    Returns:
        str: The MLflow run ID.
    """
    # MENTOR NOTE: The 'with' context manager ensures the run is properly closed even if an error occurs.
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        
        # Log the model artifact itself
        mlflow.sklearn.log_model(model, "model")
        
        # Log supplementary files if provided
        if artifacts_dir and os.path.exists(artifacts_dir):
            mlflow.log_artifacts(artifacts_dir, artifact_path="evaluation_plots")
            
        run_id = run.info.run_id
        print(f"Successfully logged run {run_id} for model {model_name}")
        return run_id

def log_comparison_table(results_df: pd.DataFrame, experiment_name: Optional[str] = None) -> None:
    """
    Log a comparison DataFrame as a CSV artifact.
    
    Args:
        results_df (pd.DataFrame): DataFrame containing model comparison results.
        experiment_name (Optional[str]): Name of the experiment to log under. 
            If None, uses a temporary run.
    """
    csv_path = "model_comparison.csv"
    results_df.to_csv(csv_path, index=False)
    
    # We can log this to the active run, or start a brief run just to log summary info.
    with mlflow.start_run(run_name="model_comparison_summary"):
        mlflow.log_artifact(csv_path)
    
    if os.path.exists(csv_path):
        os.remove(csv_path)

def get_best_run(metric_name: str = 'pr_auc') -> Dict[str, Any]:
    """
    Query MLflow for the best run based on a specific metric.
    
    Args:
        metric_name (str): The metric to optimize (assumes higher is better for PR-AUC).
        
    Returns:
        Dict[str, Any]: Dictionary with run_id, params, and metrics of the best run.
    """
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    
    if not experiment:
        raise ValueError(f"Experiment {MLFLOW_EXPERIMENT_NAME} not found.")
        
    # Search for runs and order by the specified metric descending
    query = f"metrics.{metric_name} DESC"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[query],
        max_results=1
    )
    
    if not runs:
        return {}
        
    best_run = runs[0]
    return {
        'run_id': best_run.info.run_id,
        'params': best_run.data.params,
        'metrics': best_run.data.metrics
    }
