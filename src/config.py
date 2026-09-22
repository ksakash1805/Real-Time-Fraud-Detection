"""
config.py — Central Configuration for Credit Card Fraud Detection System
=========================================================================

WHY THIS FILE EXISTS (Mentor Note):
    In any ML project, you'll have constants scattered everywhere — paths,
    random seeds, column names, hyperparameter defaults. Centralizing them
    in one file means:
      1. A single source of truth (change once, reflects everywhere).
      2. Reproducibility — the random seed is set globally.
      3. Easy onboarding — a new team member reads this file first.
"""

import os
from pathlib import Path

# ============================================================
# 1. DIRECTORY PATHS
# ============================================================
# Path(__file__).resolve().parent.parent gives us the project root,
# regardless of where the script is called from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "creditcard.csv"
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = PROJECT_ROOT / "plots"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for d in [MODELS_DIR, PLOTS_DIR, MLRUNS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. RANDOM SEED — for reproducibility
# ============================================================
# WHY: Every random operation (train-test split, model init, SMOTE)
# uses this seed so results are identical across runs.
RANDOM_SEED = 42

# ============================================================
# 3. DATA SCHEMA
# ============================================================
# The dataset has 28 PCA-transformed features (V1–V28), plus Time and Amount.
# 'Class' is the target: 0 = legitimate, 1 = fraud.
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]  # V1 through V28
SCALE_COLUMNS = ["Time", "Amount"]  # these need scaling; PCA features are already scaled
FEATURE_COLUMNS = ["Time"] + PCA_FEATURES + ["Amount"]  # all 30 input features
TARGET_COLUMN = "Class"

# ============================================================
# 4. TRAIN-TEST SPLIT
# ============================================================
TEST_SIZE = 0.2  # 80% train, 20% test
VALIDATION_SIZE = 0.1  # 10% of training for validation (used in tuning)

# ============================================================
# 5. BUSINESS COST MATRIX
# ============================================================
# WHY THIS MATTERS (Mentor Note):
#   False Negative (FN) = We MISSED a fraud → the bank loses money.
#   False Positive (FP) = We BLOCKED a legit transaction → customer is annoyed.
#
#   In fraud detection, FN is MUCH more expensive than FP.
#   A typical FN might cost $500–$5000 (the fraud amount).
#   A typical FP might cost $10–$50 (customer service call + inconvenience).
#
#   We use these costs in threshold tuning (Phase 6) to find the
#   optimal decision boundary.
COST_FALSE_NEGATIVE = 500  # cost of missing a fraud (dollars)
COST_FALSE_POSITIVE = 10   # cost of blocking a legit transaction (dollars)

# ============================================================
# 6. MODEL CONFIGURATION
# ============================================================
# Default hyperparameters — these are starting points before tuning.
MODEL_CONFIGS = {
    "logistic_regression": {
        "C": 1.0,
        "max_iter": 1000,
        "class_weight": "balanced",
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "class_weight": "balanced",
        "n_jobs": -1,
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "eval_metric": "aucpr",  # PR-AUC is better for imbalanced data
        "use_label_encoder": False,
    },
    "lightgbm": {
        "n_estimators": 100,
        "max_depth": -1,  # no limit
        "learning_rate": 0.1,
        "is_unbalance": True,  # LightGBM's way of handling imbalance
        "verbose": -1,
    },
}

# ============================================================
# 7. API CONFIGURATION
# ============================================================
API_HOST = "0.0.0.0"
API_PORT = 8000
MODEL_FILENAME = "best_model.joblib"
SCALER_FILENAME = "scaler.joblib"
THRESHOLD_FILENAME = "threshold.joblib"
DEFAULT_THRESHOLD = 0.5

# ============================================================
# 8. MLFLOW CONFIGURATION
# ============================================================
MLFLOW_EXPERIMENT_NAME = "credit-card-fraud-detection"
MLFLOW_TRACKING_URI = f"file:///{MLRUNS_DIR.as_posix()}"

# ============================================================
# 9. MONITORING THRESHOLDS
# ============================================================
PSI_THRESHOLD = 0.2  # Population Stability Index > 0.2 → significant drift
PERFORMANCE_DECAY_THRESHOLD = 0.05  # F1 drop > 5% → trigger retraining
