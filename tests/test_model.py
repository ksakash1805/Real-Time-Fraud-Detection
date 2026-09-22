"""
Unit tests for model training and prediction.

MENTOR NOTE:
Testing models directly involves verifying that the pipeline constructs correctly,
can fit on data without crashing, produces outputs in expected ranges, and can
be serialized for deployment. We use small synthetic datasets for speed, as we
are testing the *code* not the *accuracy* here.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from src.models import get_model_pipeline, train_model
from src.config import FEATURE_COLUMNS

@pytest.fixture
def synthetic_data():
    """Creates a small synthetic dataset for testing training functions."""
    np.random.seed(42)
    n_samples = 100
    
    # Create random features
    X = pd.DataFrame(
        np.random.randn(n_samples, len(FEATURE_COLUMNS)),
        columns=FEATURE_COLUMNS
    )
    
    # Create random binary target
    y = pd.Series(np.random.randint(0, 2, n_samples))
    
    return X, y

def test_model_pipeline_creation():
    """Verify get_model_pipeline returns a Pipeline for each model name."""
    valid_models = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm']
    
    for model_name in valid_models:
        try:
            pipeline = get_model_pipeline(model_name)
            assert isinstance(pipeline, Pipeline), f"Expected Pipeline for {model_name}"
            # Ensure it has an estimator step
            assert 'classifier' in pipeline.named_steps
        except ImportError as e:
            pytest.skip(f"Skipping {model_name} due to missing dependency: {e}")

def test_invalid_model_name():
    """Verify get_model_pipeline raises ValueError for unknown model."""
    with pytest.raises(ValueError):
        get_model_pipeline("unsupported_model_type")

def test_model_training(synthetic_data):
    """Verify a model can be trained on small dummy data without errors."""
    X, y = synthetic_data
    pipeline = get_model_pipeline('logistic_regression')
    
    # Should not raise any exceptions
    trained_model = train_model(pipeline, X, y, "test_lr")
    
    # Check if the model is fitted by checking classes_ attribute of classifier
    assert hasattr(trained_model.named_steps['classifier'], 'classes_'), "Model does not appear to be fitted"

def test_prediction_output_range(synthetic_data):
    """Verify model predictions are probabilities in [0, 1]."""
    X, y = synthetic_data
    pipeline = get_model_pipeline('logistic_regression')
    trained_model = train_model(pipeline, X, y, "test_lr")
    
    probas = trained_model.predict_proba(X)[:, 1]
    
    assert np.all(probas >= 0.0), "Found probabilities < 0"
    assert np.all(probas <= 1.0), "Found probabilities > 1"

def test_model_serialization(synthetic_data):
    """Verify model can be saved and loaded with joblib."""
    X, y = synthetic_data
    pipeline = get_model_pipeline('logistic_regression')
    trained_model = train_model(pipeline, X, y, "test_lr")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "test_model.joblib"
        
        # Save
        joblib.dump(trained_model, model_path)
        assert model_path.exists(), "Model file was not created"
        
        # Load
        loaded_model = joblib.load(model_path)
        
        # Ensure predictions match
        orig_preds = trained_model.predict_proba(X)
        loaded_preds = loaded_model.predict_proba(X)
        
        np.testing.assert_array_almost_equal(orig_preds, loaded_preds, err_msg="Predictions differ after reloading model")
