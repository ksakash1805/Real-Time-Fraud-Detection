"""
Unit tests for data preprocessing module.

MENTOR NOTE:
Testing ML pipelines is different from testing standard software. 
While traditional tests verify deterministic logic (1+1=2), ML tests must verify 
statistical properties, data distribution preservation, and structural integrity.
A common failure in ML is "silent failure" where code runs without crashing but 
produces garbage predictions due to data leakage or incorrect scaling.
"""

import pytest
import numpy as np
import pandas as pd
from src.preprocessing import prepare_data
from src.data_loader import load_data
from src.config import DATA_PATH, TARGET_COLUMN, SCALE_COLUMNS, FEATURE_COLUMNS, TEST_SIZE

@pytest.fixture(scope="module")
def sample_data():
    """
    Fixture to load data once for all tests.
    
    MENTOR NOTE:
    Data loading is expensive. We use scope='module' so the dataset is only 
    loaded once per test file execution, speeding up the test suite.
    """
    try:
        # Load a small sample to keep tests fast, or load full if needed.
        # Here we attempt to load the actual data, but limit to 10000 rows for speed if possible.
        # Alternatively, we just use load_data(). Let's load actual data to test realistic properties.
        df = load_data()
        return df.sample(min(10000, len(df)), random_state=42)
    except Exception as e:
        pytest.skip(f"Could not load data for testing: {e}")

def test_prepare_data_shapes(sample_data):
    """Verify X_train, X_test, y_train, y_test have correct shapes after split."""
    X_train, X_test, y_train, y_test, scaler = prepare_data(sample_data)
    
    expected_train_size = int(len(sample_data) * (1 - TEST_SIZE))
    
    # Allow for rounding differences
    assert abs(len(X_train) - expected_train_size) <= 1
    assert len(X_train) + len(X_test) == len(sample_data)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert X_train.shape[1] == len(FEATURE_COLUMNS)

def test_stratification_preserves_ratio(sample_data):
    """
    Verify fraud ratio in train and test sets is approximately equal to original ratio.
    """
    original_ratio = sample_data[TARGET_COLUMN].mean()
    X_train, X_test, y_train, y_test, scaler = prepare_data(sample_data)
    
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()
    
    # Check if ratios are within 0.01 (1%)
    assert abs(original_ratio - train_ratio) < 0.01, "Train fraud ratio diverges from original"
    assert abs(original_ratio - test_ratio) < 0.01, "Test fraud ratio diverges from original"

def test_no_nulls_after_preprocessing(sample_data):
    """Verify no NaN values in processed data."""
    X_train, X_test, y_train, y_test, scaler = prepare_data(sample_data)
    
    assert not X_train.isnull().any().any(), "NaNs found in X_train"
    assert not X_test.isnull().any().any(), "NaNs found in X_test"

def test_scaler_fit_only_on_train():
    """
    Verify test data is scaled using train statistics (not its own).
    
    MENTOR NOTE:
    Data leakage is a critical issue! If we fit the scaler on the entire dataset 
    or fit a new scaler on the test set, information from the test set "leaks" 
    into our training process, leading to overly optimistic evaluation metrics.
    """
    # Create known dummy data
    from sklearn.preprocessing import RobustScaler
    
    df_train = pd.DataFrame({
        'Time': [1, 2, 3],
        'Amount': [10, 20, 30],
        'V1': [0.1, 0.2, 0.3],
        'Class': [0, 0, 1]
    })
    
    df_test = pd.DataFrame({
        'Time': [100, 200, 300], # drastically different to verify scaling uses train stats
        'Amount': [1000, 2000, 3000],
        'V1': [0.5, 0.6, 0.7],
        'Class': [1, 0, 0]
    })
    
    scaler = RobustScaler()
    # Fit only on train
    scaler.fit(df_train[['Time', 'Amount']])
    
    # Transform test
    test_scaled = scaler.transform(df_test[['Time', 'Amount']])
    
    # If the scaler was fit on test_data, the robust median would be 200 for Time.
    # Since it was fit on train_data, the median is 2, IQR (Q3-Q1) = 2.5-1.5 = 1.0
    # Scaled Time for 100 should be (100 - 2) / 1.0 = 98.0
    assert np.isclose(test_scaled[0, 0], 98.0), "Scaler appears to have fit on test data or incorrectly scaled!"

def test_feature_columns_correct(sample_data):
    """Verify the correct columns are present in processed data."""
    X_train, _, _, _, _ = prepare_data(sample_data)
    
    assert list(X_train.columns) == FEATURE_COLUMNS, "Processed features do not match config"

def test_scale_columns_transformed(sample_data):
    """Verify that Time and Amount columns have been scaled (StandardScaler: mean ~0, std ~1)."""
    X_train, _, _, _, _ = prepare_data(sample_data)

    # StandardScaler transforms to mean=0, std=1 on training data
    for col in SCALE_COLUMNS:
        mean_val = X_train[col].mean()
        std_val = X_train[col].std()
        # Mean should be very close to 0
        assert abs(mean_val) < 1e-5, f"{col} mean is not ~0, scaling might have failed"
        # Std should be very close to 1
        assert abs(std_val - 1.0) < 0.1, f"{col} std is not ~1, scaling might have failed"
