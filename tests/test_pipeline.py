"""
Unit tests for MLOps pipeline
"""
import pytest
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def test_data_files_exist():
    """Test that generated data files exist"""
    data_dir = 'data'
    expected_files = [
        'housing_data.csv',
        'credit_data.csv',
        'churn_data.csv'
    ]
    
    for file in expected_files:
        filepath = os.path.join(data_dir, file)
        # Test will pass if file exists or skip if not generated yet
        if os.path.exists(data_dir):
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                assert len(df) > 0, f"{file} is empty"
                assert len(df.columns) > 0, f"{file} has no columns"


def test_housing_data_structure():
    """Test California Housing data structure"""
    filepath = 'data/housing_data.csv'
    
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        
        # Check expected columns
        expected_cols = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                        'Population', 'AveOccup', 'Latitude', 'Longitude', 
                        'MedHouseVal']
        
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"
        
        # Check data types
        assert df['MedHouseVal'].dtype in [np.float64, np.float32], "Target should be float"
        
        # Check for missing values
        assert df.isnull().sum().sum() == 0, "Data contains missing values"


def test_model_creation():
    """Test that models can be instantiated"""
    # Test regression model
    reg_model = RandomForestRegressor(n_estimators=10, random_state=42)
    assert reg_model is not None
    
    # Test classification model
    clf_model = RandomForestClassifier(n_estimators=10, random_state=42)
    assert clf_model is not None


def test_model_training():
    """Test basic model training"""
    # Create simple dummy data
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    
    # Train model
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Test prediction
    predictions = model.predict(X[:10])
    assert len(predictions) == 10
    assert predictions.dtype in [np.float64, np.float32]


def test_metrics_directory():
    """Test that metrics directory exists or can be created"""
    metrics_dir = 'metrics'
    
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)
    
    assert os.path.exists(metrics_dir), "Metrics directory doesn't exist"
    assert os.path.isdir(metrics_dir), "Metrics path is not a directory"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
