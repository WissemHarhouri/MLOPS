"""
Configuration file for MLOps Pipeline
Centralized settings for all experiments
"""

# MLflow Configuration
MLFLOW_TRACKING_URI = "file:./mlruns"
MLFLOW_ARTIFACT_LOCATION = "./mlartifacts"

# Dataset Configurations
DATASETS = {
    'california_housing': {
        'file_path': 'data/housing_data.csv',
        'target_column': 'MedHouseVal',
        'task_type': 'regression',
        'test_size': 0.2,
        'random_state': 42,
        'metrics': ['rmse', 'mae', 'r2'],
        'description': 'California Housing Price Prediction'
    },
    'credit_fraud': {
        'file_path': 'data/credit_data.csv',
        'target_column': 'Class',
        'task_type': 'classification_imbalanced',
        'test_size': 0.2,
        'random_state': 42,
        'metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        'class_weight': 'balanced',
        'description': 'Credit Card Fraud Detection'
    },
    'customer_churn': {
        'file_path': 'data/churn_data.csv',
        'target_column': 'Churn',
        'task_type': 'classification',
        'test_size': 0.2,
        'random_state': 42,
        'metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        'description': 'Customer Churn Prediction'
    }
}

# Model Configurations
MODELS = {
    'random_forest': {
        'regression': {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'n_jobs': -1
        },
        'classification': {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'n_jobs': -1
        }
    },
    'gradient_boosting': {
        'regression': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'random_state': 42
        },
        'classification': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'random_state': 42
        }
    },
    'logistic_regression': {
        'classification': {
            'random_state': 42,
            'max_iter': 1000,
            'solver': 'lbfgs'
        }
    }
}

# Optuna Configuration
OPTUNA_CONFIG = {
    'n_trials': 50,
    'timeout': None,
    'n_jobs': 1,
    'sampler': 'TPE',  # Tree-structured Parzen Estimator
    'pruner': 'MedianPruner',
    'direction': 'maximize',  # for most metrics
    'search_space': {
        'n_estimators': (50, 300),
        'max_depth': (3, 20),
        'min_samples_split': (2, 20),
        'min_samples_leaf': (1, 10),
        'max_features': ['sqrt', 'log2', None]
    }
}

# Evidently Configuration
EVIDENTLY_CONFIG = {
    'drift_threshold': 0.3,  # 30% of features
    'feature_drift_threshold': 0.1,
    'data_quality_threshold': 0.95,
    'output_dir': 'reports',
    'report_types': ['data_drift', 'data_quality', 'target_drift']
}

# DVC Configuration
DVC_CONFIG = {
    'metrics_dir': 'metrics',
    'plots_dir': 'plots',
    'cache_dir': '.dvc/cache'
}

# GitHub Actions Configuration
CI_CD_CONFIG = {
    'python_version': '3.9',
    'pytest_options': '-v --cov=. --cov-report=html',
    'flake8_max_line_length': 127,
    'artifact_retention_days': 90
}

# Cross-validation Configuration
CV_CONFIG = {
    'n_splits': 5,
    'shuffle': True,
    'random_state': 42
}

# Performance Thresholds (for validation in CI/CD)
PERFORMANCE_THRESHOLDS = {
    'california_housing': {
        'test_r2': 0.70,  # Minimum acceptable R² score
        'test_rmse': 1.0   # Maximum acceptable RMSE
    },
    'credit_fraud': {
        'test_f1': 0.50,      # Minimum F1 score
        'test_roc_auc': 0.90  # Minimum ROC-AUC
    },
    'customer_churn': {
        'test_accuracy': 0.70,  # Minimum accuracy
        'test_f1': 0.50         # Minimum F1 score
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S'
}

# Directory Structure
DIRS = {
    'data': 'data',
    'models': 'models',
    'metrics': 'metrics',
    'reports': 'reports',
    'plots': 'plots',
    'mlruns': 'mlruns',
    'optuna_results': 'optuna_results',
    'tests': 'tests'
}

# Feature Engineering
FEATURE_ENGINEERING = {
    'california_housing': {
        'derived_features': [
            'rooms_per_household',
            'bedrooms_per_room',
            'population_per_household'
        ]
    }
}

# Alert Configuration (for drift detection)
ALERT_CONFIG = {
    'email_enabled': False,
    'slack_enabled': False,
    'drift_alert_threshold': 0.3,
    'quality_alert_threshold': 0.90
}

# Experiment Tracking
EXPERIMENT_CONFIG = {
    'auto_log': True,
    'log_models': True,
    'log_artifacts': True,
    'log_params': True,
    'log_metrics': True,
    'nested_runs': False
}

# Production Configuration
PRODUCTION_CONFIG = {
    'model_stage': 'Production',
    'model_version': 'latest',
    'serving_port': 5001,
    'api_timeout': 30,
    'max_batch_size': 1000
}
