"""
Training Script with MLflow Integration
Supports multiple datasets and models with comprehensive tracking
"""
import argparse
import os
import json
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set MLflow tracking URI
mlflow.set_tracking_uri("file:./mlruns")

def load_and_preprocess_housing(file_path):
    """Load and preprocess California Housing dataset"""
    print("Loading California Housing dataset...")
    df = pd.read_csv(file_path)
    
    # Separate features and target
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    return X, y, 'regression'

def load_and_preprocess_credit(file_path):
    """Load and preprocess Credit Fraud dataset"""
    print("Loading Credit Fraud dataset...")
    df = pd.read_csv(file_path)
    
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    print(f"Dataset shape: {X.shape}")
    print(f"Fraud rate: {y.sum() / len(y) * 100:.2f}%")
    
    return X, y, 'classification_imbalanced'

def load_and_preprocess_churn(file_path):
    """Load and preprocess Customer Churn dataset"""
    print("Loading Customer Churn dataset...")
    df = pd.read_csv(file_path)
    
    # Drop customerID
    df = df.drop('customerID', axis=1)
    
    # Handle categorical variables
    label_encoders = {}
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        if col != 'Churn':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
    
    # Encode target
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    
    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    print(f"Dataset shape: {X.shape}")
    print(f"Churn rate: {y.sum() / len(y) * 100:.2f}%")
    
    return X, y, 'classification'

def get_model(model_type, task_type, class_weight=None):
    """Get model based on type and task"""
    if task_type == 'regression':
        if model_type == 'random_forest':
            return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        elif model_type == 'gradient_boosting':
            return GradientBoostingRegressor(n_estimators=100, random_state=42)
    else:  # classification
        if model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1,
                class_weight=class_weight
            )
        elif model_type == 'gradient_boosting':
            return GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif model_type == 'logistic_regression':
            return LogisticRegression(
                random_state=42, max_iter=1000,
                class_weight=class_weight
            )
    
    raise ValueError(f"Unknown model type: {model_type}")

def plot_feature_importance(model, feature_names, save_path):
    """Plot feature importance"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]  # Top 15 features
        
        plt.figure(figsize=(10, 6))
        plt.title('Top 15 Feature Importances')
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"✓ Feature importance plot saved: {save_path}")

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✓ Confusion matrix saved: {save_path}")

def plot_predictions(y_true, y_pred, save_path):
    """Plot predictions vs actual for regression"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predictions vs Actual Values')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✓ Predictions plot saved: {save_path}")

def train_model(dataset_name, model_type, tune=False):
    """Train model with MLflow tracking"""
    
    # Determine file path
    file_paths = {
        'california_housing': 'data/housing_data.csv',
        'credit_fraud': 'data/credit_data.csv',
        'customer_churn': 'data/churn_data.csv'
    }
    
    if dataset_name not in file_paths:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    file_path = file_paths[dataset_name]
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}. Run generate_data.py first.")
    
    # Load and preprocess data
    if dataset_name == 'california_housing':
        X, y, task_type = load_and_preprocess_housing(file_path)
    elif dataset_name == 'credit_fraud':
        X, y, task_type = load_and_preprocess_credit(file_path)
    elif dataset_name == 'customer_churn':
        X, y, task_type = load_and_preprocess_churn(file_path)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if task_type != 'regression' else None
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Set experiment name
    experiment_name = f"{dataset_name}_{model_type}"
    mlflow.set_experiment(experiment_name)
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Log parameters
        mlflow.log_param("dataset", dataset_name)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("task_type", task_type)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])
        
        # Determine class weight for imbalanced datasets
        class_weight = 'balanced' if task_type == 'classification_imbalanced' else None
        if class_weight:
            mlflow.log_param("class_weight", "balanced")
        
        # Get and train model
        print(f"\nTraining {model_type} model...")
        model = get_model(model_type, task_type, class_weight)
        
        # Log model parameters
        mlflow.log_params(model.get_params())
        
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        mlflow.log_metric("cv_mean_score", cv_scores.mean())
        mlflow.log_metric("cv_std_score", cv_scores.std())
        print(f"Cross-validation score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Predictions
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Evaluate
        print("\nEvaluating model...")
        metrics = {}
        
        if task_type == 'regression':
            # Regression metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            metrics = {
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                "train_mae": train_mae,
                "test_mae": test_mae,
                "train_r2": train_r2,
                "test_r2": test_r2
            }
            
            print(f"Train RMSE: {train_rmse:.4f}")
            print(f"Test RMSE: {test_rmse:.4f}")
            print(f"Test MAE: {test_mae:.4f}")
            print(f"Test R²: {test_r2:.4f}")
            
            # Plot predictions
            plot_predictions(y_test, y_test_pred, 'predictions_plot.png')
            mlflow.log_artifact('predictions_plot.png')
            os.remove('predictions_plot.png')
            
        else:
            # Classification metrics
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            
            # Handle binary vs multiclass
            average = 'binary' if len(np.unique(y)) == 2 else 'weighted'
            
            test_precision = precision_score(y_test, y_test_pred, average=average, zero_division=0)
            test_recall = recall_score(y_test, y_test_pred, average=average, zero_division=0)
            test_f1 = f1_score(y_test, y_test_pred, average=average, zero_division=0)
            
            metrics = {
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_f1": test_f1
            }
            
            # ROC AUC for binary classification
            if len(np.unique(y)) == 2 and hasattr(model, 'predict_proba'):
                y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
                test_roc_auc = roc_auc_score(y_test, y_test_proba)
                metrics["test_roc_auc"] = test_roc_auc
                print(f"Test ROC AUC: {test_roc_auc:.4f}")
            
            print(f"Train Accuracy: {train_acc:.4f}")
            print(f"Test Accuracy: {test_acc:.4f}")
            print(f"Test Precision: {test_precision:.4f}")
            print(f"Test Recall: {test_recall:.4f}")
            print(f"Test F1: {test_f1:.4f}")
            
            # Confusion matrix
            plot_confusion_matrix(y_test, y_test_pred, 'confusion_matrix.png')
            mlflow.log_artifact('confusion_matrix.png')
            os.remove('confusion_matrix.png')
            
            # Classification report
            report = classification_report(y_test, y_test_pred)
            print("\nClassification Report:")
            print(report)
        
        # Log all metrics
        mlflow.log_metrics(metrics)
        
        # Feature importance
        plot_feature_importance(model, X.columns.tolist(), 'feature_importance.png')
        mlflow.log_artifact('feature_importance.png')
        os.remove('feature_importance.png')
        
        # Save model
        os.makedirs("models", exist_ok=True)
        model_path = f"models/model_{dataset_name}_{model_type}.pkl"
        
        # Log model with MLflow
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name=f"{dataset_name}_{model_type}"
        )
        
        # Save metrics to JSON for DVC
        os.makedirs("metrics", exist_ok=True)
        metrics_path = f"metrics/metrics_{dataset_name}_{model_type}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        mlflow.log_artifact(metrics_path)
        
        print(f"\n✓ Model logged to MLflow")
        print(f"✓ Metrics saved to {metrics_path}")
        print(f"✓ Run ID: {mlflow.active_run().info.run_id}")
        
        # Log tags
        mlflow.set_tag("dataset", dataset_name)
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("task_type", task_type)
        
        return metrics

def main():
    parser = argparse.ArgumentParser(description='Train ML models with MLflow tracking')
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['california_housing', 'credit_fraud', 'customer_churn'],
        help='Dataset to use for training'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='random_forest',
        choices=['random_forest', 'gradient_boosting', 'logistic_regression'],
        help='Model type to train'
    )
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Enable hyperparameter tuning (requires Optuna)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MLOps Training Pipeline with MLflow")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Hyperparameter tuning: {args.tune}")
    print("=" * 60)
    
    # Train model
    metrics = train_model(args.dataset, args.model, args.tune)
    
    print("\n" + "=" * 60)
    print("✓ Training completed successfully!")
    print("=" * 60)
    print("\nTo view results in MLflow UI, run:")
    print("  mlflow ui")
    print("Then open http://localhost:5000 in your browser")

if __name__ == "__main__":
    main()
