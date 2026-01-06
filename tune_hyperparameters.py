"""
Hyperparameter Tuning with Optuna and MLflow Integration
Advanced feature for automatic model optimization
"""
import argparse
import os
import json
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import optuna
from optuna.integration.mlflow import MLflowCallback
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_squared_error, r2_score, f1_score, roc_auc_score
)
from datetime import datetime

# Suppress Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

mlflow.set_tracking_uri("file:./mlruns")


def load_dataset(dataset_name):
    """Load and preprocess dataset"""
    file_paths = {
        'california_housing': 'data/housing_data.csv',
        'credit_fraud': 'data/credit_data.csv',
        'customer_churn': 'data/churn_data.csv'
    }
    
    df = pd.read_csv(file_paths[dataset_name])
    
    if dataset_name == 'california_housing':
        X = df.drop('MedHouseVal', axis=1)
        y = df['MedHouseVal']
        task_type = 'regression'
        
    elif dataset_name == 'credit_fraud':
        X = df.drop('Class', axis=1)
        y = df['Class']
        task_type = 'classification'
        
    elif dataset_name == 'customer_churn':
        df = df.drop('customerID', axis=1)
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col != 'Churn':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
        
        df['Churn'] = (df['Churn'] == 'Yes').astype(int)
        X = df.drop('Churn', axis=1)
        y = df['Churn']
        task_type = 'classification'
    
    return X, y, task_type


def objective(trial, X_train, X_test, y_train, y_test, task_type):
    """Objective function for Optuna optimization"""
    
    # Suggest hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Create model based on task type
    if task_type == 'regression':
        model = RandomForestRegressor(**params)
        # Use negative RMSE as optimization metric (Optuna maximizes)
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=3, 
            scoring='neg_root_mean_squared_error'
        )
        return cv_scores.mean()
    else:
        params['class_weight'] = 'balanced'
        model = RandomForestClassifier(**params)
        # Use F1 score for classification
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=3, 
            scoring='f1_weighted'
        )
        return cv_scores.mean()


def tune_hyperparameters(dataset_name, n_trials=50):
    """
    Tune hyperparameters using Optuna with MLflow tracking
    
    Args:
        dataset_name: Name of the dataset
        n_trials: Number of optimization trials
    """
    print(f"\n{'='*60}")
    print(f"Hyperparameter Tuning with Optuna")
    print(f"Dataset: {dataset_name}")
    print(f"Number of trials: {n_trials}")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading dataset...")
    X, y, task_type = load_dataset(dataset_name)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if task_type == 'classification' else None
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Set MLflow experiment
    experiment_name = f"{dataset_name}_optuna_tuning"
    mlflow.set_experiment(experiment_name)
    
    # Create Optuna study
    study_name = f"{dataset_name}_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Determine optimization direction
    direction = 'maximize' if task_type == 'classification' else 'maximize'
    
    print(f"Creating Optuna study: {study_name}")
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    
    # MLflow callback
    mlflow_callback = MLflowCallback(
        tracking_uri="file:./mlruns",
        metric_name="cv_score"
    )
    
    # Run optimization
    print(f"\nStarting optimization with {n_trials} trials...")
    print("This may take several minutes...\n")
    
    study.optimize(
        lambda trial: objective(trial, X_train_scaled, X_test_scaled, y_train, y_test, task_type),
        n_trials=n_trials,
        callbacks=[mlflow_callback],
        show_progress_bar=True
    )
    
    # Get best parameters
    best_params = study.best_params
    best_score = study.best_value
    
    print(f"\n{'='*60}")
    print("Optimization completed!")
    print(f"{'='*60}")
    print(f"\nBest cross-validation score: {best_score:.4f}")
    print(f"\nBest hyperparameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Train final model with best parameters
    print(f"\n{'='*60}")
    print("Training final model with best parameters...")
    print(f"{'='*60}\n")
    
    with mlflow.start_run(run_name=f"best_model_{study_name}"):
        # Log dataset info
        mlflow.log_param("dataset", dataset_name)
        mlflow.log_param("task_type", task_type)
        mlflow.log_param("optimization_method", "optuna_tpe")
        mlflow.log_param("n_trials", n_trials)
        
        # Log best parameters
        mlflow.log_params(best_params)
        
        # Create and train model
        if task_type == 'regression':
            model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
        else:
            model = RandomForestClassifier(
                **best_params, 
                class_weight='balanced',
                random_state=42, 
                n_jobs=-1
            )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        if task_type == 'regression':
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            mlflow.log_metric("train_rmse", train_rmse)
            mlflow.log_metric("test_rmse", test_rmse)
            mlflow.log_metric("train_r2", train_r2)
            mlflow.log_metric("test_r2", test_r2)
            
            print(f"Train RMSE: {train_rmse:.4f}")
            print(f"Test RMSE: {test_rmse:.4f}")
            print(f"Test R²: {test_r2:.4f}")
            
        else:
            if hasattr(model, 'predict_proba'):
                y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
                test_roc_auc = roc_auc_score(y_test, y_test_proba)
                mlflow.log_metric("test_roc_auc", test_roc_auc)
                print(f"Test ROC AUC: {test_roc_auc:.4f}")
            
            test_f1 = f1_score(y_test, y_test_pred, average='weighted')
            mlflow.log_metric("test_f1", test_f1)
            print(f"Test F1 Score: {test_f1:.4f}")
        
        # Log model
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=f"{dataset_name}_tuned_rf"
        )
        
        # Save optimization results
        os.makedirs("optuna_results", exist_ok=True)
        results_path = f"optuna_results/{dataset_name}_optimization.json"
        
        results = {
            "best_params": best_params,
            "best_score": best_score,
            "n_trials": n_trials,
            "study_name": study_name,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        mlflow.log_artifact(results_path)
        
        print(f"\n✓ Optimization results saved: {results_path}")
        print(f"✓ Model logged to MLflow")
    
    # Print optimization insights
    print(f"\n{'='*60}")
    print("Optimization Insights")
    print(f"{'='*60}")
    print(f"Total trials: {len(study.trials)}")
    print(f"Complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    
    # Feature importance from best parameters
    if hasattr(study, 'best_trial'):
        importance = optuna.importance.get_param_importances(study)
        print(f"\nHyperparameter Importances:")
        for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {param}: {imp:.4f}")
    
    return study, model


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning with Optuna')
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['california_housing', 'credit_fraud', 'customer_churn'],
        help='Dataset to use for tuning'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=50,
        help='Number of optimization trials (default: 50)'
    )
    
    args = parser.parse_args()
    
    # Run tuning
    study, model = tune_hyperparameters(args.dataset, args.n_trials)
    
    print(f"\n{'='*60}")
    print("✓ Hyperparameter tuning completed successfully!")
    print(f"{'='*60}")
    print("\nTo view results in MLflow UI, run:")
    print("  mlflow ui")
    print("Then open http://localhost:5000 in your browser")


if __name__ == "__main__":
    main()
