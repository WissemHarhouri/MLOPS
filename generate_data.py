"""
Data Generation Script for MLOps Pipeline
Supports multiple datasets: California Housing, Credit Fraud, Customer Churn
"""
import argparse
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
import os

def generate_california_housing():
    """Generate California Housing dataset for regression (synthetic but realistic)"""
    print("Generating California Housing dataset...")
    np.random.seed(42)
    
    # Generate realistic housing data using make_regression
    n_samples = 20640
    X, y_raw = make_regression(
        n_samples=n_samples,
        n_features=8,
        n_informative=8,
        noise=15.0,
        random_state=42
    )
    
    # Create DataFrame with meaningful column names
    feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                     'Population', 'AveOccup', 'Latitude', 'Longitude']
    df = pd.DataFrame(X, columns=feature_names)
    
    # Scale features to realistic ranges
    df['MedInc'] = np.abs(df['MedInc']) * 1.5 + 2  # Median income: 2-10
    df['HouseAge'] = np.abs(df['HouseAge']) * 8 + 10  # House age: 10-60 years
    df['AveRooms'] = np.abs(df['AveRooms']) * 1.5 + 4  # Average rooms: 4-10
    df['AveBedrms'] = df['AveRooms'] / 4  # Bedrooms proportional to rooms
    df['Population'] = np.abs(df['Population']) * 400 + 500  # Population: 500-5000
    df['AveOccup'] = df['Population'] / (df['AveRooms'] * 100)  # Average occupancy
    df['Latitude'] = df['Latitude'] * 2 + 35  # California latitude range
    df['Longitude'] = df['Longitude'] * 5 - 120  # California longitude range
    
    # Scale target to realistic house values (in $100k)
    y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min()) * 4 + 0.5  # 0.5-5.0
    df['MedHouseVal'] = y
    
    # Feature engineering
    df['rooms_per_household'] = df['AveRooms'] / df['AveOccup']
    df['bedrooms_per_room'] = df['AveBedrms'] / df['AveRooms']
    df['population_per_household'] = df['Population'] / df['AveOccup']
    
    # Save dataset
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/housing_data.csv", index=False)
    print(f"[OK] California Housing dataset saved: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"  Target variable: MedHouseVal (median house value in $100k)")
    print(f"  Features: {df.columns.tolist()}")
    return df

def generate_credit_fraud():
    """Generate synthetic Credit Card Fraud dataset"""
    print("Generating Credit Card Fraud dataset...")
    np.random.seed(42)
    
    # Generate 10,000 transactions
    n_samples = 10000
    n_fraud = int(n_samples * 0.002)  # 0.2% fraud rate
    
    # Create synthetic features (similar to PCA components)
    data = {
        'Time': np.random.randint(0, 172800, n_samples),  # 48 hours in seconds
        'Amount': np.random.exponential(88, n_samples)  # Exponential distribution
    }
    
    # Add V1-V28 features (PCA components - synthetic)
    for i in range(1, 29):
        data[f'V{i}'] = np.random.randn(n_samples)
    
    # Create target variable
    fraud_indices = np.random.choice(n_samples, n_fraud, replace=False)
    data['Class'] = 0
    
    df = pd.DataFrame(data)
    df.loc[fraud_indices, 'Class'] = 1
    
    # Make fraud transactions look different
    for i in range(1, 29):
        df.loc[fraud_indices, f'V{i}'] += np.random.randn(n_fraud) * 2
    df.loc[fraud_indices, 'Amount'] = np.random.exponential(200, n_fraud)
    
    # Save dataset
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/credit_data.csv", index=False)
    print(f"[OK] Credit Fraud dataset saved: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"  Target variable: Class (0=legitimate, 1=fraud)")
    print(f"  Fraud rate: {(df['Class'].sum() / len(df)) * 100:.2f}%")
    return df

def generate_customer_churn():
    """Generate synthetic Customer Churn dataset"""
    print("Generating Customer Churn dataset...")
    np.random.seed(42)
    
    n_samples = 7043
    
    # Generate features
    df = pd.DataFrame({
        'customerID': [f'ID{i:04d}' for i in range(n_samples)],
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.84, 0.16]),
        'Partner': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples),
        'tenure': np.random.randint(0, 73, n_samples),
        'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
        'PaymentMethod': np.random.choice([
            'Electronic check', 'Mailed check', 
            'Bank transfer (automatic)', 'Credit card (automatic)'
        ], n_samples),
    })
    
    # Generate continuous features
    df['MonthlyCharges'] = np.random.uniform(18.0, 118.0, n_samples)
    df['TotalCharges'] = df['tenure'] * df['MonthlyCharges'] + np.random.randn(n_samples) * 100
    df['TotalCharges'] = df['TotalCharges'].clip(lower=0)
    
    # Generate target based on features
    churn_prob = 0.27  # Base churn rate
    churn_score = np.random.rand(n_samples)
    
    # Increase churn probability for certain conditions
    churn_score[df['Contract'] == 'Month-to-month'] += 0.2
    churn_score[df['tenure'] < 12] += 0.15
    churn_score[df['SeniorCitizen'] == 1] += 0.1
    
    df['Churn'] = (churn_score > (1 - churn_prob)).astype(int)
    df['Churn'] = df['Churn'].map({0: 'No', 1: 'Yes'})
    
    # Save dataset
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/churn_data.csv", index=False)
    print(f"[OK] Customer Churn dataset saved: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"  Target variable: Churn (Yes/No)")
    print(f"  Churn rate: {(df['Churn'] == 'Yes').sum() / len(df) * 100:.2f}%")
    return df

def main():
    parser = argparse.ArgumentParser(description='Generate datasets for MLOps pipeline')
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['california_housing', 'credit_fraud', 'customer_churn', 'all'],
        default='all',
        help='Dataset to generate'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MLOps Data Generation Pipeline")
    print("=" * 60)
    
    if args.dataset == 'all':
        generate_california_housing()
        print()
        generate_credit_fraud()
        print()
        generate_customer_churn()
    elif args.dataset == 'california_housing':
        generate_california_housing()
    elif args.dataset == 'credit_fraud':
        generate_credit_fraud()
    elif args.dataset == 'customer_churn':
        generate_customer_churn()
    
    print("\n" + "=" * 60)
    print("[OK] Data generation completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()