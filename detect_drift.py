"""
Data Drift Detection with Evidently AI
Advanced feature for monitoring data quality and model performance
"""
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset, 
    DataQualityPreset,
    TargetDriftPreset
)
from evidently.metrics import *
from datetime import datetime


def load_dataset(dataset_path):
    """Load and preprocess dataset"""
    df = pd.read_csv(dataset_path)
    
    # Handle categorical variables for churn dataset
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    return df


def detect_drift(reference_data_path, current_data_path, output_dir='reports'):
    """
    Detect data drift between reference and current datasets
    
    Args:
        reference_data_path: Path to reference (training) dataset
        current_data_path: Path to current (new) dataset
        output_dir: Directory to save reports
    """
    print(f"\n{'='*60}")
    print("Data Drift Detection with Evidently AI")
    print(f"{'='*60}\n")
    
    # Load datasets
    print(f"Loading reference data: {reference_data_path}")
    reference_data = load_dataset(reference_data_path)
    print(f"  Shape: {reference_data.shape}")
    
    print(f"\nLoading current data: {current_data_path}")
    current_data = load_dataset(current_data_path)
    print(f"  Shape: {current_data.shape}")
    
    # Ensure both datasets have same columns
    common_cols = list(set(reference_data.columns) & set(current_data.columns))
    reference_data = reference_data[common_cols]
    current_data = current_data[common_cols]
    
    print(f"\nCommon columns: {len(common_cols)}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for reports
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. Data Drift Report
    print(f"\n{'='*60}")
    print("Generating Data Drift Report...")
    print(f"{'='*60}\n")
    
    drift_report = Report(metrics=[
        DataDriftPreset(),
    ])
    
    drift_report.run(
        reference_data=reference_data,
        current_data=current_data
    )
    
    drift_report_path = os.path.join(output_dir, f'data_drift_report_{timestamp}.html')
    drift_report.save_html(drift_report_path)
    print(f"✓ Data Drift Report saved: {drift_report_path}")
    
    # 2. Data Quality Report
    print(f"\n{'='*60}")
    print("Generating Data Quality Report...")
    print(f"{'='*60}\n")
    
    quality_report = Report(metrics=[
        DataQualityPreset(),
    ])
    
    quality_report.run(
        reference_data=reference_data,
        current_data=current_data
    )
    
    quality_report_path = os.path.join(output_dir, f'data_quality_report_{timestamp}.html')
    quality_report.save_html(quality_report_path)
    print(f"✓ Data Quality Report saved: {quality_report_path}")
    
    # 3. Detailed Drift Analysis
    print(f"\n{'='*60}")
    print("Generating Detailed Drift Analysis...")
    print(f"{'='*60}\n")
    
    detailed_report = Report(metrics=[
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
        DatasetCorrelationsMetric(),
    ])
    
    detailed_report.run(
        reference_data=reference_data,
        current_data=current_data
    )
    
    detailed_report_path = os.path.join(output_dir, f'detailed_drift_report_{timestamp}.html')
    detailed_report.save_html(detailed_report_path)
    print(f"✓ Detailed Drift Report saved: {detailed_report_path}")
    
    # Extract and display key metrics
    print(f"\n{'='*60}")
    print("Drift Detection Summary")
    print(f"{'='*60}\n")
    
    # Get drift results as JSON
    drift_results = drift_report.as_dict()
    
    # Extract drift metrics
    if 'metrics' in drift_results:
        for metric in drift_results['metrics']:
            if metric.get('metric') == 'DatasetDriftMetric':
                result = metric.get('result', {})
                drift_share = result.get('drift_share', 0)
                number_of_drifted_columns = result.get('number_of_drifted_columns', 0)
                dataset_drift = result.get('dataset_drift', False)
                
                print(f"Dataset Drift Detected: {'YES ⚠️' if dataset_drift else 'NO ✓'}")
                print(f"Drift Share: {drift_share:.2%}")
                print(f"Number of Drifted Columns: {number_of_drifted_columns}")
                
                if 'drift_by_columns' in result:
                    drifted_cols = [
                        col for col, info in result['drift_by_columns'].items() 
                        if info.get('drift_detected', False)
                    ]
                    if drifted_cols:
                        print(f"\nDrifted Columns:")
                        for col in drifted_cols[:10]:  # Show first 10
                            print(f"  - {col}")
                        if len(drifted_cols) > 10:
                            print(f"  ... and {len(drifted_cols) - 10} more")
    
    # Recommendations
    print(f"\n{'='*60}")
    print("Recommendations")
    print(f"{'='*60}\n")
    
    if dataset_drift:
        print("⚠️  DRIFT DETECTED - Recommended Actions:")
        print("  1. Review the drifted features in the HTML report")
        print("  2. Consider retraining the model with recent data")
        print("  3. Investigate root causes of distribution changes")
        print("  4. Update data preprocessing pipeline if needed")
    else:
        print("✓ NO SIGNIFICANT DRIFT - Model remains valid")
        print("  - Continue monitoring data quality")
        print("  - Regular periodic checks recommended")
    
    print(f"\n{'='*60}")
    print("✓ Drift detection completed successfully!")
    print(f"{'='*60}")
    print(f"\nGenerated Reports:")
    print(f"  1. {drift_report_path}")
    print(f"  2. {quality_report_path}")
    print(f"  3. {detailed_report_path}")
    print(f"\nOpen these HTML files in your browser to view detailed analysis.")
    
    return drift_report, quality_report, detailed_report


def compare_datasets(dataset1_name, dataset2_name):
    """
    Compare two different datasets to understand their characteristics
    """
    dataset_paths = {
        'housing': 'data/housing_data.csv',
        'credit': 'data/credit_data.csv',
        'churn': 'data/churn_data.csv'
    }
    
    if dataset1_name not in dataset_paths or dataset2_name not in dataset_paths:
        raise ValueError(f"Unknown dataset. Choose from: {list(dataset_paths.keys())}")
    
    print(f"\n{'='*60}")
    print(f"Comparing Datasets: {dataset1_name} vs {dataset2_name}")
    print(f"{'='*60}\n")
    
    return detect_drift(
        dataset_paths[dataset1_name],
        dataset_paths[dataset2_name],
        output_dir=f'reports/comparison_{dataset1_name}_vs_{dataset2_name}'
    )


def main():
    parser = argparse.ArgumentParser(description='Data drift detection with Evidently AI')
    parser.add_argument(
        '--reference-data',
        type=str,
        help='Path to reference (training) dataset'
    )
    parser.add_argument(
        '--current-data',
        type=str,
        help='Path to current (new) dataset'
    )
    parser.add_argument(
        '--compare-datasets',
        nargs=2,
        metavar=('DATASET1', 'DATASET2'),
        choices=['housing', 'credit', 'churn'],
        help='Compare two project datasets (choices: housing, credit, churn)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='reports',
        help='Directory to save reports (default: reports)'
    )
    
    args = parser.parse_args()
    
    if args.compare_datasets:
        # Compare project datasets
        compare_datasets(args.compare_datasets[0], args.compare_datasets[1])
    elif args.reference_data and args.current_data:
        # Compare custom datasets
        detect_drift(args.reference_data, args.current_data, args.output_dir)
    else:
        parser.print_help()
        print("\nError: Please provide either:")
        print("  --reference-data and --current-data for custom comparison")
        print("  OR --compare-datasets for comparing project datasets")


if __name__ == "__main__":
    main()
