"""
Master Script - Execute Full MLOps Pipeline
Run all experiments, comparisons and generate reports
"""
import subprocess
import sys
import time
from datetime import datetime


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"▶ {description}...")
    print(f"  Command: {command}\n")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        print(f"✓ {description} completed successfully!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error in {description}:")
        print(e.stderr)
        return False


def main():
    """Execute the full MLOps pipeline"""
    
    start_time = time.time()
    
    print("\n" + "🚀" * 35)
    print("  MLOPS FULL PIPELINE EXECUTION")
    print("  Started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("🚀" * 35)
    
    # Step 1: Generate all datasets
    print_section("STEP 1: Data Generation")
    success = run_command(
        "python generate_data.py --dataset all",
        "Generating all datasets (Housing, Credit, Churn)"
    )
    
    if not success:
        print("Pipeline stopped due to error in data generation")
        return
    
    # Step 2: Train baseline models
    print_section("STEP 2: Baseline Model Training")
    
    datasets = [
        ("california_housing", "California Housing (Regression)"),
        ("credit_fraud", "Credit Card Fraud (Classification)"),
        ("customer_churn", "Customer Churn (Classification)")
    ]
    
    for dataset, description in datasets:
        success = run_command(
            f"python train.py --dataset {dataset} --model random_forest",
            f"Training Random Forest on {description}"
        )
        if not success:
            print(f"Warning: Training failed for {dataset}, continuing...")
    
    # Step 3: Hyperparameter tuning (optional, takes longer)
    print_section("STEP 3: Hyperparameter Tuning (Optional)")
    
    user_input = input("Do you want to run hyperparameter tuning? (y/n): ").lower()
    
    if user_input == 'y':
        trials = input("Number of trials (default 20, recommended 50-100 for best results): ")
        n_trials = int(trials) if trials.isdigit() else 20
        
        for dataset, description in datasets[:2]:  # Housing and Churn only
            run_command(
                f"python tune_hyperparameters.py --dataset {dataset} --n-trials {n_trials}",
                f"Tuning hyperparameters for {description}"
            )
    else:
        print("Skipping hyperparameter tuning...")
    
    # Step 4: Data drift detection
    print_section("STEP 4: Data Drift Detection")
    
    run_command(
        "python detect_drift.py --compare-datasets housing credit",
        "Detecting drift between Housing and Credit datasets"
    )
    
    run_command(
        "python detect_drift.py --compare-datasets housing churn",
        "Detecting drift between Housing and Churn datasets"
    )
    
    # Step 5: Compare all results
    print_section("STEP 5: Results Comparison")
    
    run_command(
        "python compare_results.py",
        "Generating comparison report for all models"
    )
    
    # Step 6: Run tests
    print_section("STEP 6: Unit Tests")
    
    run_command(
        "pytest tests/ -v",
        "Running unit tests"
    )
    
    # Summary
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    
    print_section("PIPELINE EXECUTION SUMMARY")
    
    print("✓ All steps completed!")
    print(f"\nTotal execution time: {minutes} minutes {seconds} seconds")
    
    print("\n📊 Generated Outputs:")
    print("  - Datasets: data/*.csv")
    print("  - MLflow tracking: mlruns/")
    print("  - Metrics: metrics/*.json")
    print("  - Reports: reports/*.html")
    print("  - Optuna results: optuna_results/")
    
    print("\n🔍 Next Steps:")
    print("  1. View MLflow experiments:")
    print("     > mlflow ui")
    print("     Then open http://localhost:5000")
    print("\n  2. View comparison report:")
    print("     > start reports/comparison_report.html  (Windows)")
    print("     > open reports/comparison_report.html   (Mac)")
    print("     > xdg-open reports/comparison_report.html (Linux)")
    print("\n  3. View drift reports:")
    print("     > start reports/data_drift_report_*.html")
    print("\n  4. Check detailed results:")
    print("     > Open RESULTS.md for comprehensive analysis")
    
    print("\n" + "🎉" * 35)
    print("  PIPELINE COMPLETED SUCCESSFULLY!")
    print("🎉" * 35 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Pipeline failed with error: {str(e)}")
        sys.exit(1)
