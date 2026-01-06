"""
Compare Results Across Different Datasets and Models
Generates comprehensive comparison reports and visualizations
"""
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_metrics(metrics_dir='metrics'):
    """Load all metrics from JSON files"""
    metrics_data = []
    
    if not os.path.exists(metrics_dir):
        print(f"Warning: Metrics directory '{metrics_dir}' not found")
        return pd.DataFrame()
    
    for filename in os.listdir(metrics_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(metrics_dir, filename)
            
            with open(filepath, 'r') as f:
                metrics = json.load(f)
            
            # Parse filename to extract dataset and model info
            # Expected format: metrics_<dataset>_<model>.json
            parts = filename.replace('metrics_', '').replace('.json', '').split('_')
            
            if len(parts) >= 2:
                # Handle multi-word dataset names
                if 'california' in parts:
                    dataset = 'california_housing'
                    model = '_'.join(parts[2:]) if len(parts) > 2 else 'unknown'
                elif 'credit' in parts:
                    dataset = 'credit_fraud'
                    model = '_'.join(parts[2:]) if len(parts) > 2 else 'unknown'
                elif 'customer' in parts:
                    dataset = 'customer_churn'
                    model = '_'.join(parts[2:]) if len(parts) > 2 else 'unknown'
                else:
                    dataset = parts[0]
                    model = '_'.join(parts[1:])
                
                metrics['dataset'] = dataset
                metrics['model'] = model
                metrics['filename'] = filename
                metrics_data.append(metrics)
    
    if not metrics_data:
        print("Warning: No metrics files found")
        return pd.DataFrame()
    
    return pd.DataFrame(metrics_data)


def create_comparison_plots(df, output_dir='reports'):
    """Create visualization plots for model comparison"""
    os.makedirs(output_dir, exist_ok=True)
    
    plots_created = []
    
    # 1. Performance by Dataset
    if not df.empty:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison Across Datasets', fontsize=16, fontweight='bold')
        
        datasets = df['dataset'].unique()
        
        # Plot 1: Overall Performance
        ax1 = axes[0, 0]
        for dataset in datasets:
            dataset_df = df[df['dataset'] == dataset]
            
            # Select appropriate metric
            if 'test_r2' in dataset_df.columns:
                metric = 'test_r2'
                ylabel = 'R² Score'
            elif 'test_f1' in dataset_df.columns:
                metric = 'test_f1'
                ylabel = 'F1 Score'
            elif 'test_accuracy' in dataset_df.columns:
                metric = 'test_accuracy'
                ylabel = 'Accuracy'
            else:
                continue
            
            ax1.bar(dataset, dataset_df[metric].mean(), alpha=0.7, label=dataset)
        
        ax1.set_title('Average Performance by Dataset')
        ax1.set_ylabel(ylabel)
        ax1.set_xlabel('Dataset')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Training vs Test Performance
        ax2 = axes[0, 1]
        train_metrics = []
        test_metrics = []
        labels = []
        
        for idx, row in df.iterrows():
            dataset_model = f"{row['dataset']}\n{row['model']}"
            
            if 'train_r2' in row and 'test_r2' in row:
                train_metrics.append(row['train_r2'])
                test_metrics.append(row['test_r2'])
                labels.append(dataset_model)
            elif 'train_accuracy' in row and 'test_accuracy' in row:
                train_metrics.append(row['train_accuracy'])
                test_metrics.append(row['test_accuracy'])
                labels.append(dataset_model)
        
        if train_metrics:
            x = np.arange(len(labels))
            width = 0.35
            ax2.bar(x - width/2, train_metrics, width, label='Train', alpha=0.8)
            ax2.bar(x + width/2, test_metrics, width, label='Test', alpha=0.8)
            ax2.set_title('Train vs Test Performance')
            ax2.set_ylabel('Score')
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Regression Metrics Comparison (if applicable)
        ax3 = axes[1, 0]
        regression_df = df[df['dataset'] == 'california_housing']
        if not regression_df.empty and 'test_rmse' in regression_df.columns:
            metrics_to_plot = ['test_rmse', 'test_mae']
            data_to_plot = []
            
            for metric in metrics_to_plot:
                if metric in regression_df.columns:
                    data_to_plot.append(regression_df[metric].values[0])
            
            if data_to_plot:
                ax3.bar(metrics_to_plot, data_to_plot, alpha=0.7, color=['#ff7f0e', '#2ca02c'])
                ax3.set_title('Regression Metrics (California Housing)')
                ax3.set_ylabel('Error Value')
                ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No Regression Data', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Regression Metrics')
        
        # Plot 4: Classification Metrics Comparison
        ax4 = axes[1, 1]
        classification_df = df[df['dataset'].isin(['credit_fraud', 'customer_churn'])]
        
        if not classification_df.empty:
            metrics = ['test_precision', 'test_recall', 'test_f1']
            
            for dataset in classification_df['dataset'].unique():
                dataset_data = classification_df[classification_df['dataset'] == dataset]
                values = []
                available_metrics = []
                
                for metric in metrics:
                    if metric in dataset_data.columns:
                        values.append(dataset_data[metric].values[0])
                        available_metrics.append(metric.replace('test_', '').title())
                
                if values:
                    x = np.arange(len(available_metrics))
                    ax4.plot(x, values, marker='o', label=dataset, linewidth=2, markersize=8)
            
            ax4.set_title('Classification Metrics Comparison')
            ax4.set_ylabel('Score')
            ax4.set_xticks(range(len(available_metrics)))
            ax4.set_xticklabels(available_metrics)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No Classification Data', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Classification Metrics')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'performance_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots_created.append(plot_path)
        print(f"✓ Performance comparison plot saved: {plot_path}")
    
    return plots_created


def generate_comparison_report(df, output_dir='reports'):
    """Generate comprehensive HTML comparison report"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>MLOps Model Comparison Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 40px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #34495e;
                margin-top: 30px;
                border-left: 4px solid #3498db;
                padding-left: 10px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                box-shadow: 0 0 5px rgba(0,0,0,0.1);
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border: 1px solid #ddd;
            }}
            th {{
                background-color: #3498db;
                color: white;
                font-weight: bold;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            .metric-value {{
                font-weight: bold;
                color: #27ae60;
            }}
            .dataset-housing {{
                background-color: #e8f5e9;
            }}
            .dataset-credit {{
                background-color: #fff3e0;
            }}
            .dataset-churn {{
                background-color: #e3f2fd;
            }}
            .best-score {{
                background-color: #ffd700;
                font-weight: bold;
            }}
            .timestamp {{
                color: #7f8c8d;
                font-size: 0.9em;
                text-align: right;
                margin-top: 20px;
            }}
            .summary-box {{
                background-color: #ecf0f1;
                padding: 20px;
                border-radius: 5px;
                margin: 20px 0;
            }}
            .plot-container {{
                text-align: center;
                margin: 30px 0;
            }}
            .plot-container img {{
                max-width: 100%;
                border: 1px solid #ddd;
                border-radius: 5px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 MLOps Model Comparison Report</h1>
            <p class="timestamp">Generated: {timestamp}</p>
            
            <div class="summary-box">
                <h3>📊 Executive Summary</h3>
                <p>This report compares the performance of machine learning models across three different datasets:</p>
                <ul>
                    <li><strong>California Housing</strong>: Regression task predicting house prices</li>
                    <li><strong>Credit Fraud Detection</strong>: Classification task identifying fraudulent transactions</li>
                    <li><strong>Customer Churn</strong>: Classification task predicting customer churn</li>
                </ul>
                <p>Total models evaluated: <strong>{len(df)}</strong></p>
            </div>
    """
    
    # Add detailed metrics table
    if not df.empty:
        html_content += "<h2>📈 Detailed Metrics</h2>\n"
        html_content += "<table>\n<thead>\n<tr>\n"
        html_content += "<th>Dataset</th><th>Model</th>"
        
        # Collect all unique metric columns
        metric_columns = [col for col in df.columns if col not in ['dataset', 'model', 'filename']]
        for col in metric_columns:
            html_content += f"<th>{col.replace('_', ' ').title()}</th>"
        
        html_content += "</tr>\n</thead>\n<tbody>\n"
        
        for idx, row in df.iterrows():
            dataset_class = f"dataset-{row['dataset'].split('_')[0]}"
            html_content += f"<tr class='{dataset_class}'>\n"
            html_content += f"<td><strong>{row['dataset'].replace('_', ' ').title()}</strong></td>\n"
            html_content += f"<td>{row['model'].replace('_', ' ').title()}</td>\n"
            
            for col in metric_columns:
                value = row.get(col, 'N/A')
                if isinstance(value, (int, float)):
                    html_content += f"<td class='metric-value'>{value:.4f}</td>\n"
                else:
                    html_content += f"<td>{value}</td>\n"
            
            html_content += "</tr>\n"
        
        html_content += "</tbody>\n</table>\n"
    
    # Add key findings
    html_content += "<h2>🔍 Key Findings</h2>\n<div class='summary-box'>\n"
    
    if not df.empty:
        # Housing findings
        housing_df = df[df['dataset'] == 'california_housing']
        if not housing_df.empty and 'test_r2' in housing_df.columns:
            best_r2 = housing_df['test_r2'].max()
            html_content += f"<p><strong>California Housing:</strong> Best R² Score = <span class='metric-value'>{best_r2:.4f}</span></p>\n"
        
        # Credit findings
        credit_df = df[df['dataset'] == 'credit_fraud']
        if not credit_df.empty and 'test_f1' in credit_df.columns:
            best_f1 = credit_df['test_f1'].max()
            html_content += f"<p><strong>Credit Fraud:</strong> Best F1 Score = <span class='metric-value'>{best_f1:.4f}</span></p>\n"
        
        # Churn findings
        churn_df = df[df['dataset'] == 'customer_churn']
        if not churn_df.empty and 'test_accuracy' in churn_df.columns:
            best_acc = churn_df['test_accuracy'].max()
            html_content += f"<p><strong>Customer Churn:</strong> Best Accuracy = <span class='metric-value'>{best_acc:.4f}</span></p>\n"
    
    html_content += "</div>\n"
    
    # Add visualizations
    html_content += "<h2>📊 Visualizations</h2>\n"
    html_content += "<div class='plot-container'>\n"
    html_content += "<img src='performance_comparison.png' alt='Performance Comparison'>\n"
    html_content += "</div>\n"
    
    # Add recommendations
    html_content += """
    <h2>💡 Recommendations</h2>
    <div class='summary-box'>
        <ul>
            <li><strong>Model Selection:</strong> Choose models based on the specific metrics most important for your use case</li>
            <li><strong>Hyperparameter Tuning:</strong> Consider using Optuna for automated hyperparameter optimization</li>
            <li><strong>Monitoring:</strong> Set up Evidently AI for continuous drift detection</li>
            <li><strong>Retraining:</strong> Schedule periodic model retraining with fresh data</li>
            <li><strong>A/B Testing:</strong> Deploy top models in production and compare real-world performance</li>
        </ul>
    </div>
    """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    report_path = os.path.join(output_dir, 'comparison_report.html')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ Comparison report saved: {report_path}")
    
    return report_path


def main():
    print("=" * 60)
    print("MLOps Model Comparison Tool")
    print("=" * 60)
    
    # Load metrics
    print("\nLoading metrics from JSON files...")
    df = load_metrics()
    
    if df.empty:
        print("\n⚠️  No metrics found. Please run training first:")
        print("  python train.py --dataset california_housing")
        print("  python train.py --dataset credit_fraud")
        print("  python train.py --dataset customer_churn")
        return
    
    print(f"✓ Loaded {len(df)} model results\n")
    
    # Display summary
    print("Datasets found:")
    for dataset in df['dataset'].unique():
        count = len(df[df['dataset'] == dataset])
        print(f"  - {dataset}: {count} model(s)")
    
    # Create visualizations
    print("\nGenerating comparison plots...")
    plots = create_comparison_plots(df)
    
    # Generate HTML report
    print("\nGenerating HTML comparison report...")
    report_path = generate_comparison_report(df)
    
    print("\n" + "=" * 60)
    print("✓ Comparison completed successfully!")
    print("=" * 60)
    print(f"\nGenerated files:")
    print(f"  - {report_path}")
    for plot in plots:
        print(f"  - {plot}")
    print(f"\nOpen {report_path} in your browser to view the full report.")


if __name__ == "__main__":
    main()
