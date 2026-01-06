# 🎨 Guide Visuel des Commandes - MLOps Pipeline

## 📋 Table des Matières
1. [Commandes de Base](#commandes-de-base)
2. [Workflows Complets](#workflows-complets)
3. [Exemples de Sortie](#exemples-de-sortie)
4. [Dépannage](#dépannage)

---

## 🚀 Commandes de Base

### 1. Setup Initial

```powershell
# Installation complète
pip install -r requirements.txt

# Vérification
python -c "import mlflow, sklearn, pandas, optuna, evidently; print('✓ OK')"
```

**Sortie attendue:**
```
✓ OK
```

---

### 2. Génération de Données

```powershell
# Générer TOUS les datasets
python generate_data.py --dataset all
```

**Sortie attendue:**
```
============================================================
MLOps Data Generation Pipeline
============================================================
Generating California Housing dataset...
✓ California Housing dataset saved: 20640 rows, 12 columns
  Target variable: MedHouseVal (median house value in $100k)
  Features: ['MedInc', 'HouseAge', 'AveRooms', ...]

Generating Credit Card Fraud dataset...
✓ Credit Fraud dataset saved: 10000 rows, 31 columns
  Target variable: Class (0=legitimate, 1=fraud)
  Fraud rate: 0.20%

Generating Customer Churn dataset...
✓ Customer Churn dataset saved: 7043 rows, 21 columns
  Target variable: Churn (Yes/No)
  Churn rate: 26.54%

============================================================
✓ Data generation completed successfully!
============================================================
```

**Ou générer un seul dataset:**
```powershell
python generate_data.py --dataset california_housing
python generate_data.py --dataset credit_fraud
python generate_data.py --dataset customer_churn
```

---

### 3. Entraînement de Modèles

```powershell
# Random Forest sur California Housing
python train.py --dataset california_housing --model random_forest
```

**Sortie attendue:**
```
============================================================
MLOps Training Pipeline with MLflow
============================================================
Dataset: california_housing
Model: random_forest
============================================================

Loading California Housing dataset...
Dataset shape: (20640, 11)
Target range: [0.15, 5.00]

Training random_forest model...
Cross-validation score: 0.7981 (+/- 0.0124)

Evaluating model...
Train RMSE: 0.2847
Test RMSE: 0.4923
Test MAE: 0.3254
Test R²: 0.8129

✓ Feature importance plot saved: feature_importance.png
✓ Predictions plot saved: predictions_plot.png
✓ Model logged to MLflow
✓ Metrics saved to metrics/metrics_california_housing_random_forest.json
✓ Run ID: a1b2c3d4e5f6g7h8i9j0

============================================================
✓ Training completed successfully!
============================================================

To view results in MLflow UI, run:
  mlflow ui
Then open http://localhost:5000 in your browser
```

**Autres combinaisons:**
```powershell
# Gradient Boosting sur Credit Fraud
python train.py --dataset credit_fraud --model gradient_boosting

# Logistic Regression sur Customer Churn
python train.py --dataset customer_churn --model logistic_regression
```

---

### 4. Hyperparameter Tuning avec Optuna

```powershell
# Tuning avec 50 trials (recommandé)
python tune_hyperparameters.py --dataset california_housing --n-trials 50
```

**Sortie attendue:**
```
============================================================
Hyperparameter Tuning with Optuna
Dataset: california_housing
Number of trials: 50
============================================================

Loading dataset...
Creating Optuna study: california_housing_study_20260106_143000

Starting optimization with 50 trials...
This may take several minutes...

[I 2026-01-06 14:30:05,123] Trial 0 finished with value: 0.7845
[I 2026-01-06 14:30:12,456] Trial 1 finished with value: 0.8012
[I 2026-01-06 14:30:19,789] Trial 2 finished with value: 0.8156
...
[I 2026-01-06 14:36:45,321] Trial 49 finished with value: 0.8423

============================================================
Optimization completed!
============================================================

Best cross-validation score: 0.8441

Best hyperparameters:
  n_estimators: 237
  max_depth: 18
  min_samples_split: 3
  min_samples_leaf: 1
  max_features: sqrt

============================================================
Training final model with best parameters...
============================================================

Train RMSE: 0.2103
Test RMSE: 0.4512
Test MAE: 0.2987
Test R²: 0.8441

✓ Optimization results saved: optuna_results/california_housing_optimization.json
✓ Model logged to MLflow

============================================================
Optimization Insights
============================================================
Total trials: 50
Complete trials: 50
Pruned trials: 0

Hyperparameter Importances:
  n_estimators: 0.3842
  max_depth: 0.2917
  min_samples_leaf: 0.1835
  min_samples_split: 0.0956
  max_features: 0.0450

============================================================
✓ Hyperparameter tuning completed successfully!
============================================================
```

---

### 5. Détection de Drift

```powershell
# Comparer deux datasets du projet
python detect_drift.py --compare-datasets housing churn
```

**Sortie attendue:**
```
============================================================
Data Drift Detection with Evidently AI
============================================================

Comparing Datasets: housing vs churn
============================================================

Loading reference data: data/housing_data.csv
  Shape: (20640, 12)

Loading current data: data/churn_data.csv
  Shape: (7043, 21)

Common columns: 0

============================================================
Generating Data Drift Report...
============================================================

✓ Data Drift Report saved: reports/comparison_housing_vs_churn/data_drift_report_20260106_143500.html

============================================================
Generating Data Quality Report...
============================================================

✓ Data Quality Report saved: reports/comparison_housing_vs_churn/data_quality_report_20260106_143500.html

============================================================
Drift Detection Summary
============================================================

Dataset Drift Detected: YES ⚠️
Drift Share: 100.00%
Number of Drifted Columns: 30

Drifted Columns:
  - MedInc
  - HouseAge
  - AveRooms
  - AveBedrms
  - Population
  - AveOccup
  - Latitude
  - Longitude
  - MedHouseVal
  - tenure

============================================================
Recommendations
============================================================

⚠️  DRIFT DETECTED - Recommended Actions:
  1. Review the drifted features in the HTML report
  2. Consider retraining the model with recent data
  3. Investigate root causes of distribution changes
  4. Update data preprocessing pipeline if needed

============================================================
✓ Drift detection completed successfully!
============================================================

Generated Reports:
  1. reports/.../data_drift_report_20260106_143500.html
  2. reports/.../data_quality_report_20260106_143500.html
  3. reports/.../detailed_drift_report_20260106_143500.html

Open these HTML files in your browser to view detailed analysis.
```

---

### 6. Comparaison des Résultats

```powershell
python compare_results.py
```

**Sortie attendue:**
```
============================================================
MLOps Model Comparison Tool
============================================================

Loading metrics from JSON files...
✓ Loaded 3 model results

Datasets found:
  - california_housing: 1 model(s)
  - credit_fraud: 1 model(s)
  - customer_churn: 1 model(s)

Generating comparison plots...
✓ Performance comparison plot saved: reports/performance_comparison.png

Generating HTML comparison report...
✓ Comparison report saved: reports/comparison_report.html

============================================================
✓ Comparison completed successfully!
============================================================

Generated files:
  - reports/comparison_report.html
  - reports/performance_comparison.png

Open reports/comparison_report.html in your browser to view the full report.
```

---

### 7. MLflow UI

```powershell
# Lancer l'interface MLflow
mlflow ui
```

**Sortie attendue:**
```
[2026-01-06 14:40:00 +0000] [12345] [INFO] Starting gunicorn 20.1.0
[2026-01-06 14:40:00 +0000] [12345] [INFO] Listening at: http://127.0.0.1:5000
[2026-01-06 14:40:00 +0000] [12346] [INFO] Booting worker with pid: 12346
```

**Ouvrir:** http://localhost:5000

**Interface MLflow montrera:**
- 📊 Liste de toutes les expériences
- 📈 Graphiques de métriques
- 🔍 Comparaisons de runs
- 📦 Modèles enregistrés
- 🏷️ Tags et paramètres

---

### 8. DVC Pipeline

```powershell
# Exécuter le pipeline complet
dvc repro
```

**Sortie attendue:**
```
Running stage 'generate_data':
> python generate_data.py --dataset all
✓ Stage 'generate_data' completed

Running stage 'train_housing':
> python train.py --dataset california_housing --model random_forest
✓ Stage 'train_housing' completed

Running stage 'train_credit':
> python train.py --dataset credit_fraud --model random_forest
✓ Stage 'train_credit' completed

Running stage 'train_churn':
> python train.py --dataset customer_churn --model random_forest
✓ Stage 'train_churn' completed

Running stage 'compare_results':
> python compare_results.py
✓ Stage 'compare_results' completed

Pipeline execution completed successfully!
```

**Visualiser le DAG:**
```powershell
dvc dag
```

**Sortie:**
```
        +----------------+
        | generate_data  |
        +----------------+
         ***   ***   ***
        *      *      *
       *       *       *
+-------------+ +-------------+ +-------------+
| train_housing| | train_credit| | train_churn |
+-------------+ +-------------+ +-------------+
         ***         ***         ***
            *         *         *
             *        *        *
              *       *       *
               +-------------+
               |compare_results|
               +-------------+
```

---

## 🔄 Workflows Complets

### Workflow 1: Premier Essai Rapide (5 minutes)

```powershell
# 1. Générer un seul dataset
python generate_data.py --dataset california_housing

# 2. Entraîner un modèle
python train.py --dataset california_housing --model random_forest

# 3. Voir les résultats
mlflow ui
# Ouvrir http://localhost:5000
```

---

### Workflow 2: Comparaison Complète (15 minutes)

```powershell
# 1. Générer tous les datasets
python generate_data.py --dataset all

# 2. Entraîner tous les modèles
python train.py --dataset california_housing --model random_forest
python train.py --dataset credit_fraud --model random_forest
python train.py --dataset customer_churn --model random_forest

# 3. Comparer
python compare_results.py

# 4. Visualiser
start reports/comparison_report.html
```

---

### Workflow 3: Optimisation Avancée (30 minutes)

```powershell
# 1. Baseline
python train.py --dataset california_housing --model random_forest

# 2. Tuning
python tune_hyperparameters.py --dataset california_housing --n-trials 100

# 3. Drift detection
python detect_drift.py --compare-datasets housing credit

# 4. Comparaison finale
python compare_results.py

# 5. Visualiser tout
mlflow ui
start reports/comparison_report.html
start reports/data_drift_report_*.html
```

---

### Workflow 4: Pipeline Automatique (10 minutes)

```powershell
# Exécuter tout automatiquement
python run_full_pipeline.py

# Le script exécutera:
# 1. Génération des données
# 2. Entraînement des modèles
# 3. Optionnel: Hyperparameter tuning
# 4. Détection de drift
# 5. Comparaison des résultats
# 6. Tests unitaires
```

---

## 📊 Exemples de Sortie

### Structure des Fichiers Générés

Après exécution complète:

```
mlops-mlflow-tp/
│
├── data/                              # ✓ Datasets générés
│   ├── housing_data.csv              (20,640 rows)
│   ├── credit_data.csv               (10,000 rows)
│   └── churn_data.csv                (7,043 rows)
│
├── mlruns/                           # ✓ MLflow tracking
│   ├── 0/                            (Default experiment)
│   ├── 1/                            (california_housing_random_forest)
│   ├── 2/                            (credit_fraud_random_forest)
│   └── 3/                            (customer_churn_random_forest)
│
├── metrics/                          # ✓ JSON metrics
│   ├── metrics_california_housing_random_forest.json
│   ├── metrics_credit_fraud_random_forest.json
│   └── metrics_customer_churn_random_forest.json
│
├── reports/                          # ✓ HTML reports
│   ├── comparison_report.html
│   ├── performance_comparison.png
│   ├── data_drift_report_*.html
│   └── data_quality_report_*.html
│
└── optuna_results/                   # ✓ Tuning results
    └── california_housing_optimization.json
```

---

### Exemple de Métrique JSON

**metrics/metrics_california_housing_random_forest.json:**
```json
{
  "train_rmse": 0.2847,
  "test_rmse": 0.4923,
  "train_mae": 0.2156,
  "test_mae": 0.3254,
  "train_r2": 0.9534,
  "test_r2": 0.8129
}
```

---

## 🔧 Dépannage

### Problème 1: Module non trouvé

**Erreur:**
```
ModuleNotFoundError: No module named 'mlflow'
```

**Solution:**
```powershell
pip install -r requirements.txt --upgrade
```

---

### Problème 2: MLflow UI port occupé

**Erreur:**
```
Address already in use
```

**Solution:**
```powershell
# Changer le port
mlflow ui --port 5001
```

---

### Problème 3: Fichiers de données manquants

**Erreur:**
```
FileNotFoundError: Dataset not found: data/housing_data.csv
```

**Solution:**
```powershell
# Générer les données
python generate_data.py --dataset all
```

---

### Problème 4: Mémoire insuffisante

**Symptôme:** Le script plante sans message

**Solution:**
```powershell
# Réduire le nombre d'arbres dans train.py
# Modifier: n_estimators=50 au lieu de 100

# Ou réduire les trials pour Optuna
python tune_hyperparameters.py --dataset california_housing --n-trials 20
```

---

## ⚡ Commandes Rapides (Cheat Sheet)

```powershell
# Setup
pip install -r requirements.txt

# Données
python generate_data.py --dataset all

# Training
python train.py --dataset california_housing --model random_forest

# Tuning
python tune_hyperparameters.py --dataset california_housing --n-trials 50

# Drift
python detect_drift.py --compare-datasets housing churn

# Comparaison
python compare_results.py

# MLflow
mlflow ui

# DVC
dvc repro
dvc dag
dvc metrics show

# Tests
pytest tests/ -v

# Pipeline complet
python run_full_pipeline.py
```

---

**💡 Tip:** Copiez-collez ces commandes directement dans votre terminal!
