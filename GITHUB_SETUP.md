# Guide de Configuration GitHub et GitHub Actions

## Table des Matieres

1. [Configuration du Depot GitHub](#1-configuration-du-depot-github)
2. [Structure du Projet](#2-structure-du-projet)
3. [Configuration GitHub Actions](#3-configuration-github-actions)
4. [Pipeline CI/CD Detaille](#4-pipeline-cicd-detaille)
5. [Etapes de Travail](#5-etapes-de-travail)
6. [Tracabilite MLflow](#6-tracabilite-mlflow)
7. [Verification et Tests](#7-verification-et-tests)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Configuration du Depot GitHub

### 1.1 Creer ou Utiliser un Depot Existant

#### Option A: Creer un Nouveau Depot

```bash
# Sur GitHub.com
1. Aller sur https://github.com/new
2. Nom du depot: mlops-mlflow-tp (ou autre nom)
3. Description: "MLOps Pipeline with MLflow, DVC, and GitHub Actions"
4. Visibilite: Public ou Private
5. Ne pas initialiser avec README (vous avez deja le projet)
6. Cliquer sur "Create repository"
```

#### Option B: Utiliser un Depot Existant

```bash
# Vous avez deja le depot: WissemHarhouri/MLOPS
# Branch actuelle: projet
# Aucune action necessaire
```

### 1.2 Initialiser Git Localement

```bash
# Si pas encore fait
cd C:\Users\wharhouri\Downloads\mlops-mlflow-tp

# Initialiser Git
git init

# Ajouter le remote (si pas deja fait)
git remote add origin https://github.com/WissemHarhouri/MLOPS.git

# Verifier le remote
git remote -v
```

**Sortie attendue:**
```
origin  https://github.com/WissemHarhouri/MLOPS.git (fetch)
origin  https://github.com/WissemHarhouri/MLOPS.git (push)
```

### 1.3 Configurer .gitignore

Le fichier `.gitignore` doit contenir:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# MLflow
mlruns/
mlflow.db
mlartifacts/

# DVC
.dvc/cache
.dvc/tmp
.dvc/plots

# Data (gere par DVC)
data/*.csv
!data/.gitkeep

# Models
models/*.pkl
models/*.joblib
!models/.gitkeep

# Metrics
metrics/*.json
!metrics/.gitkeep

# Reports
reports/*.html
reports/*.pdf
!reports/.gitkeep

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db
```

### 1.4 Preparer les Fichiers DVC

```bash
# Verifier les fichiers DVC essentiels
ls -la .dvc/

# Fichiers importants:
# - .dvc/config       -> Configuration DVC
# - .dvc/.gitignore   -> Ignore automatique
# - dvc.yaml          -> Pipeline definition
# - dvc.lock          -> Versions lockees
# - *.dvc             -> Fichiers de donnees trackes
```

---

## 2. Structure du Projet

### 2.1 Arborescence Complete

```
mlops-mlflow-tp/
├── .github/
│   └── workflows/
│       ├── ml-pipeline.yml          # ✅ NOUVEAU - Pipeline CI/CD complet
│       └── mlops-pipeline.yml       # Ancien workflow (optionnel)
├── .dvc/
│   ├── config                       # Configuration DVC
│   └── .gitignore                   # Ignore cache DVC
├── data/
│   ├── .gitkeep
│   ├── california_housing.csv.dvc   # DVC tracking
│   ├── credit_fraud.csv.dvc         # DVC tracking
│   └── customer_churn.csv.dvc       # DVC tracking
├── metrics/
│   ├── .gitkeep
│   ├── housing_metrics.json
│   ├── credit_metrics.json
│   └── churn_metrics.json
├── models/
│   └── .gitkeep
├── reports/
│   ├── .gitkeep
│   └── comparison_report.html
├── tests/
│   └── test_pipeline.py
├── .gitignore                       # Ignore patterns
├── dvc.yaml                         # DVC pipeline
├── dvc.lock                         # DVC versions
├── requirements.txt                 # Dependencies Python
├── generate_data.py                 # Generation datasets
├── train.py                         # Entrainement modeles
├── compare_results.py               # Comparaison resultats
├── tune_hyperparameters.py          # Optimisation Optuna
├── detect_drift.py                  # Detection drift Evidently
├── run_full_pipeline.py             # Pipeline automatique
├── config.py                        # Configuration centrale
└── README.md                        # Documentation
```

### 2.2 Fichiers a Pousser sur GitHub

**Fichiers INCLUS (commits):**
- ✅ `.github/workflows/*.yml` - Workflows CI/CD
- ✅ `.dvc/config` - Config DVC
- ✅ `*.dvc` - Fichiers de tracking DVC
- ✅ `dvc.yaml` - Pipeline DVC
- ✅ `dvc.lock` - Versions DVC
- ✅ `*.py` - Scripts Python
- ✅ `requirements.txt` - Dependencies
- ✅ `.gitignore` - Ignore patterns
- ✅ `*.md` - Documentation

**Fichiers EXCLUS (gitignore):**
- ❌ `data/*.csv` - Donnees brutes (gerees par DVC)
- ❌ `mlruns/` - Runs MLflow (local uniquement)
- ❌ `__pycache__/` - Cache Python
- ❌ `*.pyc` - Bytecode Python
- ❌ `.dvc/cache` - Cache DVC

---

## 3. Configuration GitHub Actions

### 3.1 Workflow ml-pipeline.yml

Le fichier `.github/workflows/ml-pipeline.yml` contient **10 jobs**:

#### Jobs du Pipeline:

1. **setup** - Installation environnement Python + dependencies
2. **data-generation** - Generation des 3 datasets
3. **dvc-pipeline** - Execution `dvc repro`
4. **train-housing** - Entrainement modeles California Housing
5. **train-credit** - Entrainement modeles Credit Fraud
6. **train-churn** - Entrainement modeles Customer Churn
7. **evaluate** - Comparaison des resultats
8. **validate** - Validation des metriques (seuils)
9. **mlflow-tracking** - Verification tracking MLflow
10. **notify** - Notification fin de pipeline

### 3.2 Declencheurs du Workflow

```yaml
on:
  push:
    branches:
      - main
      - projet
      - develop
  pull_request:
    branches:
      - main
      - projet
```

**Execution automatique sur:**
- ✅ Push sur branches `main`, `projet`, `develop`
- ✅ Pull Request vers `main` ou `projet`

### 3.3 Variables d'Environnement

```yaml
env:
  PYTHON_VERSION: '3.11'
  MLFLOW_TRACKING_URI: 'sqlite:///mlflow.db'
```

### 3.4 Secrets GitHub (Optionnel)

Pour des configurations avancees:

```bash
# Sur GitHub.com
1. Aller dans Settings -> Secrets and variables -> Actions
2. Cliquer "New repository secret"
3. Ajouter:
   - MLFLOW_TRACKING_URI (si remote)
   - AWS_ACCESS_KEY_ID (si S3 pour DVC)
   - AWS_SECRET_ACCESS_KEY (si S3 pour DVC)
```

---

## 4. Pipeline CI/CD Detaille

### 4.1 Job 1: Setup Environment

```yaml
setup:
  name: Setup Environment
  runs-on: ubuntu-latest
  
  steps:
    - Checkout code
    - Setup Python 3.11
    - Install dependencies from requirements.txt
    - Verify installations (dvc, mlflow)
    - Initialize DVC
```

**Duree estimee:** 1-2 minutes

### 4.2 Job 2: Data Generation

```yaml
data-generation:
  name: Generate Datasets
  needs: setup
  
  steps:
    - Generate california_housing.csv (20,640 samples)
    - Generate credit_fraud.csv (10,000 samples)
    - Generate customer_churn.csv (7,043 samples)
    - Verify datasets created
    - Upload as artifacts (retention: 7 days)
```

**Duree estimee:** 2-3 minutes

### 4.3 Job 3: DVC Pipeline

```yaml
dvc-pipeline:
  name: Execute DVC Pipeline
  needs: data-generation
  
  steps:
    - Download datasets from artifacts
    - Execute: dvc repro --verbose
    - Show: dvc metrics show
    - Show: dvc dag
    - Upload dvc.lock and metrics/*.json
```

**Duree estimee:** 5-10 minutes

### 4.4 Jobs 4-6: Model Training

Trois jobs paralleles:

```yaml
train-housing:
  - Train RandomForest on California Housing
  - Train GradientBoosting on California Housing
  - Save metrics/housing_metrics.json
  - Upload artifacts

train-credit:
  - Train RandomForest on Credit Fraud
  - Train GradientBoosting on Credit Fraud
  - Save metrics/credit_metrics.json
  - Upload artifacts

train-churn:
  - Train RandomForest on Customer Churn
  - Train GradientBoosting on Customer Churn
  - Save metrics/churn_metrics.json
  - Upload artifacts
```

**Duree estimee:** 3-5 minutes chacun (en parallele)

### 4.5 Job 7: Evaluation

```yaml
evaluate:
  needs: [train-housing, train-credit, train-churn]
  
  steps:
    - Download all metrics from artifacts
    - Execute: python compare_results.py
    - Generate comparison_report.html
    - Upload report (retention: 30 days)
```

**Duree estimee:** 1-2 minutes

### 4.6 Job 8: Validation

```yaml
validate:
  needs: evaluate
  
  steps:
    - Validate Housing: R2 >= 0.70, RMSE <= 1.0
    - Validate Credit: ROC-AUC >= 0.85, F1 >= 0.50
    - Validate Churn: Accuracy >= 0.70, F1 >= 0.50
    - Display validation summary
```

**Criteres de validation:**

| Dataset | Metrique | Seuil | Action si Echec |
|---------|----------|-------|-----------------|
| Housing | R2 Score | ≥ 0.70 | Pipeline FAIL ❌ |
| Housing | RMSE | ≤ 1.0 | Pipeline FAIL ❌ |
| Credit | ROC-AUC | ≥ 0.85 | Pipeline FAIL ❌ |
| Credit | F1 Score | ≥ 0.50 | Pipeline FAIL ❌ |
| Churn | Accuracy | ≥ 0.70 | Pipeline FAIL ❌ |
| Churn | F1 Score | ≥ 0.50 | Pipeline FAIL ❌ |

**Duree estimee:** 30 secondes

### 4.7 Job 9: MLflow Tracking

```yaml
mlflow-tracking:
  needs: [train-housing, train-credit, train-churn]
  
  steps:
    - List all MLflow experiments
    - Count runs per experiment
    - Export mlflow_runs_summary.csv
    - Upload summary (retention: 30 days)
```

**Duree estimee:** 30 secondes

### 4.8 Job 10: Notification

```yaml
notify:
  needs: [validate, mlflow-tracking]
  if: always()
  
  steps:
    - Display pipeline completion status
    - Show execution date/time
    - Summary of all jobs
```

**Duree estimee:** 10 secondes

### 4.9 Duree Totale du Pipeline

**Estimation:** 15-25 minutes (selon cache et parallelisation)

---

## 5. Etapes de Travail

### 5.1 Premier Push vers GitHub

```bash
# 1. Verifier le statut Git
git status

# 2. Ajouter tous les fichiers necessaires
git add .github/workflows/ml-pipeline.yml
git add .dvc/config
git add dvc.yaml dvc.lock
git add *.py
git add requirements.txt
git add .gitignore
git add README.md
git add data/*.dvc

# 3. Verifier ce qui sera commite
git status

# 4. Commiter avec message descriptif
git commit -m "Add GitHub Actions CI/CD pipeline with MLflow tracking"

# 5. Pousser vers GitHub
git push origin projet
```

**Sortie attendue:**
```
Enumerating objects: 42, done.
Counting objects: 100% (42/42), done.
Delta compression using up to 8 threads
Compressing objects: 100% (35/35), done.
Writing objects: 100% (42/42), 125.34 KiB | 12.53 MiB/s, done.
Total 42 (delta 15), reused 0 (delta 0)
To https://github.com/WissemHarhouri/MLOPS.git
   a1b2c3d..e4f5g6h  projet -> projet
```

### 5.2 Verifier l'Execution du Pipeline

```bash
# Sur GitHub.com
1. Aller sur: https://github.com/WissemHarhouri/MLOPS
2. Cliquer sur l'onglet "Actions"
3. Voir le workflow "MLOps Pipeline CI/CD" en execution
4. Cliquer dessus pour voir les details
```

**Interface GitHub Actions:**

```
MLOps Pipeline CI/CD
├── ✅ setup (1m 23s)
├── ✅ data-generation (2m 45s)
├── ✅ dvc-pipeline (8m 12s)
├── ✅ train-housing (4m 31s)
├── ✅ train-credit (3m 56s)
├── ✅ train-churn (4m 18s)
├── ✅ evaluate (1m 44s)
├── ✅ validate (42s)
├── ✅ mlflow-tracking (28s)
└── ✅ notify (12s)

Total duration: 18m 34s
```

### 5.3 Modifier le Dataset ou le Code

#### Exemple 1: Modifier le Dataset

```bash
# Modifier generate_data.py pour augmenter le nombre de samples
# Ligne 20: n_samples = 30000  # Avant: 20640

# Regenerer les donnees
python generate_data.py --dataset housing

# Tracker avec DVC
dvc add data/california_housing.csv

# Commiter et pousser
git add data/california_housing.csv.dvc
git commit -m "Increase housing dataset to 30,000 samples"
git push origin projet
```

**Resultat:** Pipeline CI/CD se declenche automatiquement

#### Exemple 2: Modifier le Code d'Entrainement

```bash
# Modifier train.py
# Ajouter un nouveau parametre: max_depth=15

# Commiter et pousser
git add train.py
git commit -m "Add max_depth parameter to RandomForest"
git push origin projet
```

**Resultat:** Pipeline CI/CD se declenche automatiquement

### 5.4 Observer l'Execution Automatique

```bash
# Apres le push, verifier immediatement sur GitHub Actions
# Le workflow demarre automatiquement sous 10-30 secondes

# Logs en temps reel disponibles dans chaque job
```

**Exemple de Log (Job data-generation):**

```
Run python generate_data.py --dataset housing
[INFO] Generating California Housing dataset...
[OK] California Housing dataset created successfully!
  - Shape: (30000, 9)
  - Target: Median house value (in $100,000)
  - Features: 8
[INFO] Dataset saved to data/california_housing.csv
```

---

## 6. Tracabilite MLflow

### 6.1 Creation Automatique de Runs MLflow

Chaque execution du pipeline CI/CD cree automatiquement des runs MLflow:

```python
# Dans train.py
with mlflow.start_run(run_name=f"{model_name}-{dataset_name}-CI"):
    # Log parameters
    mlflow.log_params(model_params)
    
    # Log metrics
    mlflow.log_metrics(metrics)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
```

### 6.2 Verification des Runs MLflow

#### Methode 1: Via MLflow UI (Local)

```bash
# Lancer MLflow UI localement
mlflow ui

# Ouvrir: http://localhost:5000
```

**Interface MLflow UI:**

```
Experiments
├── California-Housing-CI
│   ├── Run 1: random_forest (2026-01-06 17:05:23)
│   │   ├── Parameters: n_estimators=100, max_depth=None
│   │   ├── Metrics: R2=0.8441, RMSE=0.4512
│   │   └── Artifacts: model/, feature_importance.png
│   └── Run 2: gradient_boosting (2026-01-06 17:08:15)
│       ├── Parameters: n_estimators=100, learning_rate=0.1
│       ├── Metrics: R2=0.8297, RMSE=0.4701
│       └── Artifacts: model/, predictions.png
├── Credit-Fraud-CI
│   ├── Run 1: random_forest (2026-01-06 17:06:45)
│   └── Run 2: gradient_boosting (2026-01-06 17:09:32)
└── Customer-Churn-CI
    ├── Run 1: random_forest (2026-01-06 17:07:18)
    └── Run 2: gradient_boosting (2026-01-06 17:10:05)
```

#### Methode 2: Via Python Script

```python
import mlflow
import pandas as pd

# Lister tous les runs
all_runs = mlflow.search_runs()
print(f"Total runs: {len(all_runs)}")

# Afficher les 10 derniers runs
recent_runs = all_runs.head(10)
print(recent_runs[['run_id', 'experiment_id', 'tags.mlflow.runName', 'metrics.accuracy']])

# Comparer les metriques entre runs
housing_runs = mlflow.search_runs(
    experiment_names=["California-Housing-CI"],
    order_by=["metrics.r2_score DESC"]
)
print(housing_runs[['tags.mlflow.runName', 'metrics.r2_score', 'metrics.rmse']])
```

**Sortie attendue:**

```
Total runs: 6

run_id                            experiment_id  tags.mlflow.runName         metrics.r2_score
a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6  1              random_forest-housing-CI    0.8441
b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7  1              gradient_boosting-housing   0.8297
c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8  2              random_forest-credit-CI     NaN
...
```

### 6.3 Comparer les Metriques entre Executions

```bash
# Apres plusieurs executions du pipeline
# Comparer les performances

mlflow ui
# -> Onglet "Compare" dans MLflow UI
# -> Selectionner plusieurs runs
# -> Voir les graphiques de comparaison
```

**Exemple de Comparaison:**

| Run | Date | Model | R2 Score | RMSE | Delta |
|-----|------|-------|----------|------|-------|
| Run 1 | 2026-01-06 15:00 | RandomForest | 0.8129 | 0.4923 | Baseline |
| Run 2 | 2026-01-06 16:30 | RandomForest | 0.8297 | 0.4701 | +2.07% ✅ |
| Run 3 | 2026-01-06 17:45 | RandomForest | 0.8441 | 0.4512 | +3.84% ✅ |

### 6.4 Tracer les Changements

```bash
# Associer les runs MLflow aux commits Git

# Dans le script train.py, ajouter:
git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
mlflow.set_tag("git_commit", git_commit)
mlflow.set_tag("git_branch", "projet")
```

**Resultat dans MLflow:**

```
Tags:
├── mlflow.runName: random_forest-housing-CI
├── mlflow.source.type: LOCAL
├── git_commit: e4f5g6h7i8j9k0l1m2n3o4p5q6r7s8t9u0v1w2x3
├── git_branch: projet
└── dataset_version: california_housing.csv.dvc
```

---

## 7. Verification et Tests

### 7.1 Verifier le Workflow GitHub Actions

```bash
# Sur GitHub.com
1. Actions -> MLOps Pipeline CI/CD
2. Cliquer sur une execution
3. Verifier tous les jobs sont ✅ verts

# Cas d'echec:
# - Job rouge ❌ -> Cliquer dessus -> Voir les logs d'erreur
```

### 7.2 Tests Locaux Avant Push

```bash
# 1. Tester la generation de donnees
python generate_data.py --dataset all

# 2. Tester l'entrainement
python train.py --dataset california_housing.csv --model random_forest

# 3. Tester DVC repro
dvc repro

# 4. Tester la comparaison
python compare_results.py

# 5. Si tout fonctionne -> Push
git add .
git commit -m "Test successful - ready for CI/CD"
git push origin projet
```

### 7.3 Verifier les Artifacts GitHub

```bash
# Apres execution du pipeline
# Sur GitHub Actions -> Cliquer sur une execution
# Section "Artifacts" en bas de page:

Artifacts (7):
├── datasets (12.5 MB) - 7 days retention
├── dvc-outputs (2.3 KB) - 7 days retention
├── housing-metrics (1.2 KB) - 7 days retention
├── credit-metrics (1.1 KB) - 7 days retention
├── churn-metrics (1.0 KB) - 7 days retention
├── comparison-report (45.6 KB) - 30 days retention
└── mlflow-summary (8.9 KB) - 30 days retention
```

**Telecharger et verifier:**

```bash
# Cliquer sur "comparison-report"
# Telechargement: comparison-report.zip
# Extraire et ouvrir comparison_report.html
```

### 7.4 Verifier les Metriques DVC

```bash
# Localement
dvc metrics show

# Sur GitHub Actions -> Job "dvc-pipeline" -> Logs
```

**Sortie attendue:**

```
Path                          r2_score    rmse     accuracy  f1_score  roc_auc
metrics/housing_metrics.json  0.8441      0.4512   -         -         -
metrics/credit_metrics.json   -           -        0.9823    0.8182    0.9867
metrics/churn_metrics.json    -           -        0.8145    0.6843    -
```

---

## 8. Troubleshooting

### 8.1 Pipeline Fails - Job Setup

**Erreur:**
```
ERROR: Could not find a version that satisfies the requirement mlflow>=2.0.0
```

**Solution:**
```bash
# Verifier requirements.txt
cat requirements.txt

# Verifier que toutes les dependances sont compatibles Python 3.11
# Mettre a jour si necessaire
pip install --upgrade pip
pip install -r requirements.txt
```

### 8.2 Pipeline Fails - DVC Repro

**Erreur:**
```
ERROR: failed to reproduce 'train_housing': file 'data/california_housing.csv' does not exist
```

**Solution:**
```bash
# S'assurer que les datasets sont generes avant dvc repro
# Dans le workflow, verifier que data-generation est complete

# Ou desactiver temporairement dvc repro
# Et utiliser directement les scripts Python
```

### 8.3 Pipeline Fails - Validation

**Erreur:**
```
ERROR: R2 score too low (< 0.70)
Housing R2: 0.6543, RMSE: 0.6789
```

**Solution:**
```bash
# Option 1: Ameliorer le modele
# - Augmenter n_estimators
# - Ajouter feature engineering
# - Tuner les hyperparametres

# Option 2: Ajuster les seuils de validation
# Dans ml-pipeline.yml, modifier:
if r2 < 0.60:  # Au lieu de 0.70
```

### 8.4 MLflow Tracking Issues

**Erreur:**
```
ERROR: No MLflow runs found
```

**Solution:**
```bash
# Verifier que MLflow tracking est active dans train.py
# Verifier MLFLOW_TRACKING_URI

# Localement:
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
mlflow ui

# Dans GitHub Actions:
# Ajouter dans env:
MLFLOW_TRACKING_URI: 'sqlite:///mlflow.db'
```

### 8.5 Artifacts Not Uploaded

**Erreur:**
```
WARNING: No files were found with the provided path
```

**Solution:**
```bash
# Verifier que les fichiers existent avant upload
ls -la metrics/
ls -la reports/

# Dans le workflow, ajouter verification:
- name: Check files exist
  run: |
    if [ ! -f metrics/housing_metrics.json ]; then
      echo "ERROR: Metrics file not found"
      exit 1
    fi
```

### 8.6 Permission Denied

**Erreur:**
```
ERROR: Permission denied (publickey)
```

**Solution:**
```bash
# Configurer les credentials Git
git config --global user.name "Wissem Harhouri"
git config --global user.email "wharhouri@example.com"

# Ou utiliser HTTPS au lieu de SSH
git remote set-url origin https://github.com/WissemHarhouri/MLOPS.git
```

---

## Resume des Etapes Principales

### Checklist Complete

- [x] **1. Configuration GitHub**
  - [x] Depot cree/utilise: WissemHarhouri/MLOPS
  - [x] Branch: projet
  - [x] .gitignore configure

- [x] **2. Fichiers DVC**
  - [x] dvc.yaml (pipeline definition)
  - [x] dvc.lock (versions)
  - [x] *.dvc (tracking datasets)

- [x] **3. GitHub Actions Workflow**
  - [x] .github/workflows/ml-pipeline.yml cree
  - [x] 10 jobs configures
  - [x] Declencheurs: push sur projet, main, develop

- [x] **4. Premier Push**
  - [ ] `git add .github/workflows/ml-pipeline.yml`
  - [ ] `git commit -m "Add CI/CD pipeline"`
  - [ ] `git push origin projet`

- [ ] **5. Verification Pipeline**
  - [ ] Aller sur GitHub Actions
  - [ ] Observer execution automatique
  - [ ] Verifier tous les jobs passent ✅

- [ ] **6. Modification et Re-test**
  - [ ] Modifier dataset ou code
  - [ ] Push changements
  - [ ] Observer nouvelle execution automatique

- [ ] **7. Tracabilite MLflow**
  - [ ] Lancer `mlflow ui` localement
  - [ ] Verifier nouveaux runs crees
  - [ ] Comparer metriques entre runs

---

## Commandes Rapides de Reference

```bash
# Push initial
git add .github/workflows/ml-pipeline.yml dvc.yaml dvc.lock *.py
git commit -m "Setup GitHub Actions CI/CD with MLflow and DVC"
git push origin projet

# Verifier workflow
# -> GitHub.com -> Actions

# Modifier et tester
python generate_data.py --dataset all
python train.py --dataset california_housing.csv --model random_forest
dvc repro
git add .
git commit -m "Update: [description]"
git push origin projet

# Verifier MLflow
mlflow ui
# -> http://localhost:5000

# Voir metriques DVC
dvc metrics show
dvc metrics diff
```

---

**Date de Creation:** 2026-01-06  
**Auteur:** Wissem Harhouri  
**Version:** 1.0  
**Status:** ✅ Production Ready
