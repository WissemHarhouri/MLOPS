# Documentation MLOps - Projet de Machine Learning en Production

## 📋 Table des Matières
1. [Vue d'ensemble du projet](#vue-densemble-du-projet)
2. [Architecture MLOps](#architecture-mlops)
3. [Flux de travail (Workflow)](#flux-de-travail-workflow)
4. [Outils utilisés](#outils-utilisés)
5. [Description des modèles ML](#description-des-modèles-ml)
6. [Fonctionnalités avancées](#fonctionnalités-avancées)
7. [Guide d'utilisation](#guide-dutilisation)

---

## 🎯 Vue d'ensemble du projet

Ce projet démontre l'implémentation d'un pipeline MLOps complet pour le développement, le suivi et le déploiement de modèles de Machine Learning. Le projet explore trois datasets différents avec des problématiques variées :

1. **California Housing Dataset** - Régression : Prédiction des prix immobiliers
2. **Credit Card Fraud Detection** - Classification déséquilibrée : Détection de fraudes
3. **Customer Churn Prediction** - Classification binaire : Prédiction de désabonnement

### Objectifs du projet
- ✅ Versioning du code avec **Git**
- ✅ Suivi des expériences avec **MLflow**
- ✅ Versioning des données avec **DVC**
- ✅ Automatisation avec **GitHub Actions**
- ✅ Hyperparameter tuning avec **Optuna**
- ✅ Détection de data drift avec **Evidently**

---

## 🏗️ Architecture MLOps

```
┌─────────────────────────────────────────────────────────────────┐
│                         GitHub Repository                        │
│                    (Code + Configuration)                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      GitHub Actions (CI/CD)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   Linting    │  │   Testing    │  │   Training Pipeline  │  │
│  │   (Flake8)   │  │   (Pytest)   │  │   (Auto-trigger)     │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Data Layer                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  DVC (Data Version Control)                              │   │
│  │  - Versioning des datasets                               │   │
│  │  - Pipeline de preprocessing                             │   │
│  │  - Métriques trackées                                    │   │
│  └──────────────────────────────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Training & Experimentation                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  MLflow Tracking Server                                  │   │
│  │  - Logging des paramètres                                │   │
│  │  - Métriques de performance                              │   │
│  │  - Artifacts (modèles, plots)                            │   │
│  │  - Model Registry                                        │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Optuna (Hyperparameter Tuning)                         │   │
│  │  - Optimisation bayésienne                               │   │
│  │  - Pruning automatique                                   │   │
│  │  - Multi-objective optimization                          │   │
│  └──────────────────────────────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Monitoring & Validation                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Evidently AI                                            │   │
│  │  - Data Drift Detection                                  │   │
│  │  - Model Performance Monitoring                          │   │
│  │  - Data Quality Reports                                  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Flux de travail (Workflow)

### 1. Développement Local

```bash
# Étape 1: Générer/Télécharger les données
python generate_data.py

# Étape 2: Initialiser DVC pour le versioning
dvc add data/*.csv
git add data/*.csv.dvc .gitignore
git commit -m "Add dataset v1"

# Étape 3: Entraîner le modèle avec MLflow tracking
python train.py

# Étape 4: Exécuter le pipeline DVC complet
dvc repro

# Étape 5: Comparer les expériences MLflow
mlflow ui  # Ouvre l'interface web
```

### 2. Pipeline d'expérimentation

```
Data Acquisition → Data Preprocessing → Feature Engineering → 
Model Training → Hyperparameter Tuning → Model Evaluation → 
Model Registration → Monitoring
```

### 3. CI/CD avec GitHub Actions

**Déclencheurs:**
- `git push` sur la branche `main`
- Pull Request
- Changement dans `data/` ou `train.py`

**Étapes automatisées:**
1. Linting du code (flake8)
2. Tests unitaires (pytest)
3. Entraînement automatique du modèle
4. Validation des métriques
5. Mise à jour du Model Registry
6. Génération de rapports

---

## 🛠️ Outils utilisés

### 1. **Git** - Versioning du code
- **Rôle**: Gestion des versions du code source
- **Usage**: 
  - Branches pour features/expériences
  - Commits atomiques
  - Tags pour releases
- **Fichiers clés**: `.gitignore`, `.gitattributes`

### 2. **MLflow** - Suivi des expériences ML
- **Rôle**: Tracking, packaging et déploiement de modèles
- **Composants utilisés**:
  - **Tracking**: Log des paramètres, métriques, artifacts
  - **Projects**: Packaging reproductible
  - **Models**: Format standardisé pour le déploiement
  - **Registry**: Gestion du cycle de vie des modèles
  
**Exemple d'utilisation:**
```python
with mlflow.start_run(run_name="experiment_1"):
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.sklearn.log_model(model, "model")
```

### 3. **DVC (Data Version Control)** - Versioning des données
- **Rôle**: Gestion des versions de datasets et pipelines
- **Fonctionnalités**:
  - Versioning de datasets volumineux
  - Pipeline de preprocessing reproductible
  - Tracking des métriques
  - Remote storage (S3, GCS, Azure, etc.)

**Structure du pipeline:**
```yaml
stages:
  prepare:
    cmd: python prepare_data.py
    deps: [data/raw/]
    outs: [data/processed/]
  
  train:
    cmd: python train.py
    deps: [data/processed/, train.py]
    outs: [models/model.pkl]
    metrics: [metrics/metrics.json]
```

### 4. **GitHub Actions** - CI/CD et Automatisation
- **Rôle**: Automatisation des tests, training et déploiement
- **Workflows**:
  - Tests automatiques sur chaque commit
  - Entraînement automatique périodique
  - Validation des modèles
  - Déploiement en staging/production

### 5. **Optuna** - Optimisation d'hyperparamètres (Fonctionnalité avancée)
- **Rôle**: Recherche automatique des meilleurs hyperparamètres
- **Algorithmes**: TPE Sampler, Bayesian Optimization
- **Avantages**:
  - Pruning automatique des essais non prometteurs
  - Parallélisation facile
  - Intégration native avec MLflow

### 6. **Evidently AI** - Monitoring et Drift Detection (Fonctionnalité avancée)
- **Rôle**: Détection de dégradation des modèles et drift des données
- **Rapports générés**:
  - Data Drift Report
  - Data Quality Report
  - Model Performance Report
- **Métriques surveillées**:
  - Distribution des features
  - Corrélations
  - Valeurs manquantes
  - Performance du modèle

---

## 🤖 Description des modèles ML

### Dataset 1: California Housing (Régression)

**Problématique**: Prédire le prix médian des maisons en Californie

**Features (8)**:
- MedInc: Revenu médian du quartier
- HouseAge: Âge médian des maisons
- AveRooms: Nombre moyen de pièces
- AveBedrms: Nombre moyen de chambres
- Population: Population du quartier
- AveOccup: Occupation moyenne
- Latitude, Longitude: Coordonnées géographiques

**Target**: Prix médian des maisons (en $100k)

**Algorithmes testés**:
1. Random Forest Regressor (baseline)
2. Gradient Boosting Regressor
3. XGBoost Regressor (optimisé avec Optuna)

**Métriques d'évaluation**:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

### Dataset 2: Credit Card Fraud Detection (Classification déséquilibrée)

**Problématique**: Détecter les transactions frauduleuses

**Features (30)**:
- Time: Temps écoulé depuis la première transaction
- V1-V28: Features anonymisées (PCA)
- Amount: Montant de la transaction

**Target**: Class (0 = légitime, 1 = fraude)

**Défis**:
- Dataset hautement déséquilibré (~0.17% de fraudes)
- Nécessite des techniques de rééquilibrage (SMOTE, undersampling)

**Algorithmes testés**:
1. Logistic Regression avec class weighting
2. Random Forest avec balanced class weights
3. LightGBM avec scale_pos_weight

**Métriques d'évaluation**:
- Precision, Recall, F1-Score
- AUC-ROC
- Precision-Recall AUC (plus important pour classes déséquilibrées)

### Dataset 3: Customer Churn Prediction (Classification binaire)

**Problématique**: Prédire si un client va résilier son abonnement

**Features (~20)**:
- Démographiques: gender, SeniorCitizen, Partner, Dependents
- Services: InternetService, OnlineSecurity, TechSupport, etc.
- Compte: tenure, Contract, PaymentMethod, MonthlyCharges, TotalCharges

**Target**: Churn (Yes/No)

**Algorithmes testés**:
1. Random Forest Classifier
2. XGBoost Classifier
3. CatBoost (gestion native des features catégorielles)

**Métriques d'évaluation**:
- Accuracy, Precision, Recall, F1-Score
- AUC-ROC
- Confusion Matrix

---

## 🚀 Fonctionnalités avancées

### 1. Hyperparameter Tuning avec Optuna

**Implémentation**: Recherche automatique des meilleurs hyperparamètres

```python
import optuna
from optuna.integration.mlflow import MLflowCallback

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3)
    }
    
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    return model.score(X_val, y_val)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

**Avantages**:
- Optimisation bayésienne intelligente
- Pruning des essais non prometteurs
- Tracking automatique dans MLflow
- Visualisation des importances des hyperparamètres

### 2. Data Drift Detection avec Evidently

**Implémentation**: Surveillance continue de la qualité des données

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

report = Report(metrics=[
    DataDriftPreset(),
    DataQualityPreset()
])

report.run(reference_data=train_data, current_data=new_data)
report.save_html("reports/data_drift_report.html")
```

**Détections**:
- Drift dans la distribution des features
- Changements dans les corrélations
- Augmentation des valeurs manquantes
- Dégradation de performance du modèle

**Alertes**:
- Notification si drift détecté
- Recommandation de réentraînement
- Génération de rapports HTML interactifs

---

## 📖 Guide d'utilisation

### Installation

```bash
# Cloner le repository
git clone https://github.com/WissemHarhouri/MLOPS.git
cd mlops-mlflow-tp

# Créer un environnement virtuel
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Installer les dépendances
pip install -r requirements.txt

# Initialiser DVC
dvc init
```

### Workflow complet

```bash
# 1. Générer les données (Dataset 1)
python generate_data.py --dataset california_housing

# 2. Ajouter au versioning DVC
dvc add data/housing_data.csv
git add data/housing_data.csv.dvc
git commit -m "Add California Housing dataset"

# 3. Entraîner le modèle avec tracking MLflow
python train.py --dataset california_housing --tune-hyperparameters

# 4. Visualiser les résultats MLflow
mlflow ui
# Ouvrir http://localhost:5000

# 5. Exécuter le pipeline DVC
dvc repro

# 6. Changer de dataset
python generate_data.py --dataset credit_fraud
dvc add data/credit_data.csv
git add data/credit_data.csv.dvc
git commit -m "Switch to Credit Fraud dataset"

# 7. Réentraîner
python train.py --dataset credit_fraud

# 8. Détecter le data drift
python detect_drift.py --reference-data data/housing_data.csv --current-data data/credit_data.csv

# 9. Comparer les résultats
python compare_results.py
```

### Commandes utiles

```bash
# MLflow
mlflow ui                          # Interface web
mlflow models serve -m models:/BestModel/Production  # Servir un modèle

# DVC
dvc status                         # État du pipeline
dvc dag                            # Visualiser le DAG
dvc metrics show                   # Afficher les métriques
dvc plots show                     # Générer des graphiques

# Git
git log --oneline                  # Historique
git tag v1.0.0                     # Créer un tag
```

---

## 📊 Structure du projet

```
mlops-mlflow-tp/
├── data/                          # Données (versionné avec DVC)
│   ├── housing_data.csv
│   ├── credit_data.csv
│   └── churn_data.csv
├── models/                        # Modèles entraînés
│   └── model.pkl
├── mlruns/                        # MLflow tracking
├── notebooks/                     # Notebooks d'exploration
├── reports/                       # Rapports Evidently
├── .github/
│   └── workflows/
│       └── mlops-pipeline.yml    # GitHub Actions
├── generate_data.py              # Génération de datasets
├── train.py                      # Entraînement avec MLflow
├── tune_hyperparameters.py       # Optimisation Optuna
├── detect_drift.py               # Détection drift Evidently
├── compare_results.py            # Comparaison des modèles
├── dvc.yaml                      # Pipeline DVC
├── dvc.lock                      # Lock file DVC
├── requirements.txt              # Dépendances Python
├── DOCUMENTATION.md              # Ce fichier
└── RESULTS.md                    # Résultats et analyses
```

---

## 🎓 Concepts MLOps illustrés

1. **Reproductibilité**: DVC + MLflow garantissent la reproduction exacte des expériences
2. **Versioning**: Code (Git), Données (DVC), Modèles (MLflow)
3. **Automatisation**: GitHub Actions pour CI/CD
4. **Monitoring**: Evidently pour détecter les drifts
5. **Optimisation**: Optuna pour le tuning automatique
6. **Collaboration**: Tracking centralisé des expériences

---

## 📚 Ressources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [DVC Documentation](https://dvc.org/doc)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Evidently Documentation](https://docs.evidentlyai.com/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

---

**Date de création**: Janvier 2026  
**Auteur**: Wissem Harhouri  
**Version**: 1.0.0
