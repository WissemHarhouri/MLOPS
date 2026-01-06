# 🚀 Guide de Démarrage Rapide - MLOps Pipeline

## En 5 minutes : Testez le projet complet !

### Étape 1: Installation (2 minutes)

```powershell
# Installer les dépendances
pip install -r requirements.txt

# Vérifier l'installation
python -c "import mlflow, sklearn, pandas; print('✓ Installation réussie!')"
```

### Étape 2: Générer les données (1 minute)

```powershell
# Générer les 3 datasets
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

Generating Credit Card Fraud dataset...
✓ Credit Fraud dataset saved: 10000 rows, 31 columns
  Target variable: Class (0=legitimate, 1=fraud)
  Fraud rate: 0.20%

Generating Customer Churn dataset...
✓ Customer Churn dataset saved: 7043 rows, 21 columns
  Target variable: Churn (Yes/No)
  Churn rate: 26.54%
```

### Étape 3: Entraîner un modèle (1 minute)

```powershell
# Entraîner le modèle California Housing
python train.py --dataset california_housing --model random_forest
```

**Sortie attendue:**
```
============================================================
MLOps Training Pipeline with MLflow
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

✓ Model logged to MLflow
✓ Metrics saved to metrics/metrics_california_housing_random_forest.json
```

### Étape 4: Visualiser dans MLflow (30 secondes)

```powershell
# Lancer l'interface MLflow
mlflow ui
```

Ouvrez votre navigateur sur: **http://localhost:5000**

Vous verrez:
- 📊 Toutes vos expériences
- 📈 Graphiques de comparaison
- 🎯 Paramètres et métriques
- 📦 Modèles enregistrés

### Étape 5: Tester une fonctionnalité avancée (30 secondes)

```powershell
# Optimiser les hyperparamètres avec Optuna (version courte: 10 trials)
python tune_hyperparameters.py --dataset california_housing --n-trials 10
```

---

## 📋 Workflows complets

### Workflow 1: Comparaison de datasets

```powershell
# 1. Générer tous les datasets
python generate_data.py --dataset all

# 2. Entraîner sur chaque dataset
python train.py --dataset california_housing --model random_forest
python train.py --dataset credit_fraud --model random_forest
python train.py --dataset customer_churn --model random_forest

# 3. Comparer les résultats
python compare_results.py

# 4. Ouvrir le rapport
start reports/comparison_report.html  # Windows
```

### Workflow 2: Optimisation et Monitoring

```powershell
# 1. Optimiser les hyperparamètres
python tune_hyperparameters.py --dataset california_housing --n-trials 50

# 2. Détecter le drift entre datasets
python detect_drift.py --compare-datasets housing churn

# 3. Voir les rapports
start reports/data_drift_report_*.html
```

### Workflow 3: Pipeline DVC complet

```powershell
# 1. Initialiser DVC (si pas déjà fait)
dvc init

# 2. Ajouter les données au versioning
dvc add data/housing_data.csv
git add data/housing_data.csv.dvc .gitignore
git commit -m "Add housing data to DVC"

# 3. Exécuter le pipeline complet
dvc repro

# 4. Visualiser le DAG
dvc dag

# 5. Voir les métriques
dvc metrics show
```

---

## 🎯 Commandes essentielles par outil

### MLflow

```powershell
# Interface web
mlflow ui

# Lister les expériences
mlflow experiments list

# Chercher des runs
mlflow runs search --experiment-name "california_housing_random_forest"

# Servir un modèle
mlflow models serve -m models:/california_housing_random_forest/1 -p 5001
```

### DVC

```powershell
# Statut du pipeline
dvc status

# Exécuter le pipeline
dvc repro

# Visualiser le graphe
dvc dag

# Afficher les métriques
dvc metrics show

# Comparer des métriques
dvc metrics diff
```

### Tests

```powershell
# Exécuter tous les tests
pytest tests/ -v

# Avec couverture
pytest tests/ --cov=. --cov-report=html

# Test spécifique
pytest tests/test_pipeline.py::test_housing_data_structure -v
```

---

## 🔧 Troubleshooting

### Problème: MLflow UI ne démarre pas

```powershell
# Solution: Vérifier le port
mlflow ui --port 5001
```

### Problème: Import errors

```powershell
# Solution: Réinstaller les dépendances
pip install -r requirements.txt --upgrade
```

### Problème: DVC cache issues

```powershell
# Solution: Nettoyer le cache
dvc cache dir
dvc gc --workspace
```

### Problème: Mémoire insuffisante

```powershell
# Solution: Réduire le nombre d'arbres
python train.py --dataset california_housing --model random_forest
# Modifier dans train.py: n_estimators=50 au lieu de 100
```

---

## 📊 Résultats attendus

Après avoir exécuté tous les workflows:

### Fichiers générés

```
mlops-mlflow-tp/
├── data/
│   ├── housing_data.csv        ✓ Généré
│   ├── credit_data.csv         ✓ Généré
│   └── churn_data.csv          ✓ Généré
├── mlruns/                     ✓ Expériences MLflow
├── metrics/
│   ├── metrics_california_housing_random_forest.json  ✓
│   ├── metrics_credit_fraud_random_forest.json        ✓
│   └── metrics_customer_churn_random_forest.json      ✓
├── reports/
│   ├── comparison_report.html              ✓
│   ├── performance_comparison.png          ✓
│   ├── data_drift_report_*.html            ✓
│   └── data_quality_report_*.html          ✓
└── optuna_results/
    └── california_housing_optimization.json  ✓
```

### Métriques de référence

| Dataset | Modèle | Métrique clé | Score attendu |
|---------|--------|--------------|---------------|
| California Housing | RF | R² | ~0.81 |
| Credit Fraud | RF | ROC-AUC | ~0.98 |
| Customer Churn | RF | Accuracy | ~0.79 |

---

## 🎓 Prochaines étapes

Maintenant que vous avez testé le projet:

1. **Explorer MLflow UI** (http://localhost:5000)
   - Comparer les runs
   - Analyser les graphiques
   - Télécharger les modèles

2. **Lire la documentation complète**
   - [DOCUMENTATION.md](DOCUMENTATION.md) - Architecture détaillée
   - [RESULTS.md](RESULTS.md) - Analyses approfondies

3. **Expérimenter**
   - Modifier les hyperparamètres
   - Tester d'autres modèles
   - Créer vos propres datasets

4. **Contribuer**
   - Ajouter de nouveaux modèles
   - Améliorer les visualisations
   - Partager vos résultats

---

## 💡 Tips & Tricks

### Accélérer les entraînements

```powershell
# Utiliser moins d'arbres pour tester rapidement
python train.py --dataset california_housing --model random_forest
# Puis modifier n_estimators=50 dans train.py
```

### Sauvegarder vos expériences

```powershell
# Créer un tag Git pour marquer une version
git tag -a v1.0 -m "Baseline models"
git push origin v1.0
```

### Comparer deux versions

```powershell
# Avec DVC
dvc metrics diff v1.0 v1.1

# Avec MLflow
# Utiliser l'interface web pour comparer visuellement
```

---

## 📞 Aide

Si vous rencontrez des problèmes:

1. Vérifiez les [Issues GitHub](https://github.com/WissemHarhouri/MLOPS/issues)
2. Consultez la [Documentation](DOCUMENTATION.md)
3. Créez une nouvelle issue avec:
   - Version Python
   - Message d'erreur complet
   - Commande exécutée

---

**Bon MLOps! 🚀**

*Temps total du quick start: ~5 minutes*  
*Temps pour workflow complet: ~15 minutes*
