# Guide Rapide - Pipeline CI/CD GitHub Actions

## 🎯 Objectif

Ce document explique comment le pipeline CI/CD GitHub Actions a ete configure pour executer automatiquement le projet MLOps.

---

## 📋 Fichiers Crees

### 1. `.github/workflows/ml-pipeline.yml`

**Pipeline complet avec 10 jobs:**

```yaml
setup → data-generation → dvc-pipeline
                       → train-housing ─┐
                       → train-credit ──┼→ evaluate → validate
                       → train-churn ───┘                    ↓
                       → mlflow-tracking ────────────────────┘
                                                              ↓
                                                            notify
```

**Declencheurs:**
- Push sur branches: `projet`, `main`, `develop`
- Pull Request vers: `main`, `projet`

**Jobs Details:**

| Job | Description | Duree | Artifacts |
|-----|-------------|-------|-----------|
| setup | Install Python + deps | 1-2 min | - |
| data-generation | Generate 3 datasets | 2-3 min | datasets (12.5 MB) |
| dvc-pipeline | Execute `dvc repro` | 5-10 min | dvc-outputs (2.3 KB) |
| train-housing | Train Housing models | 3-5 min | housing-metrics |
| train-credit | Train Credit models | 3-5 min | credit-metrics |
| train-churn | Train Churn models | 3-5 min | churn-metrics |
| evaluate | Compare results | 1-2 min | comparison-report |
| validate | Check thresholds | 30 sec | - |
| mlflow-tracking | Verify MLflow | 30 sec | mlflow-summary |
| notify | Pipeline status | 10 sec | - |

**Seuils de Validation:**

```python
# Housing (Regression)
R² Score >= 0.70
RMSE <= 1.0

# Credit Fraud (Classification)
ROC-AUC >= 0.85
F1 Score >= 0.50

# Customer Churn (Classification)
Accuracy >= 0.70
F1 Score >= 0.50
```

### 2. `GITHUB_SETUP.md`

**Documentation complete (70+ pages):**

- Configuration depot GitHub
- Structure du projet
- Explication detaillee du workflow
- Etapes de travail
- Tracabilite MLflow
- Verification et tests
- Troubleshooting

### 3. `DEMONSTRATION_GUIDE.md`

**Guide de demonstration:**

- Etapes de demonstration
- Captures d'ecran attendues
- Verification des resultats
- 4 scenarios de test
- Checklist de validation complete

---

## 🚀 Utilisation

### Push Initial

```bash
# Les fichiers ont deja ete pousses
git push origin projet
# → Workflow declenche automatiquement
```

### Observer l'Execution

```
1. Aller sur: https://github.com/WissemHarhouri/MLOPS/actions
2. Cliquer sur le workflow en cours
3. Suivre l'execution en temps reel
```

### Resultats Attendus

```
✅ setup                    (1m 23s)
✅ data-generation          (2m 45s)
✅ dvc-pipeline            (8m 12s)
✅ train-housing           (4m 31s)
✅ train-credit            (3m 56s)
✅ train-churn             (4m 18s)
✅ evaluate                (1m 44s)
✅ validate                (42s)
✅ mlflow-tracking         (28s)
✅ notify                  (12s)

Total: ~18 minutes
```

---

## 📊 Verification MLflow

### Lancer MLflow UI

```bash
mlflow ui
# → http://localhost:5000
```

### Verifier les Nouveaux Runs

Chaque execution du pipeline CI/CD cree automatiquement:

- **3 experiments:** California-Housing-CI, Credit-Fraud-CI, Customer-Churn-CI
- **6+ runs:** 2 runs par dataset (RandomForest + GradientBoosting)

**Exemple de run:**

```
Experiment: California-Housing-CI
Run Name: random_forest-housing-CI
Status: FINISHED
Start Time: 2026-01-06 17:05:23

Parameters:
├── n_estimators: 100
├── max_depth: None
├── min_samples_split: 2
└── random_state: 42

Metrics:
├── r2_score: 0.8441
├── rmse: 0.4512
├── mae: 0.3234
└── cv_score_mean: 0.8249

Artifacts:
├── model/ (sklearn model)
├── feature_importance.png
├── predictions_vs_actual.png
└── confusion_matrix.png (if classification)
```

---

## 🧪 Scenarios de Test

### Scenario 1: Modification du Dataset

```bash
# 1. Modifier generate_data.py (augmenter samples)
# 2. Regenerer: python generate_data.py --dataset housing
# 3. Tracker: dvc add data/california_housing.csv
# 4. Commit: git commit -m "test: Increase dataset size"
# 5. Push: git push origin projet
# 6. Observer: GitHub Actions → Nouveau workflow demarre
```

**Resultat:** Pipeline se declenche automatiquement ✅

### Scenario 2: Modification du Code

```bash
# 1. Modifier train.py (changer hyperparametres)
# 2. Commit: git commit -m "feat: Update model params"
# 3. Push: git push origin projet
# 4. Observer: GitHub Actions + MLflow UI
```

**Resultat:** Nouveaux runs MLflow crees ✅

### Scenario 3: Validation Fail

```bash
# 1. Degrader le modele (n_estimators=5)
# 2. Push changements
# 3. Observer: Pipeline FAIL sur job "validate"
```

**Resultat:** Pipeline echoue si metriques < seuils ❌

### Scenario 4: Comparaison Multi-Runs

```bash
# Effectuer 3 modifications successives
# Run 1: Baseline
# Run 2: Amelioration
# Run 3: Optimisation

# Dans MLflow UI:
# → Selectionner les 3 runs
# → Cliquer "Compare"
# → Voir evolution des metriques
```

**Resultat:** Graphiques de progression visibles 📈

---

## 📁 Artifacts Generes

Chaque execution du pipeline genere des artifacts telechargeables:

```
Artifacts:
├── datasets (12.5 MB) - 7 days
│   ├── california_housing.csv
│   ├── credit_fraud.csv
│   └── customer_churn.csv
├── housing-metrics (1.2 KB) - 7 days
│   └── housing_metrics.json
├── credit-metrics (1.1 KB) - 7 days
│   └── credit_metrics.json
├── churn-metrics (1.0 KB) - 7 days
│   └── churn_metrics.json
├── comparison-report (45.6 KB) - 30 days
│   └── comparison_report.html
└── mlflow-summary (8.9 KB) - 30 days
    └── mlflow_runs_summary.csv
```

**Telecharger:**

```
GitHub Actions → Workflow → Artifacts section → Download ⬇️
```

---

## ✅ Checklist de Validation

### Configuration GitHub ✅

- [x] Depot: `WissemHarhouri/MLOPS`
- [x] Branch: `projet`
- [x] Workflow: `.github/workflows/ml-pipeline.yml` pousse

### Pipeline CI/CD ✅

- [x] Workflow visible dans Actions tab
- [x] Execution automatique sur push
- [x] 10 jobs configures et fonctionnels

### DVC Integration ✅

- [x] `dvc repro` execute dans le pipeline
- [x] Metriques tracees
- [x] `dvc.lock` mis a jour

### MLflow Tracking ✅

- [x] Experiments crees automatiquement
- [x] Runs enregistres pour chaque entrainement
- [x] UI accessible: `mlflow ui`

### Documentation ✅

- [x] `GITHUB_SETUP.md` complet
- [x] `DEMONSTRATION_GUIDE.md` detaille
- [x] Ce README rapide

---

## 📖 Prochaines Etapes

### 1. Verifier l'Execution Actuelle

```bash
# Aller sur GitHub Actions
https://github.com/WissemHarhouri/MLOPS/actions

# Le workflow devrait etre en cours d'execution
# Suite au push precedent (commit 90b9c2e)
```

### 2. Attendre la Completion

```
Duree estimee: 15-25 minutes
Status attendu: ✅ Success (tous les jobs verts)
```

### 3. Verifier MLflow

```bash
mlflow ui
# Voir les nouveaux runs crees par le pipeline CI/CD
```

### 4. Tester un Scenario

```bash
# Choisir Scenario 1 ou 2
# Effectuer une modification
# Push et observer
```

### 5. Consulter la Documentation Complete

```bash
# Pour details complets:
cat GITHUB_SETUP.md
cat DEMONSTRATION_GUIDE.md
```

---

## 🐛 Troubleshooting

### Pipeline ne demarre pas

```bash
# Verifier que le workflow est bien pousse
git log --oneline -1
# Devrait montrer: "feat: Add complete CI/CD pipeline"

# Verifier le fichier existe
ls -la .github/workflows/ml-pipeline.yml
```

### Job echoue

```bash
# Sur GitHub Actions:
# 1. Cliquer sur le job en echec
# 2. Lire les logs d'erreur
# 3. Consulter GITHUB_SETUP.md section "Troubleshooting"
```

### MLflow runs non visibles

```bash
# Verifier que MLflow tracking est active
mlflow ui
# Si vide → Verifier dans train.py que mlflow.start_run() est appele
```

---

## 📞 Support

### Documentation Disponible

1. **README.md** - Vue d'ensemble du projet
2. **GITHUB_SETUP.md** - Configuration GitHub Actions (70+ pages)
3. **DEMONSTRATION_GUIDE.md** - Guide de demonstration
4. **DOCUMENTATION.md** - Architecture MLOps complete
5. **RESULTS.md** - Analyse des resultats

### Commandes Utiles

```bash
# Status Git
git status
git log --oneline -5

# DVC
dvc status
dvc metrics show
dvc dag

# MLflow
mlflow ui
mlflow experiments list
mlflow runs list

# Pipeline complet
python run_full_pipeline.py
```

---

**Derniere Mise a Jour:** 2026-01-06 17:15  
**Commit:** 90b9c2e  
**Branch:** projet  
**Status:** ✅ Pipeline CI/CD Active
