# Repository guide - BESI V3

Ce document sert a lire rapidement le **repo final de remise** et a retrouver
les fichiers les plus utiles sans parcourir tout le depot.

## 1. Entree principale

Le point d'entree principal du projet final est :

- `run_v3.py`

## 2. Ce qui est actif pour la version finale

### Documentation

- `README.md`
- `docs/v3_problem_statement.md`
- `docs/data_dictionary_v3.md`
- `docs/nlp_documentation.md`
- `outputs/reports/results_v3_final.md`

### Donnees

- `data/bronze/`
- `data/silver/`
- `data/gold/model_dataset_monthly.csv`

### Code

- `src/ingestion/`
- `src/transforms/`
- `src/features/`
- `src/gold/`
- `src/evaluation/`
- `src/nlp/`

### Notebooks

- `notebooks/01_exploration_v3.ipynb`
- `notebooks/02_modeling_v3.ipynb`
- `notebooks/03_analysis_v3.ipynb`
- `notebooks/04_results_v3.ipynb`

### Rapports cles

- `outputs/reports/backtest_v3_summary.csv`
- `outputs/reports/backtest_v3_results.csv`
- `outputs/reports/warning_metrics_v3.csv`
- `outputs/reports/nlp_besi_comparison.csv`
- `outputs/reports/nlp_lasso_weights.csv`

## 3. Regle pratique

Si une information est contradictoire dans le repo, la priorite de lecture est :

1. `outputs/reports/results_v3_final.md`
2. `README.md`
3. `outputs/reports/*.csv`
4. `docs/*`

En cas de divergence, les rapports V3 finaux et les CSV de sortie priment sur
les formulations plus anciennes.
