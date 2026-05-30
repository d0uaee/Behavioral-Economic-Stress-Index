# Quickstart

Cette page fournit le parcours le plus court pour comprendre et executer la
version finale du projet.

## 1. Comprendre le point d'entree

Le projet s'execute a partir de :

```bash
python run_v3.py
```

L'orchestrateur supporte une execution complete ou etape par etape.

## 2. Cas recommande pour la version finale

La version finale exploite l'IPC HCP reel sur **2017-2024**. Pour cette plage,
la commande la plus simple est :

```bash
python run_v3.py --skip-ingest --start-date 2017-01-01
```

Cette commande :

- saute l'ingestion si les fichiers bronze sont deja presents ;
- active la plage 2017-2024 ;
- produit les jeux silver, gold et les rapports d'evaluation.

## 3. Execution etape par etape

### Transform Bronze -> Silver

```bash
python run_v3.py --step transform
```

### Construction des indices BESI

```bash
python run_v3.py --step indexes
```

### Construction du Gold dataset

```bash
python run_v3.py --step gold
```

### Backtest walk-forward

```bash
python run_v3.py --step backtest
```

### Metriques warning

```bash
python run_v3.py --step warnings
```

## 4. Artefacts principaux a consulter

- `data/gold/model_dataset_monthly.csv`
- `outputs/reports/backtest_v3_summary.csv`
- `outputs/reports/warning_metrics_v3.csv`
- `outputs/reports/results_v3_final.md`

## 5. Lecture recommandee apres execution

1. Lire [](results.md)
2. Ouvrir `outputs/figures/`
3. Consulter [](nlp_documentation.md) pour l'extension presse

## Questions auxquelles cette page repond

- Quelle est la commande minimale utile ?
- Comment relancer seulement une partie du pipeline ?
- Quels fichiers verifier en premier apres execution ?

