# Scripts et commandes

Cette page sert de reference pratique pour executer le projet.

## Point d'entree principal

Le script central est :

- `run_v3.py`

Il orchestre :

- l'ingestion ;
- les transformations ;
- la construction des indices ;
- l'assemblage gold ;
- le backtest ;
- les metriques warning.

## Commandes principales

### Pipeline complet

```bash
python run_v3.py
```

### Pipeline complet sur la plage finale 2017-2024

```bash
python run_v3.py --skip-ingest --start-date 2017-01-01
```

### Ingestion uniquement

```bash
python run_v3.py --step ingest
```

### Bronze -> Silver

```bash
python run_v3.py --step transform
```

### Construction des indices BESI

```bash
python run_v3.py --step indexes
```

### Assemblage Gold

```bash
python run_v3.py --step gold
```

### Backtest walk-forward

```bash
python run_v3.py --step backtest
```

### Metriques early warning

```bash
python run_v3.py --step warnings
```

## Modules importants

### Ingestion

- `src/ingestion/fao.py`
- `src/ingestion/bam_fx.py`
- `src/ingestion/google_trends_v3.py`
- `src/ingestion/cpi_hcp.py`

### Transformations

- `src/transforms/cpi.py`
- `src/transforms/trends.py`
- `src/transforms/macro.py`

### Features et Gold

- `src/features/indexes.py`
- `src/gold/build_model_dataset.py`

### Evaluation

- `src/evaluation/backtest.py`
- `src/evaluation/warning_metrics.py`

### NLP

- `src/nlp/scraper_hespress.py`
- `src/nlp/preprocess_darija.py`
- `src/nlp/sentiment_scorer.py`
- `src/nlp/besi_v2.py`

## Notebooks

Les notebooks V3 servent d'appui analytique et de visualisation :

- `notebooks/01_exploration_v3.ipynb`
- `notebooks/02_modeling_v3.ipynb`
- `notebooks/03_analysis_v3.ipynb`
- `notebooks/04_results_v3.ipynb`

## Questions auxquelles cette page repond

- Quelle commande lancer pour chaque etape ?
- Quels scripts contiennent la logique principale ?
- Quels notebooks accompagnent l'analyse finale ?

