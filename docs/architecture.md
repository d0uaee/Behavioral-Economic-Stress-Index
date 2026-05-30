# Architecture du projet

L'une des contributions importantes de **BESI V3** est d'avoir transforme un
ensemble de scripts en une **pipeline de donnees lisible, traçable et
reproductible**.

## Vue d'ensemble

```text
Bronze  ->  Silver  ->  Gold  ->  Evaluation
raw         clean       model      rapports et figures
```

## Bronze

Le niveau **Bronze** contient les donnees brutes, telles que recuperees ou
deposees localement.

Exemples :

- IPC HCP brut
- Google Trends brut
- FAO brut
- taux MAD/EUR brut
- corpus Hespress brut

Principe :

- on ne re-ecrit pas la source ;
- on conserve la granularite et la trace d'origine ;
- on prepare la reproductibilite des traitements aval.

## Silver

Le niveau **Silver** contient les series nettoyees et harmonisees.

Exemples :

- `cpi_monthly.csv`
- `google_trends_monthly.csv`
- `macro_signals_monthly.csv`
- `behavioral_index_pure.csv`
- `hybrid_macro_index.csv`
- `sentiment_monthly.csv`

Principe :

- une ligne par mois ;
- conventions communes de colonnes ;
- dates alignees sur la frequence mensuelle ;
- transformations explicites et lisibles.

## Gold

Le niveau **Gold** contient le dataset final de modelisation :

- lags explicites ;
- cibles `t+1` ;
- metadonnees temporelles ;
- labels de split d'evaluation.

Fichier central :

- `data/gold/model_dataset_monthly.csv`

## Evaluation

Le niveau **Evaluation** contient :

- les backtests walk-forward ;
- les metriques de warning ;
- les rapports de synthese ;
- les figures finales.

Exemples :

- `outputs/reports/backtest_v3_summary.csv`
- `outputs/reports/warning_metrics_v3.csv`
- `outputs/reports/results_v3_final.md`

## Pourquoi cette architecture est importante

Cette structure apporte :

- une **separation nette des responsabilites** ;
- une meilleure **maintenabilite** ;
- une **traçabilite** des transformations ;
- une documentation plus facile ;
- un projet plus credible devant un jury technique.

## Organisation du code

Le code source suit la meme logique :

- `src/ingestion/`
- `src/transforms/`
- `src/features/`
- `src/gold/`
- `src/evaluation/`
- `src/nlp/`

## Questions auxquelles cette page repond

- Comment le projet est-il organise ?
- Pourquoi `bronze -> silver -> gold` ?
- Quels dossiers consulter selon le type de question ?

