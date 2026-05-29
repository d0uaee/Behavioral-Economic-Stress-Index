# BESI V3 - Detection precoce des regimes d'inflation au Maroc

**Auteurs :** Douae Ahadji & Adama Basse  
**Cadre :** ENSAM Meknes - Series Temporelles  
**Version :** V3 finalisee sur donnees reelles  
**Periode analysee :** 2017-01 a 2024-12

## Resume

Ce projet teste si des signaux digitaux marocains peuvent aider a detecter des
regimes d'inflation avant ou autour de la publication officielle de l'IPC.

Le coeur du projet est un indice comportemental, **BESI**, construit a partir de
Google Trends marocain. Ce signal est ensuite utilise dans un cadre SARIMAX et
evalue selon deux angles :

- la prevision quantitative de l'IPC
- la detection de regimes d'inflation elevee

La version V3 retire toute fuite directe de l'IPC hors de l'indice, utilise un
pipeline `bronze -> silver -> gold`, documente les limites, et conserve aussi
les resultats negatifs.

## Resultats principaux

- **Naif** reste le meilleur modele en RMSE global : `1.609`
- **SARIMAX + BESI behavioral** bat **SARIMA** en RMSE global : `1.891` vs `1.923`
- **SARIMAX + BESI behavioral** ameliore l'AIC de `-7.77` points vs SARIMA
- Le signal **behavioral** detecte `100%` des mois a inflation elevee sur le
  bloc 2022-2024 (`Recall test_B = 1.00`)
- Le signal **behavioral** obtient au global :
  - `AUC = 0.574`
  - `F1 = 0.703`
  - `Recall = 1.000`
- **H1 est partiellement validee**
- **H2 est rejetee**
- L'extension NLP Hespress est **CAS C** :
  - `alpha = 1.0`
  - `beta = 0.0`
  - le signal presse n'ajoute pas d'information conditionnelle supplementaire

## Conclusion honnete

Le projet montre qu'un signal comportemental digital marocain peut aider a
detecter des regimes d'inflation, surtout pendant le choc 2022-2024.

En revanche, il ne demontre pas encore une superiorite robuste en prevision
point par point face a une baseline naive. La contribution la plus forte du
projet est donc :

- une pipeline data propre
- une evaluation honnete
- une preuve partielle de valeur en early warning

## Question de recherche retenue

> Est-ce que des signaux comportementaux digitaux marocains peuvent aider a
> detecter des regimes d'inflation avant ou autour de la publication officielle
> de l'IPC ?

## Hypotheses

- **H1** : le BESI behavioral apporte une valeur informative utile pour la
  detection de regimes d'inflation
- **H2** : l'ajout d'un signal macro ameliore cette detection
- **Extension exploratoire** : un signal NLP presse marocaine ajoute-t-il une
  information complementaire ?

## Structure active du projet

Les elements V3 a utiliser en priorite sont :

```text
project/
|-- run_v3.py
|-- README.md
|-- docs/
|   |-- repo_guide.md
|   |-- v3_problem_statement.md
|   |-- data_dictionary_v3.md
|   `-- nlp_documentation.md
|-- data/
|   |-- bronze/
|   |-- silver/
|   `-- gold/model_dataset_monthly.csv
|-- src/
|   |-- ingestion/
|   |-- transforms/
|   |-- features/
|   |-- gold/
|   |-- evaluation/
|   `-- nlp/
|-- notebooks/
|   |-- 01_exploration_v3.ipynb
|   |-- 02_modeling_v3.ipynb
|   |-- 03_analysis_v3.ipynb
|   `-- 04_results_v3.ipynb
`-- outputs/reports/
    |-- backtest_v3_results.csv
    |-- backtest_v3_summary.csv
    |-- warning_metrics_v3.csv
    |-- results_v3_final.md
    |-- nlp_besi_comparison.csv
    `-- nlp_lasso_weights.csv
```

## Elements legacy conserves pour reference

Le repo contient encore des fichiers V1/V2 ou de soutenance intermediaire.
Ils sont utiles pour la trace du travail, mais ils ne doivent pas etre pris
comme source principale de verite pour la version finale.

Exemples :

- `DOCUMENTATION.md`
- `PRESENTATION_FINALE.md`
- `SCRIPT_ORAL.md`
- anciens notebooks hors prefixe `01_` a `04_`
- anciens scripts monolithiques dans `src/*.py`

## Donnees utilisees en V3

| Source | Variable | Periode | Statut |
|---|---|---|---|
| HCP Maroc | IPC mensuel base 2017=100 | 2017-2024 | OK |
| Google Trends | Signaux comportementaux | 2010-2024 | OK |
| FAO | Food Price Index et sous-indices | 2010-2024 | OK |
| Source externe documentee | Taux MAD/EUR mensuel | 2010-2024 | OK |
| Hespress API | Titres + extraits economiques | 2017-2024 | Extension NLP |
| Reddit | NLP inflation | -- | Non retenu |
| YouTube | Commentaires economiques | -- | Non retenu |

**Gold dataset V3 :**

- `96` observations mensuelles
- `45` colonnes
- `0` simulation sur le chemin principal

## Indices BESI

### BESI behavioral

Indice comportemental construit a partir de Google Trends uniquement.

- aucune composante IPC directe
- pas de `ipc_change`
- poids calibres par LassoCV quand possible
- fallback a des poids egaux uniquement sur colonnes disponibles

### BESI hybrid macro

Indice hybride combinant :

- BESI behavioral
- FAO Food Price Index
- taux de change MAD/EUR

Cet indice a ete garde pour tester H2, mais les resultats finaux sont
defavorables.

## Modeles compares

| Modele | RMSE | MAE | MAPE |
|---|---:|---:|---:|
| Naif | 1.609 | 1.200 | 1.06% |
| SARIMA | 1.923 | 1.537 | 1.38% |
| SARIMAX + BESI behavioral | 1.891 | 1.522 | 1.36% |
| SARIMAX + Hybrid macro | 1.997 | 1.576 | 1.42% |

## Metriques early warning

| Scope | Signal | AUC | F1 | Recall |
|---|---|---:|---:|---:|
| test_B | behavioral | 0.311 | 0.814 | 1.000 |
| global | behavioral | 0.574 | 0.703 | 1.000 |
| global | hybrid | 0.376 | 0.466 | 0.531 |

## Rupture structurelle 2022

| Statistique | Valeur |
|---|---|
| Inflation moyenne pre-2022 | +0.74% YoY |
| Inflation moyenne post-2022 | +8.53% YoY |
| Facteur multiplicatif | x11.6 |
| Significativite | p < 0.0001 |

## Extension NLP Hespress

Le module NLP a ete implemente comme extension exploratoire isolee.

- source : API WordPress Hespress
- signal : `title + excerpt`
- couverture : `96/96` mois
- corpus : `5 788` textes
- verdict : **CAS C**

Interpretation :

- le signal presse est proprement documente
- il enrichit l'analyse locale
- il n'ajoute pas de valeur predictive conditionnelle au BESI principal

## Execution

### Pipeline principal

```bash
python run_v3.py --skip-ingest --start-date 2017-01-01
```

### Etapes principales

```bash
python run_v3.py --step gold
python run_v3.py --step backtest
python run_v3.py --step warnings
```

## Documents a citer en soutenance

- [Problem statement](/C:/Users/ahadj/OneDrive/project/docs/v3_problem_statement.md)
- [Data dictionary](/C:/Users/ahadj/OneDrive/project/docs/data_dictionary_v3.md)
- [NLP documentation](/C:/Users/ahadj/OneDrive/project/docs/nlp_documentation.md)
- [Final results report](/C:/Users/ahadj/OneDrive/project/outputs/reports/results_v3_final.md)
- [Repository guide](/C:/Users/ahadj/OneDrive/project/docs/repo_guide.md)

## Message final a retenir

> Le projet ne presente pas un modele miracle de prevision de l'inflation.
> Il presente une pipeline rigoureuse et un signal comportemental marocain utile
> surtout pour la detection de regimes inflationnistes.
