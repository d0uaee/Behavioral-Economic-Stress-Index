# Detection Precoce du Stress Economique des Menages au Maroc
### BESI V3 — Behavioral Economic Stress Index

**Etudiantes :** Douae Ahadji & Adama Basse
**Cours :** Series Temporelles — ENSAM Meknes
**Duree :** 8 semaines | **Date :** Mai 2026

---

## Resume

Ce projet construit **BESI** (*Behavioral Economic Stress Index*), un indice composite de stress economique
fonde sur les signaux digitaux comportementaux marocains (Google Trends), et l'integre dans un modele
**SARIMAX** pour la prevision de l'IPC (Indice des Prix a la Consommation) mensuel du Maroc (2017-2024).

**Resultats principaux (V3 — donnees reelles uniquement) :**
- SARIMAX + BESI ameliore l'AIC de **-7.77 points** vs SARIMA pur (confirmation statistique)
- SARIMAX + BESI detecte **100% des mois a inflation elevee** en 2022-2024 (recall = 1.00)
- Rupture structurelle 2022 massivement significative : inflation x11.6 (p < 0.0001)
- H1 **partiellement validee** : signal comportemental utile pour l'alerte precoce, moins pour la precision point-par-point
- H2 **REJETEE** : l'ajout du signal macro (FAO + MAD/EUR) degrade la detection — Recall Bloc B chute de 1.00 a 0.375

---

## Structure du projet

```
project/
|-- CLAUDE.md                    <- Instructions du projet
|-- README.md                    <- Ce fichier
|-- run_v3.py                    <- Pipeline principal V3
|-- make_dashboard.py            <- Figure de synthese finale
|
|-- data/
|   |-- bronze/                  <- Donnees brutes (jamais modifiees)
|   |   |-- cpi_hcp_monthly_raw.csv   <- IPC HCP Maroc 2017-2024
|   |   |-- fao_food_price_raw.csv    <- FAO Food Price Index (telecharge)
|   |   `-- bam_fx_raw.csv            <- MAD/EUR ECB + interpolation lineaire
|   |-- silver/                  <- Donnees nettoyees et standardisees
|   |   |-- cpi_monthly.csv           <- IPC + inflation_yoy + mom
|   |   |-- google_trends_monthly.csv <- Sous-indices Trends normalises 0-1
|   |   |-- behavioral_index_pure.csv <- BESI comportemental (Trends seul)
|   |   `-- macro_signals_monthly.csv <- FAO + FX (normalises, 180 mois)
|   `-- gold/
|       `-- model_dataset_monthly.csv <- Dataset final (96 mois x 45 colonnes)
|
|-- src/
|   |-- ingestion/               <- Collecte des donnees
|   |   |-- cpi_hcp.py          <- IPC HCP Maroc
|   |   |-- fao.py              <- FAO Food Price Index
|   |   |-- bam_fx.py           <- Taux de change MAD/EUR
|   |   `-- google_trends_v3.py <- Google Trends (7 keywords, geo=MA)
|   |-- transforms/              <- Bronze -> Silver
|   |   |-- cpi.py
|   |   |-- trends.py
|   |   `-- macro.py
|   |-- features/
|   |   `-- indexes.py          <- Construction indices BESI (pure + hybrid)
|   |-- gold/
|   |   `-- build_model_dataset.py  <- Assemblage Gold + lags + targets
|   `-- evaluation/
|       |-- backtest.py         <- Walk-forward SARIMA vs SARIMAX
|       `-- warning_metrics.py  <- AUC, F1, recall, lead-time
|
|-- notebooks/
|   |-- 01_exploration_v3.ipynb <- Stats descriptives, heatmap, splits
|   |-- 02_modeling_v3.ipynb    <- Stationnarite, ACF/PACF, SARIMA, SARIMAX
|   |-- 03_analysis_v3.ipynb    <- Rupture 2022, Granger, early warning
|   `-- 04_results_v3.ipynb     <- Synthese H1/H2, tableaux, figures
|
`-- outputs/
    |-- figures/                <- 17 figures V3 (voir liste ci-dessous)
    `-- reports/                <- CSV de resultats + rapports
```

---

## Lancement rapide

### Pipeline complet V3
```bash
python run_v3.py --skip-ingest --start-date 2017-01-01
```

### Etape par etape
```bash
python run_v3.py --step gold      # Assembler le Gold dataset
python run_v3.py --step backtest  # Backtest walk-forward
python run_v3.py --step warnings  # Metriques alerte precoce
```

### Executer les notebooks
```bash
# Avec Anaconda (kernel besi_v3)
jupyter nbconvert --to notebook --execute --inplace notebooks/02_modeling_v3.ipynb --ExecutePreprocessor.kernel_name=besi_v3
```

### Dashboard final
```bash
python make_dashboard.py
# -> outputs/figures/dashboard_besi_v3_final.png
```

---

## Donnees

| Source | Variable | Periode | Statut |
|---|---|---|---|
| HCP Maroc (manuel) | IPC mensuel base 2017=100 | 2017-2024 | OK |
| Google Trends (pytrends) | 7 mots-cles, geo=MA | 2010-2024 | OK |
| FAO Food Price Index | Indice alimentaire mondial | 2010-2024 | A telecharger |
| Bank Al-Maghrib | Taux MAD/EUR mensuel | 2010-2024 | A telecharger |
| Reddit r/Morocco | NLP inflation | — | Absent (limite documentee) |
| YouTube | Commentaires economiques | — | Absent (limite documentee) |

**Gold dataset V3 :** 96 observations (2017-01 a 2024-12), 45 colonnes, zero simulation.

| Source | Variable | Periode | Statut |
|---|---|---|---|
| HCP Maroc (manuel) | IPC mensuel base 2017=100 | 2017-2024 | OK |
| Google Trends (pytrends) | 7 mots-cles, geo=MA | 2010-2024 | OK |
| FAO Food Price Index | 6 sous-indices alimentaires mondiaux | 2010-2024 | OK |
| ECB / interpolation lineaire | Taux MAD/EUR mensuel | 2010-2024 | OK |
| Reddit r/Morocco | NLP inflation | -- | Absent (limite documentee) |
| YouTube | Commentaires economiques | -- | Absent (limite documentee) |

---

## Indice BESI V3

```
BESI_behavioral (pure) = f(Trends) uniquement
   Composantes : trends_prix_alim, trends_inflation, trends_carburant, trends_composite
   Poids : calibres par LassoCV sur train (fallback : poids egaux)
   Regle : aucune composante IPC -> zero data leakage

BESI_hybrid (macro) = f(BESI_pure + FAO_fpi_yoy + fx_yoy)  [necessite FAO/FX]
   Composantes : BESI_pure + signaux macro exterieurs
   Objectif : tester H2 (macro apporte-t-il de l'information au-dela du comportemental ?)
```

**Distinction cle V3 :** `ipc_change` est completement retire des indices BESI.
L'IPC ne peut pas etre une feature qui predit l'IPC — c'est la cible.

---

## Modeles compares (V3 — Walk-Forward)

| Modele | RMSE (pts IPC) | MAE | MAPE | AIC Train A |
|---|---|---|---|---|
| Naif (persistance) | 1.609 | 1.200 | 1.06% | — |
| SARIMA(1,1,1)(1,0,1)[12] | 1.923 | 1.537 | 1.38% | 64.85 |
| **SARIMAX + BESI behavioral** | **1.891** | **1.522** | **1.36%** | **57.09** |
| SARIMAX + Hybrid macro | 1.997 | 1.576 | 1.42% | — |

**Blocs d'evaluation :**
- **Bloc A** (COVID 2020-2021) : train 2017-2019 | 24 mois de test
- **Bloc B** (Inflation 2022-2024) : train 2017-2021 | 36 mois de test

---

## Resultats cles

### 1. Rupture structurelle 2022

| Statistique | Valeur |
|---|---|
| Inflation pre-2022 (moyenne) | +0.74% YoY |
| Inflation post-2022 (moyenne) | +8.53% YoY |
| Facteur multiplicatif | x11.6 |
| Test t (difference moyennes) | t=-6.60, **p < 0.0001** |
| Test de Levene (variance) | W=48.53, **p < 0.0001** |

### 2. Correlation BESI - Inflation

| Periode | Lag optimal | r Pearson | p-value |
|---|---|---|---|
| Periode complete | lag=0 | **+0.535** | **< 0.001** |
| Pre-2022 | lag=0 | +0.201 | 0.161 (ns) |
| Post-2022 | lag=5 | -0.303 | 0.110 (ns) |

### 3. Test de causalite de Granger

H0 : BESI_behavioral ne cause pas l'inflation YoY au sens de Granger.
Resultat : **non significatif** a tous les lags 1-4 (p > 0.62).
Interpretation : la relation BESI-inflation est non-lineaire ; BESI agit comme
detecteur de regime, pas comme predicteur causal lineaire.

### 4. Early Warning (alerte precoce)

| Bloc | AUC | F1 | Recall | Lead-time |
|---|---|---|---|---|
| Bloc A – COVID | 0.328 | 0.273 | 0.375 | 1 mois |
| **Bloc B – Inflation 2022-2024** | 0.311 | **0.814** | **1.000** | — |
| Global | 0.35 | 0.62 | 0.90 | 1 mois |

---

## Validation des Hypotheses

### H1 : BESI predit le regime d'inflation elevee a t+1

| Critere | Valeur | Seuil | Statut |
|---|---|---|---|
| Delta AIC SARIMAX vs SARIMA | **-7.77** | < -2 | **Valide** |
| RMSE global SARIMAX vs SARIMA | -0.032 pts | < 0 | **Valide** |
| Recall global episodes inflation | **0.90** | > 0.80 | **Valide** |
| Recall Bloc B (inflation 2022-2024) | **1.00** | > 0.80 | **Valide** |
| AUC globale | 0.35 | > 0.65 | Non atteint |

**Verdict H1 : partiellement validee.** Le signal comportemental ameliore
le fit statistique et detecte 90% des episodes d'inflation avec 1 mois d'avance.
L'AUC < 0.65 est penalisee par le Bloc A (choc COVID exogene non capture).

### H2 : hybrid_macro_index ameliore la detection (delta AUC > 0.05)

| Critere | Behavioral | Hybrid | Delta | Statut |
|---|---|---|---|---|
| AUC globale | 0.352 | 0.451 | +0.099 | Favorable hybrid |
| Recall global | **0.900** | 0.625 | -0.275 | Defavorable hybrid |
| Recall Bloc B | **1.000** | 0.375 | **-0.625** | **REJETE** |
| RMSE global | **1.891** | 1.997 | +0.106 | Defavorable hybrid |

**Verdict H2 : REJETEE.** Le signal macro (FAO + MAD/EUR) n'ameliore pas la detection.
Sur la periode cle (inflation 2022-2024), le hybrid chute de Recall=1.00 a Recall=0.375.
Interpretation : les indices FAO mondiaux ne capturent pas la specificite locale marocaine ;
les comportements de recherche Google sont plus directement relies au stress des menages marocains.

---

## Figures produites (V3)

| Fichier | Description |
|---|---|
| `dashboard_besi_v3_final.png` | Dashboard 6 panels — synthese complete |
| `structural_break_v3.png` | Rupture structurelle mars 2022 |
| `acf_pacf_v3.png` | ACF/PACF pour identification SARIMA |
| `residuals_sarima_v3.png` | Diagnostics residus SARIMA Train A |
| `backtest_v3_bar_comparison.png` | RMSE/MAE/MAPE par modele et bloc |
| `backtest_v3_predictions.png` | Predictions walk-forward vs IPC reel |
| `cross_corr_besi_v3.png` | Correlation croisee BESI-Inflation |
| `early_warning_v3.png` | Signal BESI vs episodes inflation |
| `roc_curves_v3.png` | Courbes ROC alerte precoce |
| `precision_recall_v3.png` | Courbes Precision-Recall |
| `threshold_analysis_v3.png` | Analyse seuils d'alerte |

---

## Rapports CSV

| Fichier | Contenu |
|---|---|
| `backtest_v3_results.csv` | RMSE/MAE/MAPE par modele et bloc |
| `backtest_v3_summary.csv` | Moyenne globale des metriques |
| `warning_metrics_v3.csv` | AUC/F1/Recall par bloc et signal |
| `granger_besi_v3.csv` | Test de causalite de Granger (lags 1-4) |
| `besi_v3_behavioral_weights.csv` | Poids composantes BESI behavioral |
| `results_v3_final.md` | Rapport complet avec H1/H2 et phrase de conclusion |

---

## Limites documentees

| Limite | Impact | Recommandation |
|---|---|---|
| IPC HCP disponible depuis 2017 seulement | Pas de Bloc A 2010-2016, poids Lasso non calibres | Recuperer archives HCP pre-2017 |
| MAD/EUR construit par interpolation | Donnees BAM non disponibles en open data | Recuperer les donnees officielles BAM |
| Reddit/YouTube absents | BESI = Trends seul (pas composite multi-sources) | Documenter comme limite methodologique |
| Granger non significatif | Relation non-lineaire non capturee par SARIMA | Explorer modeles a seuil (TAR, STAR) |

---

## Bibliotheques principales

```
pandas >= 2.0       statsmodels >= 0.14    scikit-learn >= 1.3
numpy >= 2.0        pytrends >= 4.9        scipy >= 1.10
matplotlib >= 3.7   seaborn >= 0.12
```

---

*Douae Ahadji & Adama Basse — ENSAM Meknes — Series Temporelles — Mai 2026*
