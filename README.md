# Détection Précoce du Stress Économique des Ménages au Maroc
### BESI — Behavioral Economic Stress Index

**Étudiantes :** Douae Ahadji & Adama Basse  
**Cours :** Séries Temporelles — ENSAM Meknès  
**Durée :** 8 semaines | **Date :** Mai 2026

---

## Résumé

Ce projet construit **BESI** (*Behavioral Economic Stress Index*), un indice composite de stress économique fondé sur les signaux digitaux comportementaux marocains (Google Trends, Reddit, YouTube, NLP presse marocaine), et l'intègre dans un modèle **SARIMAX** pour la prévision de l'IPC (Indice des Prix à la Consommation) mensuel du Maroc (2010-2024).

**Résultat principal :** SARIMA(2,1,1)x(0,1,1)[12] reste le meilleur modèle en validation walk-forward (RMSE = 0.00272), mais BESI_trends améliore l'AIC de -502 a -546, confirmant une information statistique supplémentaire. BESI détecte le stress économique **~12 mois avant l'IPC**, validant partiellement l'hypothèse H1 sur un horizon macroéconomique.

---

## Structure du projet

```
project/
├── CLAUDE.md                  <- Instructions du projet
├── README.md                  <- Ce fichier
├── requirements.txt           <- Dépendances Python
├── run_all.py                 <- Pipeline complet (7 étapes)
├── run_v2.py                  <- Comparaison modèles v2 + LSTM + Prophet
│
├── data/
│   ├── raw/                   <- Données brutes téléchargées
│   ├── processed/
│   │   └── master_dataset.csv <- Dataset aligné (180 obs, 11 colonnes)
│   └── ipc_maroc.csv          <- IPC mensuel HCP (variable cible)
│
├── src/
│   ├── data_pipeline.py       <- Collecte IPC + Google Trends + BESI
│   ├── features.py            <- Feature selection, lag analysis, BESI composite
│   ├── models.py              <- SARIMA, SARIMAX, walk-forward validation
│   ├── analysis.py            <- Rupture structurelle, Granger, early warning
│   ├── deep_learning.py       <- LSTM + comparaison tailles de fenêtre
│   ├── prophet_model.py       <- Modèle Prophet (prévision IPC)
│   ├── nlp_morocco.py         <- NLP presse marocaine + YouTube (Darija/Arabe)
│   └── visualization.py       <- Dashboard + figures
│
├── notebooks/
│   ├── exploration.ipynb      <- Analyse exploratoire, stationnarité, ACF/PACF
│   ├── modeling.ipynb         <- SARIMA, SARIMAX, walk-forward
│   ├── analysis.ipynb         <- Rupture 2022, Granger, early warning
│   └── results.ipynb          <- Synthèse finale, table de comparaison complète
│
└── outputs/
    ├── figures/               <- 43 figures PNG
    ├── models/                <- Modèles sauvegardés (.pkl)
    └── reports/               <- CSV de résultats + rapports texte
```

---

## Installation

```bash
# Ouvrir le dossier projet
cd project/

# Installer les dépendances (Anaconda recommandé)
pip install -r requirements.txt
```

**Python requis :** 3.10+  
**Recommandé :** Anaconda

---

## Lancement rapide

### Pipeline complet (toutes les étapes)
```bash
python run_all.py
```

### Comparaison modèles v2 + LSTM + Prophet
```bash
python run_v2.py
```

### Options CLI de run_all.py
```bash
python run_all.py --skip-data    # Sauter le téléchargement des données
python run_all.py --skip-dl      # Sauter LSTM (long a entrainer)
python run_all.py --skip-nlp     # Sauter le scraping NLP
python run_all.py --step 5       # Démarrer a l'étape 5
```

---

## Données

| Source | Variable | Période | Fréquence |
|--------|----------|---------|-----------|
| HCP Maroc / World Bank | IPC mensuel (variable cible) | 2010-2024 | Mensuelle |
| Google Trends (pytrends) | 7 mots-clés économiques, geo=MA | 2010-2024 | Mensuelle |
| Reddit r/Morocco (praw) | Sentiments inflation/prix | 2015-2024 | Mensuelle |
| YouTube Data API v3 | Commentaires vidéos économiques | 2018-2024 | Mensuelle |
| Presse marocaine (NLP) | Hespress, Le360, Medias24, L'Économiste | 2020-2024 | Mensuelle |

**master_dataset.csv** contient 180 observations et 11 colonnes :  
`ipc`, `trends_composite`, `reddit_composite`, `youtube_composite`, `besi`, `ipc_yoy`, `ipc_change`, `ipc_mom`, `stress_level`, `besi_trends`, `besi_enrichi`

---

## Indice BESI

```
BESI composite  = 0.40 x Trends + 0.30 x Reddit + 0.20 x YouTube + 0.10 x |dIPC|
BESI_trends     = 0.70 x Trends + 0.30 x |dIPC|   (version robuste, sans données simulées)
BESI_enrichi    = 0.35 x Trends + 0.25 x NLP_Maroc + 0.20 x YouTube + 0.10 x Reddit + 0.10 x |dIPC|
```

Tous les signaux sont normalisés 0-1 avant pondération.

---

## Modèles comparés

| Modèle | RMSE (Walk-Forward) | MAE | MAPE | AIC |
|--------|---------------------|-----|------|-----|
| Naïf (Random Walk) | 0.00409 | 0.00339 | 0.28% | — |
| **SARIMA(2,1,1)x(0,1,1)[12]** | **0.00272** | **0.00232** | **0.19%** | -502.6 |
| SARIMAX + Trends | 0.00327 | 0.00253 | 0.21% | -515.5 |
| SARIMAX + BESI_trends | 0.00304 | 0.00241 | 0.20% | **-545.9** |
| LSTM (window=12) | 0.02110 | 0.01743 | 1.46% | — |
| Prophet | voir outputs/reports/prophet_results.csv | — | — | — |

**Période test :** 2022-01 -> 2024-12 (36 mois, walk-forward h=1)  
**Période train :** 2015-01 -> 2021-12

---

## Résultats clés

### 1. Rupture structurelle 2022
- Test de Chow : rupture confirmée (p < 0.05) au 2022-01-01
- CUSUM : dépassement détecté mi-2022
- Coefficients pré/post-2022 : variations de 15-50% selon la variable

### 2. Lead time BESI
- Corrélation croisée optimale au **lag 12 mois**
- Causalité de Granger significative aux lags 1-3 (p < 0.05)
- BESI anticipe l'inflation de ~12 mois (horizon macroéconomique)

### 3. Early Warning
- Rappel = 100% (TP=1, FN=0) — sur un seul événement détecté
- Précision = 7.7% (FP=12) — 12 fausses alertes pour 1 vraie
- F1-Score = 0.14 (pas opérationnel — seuil à optimiser en v3)
- **Note v3** : résultats basés sur données partiellement simulées (Reddit/YouTube).
  La v3 recalibrera sur données réelles uniquement.

### 4. Performances par sous-période

| Modèle | 2022 (choc) RMSE | 2023-2024 (post) RMSE |
|--------|------------------|-----------------------|
| SARIMA | **0.00197** | 0.00303 |
| SARIMAX_BT | 0.00362 | **0.00271** |

SARIMA domine pendant le choc ; SARIMAX_BT légèrement meilleur apres.

---

## Hypothèse H1

> *"Les signaux comportementaux digitaux (Google Trends, Reddit, YouTube) intégrés dans BESI permettent de détecter le stress économique des ménages marocains 1 a 2 mois avant l'IPC du HCP."*

**Verdict : H1 partiellement rejetée** — Le lead time détecté est de ~12 mois (non 1-2 mois). BESI offre un signal macroéconomique robuste sur un horizon de moyen terme, utile pour les politiques contracycliques (crédit, subventions).

---

## Contribution originale

1. **BESI** : Premier indice composite marocain fusionnant signaux digitaux comportementaux en Darija, Arabe et Français
2. **Pipeline NLP multilingue** : scoring sur presse marocaine + YouTube avec dictionnaire de 80+ mots-clés économiques
3. **Analyse de rupture** : Test de Chow + CUSUM sur le choc inflationniste 2022
4. **Comparaison exhaustive** : 7 modèles (Naïf, SARIMA, SARIMAX x2, LSTM x2, Prophet) sur même période test

---

## Fichiers de résultats importants

| Fichier | Contenu |
|---------|---------|
| `outputs/reports/final_model_comparison_all.csv` | Table complète tous modèles |
| `outputs/reports/model_comparison_v2.csv` | SARIMA vs SARIMAX walk-forward |
| `outputs/reports/period_performance_v2.csv` | Métriques par sous-période |
| `outputs/reports/early_warning_events.csv` | Événements d'alerte précoce |
| `outputs/reports/prophet_results.csv` | Métriques Prophet |
| `outputs/figures/compare_all_predictions_v2.png` | Graphique prédictions |
| `outputs/figures/nb04_dashboard.png` | Dashboard final (6 panneaux) |

---

## Bibliothèques principales

```
pandas >= 2.0       statsmodels >= 0.14    scikit-learn >= 1.3
numpy >= 1.24       pytrends >= 4.9        tensorflow >= 2.13
matplotlib >= 3.7   praw >= 7.7            prophet >= 1.1.4
seaborn >= 0.12     vaderSentiment >= 3.3  beautifulsoup4 >= 4.12
scipy >= 1.10       requests >= 2.31       pmdarima >= 2.0
```

---

*Douae Ahadji & Adama Basse — ENSAM Meknès — Séries Temporelles — Mai 2026*
