# Morocco Economic Stress Index — Project Context

## Identité du projet
**Titre:** Détection Précoce du Stress Économique des Ménages au Maroc  
**Étudiantes:** Douae Ahadji & Adama Basse
**Durée:** 8 semaines  
**Cours:** Séries Temporelles — ENSAM Meknès  

## Objectif principal
Prédire et détecter le stress économique des ménages marocains 
AVANT qu'il apparaisse dans les statistiques officielles (IPC du HCP).

## Contribution originale
→ BESI (Behavioral Economic Stress Index) : indice composite pondéré
   construit à partir de Google Trends + Reddit + YouTube
→ Test de rupture structurelle 2022 (choc inflationniste)
→ Early warning : BESI(t) prédit le stress à t+1 et t+2

## Variable cible
- IPC mensuel Maroc (HCP) — 2010 à 2024
- Fréquence : mensuelle (MS)
- Source : hcp.ma ou World Bank

## Variables exogènes (signaux comportementaux)
1. Google Trends — pytrends, geo='MA'
   Keywords : ["inflation maroc", "prix huile", "hausse prix",
               "credit consommation", "chomage maroc",
               "prix alimentaires", "pouvoir achat"]
2. Reddit r/Morocco — praw
   Keywords : ["inflation", "prix", "cherté", "économie"]
3. YouTube Data API v3
   Queries : ["inflation maroc", "hausse prix maroc"]
4. BESI composite = 0.40*Trends + 0.30*Reddit + 0.20*YouTube + 0.10*IPC_change

## Structure des fichiers
```
project/
├── CLAUDE.md              ← ce fichier (toujours lire en premier)
├── data/
│   ├── raw/               ← données brutes téléchargées
│   ├── processed/         ← données nettoyées et alignées
│   └── ipc_maroc.csv      ← variable cible (à télécharger sur hcp.ma)
├── src/
│   ├── data_pipeline.py   ← collecte et nettoyage de toutes les sources
│   ├── features.py        ← feature selection, lag analysis, BESI index
│   ├── models.py          ← SARIMA, SARIMAX, walk-forward validation
│   ├── analysis.py        ← structural break, Granger, early warning
│   ├── deep_learning.py   ← LSTM comparison (optionnel, semaine 7-8)
│   └── visualization.py   ← tous les graphiques et dashboard
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_modeling.ipynb
│   ├── 03_analysis.ipynb
│   └── 04_results.ipynb
├── outputs/
│   ├── figures/           ← tous les PNG exportés
│   ├── models/            ← modèles sauvegardés (.pkl)
│   └── reports/           ← CSV de résultats
├── requirements.txt
└── README.md
```

## Règles de code (TOUJOURS respecter)
- Python 3.10+
- statsmodels pour SARIMA/SARIMAX (pas sklearn)
- Fréquence mensuelle uniquement (freq='MS')
- Toujours tester la stationnarité (ADF + KPSS) avant tout modèle
- Normalisation 0-1 pour tous les signaux comportementaux
- Commentaires en français
- Pas de modèles lourds — tout doit tourner sur un laptop normal
- Seed fixe : np.random.seed(42) partout
- Sauvegarder tous les résultats dans outputs/

## Bibliothèques autorisées
pandas, numpy, matplotlib, seaborn, statsmodels,
pytrends, praw, google-api-python-client,
scikit-learn, scipy, vaderSentiment,
tensorflow (LSTM uniquement, semaine 7-8)

## Métriques d'évaluation
- RMSE, MAE, MAPE pour la prévision
- AIC/BIC pour la sélection de modèle
- p-value pour les tests statistiques (seuil : 0.05)
- Lead time en mois pour l'early warning

## Ce que le prof attend
1. Analyse de stationnarité rigoureuse
2. Identification SARIMA correcte (ACF/PACF)
3. Comparaison SARIMA baseline vs SARIMAX+BESI
4. Analyse de rupture structurelle 2022
5. Interprétation des résultats (pas juste des chiffres)
6. Question de recherche claire + hypothèse H1 testée

## Phrase de positionnement (pour l'oral)
"Je reste dans le cadre SARIMA/SARIMAX du cours, mais 
j'introduis une dimension comportementale multi-sources 
pour tester la stabilité structurelle après 2022 et 
quantifier la capacité d'alerte précoce des signaux digitaux."
