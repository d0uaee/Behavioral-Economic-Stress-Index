# PRESENTATION FINALE
## Détection Précoce du Stress Économique des Ménages au Maroc
### Behavioral Economic Stress Index (BESI)

**Étudiantes :** Douae Ahadji & Adama Basse  
**Cours :** Séries Temporelles — ENSAM Meknès  
**Durée :** 8 semaines | **Date :** Mai 2026  

---

---

# DIAPOSITIVE 1 — ETAT D'AVANCEMENT GLOBAL

```
╔══════════════════════════════════════════════════════════════════╗
║           ETAT D'AVANCEMENT DU PROJET — 8 SEMAINES              ║
╠══════════════╦══════════════════════════════════╦═══════════════╣
║   Module     ║   Description                    ║   Statut      ║
╠══════════════╬══════════════════════════════════╬═══════════════╣
║ Données      ║ IPC HCP + Google Trends + BESI   ║  [TERMINE]    ║
║ SARIMA       ║ Identification + Walk-forward     ║  [TERMINE]    ║
║ SARIMAX      ║ +BESI_trends, comparaison v2      ║  [TERMINE]    ║
║ Analyse      ║ Chow, Granger, Early Warning      ║  [TERMINE]    ║
║ Deep Learning║ LSTM (4 fenetres) + Prophet       ║  [TERMINE]    ║
║ NLP          ║ Scraping + scoring Darija/Arabe   ║  [TERMINE]    ║
║ Notebooks    ║ 4 notebooks avec outputs          ║  [TERMINE]    ║
║ Rapport      ║ README + Présentation finale      ║  [TERMINE]    ║
╚══════════════╩══════════════════════════════════╩═══════════════╝

Avancement global : 100%  [████████████████████] 8/8 modules
```

---

---

# DIAPOSITIVE 2 — CONTEXTE ET QUESTION DE RECHERCHE

## Problématique

Les statistiques officielles de l'IPC (HCP) sont publiées avec **1 mois de délai**
et ne capturent pas le ressenti des ménages en temps réel.

> *"Peut-on détecter le stress économique des ménages marocains  
> AVANT qu'il apparaisse dans les statistiques officielles ?"*

## Hypothèse H1

> **"Les signaux comportementaux digitaux (Google Trends, Reddit, YouTube)  
> intégrés dans BESI permettent de détecter le stress économique  
> 1 à 2 mois avant l'IPC du HCP."**

## Contribution originale

- **BESI** : Premier indice composite marocain de stress économique
  basé sur les signaux digitaux comportementaux (Darija + Arabe + Français)
- Pipeline NLP multilingue sur la presse marocaine
- Comparaison rigoureuse : 7 modèles sur même période test (2022-2024)

---

---

# DIAPOSITIVE 3 — DONNÉES ET PIPELINE

## Variable cible
- **IPC mensuel Maroc (HCP)** : 2010-01 à 2024-12 → **180 observations**
- Fréquence mensuelle (MS), seed fixe np.random.seed(42)

## Signaux comportementaux collectés

| Source | Outil | Mots-clés / Requêtes | Période |
|--------|-------|----------------------|---------|
| Google Trends | pytrends | "inflation maroc", "prix huile", "hausse prix", "crédit consommation", "chomage maroc", "prix alimentaires", "pouvoir achat" | 2010-2024 |
| Reddit r/Morocco | praw | "inflation", "prix", "cherté", "économie" | 2015-2024 |
| YouTube API v3 | google-api-python-client | "inflation maroc", "hausse prix maroc" | 2018-2024 |
| Presse marocaine | BeautifulSoup4 | Hespress, Le360, Medias24, L'Économiste | 2020-2024 |

## master_dataset.csv — 11 colonnes, 180 obs

```
ipc | trends_composite | reddit_stress | youtube_stress | besi
ipc_yoy | ipc_change | stress_level | stress_binary | besi_trends | besi_enrichi
```

---

---

# DIAPOSITIVE 4 — BESI : L'INDICE COMPOSITE

## Trois versions de BESI

```
┌─────────────────────────────────────────────────────────────────┐
│  BESI composite                                                  │
│  = 0.40 × Trends + 0.30 × Reddit + 0.20 × YouTube + 0.10 × dIPC│
│  → Version complète (poids arbitraires, données partiell. simul.)│
├─────────────────────────────────────────────────────────────────┤
│  BESI_trends (VERSION ROBUSTE — utilisée en modélisation)        │
│  = 0.70 × Trends + 0.30 × |ΔIPC|                               │
│  → Uniquement données réelles (Google Trends + IPC réel)         │
├─────────────────────────────────────────────────────────────────┤
│  BESI_enrichi (VERSION NLP)                                      │
│  = 0.35×Trends + 0.25×NLP_Maroc + 0.20×YouTube + 0.10×Reddit    │
│    + 0.10×|ΔIPC|                                                │
│  → Enrichi avec NLP presse marocaine Darija/Arabe               │
└─────────────────────────────────────────────────────────────────┘
```

Tous les signaux normalisés **0-1** avant pondération.

## Statistiques BESI_trends
- Moyenne : 0.376 | Écart-type : 0.228
- Pic maximum : **Choc inflationniste 2022** (confirmé)

---

---

# DIAPOSITIVE 5 — ANALYSE DE STATIONNARITÉ

## Tests réalisés : ADF + KPSS

| Série | ADF p-value | KPSS p-value | Décision |
|-------|-------------|--------------|----------|
| IPC niveau | > 0.05 | < 0.05 | Non stationnaire |
| IPC diff(1) | < 0.05 | > 0.05 | Stationnaire |
| IPC diff(1).diff(12) | < 0.05 | > 0.05 | Stationnaire (SARIMA d=1, D=1) |

## Conclusion

```
IPC brut → Non stationnaire (tendance + saisonnalité)
     ↓  différenciation simple (d=1)
IPC_d1  → Stationnaire en tendance
     ↓  différenciation saisonnière (D=1, s=12)
IPC_d1_D1 → Pleinement stationnaire → modélisation SARIMA possible
```

## Décomposition STL
- Tendance haussière 2010-2024 (IPC 0.90 → 1.25)
- Saisonnalité mensuelle stable (amplitude ±0.01)
- Résidu : choc visible début 2022

---

---

# DIAPOSITIVE 6 — IDENTIFICATION ET SÉLECTION SARIMA

## Lecture ACF / PACF

| Graphique | Observation | Implication |
|-----------|-------------|-------------|
| ACF | Décroissance lente, pics aux lags 12, 24 | Saisonnalité s=12, D=1 |
| PACF | Coupure nette au lag 2 | AR(2) → p=2 |
| ACF résiduelle | Pic significatif lag 1 seulement | MA(1) → q=1 |

## Grille d'identification (AIC comparés)

```
Modèle testés : (p,d,q)×(P,D,Q)[12]  avec p,q ∈ {0,1,2}, P,Q ∈ {0,1}

Ordre retenu : SARIMA(2,1,1) × (0,1,1)[12]
  → AIC = -502.56  (minimal sur 6 modèles testés)
  → BIC = -492.34
  → Résidus : bruit blanc confirmé (test Ljung-Box, p > 0.05)
```

## Diagnostic des résidus
- Ljung-Box : OK (p > 0.05) — pas d'autocorrélation résiduelle
- QQ-plot : distribution approximativement normale
- Hétéroscédasticité : légère après choc 2022

---

---

# DIAPOSITIVE 7 — COMPARAISON DES MODÈLES

## Résultats Walk-Forward (h=1, test 2022-2024, n=36 mois)

```
╔═══════════════════╦══════════╦══════════╦════════╦══════════╦═══════════════╗
║ Modèle            ║   RMSE   ║   MAE    ║  MAPE  ║   AIC    ║ Gain vs SARIMA║
╠═══════════════════╬══════════╬══════════╬════════╬══════════╬═══════════════╣
║ Naïf (Rand. Walk) ║ 0.00409  ║ 0.00339  ║ 0.28%  ║    —     ║    -50.3%     ║
║ ★ SARIMA(2,1,1)   ║ 0.00272  ║ 0.00232  ║ 0.19%  ║ -502.56  ║     0.0%      ║
║ SARIMAX + Trends  ║ 0.00327  ║ 0.00253  ║ 0.21%  ║ -515.48  ║    -20.3%     ║
║ SARIMAX + BESI_t  ║ 0.00304  ║ 0.00241  ║ 0.20%  ║ -545.88  ║    -11.9%     ║
║ LSTM (window=12)  ║ 0.01885  ║ 0.01647  ║ 1.37%  ║    —     ║   -593%       ║
║ Prophet           ║ 0.06082  ║ 0.05950  ║ 4.88%  ║    —     ║  -2136%       ║
╚═══════════════════╩══════════╩══════════╩════════╩══════════╩═══════════════╝

★ Meilleur RMSE    ★★ Meilleur AIC : SARIMAX_BT (-545.88)
```

## Lecture des résultats

- **SARIMA** : meilleur RMSE global → robustesse confirmée
- **SARIMAX_BT** : meilleur AIC → BESI_trends apporte une **information statistique** significative même sans gain RMSE
- **LSTM et Prophet** : performances très inférieures → séries trop courtes (180 obs), choc non-stationnarité pénalisant pour le DL

---

---

# DIAPOSITIVE 8 — PERFORMANCES PAR SOUS-PÉRIODE

## Décomposition du test en 3 phases

```
2022-01 → 2022-12 : "Choc 2022"      (12 mois, inflation record +6.57%)
2023-01 → 2024-12 : "Post-choc"      (24 mois, retour progressif)
2022-01 → 2024-12 : "Test complet"   (36 mois, période complète)
```

## Résultats par période

| Modèle | Choc 2022 RMSE | Post-Choc RMSE | Test complet RMSE |
|--------|---------------|----------------|-------------------|
| Naïf | 0.00597 | 0.00269 | 0.00409 |
| **SARIMA** | **0.00197** | 0.00303 | **0.00272** |
| SARIMAX_T | 0.00380 | 0.00297 | 0.00327 |
| SARIMAX_BT | 0.00362 | **0.00271** | 0.00304 |

## Interprétation clé

> **SARIMA domine pendant le choc 2022** (RMSE 0.00197 vs 0.00362 SARIMAX_BT)  
> → La variable exogène BESI perturbe la prévision en période de rupture  
>  
> **SARIMAX_BT légèrement meilleur post-2022** (RMSE 0.00271 vs 0.00303)  
> → Une fois le régime stabilisé, BESI apporte une information marginale utile

---

---

# DIAPOSITIVE 9 — RUPTURE STRUCTURELLE 2022

## Test de Chow — Résultat

```
Breakpoint testé     : 2022-01-01 (choc inflationniste mondial)
Décision             : RUPTURE CONFIRMÉE (p < 0.05)
Contexte             : Guerre Ukraine, prix énergie/alimentaires +30-50%
Inflation YoY max    : 6.57% (mars 2022) — record depuis 30 ans
```

## Test CUSUM
- Dépassement des bandes de confiance détecté **mi-2022**
- Trajectoire cumulative sort définitivement des limites : **changement de régime permanent**

## Matrice de transition de Markov (états BESI)

```
                  Normal    Warning   High Stress
    Normal  →     82.2%      17.8%        0.0%
    Warning →     19.3%      74.2%        6.5%
 High Stress →     2.3%       6.8%       90.9%
```

**Lecture :** Une fois en état "High Stress", probabilité de rester = **90.9%**  
→ Le choc de 2022 est un régime persistant, non transitoire

## Coefficients pré/post 2022
- Constante : +10% | Tendance : +60% | Coefficient BESI : +67%
- Le rôle de BESI s'amplifie après 2022 — signal plus lisible en période de crise

---

---

# DIAPOSITIVE 10 — CAUSALITÉ DE GRANGER ET EARLY WARNING

## Causalité de Granger — Features significatives

Toutes les variables comportementales testées sont significatives (p < 0.05) :

| Feature | Significativité |
|---------|----------------|
| trends_composite | p < 0.05 |
| reddit_composite | p < 0.05 |
| youtube_composite | p < 0.05 |
| besi | p < 0.05 |
| ipc_change | p < 0.05 |
| ipc_mom | p < 0.05 |
| ipc_yoy | p < 0.05 |

**7/7 features** avec causalité de Granger significative → BESI Granger-cause l'IPC

## Early Warning — Résultat

```
Événement détecté    : onset 2021-05-01
Lead time            : 12 mois avant le pic d'inflation (mars 2022)
Détection            : TRUE

Performance système d'alerte :
  Rappel    = 100%  (aucun faux négatif, toutes les crises détectées)
  F1-Score  = ~0.82
```

> BESI a sonné l'alerte **12 mois avant** le pic d'inflation de 2022

---

---

# DIAPOSITIVE 11 — DEEP LEARNING : LSTM

## Architecture LSTM de base

```
Input(12, 1 ou 2) → LSTM(64) → Dropout(0.1) → LSTM(32) → Dense(1)
Entraînement : Early stopping | Epochs max : 50 | Batch : 16
```

## Résultats LSTM

| Configuration | RMSE | MAE | MAPE | Epochs |
|--------------|------|-----|------|--------|
| LSTM IPC seul (w=12) | 0.01885 | 0.01647 | 1.37% | 19 |
| LSTM + BESI (w=12) | 0.06334 | 0.06219 | 5.10% | 9 |

## Comparaison tailles de fenêtre glissante

| Fenêtre | Sans exog RMSE | Avec BESI RMSE |
|---------|---------------|----------------|
| 6 mois | 0.08842 | 0.08947 |
| 12 mois | 0.05467 | 0.07440 |
| 18 mois | 0.05743 | 0.06577 |
| **24 mois** | **0.04627** | 0.06578 |

**Conclusion :** La fenêtre 24 mois sans exogène est la meilleure LSTM configuration,  
mais reste **17× pire que SARIMA** (RMSE 0.04627 vs 0.00272).

> Ajouter BESI au LSTM **dégrade** systématiquement les performances  
> → BESI est un signal macroéconomique (lead 12 mois), inadapté au LSTM court terme

---

---

# DIAPOSITIVE 12 — PROPHET

## Configuration Prophet

```python
Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    seasonality_mode='multiplicative'
)
Train : 2010-01 -> 2021-12 (144 observations)
Test  : 2022-01 -> 2024-12 (36 observations)
```

## Résultats Prophet

```
RMSE  : 0.06082    (22× pire que SARIMA)
MAE   : 0.05950
MAPE  : 4.88%
```

## Pourquoi Prophet échoue ici ?

Prophet est conçu pour :
- Longues séries à forte saisonnalité (commerce, météo...)
- Données journalières ou hebdomadaires
- Séries sans rupture structurelle majeure

L'IPC mensuel marocain avec rupture 2022 = contre-indication pour Prophet.  
→ **Résultat scientifiquement valide** : confirme la supériorité de SARIMA.

---

---

# DIAPOSITIVE 13 — NLP PRESSE MAROCAINE

## Pipeline NLP (src/nlp_morocco.py)

```
Module 1 : Scraping presse
  → Hespress, Le360, Medias24, L'Économiste
  → Sélecteurs CSS multi-fallback

Module 2 : Commentaires YouTube
  → 4 chaînes : 2M Maroc, Medi1TV, Hespress TV, Medias24
  → YouTube Data API v3

Module 3 : Scoring NLP
  → Dictionnaire 80+ mots-clés (Darija / Arabe / Français)
  → score = (0.6 × keywords + 0.4 × intensité) × engagement_weight
  → Normalisation 0-1

Module 4 : Agrégation mensuelle
  → Pondéré par nb commentaires + likes + vues

Module 5 : BESI_enrichi
  → Mise à jour automatique de master_dataset.csv

Module 6 : Visualisation
  → Figure dual-axis NLP vs IPC (300 DPI)
```

## Dictionnaire de stress (extrait)

| Catégorie | Exemples (Darija) | Exemples (Arabe) |
|-----------|------------------|------------------|
| Prix élevés | "ghali", "ghla", "cher" | "غالي", "غلاء", "ارتفاع الأسعار" |
| Manque argent | "ma b9ach", "flouss", "mskine" | "فلوس", "فقر", "محتاج" |
| Frustration | "hshuma", "crise" | "عيب", "أزمة", "مشكل" |
| Produits base | "zit", "sokkar", "carburant" | "زيت", "سكر", "وقود" |

---

---

# DIAPOSITIVE 14 — STRUCTURE DU CODE

## Fichiers Python (src/)

| Fichier | Lignes | Rôle | Auteure |
|---------|--------|------|---------|
| data_pipeline.py | ~400 | Collecte IPC + Trends + BESI | Douae |
| features.py | ~350 | Feature selection + lag analysis | Douae |
| models.py | ~600 | SARIMA + SARIMAX + walk-forward | Douae |
| analysis.py | ~550 | Chow + Granger + Early Warning + Markov | Douae |
| visualization.py | ~700 | Dashboard + 43 figures | Douae |
| nlp_morocco.py | ~1550 | NLP presse + YouTube + BESI_enrichi | Douae |
| deep_learning.py | ~400 | LSTM + comparaison fenêtres | Adama |
| prophet_model.py | ~160 | Prophet IPC | Adama |

## Scripts de lancement

```
run_v2.py    → Comparaison modèles + LSTM + Prophet (complet)
run_all.py   → Pipeline 7 étapes avec options CLI
               --skip-data | --skip-dl | --skip-nlp | --step N
```

## Notebooks (4 notebooks, tous avec outputs)

```
exploration.ipynb  (8 cells, 7 outputs)  → Stationnarité, ACF/PACF, BESI
modeling.ipynb    (10 cells, 9 outputs)  → SARIMA, SARIMAX, Walk-forward
analysis.ipynb    (8 cells, 7 outputs)   → Chow, Granger, Early Warning
results.ipynb     (6 cells, 5 outputs)   → Table finale, Dashboard 6 panneaux
```

---

---

# DIAPOSITIVE 15 — OUTPUTS GÉNÉRÉS

## Fichiers de résultats (outputs/reports/)

| Fichier CSV | Contenu |
|-------------|---------|
| model_comparison_v2.csv | RMSE/MAE/MAPE/AIC pour 4 modèles |
| period_performance_v2.csv | Métriques par sous-période (3×4=12 lignes) |
| final_model_comparison_all.csv | Table unifiée tous modèles (7 modèles) |
| early_warning_events.csv | onset 2021-05, lead_time=12 mois |
| granger_significant_features.csv | 7 features significatives |
| stress_transition_matrix.csv | Matrice Markov 3×3 |
| lstm_window_comparison.csv | 8 configurations LSTM |
| prophet_results.csv | RMSE=0.06082, MAE=0.05950, MAPE=4.88% |

## Figures (outputs/figures/) — 43 fichiers PNG

```
Exploration    : fig1_ipc, fig2_besi, fig3_signals, acf_pacf, stl...
Modélisation   : compare_all_predictions_v2, walk_forward, residus...
Analyse        : chow_test, early_warning, stress_heatmap, lag_corr...
Deep Learning  : lstm_predictions, lstm_window_comparison...
Prophet        : prophet_forecast.png
Dashboard      : nb04_dashboard.png (6 panneaux synthétiques)
```

---

---

# DIAPOSITIVE 16 — VERDICT H1 ET CONCLUSION

## Hypothèse H1 — Verdict final

```
╔════════════════════════════════════════════════════════════════╗
║                H1 : PARTIELLEMENT REJETÉE                      ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  Ce qui est REJETE :                                           ║
║  ✗  Lead time de 1-2 mois  →  lead time réel = 12 mois        ║
║  ✗  SARIMAX+BESI améliore le RMSE  →  SARIMA reste meilleur   ║
║                                                                ║
║  Ce qui est CONFIRME :                                         ║
║  ✓  BESI précède l'IPC de ~12 mois (CCF significatif)         ║
║  ✓  Causalité Granger : 7/7 features significatives (p<0.05)   ║
║  ✓  Rupture structurelle 2022 confirmée (Test de Chow)         ║
║  ✓  BESI_trends améliore l'AIC : -502 → -546 (info statistique)║
║  ✓  Early Warning : Rappel=100%, F1≈0.82                       ║
║  ✓  Régime persistant : P(High Stress|High Stress) = 90.9%     ║
╚════════════════════════════════════════════════════════════════╝
```

## Implication

L'horizon temporel de BESI est **macroéconomique (12 mois)**, non tactique (1-2 mois).  
Il est utile pour les **politiques contracycliques** (crédit, subventions alimentaires)  
mais pas pour les interventions court terme.

---

---

# DIAPOSITIVE 17 — PHRASE DE POSITIONNEMENT (ORAL)

---

> *"Je reste dans le cadre SARIMA/SARIMAX du cours, mais j'introduis  
> une dimension comportementale multi-sources pour tester la stabilité  
> structurelle après 2022 et quantifier la capacité d'alerte précoce  
> des signaux digitaux.*  
>  
> *Mes résultats montrent que SARIMA(2,1,1)×(0,1,1)[12] reste le meilleur  
> modèle en validation walk-forward (RMSE=0.00272), mais que BESI_trends  
> améliore l'AIC de -502 à -546 — confirmant qu'il apporte une information  
> statistique réelle.*  
>  
> *BESI détecte le stress économique 12 mois d'avance — non 1-2 mois  
> comme l'hypothèse initiale — offrant un signal macroéconomique robuste  
> pour les politiques contracycliques au Maroc."*

---

---

# DIAPOSITIVE 18 — RÉCAPITULATIF FINAL

## Ce que le professeur attend — Check-list

| Critère | Réalisé | Détail |
|---------|---------|--------|
| Analyse stationnarité rigoureuse | OUI | ADF + KPSS + décomposition STL |
| Identification SARIMA correcte | OUI | Grille AIC, ACF/PACF, Ljung-Box |
| Comparaison SARIMA vs SARIMAX+BESI | OUI | 4 modèles + walk-forward 36 mois |
| Analyse rupture structurelle 2022 | OUI | Test de Chow + CUSUM + Markov |
| Interprétation des résultats | OUI | H1 nuancée, implications concrètes |
| Question de recherche + H1 testée | OUI | H1 partiellement rejetée, argumentée |

## Bonus réalisés (au-delà du cours)

| Extra | Description |
|-------|-------------|
| LSTM multi-fenêtre | 4 tailles × 2 configs = 8 modèles LSTM comparés |
| Prophet | Modèle Bayésien testé et comparé |
| NLP multilingue | Darija + Arabe + Français, 4 sites + YouTube |
| Causalité de Granger | 7 features testées |
| Matrice de Markov | Dynamique des états de stress |
| BESI_enrichi | Indice enrichi avec NLP presse marocaine |
| Pipeline complet | run_all.py + run_v2.py automatisés |

---

---

# ANNEXE — CHIFFRES CLÉS À RETENIR

```
┌──────────────────────────────────────────────────────────────┐
│                  CHIFFRES CLÉS DU PROJET                     │
├──────────────────────────────────────────────────────────────┤
│  180 obs      Période 2010-2024, fréquence mensuelle         │
│  SARIMA(2,1,1)×(0,1,1)[12]   Modèle optimal (AIC=-502.56)   │
│  RMSE = 0.00272   Meilleure performance walk-forward         │
│  AIC  = -545.88   Meilleur AIC (SARIMAX_BT)                  │
│  12 mois          Lead time BESI avant IPC                   │
│  6.57%            Inflation YoY max (mars 2022, record)      │
│  90.9%            Persistance état "High Stress" (Markov)    │
│  100%             Rappel Early Warning (0 faux négatif)      │
│  43 figures       Générées dans outputs/figures/             │
│  7 modèles        Comparés sur même période test 2022-2024   │
│  8 semaines       Durée totale du projet                     │
└──────────────────────────────────────────────────────────────┘
```

---

*Douae Ahadji & Adama Basse — ENSAM Meknès — Séries Temporelles — Mai 2026*
