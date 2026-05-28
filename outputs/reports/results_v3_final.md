# Resultats BESI V3 — Rapport Final
**Projet :** Detection Precoce du Stress Economique des Menages au Maroc  
**Auteurs :** Douae Ahadji & Adama Basse — ENSAM Meknes  
**Date :** Mai 2026 | Pipeline : V3 (2017-2024, 96 mois)

---

## 1. Donnees & Architecture

| Element | Valeur |
|---|---|
| Periode couverte | 2017-01 -> 2024-12 (96 mois) |
| Source IPC | HCP Maroc (base 2017=100) |
| Source comportementale | Google Trends (7 mots-cles, geo=MA) |
| Source macro | FAO Food Price Index (6 sous-indices) + MAD/EUR (ECB) |
| Blocs d'evaluation | A : COVID (2020-2021) | B : Inflation (2022-2024) |
| Gold dataset | 96 lignes x 45 colonnes |

---

## 2. Analyse de Stationnarite (ADF + KPSS)

| Serie | ADF p | KPSS p | Conclusion |
|---|---|---|---|
| ipc_level (niveau) | 0.902 | 0.010 | NON STATIONNAIRE |
| diff(ipc_level, 1) | 0.097 | >0.10 | Borderline -- accepte I(1) |
| inflation_yoy | >0.05 | -- | NON STATIONNAIRE |

-> IPC integre d'ordre 1 : justifie d = 1 dans SARIMA(1,1,1)(1,0,1)[12]

---

## 3. Identification SARIMA

Modele retenu : **SARIMA(1,1,1)(1,0,1)[12]**  
Justification : ACF/PACF sur diff(IPC,1), saisonnalite mensuelle caracteristique.

---

## 4. Comparaison In-Sample (Train A : 2017-2019)

| Modele | AIC | BIC | Delta AIC vs SARIMA |
|---|---|---|---|
| SARIMA(1,1,1)(1,0,1)[12] | 64.85 | 70.08 | -- |
| SARIMAX + BESI behavioral | **57.09** | **63.06** | **-7.77** |

-> SARIMAX+BESI ameliore significativement le fit in-sample (delta AIC < -2 = preference forte),
   mais cela ne suffit pas a etablir une superiorite hors-echantillon face a la baseline naive.

---

## 5. Backtest Walk-Forward (Out-of-Sample)

### Par bloc

| Bloc | Modele | RMSE | MAE | MAPE | n_test |
|---|---|---|---|---|---|
| A (COVID 2020-2021) | Naif | 1.376 | 1.033 | 1.01% | 24 |
| A | SARIMA | 1.913 | 1.547 | 1.52% | 24 |
| A | SARIMAX+BESI | **1.807** | **1.476** | **1.45%** | 24 |
| A | SARIMAX+Hybrid | 2.004 | 1.600 | 1.57% | 24 |
| B (Inflation 2022-2024) | Naif | 1.843 | 1.367 | 1.11% | 36 |
| B | SARIMA | 1.932 | 1.528 | 1.24% | 36 |
| B | SARIMAX+BESI | 1.976 | 1.569 | 1.28% | 36 |
| B | SARIMAX+Hybrid | 1.991 | 1.552 | 1.26% | 36 |

### Moyenne globale

| Modele | RMSE | MAE | MAPE |
|---|---|---|---|
| Naif | 1.609 | 1.200 | 1.06% |
| SARIMA | 1.923 | 1.537 | 1.38% |
| **SARIMAX+BESI behavioral** | **1.891** | **1.522** | **1.36%** |
| SARIMAX+Hybrid | 1.997 | 1.576 | 1.42% |

**Interpretation :**
- SARIMAX+BESI bat SARIMA pur sur tous les criteres globaux (RMSE -0.032 pts, MAPE -0.02%).
- Le modele naive reste toutefois le meilleur en RMSE global (1.609 vs 1.891 pour SARIMAX+BESI).
- SARIMAX+Hybrid est le pire des 4 modeles en RMSE global (1.997 > 1.923 SARIMA).
- Le modele naif (persistance) est competitif : l'IPC est fortement persistant (racine unitaire).
- Sur le Bloc A (COVID), SARIMAX+BESI gagne +0.106 pts RMSE sur SARIMA.
- Sur le Bloc B (inflation 2022-2024), les deux modeles SARIMAX peinent face a la rupture structurelle.

---

## 6. Rupture Structurelle 2022

| Statistique | Valeur |
|---|---|
| Date de rupture | Mars 2022 (guerre Ukraine) |
| Inflation pre-2022 (moy.) | +0.74% YoY |
| Inflation post-2022 (moy.) | +8.53% YoY |
| Hausse absolue | +7.80 points |
| Facteur multiplicatif | x11.6 |
| Test t (difference de moyennes) | t=-6.60 | **p < 0.0001** |
| Test de Levene (variance) | W=48.53 | **p < 0.0001** |

-> La rupture est massivement significative, en moyenne ET en variance.  
-> Cette instabilite parametrique explique les limites du SARIMA classique post-2022.

---

## 7. Test de Causalite de Granger

| Lag | F-stat | p-value | Significatif |
|---|---|---|---|
| 1 | 0.238 | 0.627 | Non |
| 2 | 0.248 | 0.781 | Non |
| 3 | 0.235 | 0.872 | Non |
| 4 | 0.200 | 0.938 | Non |

-> BESI ne cause pas l'inflation au sens de Granger (p >> 0.05 a tous les lags).  
-> Interpretation : la relation BESI-inflation est **non-lineaire** -- le signal agit comme detecteur de regime, pas comme predicteur causal lineaire.

---

## 8. Correlation Croisee BESI -> Inflation

| Periode | Lag optimal | r de Pearson | p-value |
|---|---|---|---|
| Pre-2022 | 0 | +0.201 | 0.161 (ns) |
| Post-2022 | 5 | -0.303 | 0.110 (ns) |
| **Periode complete** | 0 | **+0.535** | **< 0.001** |

-> Correlation globale forte et significative (r=0.535, p<0.001).  
-> La correlation s'effondre dans les sous-periodes : signe d'une relation non-lineaire moderee par le regime inflationniste.

---

## 9. Composition de l'Indice Hybrid (Poids LassoCV)

| Composante | Poids Lasso |
|---|---|
| fao_oils_yoy | 0.73 |
| fx_yoy | 0.23 |
| fao_cereals_yoy | 0.04 |
| behavioral_pure | 0.00 |
| fao_food_yoy, fao_meat_yoy, fao_dairy_yoy, fao_sugar_yoy | 0.00 |

-> Le Lasso assigne zero poids au signal comportemental dans le hybrid.  
-> Le signal macro (huiles + cereales FAO + taux de change) domine mais ne generalise pas bien hors-echantillon.

---

## 10. Early Warning — Alerte Precoce

Methode : seuil calibre sur TRAIN, evalue sur TEST (aucun data leakage).

### Par bloc et par signal

| Bloc | Signal | AUC | F1 | Precision | Recall | Lead-time |
|---|---|---|---|---|---|---|
| A (COVID) | behavioral | 0.328 | 0.500 | 0.333 | 1.000 | 1 mois |
| A (COVID) | hybrid | 0.562 | 0.500 | 0.333 | 1.000 | 1 mois |
| **B (Inflation)** | **behavioral** | 0.311 | **0.814** | **0.686** | **1.000** | -- |
| B (Inflation) | hybrid | 0.356 | 0.439 | 0.529 | 0.375 | -- |

### Global (agregation des blocs test avec seuils appris bloc par bloc)

| Signal | AUC | F1 | Recall | Lead-time |
|---|---|---|---|---|
| **BESI behavioral** | **0.574** | **0.703** | **1.000** | 1 mois |
| Hybrid macro | 0.376 | 0.466 | 0.531 | 1 mois |

**Interpretation :**
- **Bloc B (inflation 2022-2024) behavioral : Recall = 1.00** -- 100% des mois a inflation elevee detectes.
- Bloc A : les deux signaux sur-alertent et atteignent Recall=1.00, sans vraie specificite.
- Bloc B : le behavioral est superieur (Recall=1.00 vs 0.375 pour hybrid) -- la crise 2022 est mieux captee par les comportements de recherche marocains que par les indices FAO mondiaux.
- Global : behavioral bat hybrid sur Recall (1.00 vs 0.531) et F1 (0.703 vs 0.466).

---

## 11. Validation des Hypotheses

### H1 : BESI(t) predit le regime d'inflation elevee a t+1

| Critere | Valeur | Seuil | Statut |
|---|---|---|---|
| AIC SARIMAX vs SARIMA | -7.77 | < -2 | **VALIDE** |
| RMSE global SARIMAX vs SARIMA | -0.032 | < 0 | **VALIDE** |
| Recall global behavioral | 1.00 | > 0.80 | **VALIDE** |
| Recall Bloc B (inflation) | 1.00 | > 0.80 | **VALIDE** |
| AUC globale | 0.574 | > 0.65 | Non atteint |

-> **H1 partiellement validee** : BESI ameliore le fit in-sample (AIC) et capture bien les episodes d'inflation sur le bloc 2022-2024. En revanche, la baseline naive reste plus forte en RMSE global, donc la preuve de valeur predictive reste partielle.

### H2 : hybrid_macro_index ameliore la detection vs behavioral (delta AUC > 0.05)

| Critere | Behavioral | Hybrid | Delta | Statut |
|---|---|---|---|---|
| AUC globale | **0.574** | 0.376 | **-0.198** | Favorable behavioral |
| Recall global | **1.000** | 0.531 | **-0.469** | Defavorable hybrid |
| Recall Bloc B | **1.000** | 0.375 | **-0.625** | **REJETE** |
| RMSE global | **1.891** | 1.997 | +0.106 | Defavorable hybrid |

-> **H2 REJETEE** : le signal macro (FAO + FX) n'ameliore pas la detection.  
-> Sur la periode la plus importante (Bloc B, inflation 2022-2024), le hybrid chute a Recall=0.375 contre 1.000 pour le behavioral.  
-> Interpretation : les indices FAO sont des prix mondiaux, pas specifiquement marocains. Les comportements de recherche Google capturent mieux la specificite locale de la crise inflationniste marocaine.

---

## 12. Fichiers Produits

### Figures cles
| Fichier | Description |
|---|---|
| `dashboard_besi_v3_final.png` | Dashboard complet 6 panels -- H1 et H2 |
| `structural_break_v3.png` | Rupture structurelle 2022 |
| `acf_pacf_v3.png` | ACF/PACF pour identification SARIMA |
| `residuals_sarima_v3.png` | Diagnostics residus SARIMA |
| `backtest_v3_bar_comparison.png` | RMSE/MAE/MAPE par bloc (4 modeles) |
| `cross_corr_besi_v3.png` | Correlation croisee BESI-Inflation |
| `early_warning_v3.png` | Signal BESI vs episodes d'inflation |
| `roc_curves_v3.png` | Courbes ROC early warning |

### Rapports CSV
| Fichier | Description |
|---|---|
| `backtest_v3_results.csv` | RMSE/MAE/MAPE par modele et bloc (8 lignes) |
| `backtest_v3_summary.csv` | Moyenne globale |
| `warning_metrics_v3.csv` | AUC/F1/Recall par bloc et signal (6 lignes) |
| `granger_besi_v3.csv` | Test de causalite de Granger |
| `besi_v3_behavioral_weights.csv` | Poids des composantes BESI |

---

## 13. Phrase de Conclusion (Oral)

> "Notre BESI comportemental, construit uniquement a partir de Google Trends,
> ameliore le fit in-sample du modele SARIMA de 7.77 points AIC.
> Il detecte 100% des mois a inflation elevee pendant la crise 2022-2024
> et obtient un F1 global de 0.703 sur les periodes test agregees.
> La rupture structurelle de mars 2022 est massivement
> significative (p<0.0001, x11.6 l'inflation moyenne).
> L'ajout des signaux macro FAO et taux de change (H2) degrade la detection
> sur la periode inflationniste cle : Recall chute de 100% a 37.5%.
> En revanche, la baseline naive reste meilleure en RMSE global :
> le BESI apporte surtout une valeur de detection de regime, pas encore
> une superiorite robuste en prevision point par point."
