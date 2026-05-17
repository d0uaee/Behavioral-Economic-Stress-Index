# Data Dictionary — BESI V3 Gold Dataset

Fichier documenté : `data/gold/model_dataset_monthly.csv`

Fréquence : mensuelle (`MS`)  
Plage cible du projet : `2010-01-01` → `2024-12-01`

Ce document décrit toutes les colonnes attendues dans le Gold dataset V3, leur type, leur source et leur rôle analytique.

## Règle As-Of-Date

Objectif du Gold dataset :
- prédire `IPC(t+1)` ou `inflation_yoy(t+1)`
- détecter le régime d'inflation du mois suivant

Règle fondamentale :
- les colonnes contemporaines `ipc_level`, `inflation_mom`, `inflation_yoy` existent dans le dataset comme vérité terrain et contexte analytique
- pour la modélisation prédictive, **seuls les lags >= 1 de l'IPC sont autorisés comme features**
- `ipc_level_lag0` est interdit comme feature car trop proche de la cible

En pratique :
- `ipc_level`, `inflation_mom`, `inflation_yoy` = colonnes de référence / cibles / diagnostic
- `ipc_level_lag1`, `inflation_yoy_lag1`, etc. = colonnes autorisées côté features

---

## 1. Identifiant

| Nom | Type | Source | Description |
|---|---|---|---|
| `month` | `DatetimeIndex[MS]` | Gold | Identifiant temporel mensuel du dataset. Une ligne = un mois. |

---

## 2. IPC Silver

Source primaire : HCP Maroc (base 2017=100)

| Nom | Type | Source | Description |
|---|---|---|---|
| `ipc_level` | `float` | `silver/cpi_monthly.csv` | Niveau brut de l'IPC mensuel (base 2017=100). |
| `inflation_mom` | `float` | `silver/cpi_monthly.csv` | Variation mensuelle de l'IPC en pourcentage (`MoM`). |
| `inflation_yoy` | `float` | `silver/cpi_monthly.csv` | Variation annuelle de l'IPC en pourcentage (`YoY`). |
| `publication_date` | `date` ou `str` | `silver/cpi_monthly.csv` | Date estimée de publication officielle HCP pour le mois courant (approx. fin de mois + 20 jours). |

---

## 3. Trends Silver

Sous-indices thématiques Google Trends, normalisés individuellement sur `[0, 1]`.

| Nom | Type | Source | Description |
|---|---|---|---|
| `trends_prix_alim` | `float` | `silver/google_trends_monthly.csv` | Intensité de recherche liée aux prix alimentaires (`prix huile`, `hausse prix`, etc.). |
| `trends_inflation` | `float` | `silver/google_trends_monthly.csv` | Intensité de recherche liée à l'inflation générale (`inflation maroc`, arabe). |
| `trends_carburant` | `float` | `silver/google_trends_monthly.csv` | Intensité de recherche liée au carburant et à l'énergie. |
| `trends_subvention` | `float` | `silver/google_trends_monthly.csv` | Intensité de recherche liée aux subventions / politique de prix. |
| `trends_composite` | `float` | `silver/google_trends_monthly.csv` | Moyenne simple des sous-indices thématiques Trends disponibles. |

---

## 4. Macro Silver

Signaux macroéconomiques réels alignés au mois.

| Nom | Type | Source | Description |
|---|---|---|---|
| `fao_food_index` | `float` | `silver/macro_signals_monthly.csv` | FAO Food Price Index global (base FAO 2014-2016=100). |
| `fao_food_yoy` | `float` | `silver/macro_signals_monthly.csv` | Variation annuelle (%) du FAO Food Price Index global. |
| `fao_oils_yoy` | `float` | `silver/macro_signals_monthly.csv` | Variation annuelle (%) du sous-indice FAO des huiles. |
| `mad_eur` | `float` | `silver/macro_signals_monthly.csv` | Taux de change MAD/EUR mensuel. |
| `fx_yoy` | `float` | `silver/macro_signals_monthly.csv` | Variation annuelle (%) du taux MAD/EUR. |

---

## 5. Indices BESI V3

Deux indices séparés, sans composante IPC directe.

| Nom | Type | Source | Description |
|---|---|---|---|
| `behavioral_index_pure` | `float` | `silver/behavioral_index_pure.csv` | Indice comportemental pur construit uniquement à partir des signaux Google Trends. Normalisé sur `[0, 1]`. Poids calibrés via `LassoCV` ou fallback simple. |
| `hybrid_macro_index` | `float` | `silver/hybrid_macro_index.csv` | Indice hybride combinant signaux comportementaux + macro (FAO, FX). Normalisé sur `[0, 1]`. |

---

## 6. Lags Explicites

Convention :
- suffixe `_lag1` = valeur du mois précédent
- suffixe `_lag2` = valeur décalée de 2 mois
- etc.

### 6.1 Lags des indices BESI

| Nom | Type | Source | Description |
|---|---|---|---|
| `behavioral_index_pure_lag1` | `float` | Gold dérivé | Valeur de `behavioral_index_pure` décalée de 1 mois. |
| `behavioral_index_pure_lag2` | `float` | Gold dérivé | Valeur de `behavioral_index_pure` décalée de 2 mois. |
| `hybrid_macro_index_lag1` | `float` | Gold dérivé | Valeur de `hybrid_macro_index` décalée de 1 mois. |
| `hybrid_macro_index_lag2` | `float` | Gold dérivé | Valeur de `hybrid_macro_index` décalée de 2 mois. |

### 6.2 Lags des Trends

| Nom | Type | Source | Description |
|---|---|---|---|
| `trends_prix_alim_lag1` | `float` | Gold dérivé | Sous-indice alimentaire Trends décalé de 1 mois. |
| `trends_inflation_lag1` | `float` | Gold dérivé | Sous-indice inflation Trends décalé de 1 mois. |
| `trends_carburant_lag1` | `float` | Gold dérivé | Sous-indice carburant Trends décalé de 1 mois. |
| `trends_composite_lag1` | `float` | Gold dérivé | Composite Trends décalé de 1 mois. |

### 6.3 Lags macro

| Nom | Type | Source | Description |
|---|---|---|---|
| `fao_food_index_lag1` | `float` | Gold dérivé | FAO Food Index décalé de 1 mois. |
| `fao_food_yoy_lag1` | `float` | Gold dérivé | Variation annuelle du FAO Food Index décalée de 1 mois. |
| `fao_oils_yoy_lag1` | `float` | Gold dérivé | Variation annuelle du sous-indice FAO huiles décalée de 1 mois. |
| `mad_eur_lag1` | `float` | Gold dérivé | Taux MAD/EUR décalé de 1 mois. |
| `fx_yoy_lag1` | `float` | Gold dérivé | Variation annuelle du taux MAD/EUR décalée de 1 mois. |

### 6.4 Lags IPC historiques

Ces colonnes sont les seules versions de l'IPC autorisées comme features prédictives.

| Nom | Type | Source | Description |
|---|---|---|---|
| `ipc_level_lag1` | `float` | Gold dérivé | IPC du mois `t-1`. |
| `ipc_level_lag2` | `float` | Gold dérivé | IPC du mois `t-2`. |
| `ipc_level_lag3` | `float` | Gold dérivé | IPC du mois `t-3`. |
| `inflation_yoy_lag1` | `float` | Gold dérivé | Inflation YoY du mois `t-1`. |
| `inflation_yoy_lag2` | `float` | Gold dérivé | Inflation YoY du mois `t-2`. |
| `inflation_mom_lag1` | `float` | Gold dérivé | Inflation MoM du mois `t-1`. |
| `inflation_mom_lag2` | `float` | Gold dérivé | Inflation MoM du mois `t-2`. |

---

## 7. Cibles

Toutes les cibles sont décalées d'un mois (`t+1`) pour prédiction.

| Nom | Type | Source | Description |
|---|---|---|---|
| `target_inflation_yoy_t1` | `float` | Gold dérivé | Inflation YoY du mois suivant. Utilisée pour la régression. |
| `target_high_inflation_regime_t1` | `float` ou `int` | Gold dérivé | Variable binaire : `1` si `inflation_yoy(t+1) >= 2%`, sinon `0`. Utilisée pour la classification du régime d'inflation. |
| `target_ipc_level_t1` | `float` | Gold dérivé | Niveau d'IPC du mois suivant. |

---

## 8. Métadonnées

| Nom | Type | Source | Description |
|---|---|---|---|
| `as_of_date` | `date` ou `str` | Gold dérivé | Fin du mois courant. Sert de repère de disponibilité logique du jeu de données. |
| `feature_available_at` | `date` ou `str` | Gold dérivé | Date à partir de laquelle la ligne peut être utilisée dans la logique de prévision (lendemain de `as_of_date`). |
| `split_label` | `str` | Gold dérivé | Appartenance aux blocs d'évaluation : `train_A`, `test_A`, `train_B`, `test_B`, `train_C`, `test_C`, ou `unused`. |

---

## 9. Colonnes Potentiellement Partielles

Selon l'état réel du pipeline et la disponibilité des sources, certaines colonnes peuvent être absentes ou fortement incomplètes :

| Nom | Cause possible |
|---|---|
| `trends_carburant` / `trends_subvention` | keywords non présents dans la source Trends brute |
| `hybrid_macro_index` | macro silver absent (FAO / FX non disponibles) |
| `fao_*`, `mad_eur`, `fx_yoy` | ingestion macro non exécutée ou source indisponible |

Ces cas doivent être explicitement contrôlés avant modélisation finale.

---

## 10. Interdictions Méthodologiques

Colonnes à ne **pas** utiliser comme features directes pour prédire `t+1` :

| Nom | Pourquoi |
|---|---|
| `ipc_level` | colonne contemporaine quasi cible |
| `inflation_mom` | colonne contemporaine |
| `inflation_yoy` | colonne contemporaine |
| toute colonne `target_*` | vérité terrain future |
| `ipc_change` | transformation de la cible historique, explicitement interdite en V3 |

---

## 11. Résumé d'Usage

Pour l'exploration :
- utiliser toutes les colonnes descriptives et cibles

Pour la modélisation prédictive :
- privilégier les colonnes laguées (`*_lag1`, `*_lag2`, `*_lag3`)
- utiliser `split_label` pour séparer train/test
- vérifier la complétude des colonnes macro avant d'entraîner `hybrid_macro_index`

Pour l'interprétation :
- `behavioral_index_pure` = signal digital pur
- `hybrid_macro_index` = signal combiné digital + macro

