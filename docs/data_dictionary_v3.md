# Data Dictionary - BESI V3 Gold Dataset

Fichier documente : `data/gold/model_dataset_monthly.csv`  
Frequence : mensuelle (`MS`)  
Plage effective : `2017-01-01` -> `2024-12-01`

Ce document decrit les colonnes effectives du Gold dataset final.

## Regle centrale

Le dataset contient a la fois :

- des colonnes de verite terrain et de diagnostic
- des colonnes utilisables comme features predictives

Regle :

- les colonnes contemporaines `ipc_level`, `inflation_mom`, `inflation_yoy`
  ne doivent pas etre utilisees comme features predictives directes
- seules les versions laggees de l'IPC sont autorisees cote modelisation

## 1. Identifiant

| Nom | Type | Source | Description |
|---|---|---|---|
| `month` | date | gold | Identifiant mensuel du dataset. |

## 2. IPC courant

| Nom | Type | Source | Description |
|---|---|---|---|
| `ipc_level` | float | `silver/cpi_monthly.csv` | Niveau brut de l'IPC HCP, base 2017=100. |
| `inflation_mom` | float | `silver/cpi_monthly.csv` | Variation mensuelle de l'IPC en pourcentage. |
| `inflation_yoy` | float | `silver/cpi_monthly.csv` | Variation annuelle de l'IPC en pourcentage. |
| `publication_date` | date | `silver/cpi_monthly.csv` | Date estimee de publication officielle HCP. |

## 3. Trends Silver

| Nom | Type | Source | Description |
|---|---|---|---|
| `trends_prix_alim` | float | `silver/google_trends_monthly.csv` | Intensite de recherche sur les prix alimentaires. |
| `trends_inflation` | float | `silver/google_trends_monthly.csv` | Intensite de recherche sur l'inflation generale. |
| `trends_carburant` | float | `silver/google_trends_monthly.csv` | Intensite de recherche sur le carburant. |
| `trends_subvention` | float | `silver/google_trends_monthly.csv` | Intensite de recherche sur les subventions. |
| `trends_composite` | float | `silver/google_trends_monthly.csv` | Composite simple des sous-indices Trends. |

## 4. Macro Silver

| Nom | Type | Source | Description |
|---|---|---|---|
| `fao_food_index` | float | `silver/macro_signals_monthly.csv` | FAO Food Price Index. |
| `fao_food_yoy` | float | `silver/macro_signals_monthly.csv` | Variation annuelle du FAO Food Index. |
| `fao_cereals_index` | float | `silver/macro_signals_monthly.csv` | Sous-indice FAO des cereales. |
| `fao_cereals_yoy` | float | `silver/macro_signals_monthly.csv` | Variation annuelle du sous-indice FAO des cereales. |
| `fao_oils_index` | float | `silver/macro_signals_monthly.csv` | Sous-indice FAO des huiles. |
| `fao_oils_yoy` | float | `silver/macro_signals_monthly.csv` | Variation annuelle du sous-indice FAO des huiles. |
| `mad_eur` | float | `silver/macro_signals_monthly.csv` | Taux de change mensuel MAD/EUR. |
| `fx_yoy` | float | `silver/macro_signals_monthly.csv` | Variation annuelle du taux MAD/EUR. |

## 5. Indices BESI

| Nom | Type | Source | Description |
|---|---|---|---|
| `behavioral_index_pure` | float | `features/indexes.py` | Indice comportemental fonde sur Google Trends. |
| `hybrid_macro_index` | float | `features/indexes.py` | Indice hybride combinant behavioral et macro. |

## 6. Lags des indices

| Nom | Type | Source | Description |
|---|---|---|---|
| `behavioral_index_pure_lag1` | float | gold | BESI behavioral decale de 1 mois. |
| `behavioral_index_pure_lag2` | float | gold | BESI behavioral decale de 2 mois. |
| `hybrid_macro_index_lag1` | float | gold | Indice hybrid decale de 1 mois. |
| `hybrid_macro_index_lag2` | float | gold | Indice hybrid decale de 2 mois. |

## 7. Lags des features Trends et macro

| Nom | Type | Source | Description |
|---|---|---|---|
| `trends_prix_alim_lag1` | float | gold | Sous-indice prix alimentaires, lag 1. |
| `trends_inflation_lag1` | float | gold | Sous-indice inflation, lag 1. |
| `trends_carburant_lag1` | float | gold | Sous-indice carburant, lag 1. |
| `trends_composite_lag1` | float | gold | Composite Trends, lag 1. |
| `fao_food_index_lag1` | float | gold | FAO Food Index, lag 1. |
| `fao_food_yoy_lag1` | float | gold | FAO Food YoY, lag 1. |
| `fao_oils_yoy_lag1` | float | gold | FAO Oils YoY, lag 1. |
| `mad_eur_lag1` | float | gold | MAD/EUR, lag 1. |
| `fx_yoy_lag1` | float | gold | FX YoY, lag 1. |

## 8. Lags IPC autorises en features

| Nom | Type | Source | Description |
|---|---|---|---|
| `ipc_level_lag1` | float | gold | IPC du mois precedent. |
| `ipc_level_lag2` | float | gold | IPC avec 2 mois de decalage. |
| `ipc_level_lag3` | float | gold | IPC avec 3 mois de decalage. |
| `inflation_yoy_lag1` | float | gold | Inflation YoY du mois precedent. |
| `inflation_yoy_lag2` | float | gold | Inflation YoY avec 2 mois de decalage. |
| `inflation_mom_lag1` | float | gold | Inflation MoM du mois precedent. |
| `inflation_mom_lag2` | float | gold | Inflation MoM avec 2 mois de decalage. |

## 9. Cibles

| Nom | Type | Source | Description |
|---|---|---|---|
| `target_inflation_yoy_t1` | float | gold | Inflation YoY du mois suivant. |
| `target_high_inflation_regime_t1` | int | gold | Regime binaire du mois suivant. |
| `target_ipc_level_t1` | float | gold | Niveau IPC du mois suivant. |

## 10. Metadonnees

| Nom | Type | Source | Description |
|---|---|---|---|
| `as_of_date` | date | gold | Fin du mois courant. |
| `feature_available_at` | date | gold | Date theorique d'utilisation de la ligne. |
| `split_label` | string | gold | Bloc d'evaluation: train_A, test_A, train_B, test_B, etc. |
