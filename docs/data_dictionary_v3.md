# Dictionnaire de Données — BESI Maroc V3

Gold dataset : `data/gold/model_dataset_monthly.csv`  
Période : 2010-01-01 → 2024-12-01 (fréquence mensuelle, `freq=MS`)  
Généré par : `src/gold/build_model_dataset.py`

---

## Règle fondamentale as-of-date

> Pour prédire `target_*_t1` (valeur au mois `t+1`), seules les features
> disponibles **avant la publication HCP du mois `t`** sont utilisables.
> La publication HCP intervient environ **J+20** après la fin du mois.
>
> **Colonnes contemporaines interdites comme features :**
> `ipc_level`, `inflation_yoy`, `inflation_mom`, `inflation_regime`
>
> **Utiliser uniquement leurs versions laggées :** `_lag1`, `_lag2`, `_lag3`

---

## 1. Identifiant temporel

| Colonne | Type | Description |
|---|---|---|
| `month` | DatetimeIndex (MS) | Index du dataset — 1er jour du mois |

---

## 2. CPI Silver — Source HCP

> Source : Haut-Commissariat au Plan (hcp.ma), indice base 2017=100.
> Bronze : `data/bronze/cpi_hcp_monthly_raw.csv`
> Silver : `data/silver/cpi_monthly.csv`

| Colonne | Type | Unité | Description |
|---|---|---|---|
| `ipc_level` | float | base 2017=100 | Indice des prix à la consommation brut |
| `inflation_mom` | float | % | Variation mensuelle : `(IPC_t / IPC_{t-1} - 1) × 100` |
| `inflation_yoy` | float | % | Variation annuelle : `(IPC_t / IPC_{t-12} - 1) × 100` — NaN pour les 12 premiers mois |
| `inflation_regime` | float | 0 ou 1 | 1 si `inflation_yoy >= 2%` (seuil Bank Al-Maghrib) — NaN si `inflation_yoy` est NaN |
| `publication_date` | date | — | Date estimée de publication HCP : fin du mois + 20 jours |

⚠️ `inflation_yoy` et `inflation_mom` sont **NaN** pour les 12/1 premiers mois (pas d'historique suffisant).

---

## 3. Trends Silver — Google Trends Maroc

> Source : pytrends, `geo='MA'`, agrégé mensuellement.
> Mots-clés en arabe et français (voir `src/ingestion/google_trends_v3.py`).
> Bronze : `data/bronze/google_trends_raw_v3.csv`
> Silver : `data/silver/google_trends_monthly.csv`
> Normalisation : min-max 0-1 indépendante pour chaque sous-indice.

| Colonne | Type | Plage | Description |
|---|---|---|---|
| `trends_prix_alim` | float | 0–1 | Sous-indice "prix alimentaires" (huile, hausse prix, légumes, arabe) |
| `trends_inflation` | float | 0–1 | Sous-indice "inflation" ("inflation maroc", التضخم في المغرب) |
| `trends_carburant` | float | 0–1 | Sous-indice "carburant" ("prix carburant maroc", ارتفاع الأسعار) |
| `trends_subvention` | float | 0–1 | Sous-indice "subventions" ("subvention maroc") — faible signal avant 2021 |
| `trends_composite` | float | 0–1 | Moyenne simple des 4 sous-indices disponibles |

---

## 4. Macro Silver — FAO FPI + BAM FX

> Sources : FAO Food Price Index (fao.org) + Bank Al-Maghrib (taux MAD/EUR).
> Bronze : `data/bronze/fao_food_price_raw.csv`, `data/bronze/bam_fx_raw.csv`
> Silver : `data/silver/macro_signals_monthly.csv`

| Colonne | Type | Unité | Description |
|---|---|---|---|
| `fao_food_index` | float | base 2014-2016=100 | Indice FAO des prix alimentaires mondial |
| `fao_cereals_index` | float | base 2014-2016=100 | Sous-indice céréales FAO (blé, maïs, riz) |
| `fao_oils_index` | float | base 2014-2016=100 | Sous-indice huiles FAO |
| `fao_food_yoy` | float | % | Variation annuelle FAO global |
| `fao_oils_yoy` | float | % | Variation annuelle huiles — pertinent pour le panier HCP Maroc |
| `mad_eur` | float | MAD/€ | Taux de change Dirham / Euro |
| `fx_yoy` | float | % | Dépréciation annuelle MAD/EUR : `(MAD_t / MAD_{t-12} - 1) × 100` |

**Justification économique :**
- Maroc importe ~60% de ses céréales → `fao_cereals_index` est un driver direct de l'IPC
- Panier HCP contient huile d'olive + tournesol → `fao_oils_index`
- Zone Euro = principal partenaire commercial → `fx_yoy` impacte le coût des importations

---

## 5. Indices BESI v3

> Générés par : `src/features/indexes.py`
> Silver : `data/silver/behavioral_index_pure.csv`, `data/silver/hybrid_macro_index.csv`
> Poids calibrés par LassoCV sur la période train 2010–2018.
> **Règle absolue : ipc_level, inflation_yoy, inflation_mom sont interdits dans les deux indices.**

| Colonne | Type | Plage | Description |
|---|---|---|---|
| `behavioral_index_pure` | float | 0–1 | Trends uniquement — 100% comportemental, 0% IPC, poids LassoCV |
| `hybrid_macro_index` | float | 0–1 | Trends + FAO FPI + MAD/EUR, poids LassoCV |

---

## 6. Lags explicites

> Format de nommage : `{feature}_lag{n}` où `n` = nombre de mois de décalage.
> Créés par `build_gold_dataset()` à partir de la configuration `FEATURE_LAGS`.

| Pattern | Lags | Disponibilité |
|---|---|---|
| `behavioral_index_pure_lag{1,2}` | 1, 2 | lag1 = BESI du mois précédent |
| `hybrid_macro_index_lag{1,2}` | 1, 2 | lag1 = hybrid du mois précédent |
| `trends_prix_alim_lag1` | 1 | Trends alimentaires du mois précédent |
| `trends_inflation_lag1` | 1 | Trends inflation du mois précédent |
| `trends_carburant_lag1` | 1 | Trends carburant du mois précédent |
| `trends_composite_lag1` | 1 | Composite Trends du mois précédent |
| `fao_food_index_lag1` | 1 | FAO global du mois précédent |
| `fao_food_yoy_lag1` | 1 | FAO YoY du mois précédent |
| `fao_oils_yoy_lag1` | 1 | FAO huiles YoY du mois précédent |
| `mad_eur_lag1` | 1 | MAD/EUR du mois précédent |
| `fx_yoy_lag1` | 1 | Dépréciation FX du mois précédent |
| `ipc_level_lag{1,2,3}` | 1, 2, 3 | **Seule forme légitime de l'IPC comme feature** |
| `inflation_yoy_lag{1,2}` | 1, 2 | YoY historique |
| `inflation_mom_lag{1,2}` | 1, 2 | MoM historique |

---

## 7. Cibles (variables à prédire)

> Toutes les cibles sont décalées d'un mois en avant (`shift(-1)`).
> Le **dernier mois du dataset est toujours NaN** pour toutes les cibles.

| Colonne | Type | Tâche | Description |
|---|---|---|---|
| `target_inflation_yoy_t1` | float | Régression | `inflation_yoy` du mois suivant — en % |
| `target_high_inflation_regime_t1` | float | Classification | 1.0 si `inflation_yoy(t+1) >= 2%`, 0.0 sinon, **NaN si inconnu** |
| `target_ipc_level_t1` | float | Régression | `ipc_level` du mois suivant |

---

## 8. Métadonnées

| Colonne | Type | Description |
|---|---|---|
| `as_of_date` | str (YYYY-MM-DD) | Dernier jour du mois courant — date à laquelle les features sont disponibles |
| `feature_available_at` | str (YYYY-MM-DD) | `as_of_date + 1 jour` — date minimale pour utiliser cette ligne en prédiction |
| `split_label` | str | Label de partition (voir ci-dessous) |

### Split labels

| Valeur | Période | Rôle |
|---|---|---|
| `train_A` | 2010-01 → 2017-12 | Entraînement bloc A |
| `test_A`  | 2018-01 → 2019-12 | Test bloc A (24 mois) |
| `train_B` | 2010-01 → 2019-12 | Entraînement bloc B (expanding) |
| `test_B`  | 2020-01 → 2021-12 | Test bloc B (COVID) |
| `train_C` | 2010-01 → 2021-12 | Entraînement bloc C (expanding) |
| `test_C`  | 2022-01 → 2024-12 | Test bloc C (choc inflationniste) |
| `unused`  | hors fenêtres | Non utilisé dans l'évaluation |

Une même date peut apparaître dans plusieurs labels (ex: `train_B|train_C`).

---

## 9. Features recommandées pour la modélisation

**Pour SARIMA/SARIMAX :**
```python
TARGET  = "ipc_level"           # série principale
EXOG_BEH = "behavioral_index_pure_lag1"
EXOG_HYB = "hybrid_macro_index_lag1"
```

**Pour la classification (régime inflation) :**
```python
TARGET = "target_high_inflation_regime_t1"
FEATURES = [
    "behavioral_index_pure_lag1", "hybrid_macro_index_lag1",
    "ipc_level_lag1", "inflation_yoy_lag1", "inflation_mom_lag1",
    "fao_food_yoy_lag1", "fx_yoy_lag1",
    "trends_prix_alim_lag1", "trends_inflation_lag1",
]
```

**Colonnes à NE JAMAIS utiliser comme features :**
```python
FORBIDDEN = [
    "ipc_level",        # contemporain = quasi-cible
    "inflation_yoy",    # contemporain = quasi-cible
    "inflation_mom",    # contemporain, publication J+20
    "inflation_regime", # dérivé de inflation_yoy courant
    "ipc_change",       # dérivée directe de la cible
    "target_*",         # les cibles elles-mêmes
]
```
