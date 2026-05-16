# BESI v3 — Problem Statement

**Projet :** Détection Précoce des Régimes d'Inflation au Maroc  
**Version :** v3.0 (données réelles, zéro simulation)  
**Date :** Mai 2026

---

## Question de recherche

> Est-ce que des signaux digitaux comportementaux (Google Trends) et des
> variables macro observables (FAO Food Price Index, taux de change MAD/EUR)
> améliorent la détection ou la prévision des régimes d'inflation marocains,
> au-delà d'un modèle SARIMA baseline ?

---

## Cibles

| Cible | Variable | Type | Horizon |
|---|---|---|---|
| **Principale** | `target_inflation_yoy_t1` | Float (%) | t+1 mois |
| **Secondaire** | `target_high_inflation_regime_t1` | Binaire (0/1) | t+1 mois |

**Définition du régime haute inflation :**
`high_inflation_regime = 1` si `ipc_yoy(t+1) >= 2.0%`  
Seuil basé sur la cible de stabilité des prix de Bank Al-Maghrib.

---

## Hypothèses testables

**H1 :** Les signaux Google Trends sur les prix alimentaires précèdent
les variations de l'IPC marocain d'au moins 1 mois (causalité de Granger).

**H2 :** Un SARIMAX incluant le FAO Food Price Index (disponible avant publication HCP)
produit un RMSE inférieur au SARIMA baseline sur la période 2019-2024.

**H3 :** Un indice composite de signaux digitaux + macro permet de détecter
les transitions vers un régime haute inflation avec F1 > 0.5 sur données test.

---

## Ce qui est hors scope (v3)

- Prédiction du "stress des ménages" au sens large (revenu, dette, chômage)
- Prédiction de l'IPC à horizon > 3 mois
- Modèles deep learning (données trop courtes : 180 obs max)
- Reddit / YouTube comme sources primaires (données pré-2016 insuffisantes)

---

## Sources de données autorisées en v3

| Source | Variable | Disponible avant IPC(t) ? | Statut |
|---|---|---|---|
| HCP Maroc | IPC mensuel (base 2017=100) | Non — c'est la cible | Obligatoire |
| Google Trends | 3 sous-indices thématiques | Oui (J-30) | Obligatoire |
| FAO Food Price Index | Indice prix alimentaires mondiaux | Oui (J-7) | Obligatoire |
| Bank Al-Maghrib | Taux MAD/EUR mensuel | Oui (temps réel) | Obligatoire |
| IMF Commodity Prices | Blé, huile végétale | Oui (mensuel) | Recommandé |
| Presse marocaine NLP | Score sentiment articles | Oui si scraping réel | Optionnel |
| Reddit r/Morocco | Sentiment posts (NLP réel) | Oui | Optionnel (données < 2016 absentes) |

---

## Règle as-of-date

Pour prédire `IPC(t)`, seules les features disponibles
avant la date de publication officielle de `IPC(t)` (~J+20 du mois t) sont autorisées.

```
Exemple : prédire IPC Mars 2024 (publié ~20 Avril 2024)
  ✅ Autorisé  : Google Trends Mars 2024, FAO Février 2024, taux MAD/EUR Mars 2024
  ❌ Interdit  : IPC Mars 2024, IPC_yoy Mars 2024, ipc_change Mars 2024
```

---

## Définition BESI v3

Deux indices séparés et traçables :

```
behavioral_index_pure =
    w1 * trends_prix_alim    # Google Trends "prix huile" + "hausse prix"
  + w2 * trends_inflation    # Google Trends "inflation maroc" (anchor)
  + w3 * trends_carburant    # Google Trends "prix carburant maroc"
  (poids w1/w2/w3 calibrés par LassoCV sur train 2010-2018 uniquement)

hybrid_macro_index =
    a1 * behavioral_index_pure
  + a2 * fao_food_index_yoy  # variation annuelle FAO FPI (publié avant IPC)
  + a3 * mad_eur_change      # dépréciation MAD vs EUR (pression importations)
  (poids calibrés par LassoCV sur train 2010-2018 uniquement)
```

**Interdit dans les deux indices :**
- `ipc_change`, `ipc_yoy`, `ipc_mom`, `ipc_level` — data leakage direct
- Toute donnée simulée ou interpolée annuelle → mensuelle

---

## Métriques d'évaluation

### Forecasting (cible continue)
- RMSE, MAE, MAPE sur `inflation_yoy_t1`
- Comparaison baseline : SARIMA / Naif / SARIMAX_Trends / SARIMAX_Hybrid

### Warning (cible binaire)
- Courbe Precision-Recall (pas juste un point)
- F1 optimal sur train, évalué sur test
- Lead time moyen en mois (si F1 > 0.4)

### Fenêtres d'évaluation (3 périodes distinctes)
- Bloc A : test 2018-2019 (train 2010-2017) — période stable
- Bloc B : test 2020-2021 (train 2010-2019) — covid
- Bloc C : test 2022-2024 (train 2010-2021) — choc inflationniste

---

## Ce que le projet NE prétend PAS

- BESI ne "prédit" pas l'inflation — il fournit un signal corrélé en avance
- Un lead time de 12 mois n'est pas une prédiction causale de la guerre en Ukraine
- Reddit et YouTube ne sont pas des données comportementales représentatives des ménages marocains
- SARIMA peut rester le meilleur modèle — c'est un résultat honnête, pas un échec

---

*Document de référence v3 — toute claim dans README ou soutenance doit être alignée avec ce fichier.*
