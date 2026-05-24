# Audit Leakage — Seuil de stress élevé

Date de l'audit : 2026-05-19

## Conclusion courte

Le pipeline d'évaluation ne recalcule pas le seuil sur le bloc test. Le point faible était la définition du régime cible : avant correction, le label de stress élevé reposait sur un seuil fixe de 2.0% YoY dans [src/transforms/cpi.py](../../src/transforms/cpi.py) et [src/gold/build_model_dataset.py](../../src/gold/build_model_dataset.py). Le code d'évaluation utilise maintenant un seuil appris uniquement sur le train du bloc, défini comme le 75e percentile de l'inflation YoY observée dans l'entraînement.

## Réponses à l'audit

### a) Comment le seuil de "stress élevé" est défini actuellement ?

Il est désormais défini comme le 75e percentile de l'inflation YoY observée sur la période d'entraînement du bloc, via `STRESS_REGIME_PERCENTILE = 75` et `_stress_regime_threshold(...)` dans [src/evaluation/warning_metrics.py](../../src/evaluation/warning_metrics.py#L71).

### b) Est-il calculé sur l'inflation YoY de quelle période ?

Sur le train uniquement, sans utiliser le bloc test. Pour le Bloc B, le seuil est appris sur 2017-2021 et appliqué tel quel sur 2022-2024.

### c) Est-ce qu'il y a recalibration sur le Bloc B ?

Non. Le seuil de stress est figé après apprentissage sur le train. Le seuil de détection du signal est aussi calibré sur train uniquement, mais il s'agit d'un seuil différent: celui du score BESI, pas du régime cible.

### d) Y a-t-il d'autres formes de leakage ?

Oui, une fuite potentielle existait dans la branche PCA de [src/features/indexes.py](../../src/features/indexes.py#L138) : le scaler et le PCA étaient ajustés sur toute la série. Cela a été corrigé pour fitter uniquement sur le train. Dans [src/evaluation/backtest.py](../../src/evaluation/backtest.py#L166), le walk-forward reste propre: les exogènes sont reindexés et remplis uniquement à l'intérieur de la fenêtre d'entraînement courante.

## Transformations et points de fit

| Composant | Transformation | Fit sur train uniquement ? | Remarque |
|---|---|---:|---|
| `src/transforms/cpi.py` | `ipc_level` → `inflation_mom`, `inflation_yoy`, `inflation_regime` | Non | Transformation déterministe sur toute la série; le seuil 2.0% était fixe avant correction. |
| `src/gold/build_model_dataset.py` | lags explicites, `shift(-1)` pour les targets | Non | Pas d'ajustement statistique, uniquement des décalages temporels. |
| `src/evaluation/warning_metrics.py` | seuil stress = 75e percentile de `inflation_yoy` | Oui | Appliqué sans recalibration au bloc test. |
| `src/evaluation/warning_metrics.py` | seuil score BESI maximisant F1 | Oui | Calibré sur le train du bloc uniquement. |
| `src/evaluation/backtest.py` | walk-forward expanding window | Oui | Chaque itération refit le modèle sur l'historique disponible jusqu'à t-1. |
| `src/features/indexes.py` | StandardScaler + PCA (branche `method="pca"`) | Oui, après correction | Avant correction, cette branche fit sur toute la série. |

## Avant / après

Les métriques ci-dessous comparent l'ancien export `classification_metrics.csv` avec le recalcul corrigé.

| Modèle | Bloc | Recall ancien | Recall nouveau | Precision ancienne | Precision nouvelle | F1 ancien | F1 nouveau | Specificity ancienne | Specificity nouvelle | Bal.Acc ancienne | Bal.Acc nouvelle | AUC ancien | AUC nouveau | AP ancien | AP nouveau |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SARIMAX + BESI behavioral | A | 0.3750 | 1.0000 | 0.2143 | 0.3333 | 0.2727 | 0.5000 | 0.3125 | 0.0000 | 0.3438 | 0.5000 | 0.3281 | 0.3281 | 0.2416 | 0.2416 |
| SARIMAX + Hybrid macro | A | 1.0000 | 1.0000 | 0.3333 | 0.3333 | 0.5000 | 0.5000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 0.5625 | 0.5625 | 0.3123 | 0.3123 |
| SARIMAX + BESI behavioral | B | 1.0000 | 1.0000 | 0.6857 | 0.6857 | 0.8136 | 0.8136 | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 0.3106 | 0.3106 | 0.5685 | 0.5685 |
| SARIMAX + Hybrid macro | B | 0.3750 | 0.3750 | 0.5294 | 0.5294 | 0.4390 | 0.4390 | 0.2727 | 0.2727 | 0.3239 | 0.3239 | 0.3561 | 0.3561 | 0.6769 | 0.6769 |
| SARIMAX + BESI behavioral | global | 0.9000 | 0.0000 | 0.4865 | 0.0000 | 0.6316 | 0.0000 | 0.1364 | 1.0000 | 0.5182 | 0.5000 | 0.6233 | 0.6584 | 0.5475 | 0.5412 |
| SARIMAX + Hybrid macro | global | 0.6250 | 0.0000 | 0.4717 | 0.0000 | 0.5376 | 0.0000 | 0.3636 | 1.0000 | 0.4943 | 0.5000 | 0.4511 | 0.4641 | 0.4515 | 0.4332 |

## Seuils appris

| Bloc | Seuil YoY appris sur train |
|---|---:|
| A | 2.3191 |
| B | 2.4169 |
| global | 2.4169 |

## Recommandations

1. Conserver la règle train-only pour le seuil de stress élevé et l'annoter dans les rapports.
2. Ajouter un test de non-régression qui vérifie que le seuil de stress est calculé sur `train_*` uniquement.
3. Garder la branche PCA de [src/features/indexes.py](../../src/features/indexes.py#L138) sur train-only et éviter tout `fit_transform` sur la série complète.
4. Éviter d'exposer dans l'oral le seuil fixe 2.0% comme règle de décision: il faut parler du seuil appris sur l'entraînement.
