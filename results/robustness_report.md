# Robustesse BESI hors fenêtre 2022

## Méthodologie
- Scénario 1 : exclusion de 2021-12 à 2022-06 (7 mois).
- Scénario 2 : exclusion de tout le Bloc B (2022-2024).
- Modèles : SARIMA(1,1,1)(1,0,1)[12] et SARIMAX + BESI behavioral.
- Seuil de stress élevé : 75e percentile de l'inflation YoY appris sur l'entraînement uniquement.

## Référence modèle complète

| Modèle | AIC | RMSE | Delta AIC vs SARIMA |
|---|---:|---:|---:|
| SARIMA | 64.85 | 1.923 | — |
| SARIMAX+BESI | 57.09 | 1.891 | -7.77 |

## Résultats robustesse

### Sans mars 2022 ±3 mois

| Modèle | Train | Test | AIC | BIC | RMSE | Recall | AUC | Stress % test | YoY moyen test |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| SARIMA | 2017-01-01 → 2021-11-01 | 2022-07-01 → 2024-12-01 | 147.99 | 156.91 | 2.8126 | 1.0000 | 1.0000 | 38.89% | 3.5675 |
| SARIMAX+BESI | 2017-01-01 → 2021-11-01 | 2022-07-01 → 2024-12-01 | 149.76 | 160.47 | 2.8866 | 1.0000 | 1.0000 | 38.89% | 3.5675 |

- Delta AIC (BESI - SARIMA) = 1.77
- Delta RMSE (BESI - SARIMA) = +0.0740
- Conclusion orale : BESI utilise principalement le signal de crise, à reconnaître honnêtement.

### Sans Bloc B entier

| Modèle | Train | Test | AIC | BIC | RMSE | Recall | AUC | Stress % test | YoY moyen test |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| SARIMA | 2017-01-01 → 2019-12-01 | 2020-01-01 → 2021-12-01 | 64.85 | 70.08 | 4.3933 | 0.3333 | 0.9259 | 25.00% | 0.6349 |
| SARIMAX+BESI | 2017-01-01 → 2019-12-01 | 2020-01-01 → 2021-12-01 | 58.10 | 64.36 | 1.6564 | 0.3333 | 0.7778 | 25.00% | 0.6349 |

- Delta AIC (BESI - SARIMA) = -6.75
- Delta RMSE (BESI - SARIMA) = -2.7369
- Conclusion orale : BESI apporte de l'info AUSSI hors crise.

## Phrase défensive pour l'oral
Le seuil de stress élevé est fixé à partir du train uniquement, au 75e percentile de l'inflation YoY, puis appliqué tel quel aux périodes de test sans recalibration ni fuite d'information future.
