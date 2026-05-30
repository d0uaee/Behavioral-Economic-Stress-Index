# Resultats

Cette page synthetise les resultats finaux et leur interpretation.

## Message principal

Le projet est **plus fort en detection de regime** qu'en prevision point par
point.

## Forecasting global

| Modele | RMSE |
|---|---:|
| Naif | 1.609 |
| SARIMA | 1.923 |
| SARIMAX + BESI behavioral | 1.891 |
| SARIMAX + Hybrid macro | 1.997 |

### Lecture correcte du forecasting

- la baseline **naive** reste la meilleure ;
- le **BESI behavioral** ameliore SARIMA ;
- l'indice **hybrid macro** degrade la performance ;
- le projet ne doit pas pretendre battre toutes les baselines.

## Early warning

### Behavioral global

- `AUC = 0.574`
- `F1 = 0.703`
- `Recall = 1.000`

### Behavioral Bloc B (2022-2024)

- `F1 = 0.814`
- `Recall = 1.000`

### Hybrid global

- `AUC = 0.376`
- `F1 = 0.466`
- `Recall = 0.531`

### Hybrid Bloc B

- `Recall = 0.375`

### Lecture correcte de l'early warning

- le **BESI behavioral** detecte tous les mois a inflation elevee ;
- sa faiblesse reste la **specificite** ;
- sa vraie force est la **detection de regime** ;
- l'ajout macro **affaiblit** la detection dans cette version.

## Fit in-sample

Le BESI behavioral ameliore l'AIC de `-7.77` points par rapport a SARIMA.

Interpretation :

- le signal apporte de l'information structurelle utile ;
- mais un meilleur AIC ne garantit pas automatiquement un meilleur RMSE global.

## Rupture structurelle 2022

| Statistique | Valeur |
|---|---|
| Inflation moyenne pre-2022 | +0.74% YoY |
| Inflation moyenne post-2022 | +8.53% YoY |
| Facteur multiplicatif | x11.6 |
| Significativite | p < 0.0001 |

Interpretation :

- 2022 constitue une vraie rupture de regime ;
- cela renforce l'interet d'un signal d'alerte plutot que d'une simple
  extrapolation lineaire.

## Verdict des hypotheses

### H1 - Partiellement validee

H1 est **partiellement validee** parce que :

- le BESI behavioral est utile pour detecter les regimes inflationnistes ;
- il ameliore SARIMA ;
- mais il ne bat pas la baseline naive en RMSE global.

### H2 - Rejetee

H2 est **rejetee** parce que :

- l'ajout macro degrade la detection ;
- cet effet est particulierement visible sur 2022-2024.

## Resultat a retenir pour la soutenance

> Le projet ne montre pas un modele miracle de prevision de l'inflation. Il
> montre qu'un signal comportemental marocain peut etre utile pour detecter des
> regimes inflationnistes dans un cadre methodologiquement propre.

## Sources de detail

Pour les chiffres complets, voir :

- `outputs/reports/backtest_v3_summary.csv`
- `outputs/reports/warning_metrics_v3.csv`
- `outputs/reports/results_v3_final.md`
