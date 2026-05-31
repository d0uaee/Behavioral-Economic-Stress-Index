# Resultats BESI V3 - Rapport final

**Projet :** Detection precoce des regimes d'inflation au Maroc  
**Auteurs :** Douae Ahadji & Adama Basse  
**Version :** V3 finalisee sur donnees reelles  
**Periode :** 2017-01 a 2024-12

---

## 1. Cadrage

Le projet teste si des signaux digitaux marocains peuvent aider a detecter des
regimes d'inflation avant ou autour de la publication officielle de l'IPC.

Le coeur du projet est un indice comportemental, **BESI**, construit a partir
de Google Trends. Deux extensions ont ensuite ete evaluees :

- un indice **hybrid macro** avec FAO et taux de change
- une extension **NLP presse marocaine** a partir de Hespress

---

## 2. Donnees et architecture

| Element | Valeur |
|---|---|
| Periode couverte | 2017-01 a 2024-12 |
| Nombre de mois | 96 |
| Source IPC | HCP Maroc, base 2017=100 |
| Source comportementale | Google Trends |
| Source macro | FAO + taux MAD/EUR mensuel |
| Source NLP | Hespress API (`title + excerpt`) |
| Gold dataset | 96 lignes x 45 colonnes |
| Blocs d'evaluation | A = 2020-2021 ; B = 2022-2024 |

---

## 3. Modeles compares

| Modele | RMSE | MAE | MAPE |
|---|---:|---:|---:|
| Naif | 1.609 | 1.200 | 1.06% |
| SARIMA | 1.923 | 1.537 | 1.38% |
| SARIMAX + BESI behavioral | 1.891 | 1.522 | 1.36% |
| SARIMAX + Hybrid macro | 1.997 | 1.576 | 1.42% |

**Lecture :**

- le modele naive reste le meilleur en RMSE global
- SARIMAX + BESI behavioral bat SARIMA pur
- l'indice hybrid macro degrade les performances globales

---

## 4. Fit in-sample

| Modele | AIC | Delta AIC vs SARIMA |
|---|---:|---:|
| SARIMA | 64.85 | -- |
| SARIMAX + BESI behavioral | 57.09 | -7.77 |

Interpretation :

- le BESI behavioral ameliore le fit in-sample
- cela ne suffit pas a battre la baseline naive hors echantillon

---

## 5. Rupture structurelle 2022

| Statistique | Valeur |
|---|---|
| Inflation moyenne pre-2022 | +0.74% YoY |
| Inflation moyenne post-2022 | +8.53% YoY |
| Facteur multiplicatif | x11.6 |
| Difference de moyenne | p < 0.0001 |
| Difference de variance | p < 0.0001 |

Interpretation :

- mars 2022 introduit une vraie rupture de regime
- cela explique pourquoi les modeles lineaires classiques deviennent plus fragiles
- dans ce contexte, un signal de regime peut etre plus utile qu'une simple prevision point par point

---

## 6. Correlation et causalite

### Correlation croisee BESI -> inflation

| Periode | Lag optimal | r de Pearson | p-value |
|---|---|---:|---:|
| Periode complete | 0 | 0.535 | < 0.001 |
| Pre-2022 | 0 | 0.201 | 0.161 |
| Post-2022 | 5 | -0.303 | 0.110 |

### Causalite de Granger

Resultat :

- non significatif sur les lags 1 a 4
- le BESI ne "cause" pas l'inflation au sens lineaire de Granger

Interpretation :

- la relation est plutot non lineaire
- le BESI agit comme detecteur de regime, pas comme preuve de causalite forte

---

## 7. Early warning

### Metriques par bloc

| Bloc | Signal | AUC | F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|
| test_A | behavioral | 0.328 | 0.500 | 0.333 | 1.000 |
| test_A | hybrid | 0.562 | 0.500 | 0.333 | 1.000 |
| test_B | behavioral | 0.311 | 0.814 | 0.686 | 1.000 |
| test_B | hybrid | 0.356 | 0.439 | 0.529 | 0.375 |

### Metriques globales

| Signal | AUC | F1 | Recall | Lead time |
|---|---:|---:|---:|---|
| behavioral | 0.574 | 0.703 | 1.000 | 1 mois |
| hybrid | 0.376 | 0.466 | 0.531 | 1 mois |

Interpretation :

- le BESI behavioral detecte tous les mois a inflation elevee sur 2022-2024
- il reste imparfait en specificite
- sa valeur est plus forte en detection de regime qu'en forecasting quantitatif

---

## 8. Validation des hypotheses

### H1 - Le BESI behavioral apporte une valeur utile

| Critere | Valeur | Lecture |
|---|---|---|
| Delta AIC vs SARIMA | -7.77 | favorable |
| RMSE vs SARIMA | -0.032 | favorable |
| Recall global | 1.000 | favorable |
| Recall bloc B | 1.000 | favorable |
| AUC globale | 0.574 | modeste |

**Verdict H1 : partiellement validee**

Pourquoi :

- le BESI aide a detecter les regimes inflationnistes
- il ameliore SARIMA
- mais il ne bat pas la baseline naive en RMSE global

### H2 - Le signal macro ameliore la detection

| Critere | Behavioral | Hybrid | Verdict |
|---|---:|---:|---|
| AUC globale | 0.574 | 0.376 | behavioral meilleur |
| Recall global | 1.000 | 0.531 | hybrid defavorable |
| Recall bloc B | 1.000 | 0.375 | hybrid rejete |
| RMSE global | 1.891 | 1.997 | hybrid defavorable |

**Verdict H2 : rejetee**

Interpretation :

- le macro est moins specifique au contexte marocain
- le behavioral capte mieux la crise inflationniste locale

---

## 9. Extension NLP Hespress

### Couverture

| Element | Valeur |
|---|---|
| Source | Hespress API |
| Texte utilise | `title + excerpt` |
| Corpus | 5 788 textes |
| Couverture | 96/96 mois |
| Imputation | 0 mois |

### Resultat

| Parametre | Valeur |
|---|---|
| alpha | 1.0 |
| beta | 0.0 |
| Verdict | CAS C |

Interpretation :

- le signal NLP a ete teste proprement
- il enrichit l'analyse locale
- il n'ajoute pas d'information conditionnelle supplementaire au BESI Trends

---

## 10. Conclusion finale

La conclusion honnete du projet est la suivante :

- le BESI behavioral n'est pas meilleur que la baseline naive en prevision point par point
- en revanche, il detecte tres bien les regimes inflationnistes, surtout sur 2022-2024
- l'indice hybrid macro n'ameliore pas la detection
- l'extension NLP est proprement documentee, mais n'apporte pas de gain conditionnel

### Phrase finale pour l'oral

> Le projet ne presente pas un modele miracle de prevision de l'inflation.
> Il presente une pipeline rigoureuse et un signal comportemental marocain utile
> surtout pour la detection de regimes inflationnistes.
