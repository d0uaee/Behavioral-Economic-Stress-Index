# Methodologie

Cette page explique la logique methodologique retenue dans la version finale.

## Question scientifique

Le projet teste si des signaux comportementaux digitaux marocains peuvent aider
a detecter des regimes d'inflation avant ou autour de la publication officielle
de l'IPC.

## Pourquoi une baseline naive ?

Une baseline naive est indispensable pour les series macroeconomiques
persistantes.

Dans le cas de l'IPC :

- la derniere valeur observee contient deja beaucoup d'information ;
- un modele simple peut donc etre difficile a battre ;
- il serait trompeur d'evaluer BESI seulement contre des modeles plus
  sophistiques.

## Pourquoi SARIMA / SARIMAX ?

Le choix de SARIMA / SARIMAX repose sur plusieurs raisons :

- cadre standard pour series temporelles mensuelles ;
- interpretation economique relativement claire ;
- integration naturelle de signaux exogenes ;
- comparaison raisonnable avec une baseline simple.

SARIMA sert de reference structurelle. SARIMAX permet de tester si BESI ajoute
une information utile.

## Deux niveaux d'evaluation

Le projet distingue volontairement :

### 1. Prevision point par point

Objectif :

- predire la valeur future de l'IPC ou de l'inflation.

Metriques :

- RMSE
- MAE
- MAPE
- AIC

### 2. Detection de regime

Objectif :

- identifier les episodes d'inflation elevee.

Metriques :

- AUC
- F1
- Recall
- Precision

Cette distinction est centrale :

> un signal peut etre utile pour l'alerte sans etre meilleur en RMSE global.

## Interdiction du leakage

La version finale V3 retire toute composante directe de l'IPC hors de
l'indice BESI.

Concretement :

- `ipc_change` a ete retire du BESI ;
- les features contemporaines de la cible ne doivent pas etre injectees ;
- seuls les lags autorises de l'IPC sont utilisables pour predire `t+1`.

## Protocole d'evaluation

Le protocole repose sur un **walk-forward expanding window**.

Blocs d'evaluation :

- **Bloc A** : 2020-2021
- **Bloc B** : 2022-2024

Pourquoi cette separation ?

- Bloc A capture la periode COVID ;
- Bloc B capture la crise inflationniste 2022-2024 ;
- cela permet de distinguer des environnements economiques differents.

## Hypotheses testees

- **H1** : le BESI comportemental apporte une information utile pour la
  detection des regimes d'inflation.
- **H2** : l'ajout de variables macroeconomiques au BESI comportemental
  ameliore cette detection.
- **Extension exploratoire** : un signal NLP presse marocaine ajoute-t-il une
  information conditionnelle supplementaire ?

## Questions auxquelles cette page repond

- Pourquoi comparer au naive ?
- Pourquoi separer forecasting et warning ?
- Pourquoi dire qu'un meilleur AIC ne garantit pas un meilleur RMSE ?
- Pourquoi le leakage est-il interdit ?

