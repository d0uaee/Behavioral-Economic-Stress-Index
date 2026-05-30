# Donnees

Cette page decrit les donnees effectivement utilisees dans la **version finale**
du projet.

## Perimetre temporel

- **Periode analysee** : 2017-01 a 2024-12
- **Frequence** : mensuelle
- **Nombre de points** : 96 mois

## Source cible : IPC HCP

La verite terrain principale est l'**IPC mensuel du HCP**.

- Source : Haut-Commissariat au Plan
- Base : `2017=100`
- Variable brute : `ipc_level`
- Variables derivees :
  - `inflation_mom`
  - `inflation_yoy`

## Signal comportemental : Google Trends

Google Trends constitue le coeur du **BESI behavioral**.

Sous-indices thematiques :

- `trends_prix_alim`
- `trends_inflation`
- `trends_carburant`
- `trends_subvention`
- `trends_composite`

Interpretation :

- il s'agit d'un **signal de comportement de recherche** ;
- ce n'est ni une preuve de causalite, ni une mesure exhaustive du stress des
  menages.

## Signal macro : FAO + MAD/EUR

L'extension macro agrupe :

- `fao_food_index`
- `fao_food_yoy`
- `fao_oils_yoy`
- `mad_eur`
- `fx_yoy`

Objectif :

- tester si des facteurs macro externes ameliorent la detection des regimes
  d'inflation.

## Extension NLP : Hespress

L'extension NLP repose sur un corpus presse local.

- Source : Hespress API WordPress
- Texte utilise : `title + excerpt`
- Corpus total : **5 788 textes**
- Couverture : **96/96 mois**
- Imputation : **0**

Point cle :

> Ce signal est un **signal presse editorial**. Ce n'est pas un veritable
> signal conversationnel de lecteurs.

## Gold dataset final

Le fichier central de modelisation est :

- `data/gold/model_dataset_monthly.csv`

Il contient :

- les variables IPC ;
- les signaux Trends ;
- les signaux macro ;
- les indices BESI ;
- les lags explicites ;
- les cibles `t+1` ;
- les labels de split.

Pour le detail colonne par colonne, voir [](data_dictionary_v3.md).

## Questions auxquelles cette page repond

- Quelles donnees sont vraiment utilisees en V3 ?
- Quelle est la cible du projet ?
- Quelle est la difference entre signal comportemental, macro et NLP ?

