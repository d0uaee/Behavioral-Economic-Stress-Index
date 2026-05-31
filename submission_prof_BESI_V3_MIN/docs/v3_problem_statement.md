# BESI V3 - Problem statement

**Projet :** Detection precoce des regimes d'inflation au Maroc  
**Version :** V3  
**Date :** Mai 2026

## Question de recherche

> Est-ce que des signaux digitaux comportementaux marocains peuvent aider a
> detecter des regimes d'inflation avant ou autour de la publication officielle
> de l'IPC ?

## Cibles retenues

| Cible | Variable | Type | Horizon |
|---|---|---|---|
| Principale | `target_inflation_yoy_t1` | continue | t+1 mois |
| Secondaire | `target_high_inflation_regime_t1` | binaire | t+1 mois |

Definition du regime :

- `1` si `inflation_yoy(t+1) >= 2.0%`
- `0` sinon

## Hypotheses

- **H1** : le BESI behavioral apporte une information utile pour detecter les
  regimes d'inflation
- **H2** : l'ajout d'un signal macro ameliore cette detection
- **Extension exploratoire** : un signal presse local ajoute-t-il quelque chose
  au BESI principal ?

## Perimetre V3

Le projet V3 est volontairement restreint a :

- l'IPC HCP reel 2017-2024
- Google Trends comme coeur comportemental
- FAO + taux de change comme extension macro
- Hespress comme extension NLP exploratoire

Le projet ne pretend pas :

- mesurer tout le stress economique des menages
- prouver une causalite forte entre Google Trends et inflation
- battre forcement la baseline naive

## Regle methodologique cle

Pour predire `t+1`, on ne doit jamais injecter directement la cible courante
dans les features.

Donc :

- `ipc_change` est retire du BESI V3
- seuls les lags autorises de l'IPC sont utilisables comme features
- les evaluations doivent etre faites sans leakage

## Lecture attendue des resultats

Trois cas sont possibles :

1. le BESI ameliore forecasting et detection
2. le BESI aide surtout la detection de regime
3. le BESI n'apporte pas d'information utile

Le projet final se situe dans le **cas 2**.
