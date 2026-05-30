# FAQ

## Le projet mesure-t-il le stress economique complet des menages ?

Non. La version finale est recentree sur la **detection des regimes
d'inflation** a partir de signaux digitaux.

## Pourquoi la baseline naive est-elle si forte ?

Parce que l'IPC est une serie persistante. La derniere valeur observee contient
deja beaucoup d'information sur la suivante.

## Si le naive gagne, a quoi sert BESI ?

Le BESI apporte surtout de la valeur en **early warning** et en detection de
regime, pas encore en prevision point par point.

## Pourquoi H1 est-elle seulement partiellement validee ?

Parce que le BESI behavioral :

- ameliore SARIMA ;
- detecte bien les episodes inflationnistes ;
- mais ne bat pas la baseline naive en RMSE global.

## Pourquoi H2 est-elle rejetee ?

Parce que l'ajout du macro degrade la detection, en particulier sur la periode
inflationniste 2022-2024.

## Pourquoi l'extension NLP est-elle CAS C ?

Parce que le Lasso attribue un poids nul au signal NLP (`beta = 0.0`) une fois
le BESI Trends deja present.

## Le projet prouve-t-il que Google Trends cause l'inflation ?

Non. Le projet montre une relation informative utile, mais ne prouve pas une
causalite forte.

## Le projet est-il pret pour une utilisation en production ?

Non. La version finale est solide académiquement et methodologiquement, mais
reste un projet de soutenance et d'exploration rigoureuse.

