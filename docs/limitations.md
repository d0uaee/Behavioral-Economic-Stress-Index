# Limites

Documenter les limites de maniere explicite renforce la credibilite du projet.

## 1. Taille de l'echantillon

La cible principale repose sur **96 observations mensuelles** entre 2017 et
2024.

Implication :

- le projet permet une evaluation serieuse ;
- mais la profondeur historique reste limitee pour des conclusions tres fortes.

## 2. Force de la baseline naive

Le modele naive reste meilleur en RMSE global.

Implication :

- le BESI apporte surtout une valeur de **detection de regime** ;
- il ne faut pas sur-vendre la prevision quantitative.

## 3. Specificite imparfaite du signal behavioral

Le BESI behavioral detecte tous les mois a inflation elevee, mais il genere
encore des fausses alertes.

Implication :

- bonne sensibilite ;
- specificite encore perfectible.

## 4. Extension macro insuffisamment specifique

L'ajout de FAO et du taux MAD/EUR ne renforce pas la detection.

Implication :

- un signal macro global peut etre trop general pour saisir le contexte local
  marocain sur cette periode.

## 5. Nature du signal NLP

Le signal Hespress est base sur `title + excerpt` et non sur des commentaires
lecteurs.

Implication :

- il s'agit d'un signal **editorial** ;
- il ne faut pas l'interpreter comme un veritable sentiment conversationnel des
  menages.

## 6. Portee scientifique

Le projet ne demontre pas :

- une causalite forte ;
- un systeme deja deployable en production ;
- une mesure complete du stress economique des menages.

## Formulation recommandee

La bonne facon de presenter les limites est la suivante :

> Les resultats sont suffisamment solides pour soutenir une contribution
> methodologique et exploratoire credible, mais pas pour pretendre a un systeme
> de prevision definitif.

