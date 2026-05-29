L'Indice BESI
=============

Definition
----------

Le **BESI** (*Behavioral Economic Stress Index*) est un indice composite
qui mesure l'intensite des signaux comportementaux lies aux regimes
d'inflation au Maroc, a partir de donnees digitales librement accessibles.

Formule BESI V3 (behavioral pur)
----------------------------------

.. math::

   BESI_{behavioral}(t) = \sum_{k=1}^{K} w_k \cdot Trends_k(t)

Ou :

- :math:`Trends_k(t)` = score normalise du keyword k au mois t (entre 0 et 1)
- :math:`w_k` = poids calibre par **LassoCV** sur le jeu d'entrainement
- K = 7 keywords selectionnes automatiquement

.. important::

   **Difference cle avec V1/V2 :** la composante IPC est completement
   retiree de la formule BESI. L'IPC est la variable cible — l'inclure
   dans les features serait du data leakage.

Formule BESI Hybrid (V3 — test H2)
------------------------------------

.. math::

   BESI_{hybrid}(t) = f(BESI_{pure}(t),\ FAO\_fpi\_yoy(t),\ fx\_yoy(t))

Utilise pour tester H2. **Resultat : H2 rejetee** — le Recall Bloc B
chute de 1.00 a 0.375 avec le signal macro.

NLP Hespress — CAS C
----------------------

**Contexte :** Reddit et YouTube n'ont pas pu etre collectes (API refusee,
quota depasse). Ils ont ete remplaces par la **presse marocaine via flux RSS**
(Hespress, Le360, Medias24).

**Ce que le NLP Hespress mesure :** volume mensuel d'articles economiques
publies par la presse marocaine, normalise entre 0 et 1. C'est un signal
textuel qui capte l'intensite de la couverture mediatique de l'inflation.

**Verdict CAS C :**

+----------------------------+-------------------------------------------+
| Caracteristique            | Valeur                                    |
+============================+===========================================+
| Couverture historique      | 30 a 90 jours (limite RSS)                |
+----------------------------+-------------------------------------------+
| Usage dans le projet       | Validation recente uniquement             |
+----------------------------+-------------------------------------------+
| Integre dans BESI principal| Non — signal complementaire               |
+----------------------------+-------------------------------------------+
| Avantage vs Reddit         | Plus representatif des menages marocains  |
+----------------------------+-------------------------------------------+
| Fichier                    | data/silver/press_signal_monthly.csv      |
+----------------------------+-------------------------------------------+

.. note::

   Le CAS C signifie que le NLP Hespress est un **signal de validation
   recent** — il confirme que les comportements detectes par Google Trends
   se retrouvent aussi dans la couverture mediatique, mais il ne peut pas
   remplacer le BESI sur toute la periode 2017-2024 faute d'historique RSS.

Technique du chunking ancre (Google Trends)
--------------------------------------------

Google Trends normalise chaque serie independamment entre 0 et 100.
Pour rendre les series comparables entre batchs, chaque requete inclut
le meme mot-cle ancre "inflation maroc" :

.. code-block:: text

   Batch 1 : ["inflation maroc", "prix huile", "hausse prix"]
   Batch 2 : ["inflation maroc", "ghla lprix", "التضخم في المغرب"]
   Batch 3 : ["inflation maroc", "prix carburant", "subvention"]

Le ratio entre les valeurs de l'ancre dans chaque batch permet de
recalibrer toutes les series sur une echelle commune.

Selection automatique des keywords
------------------------------------

La selection des mots-cles suit 4 etapes pour repondre a la critique
"pourquoi ces keywords ?" :

1. **Generation automatique** : pytrends genere 50+ candidats
2. **Decomposition STL** : suppression de la saisonnalite (Ramadan)
   avant correlation — la correlation brute passe de r=0.535 a r=0.228
   apres correction
3. **Filtrage** : seuls les keywords avec r > 0.25 sur les residus STL
4. **Clustering K-Means** : elimination des redondances, 7 groupes finaux

Interpretation des valeurs BESI
---------------------------------

+---------------------+--------------------+-----------------------------------+
| Valeur BESI         | Etat               | Interpretation                    |
+=====================+====================+===================================+
| 0.00 a 0.35         | Normal             | Regime d'inflation stable         |
+---------------------+--------------------+-----------------------------------+
| 0.35 a 0.65         | Warning            | Transition vers regime inflatoire |
+---------------------+--------------------+-----------------------------------+
| 0.65 a 1.00         | High Stress        | Regime d'inflation elevee         |
+---------------------+--------------------+-----------------------------------+

Pourquoi le BESI perd de la puissance apres 2022
-------------------------------------------------

Avant 2022, l'inflation etait **transitoire** — les menages cherchaient
sur Google parce que la situation etait inhabituelle. Le BESI precedait
bien les changements de regime.

Apres 2022, l'inflation est devenue **structurelle et persistante**.
Les menages ont integre cette realite et ne cherchent plus autant
"hausse des prix". Le signal comportemental s'affaiblit meme si l'inflation
reste elevee — c'est pour cela que le coefficient BESI dans SARIMAX
chute de -60% apres la rupture de 2022.
