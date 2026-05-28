L'Indice BESI
=============

Definition
----------

Le **BESI** (*Behavioral Economic Stress Index*) est un indice composite
qui mesure, sur une echelle de 0 a 1, l'intensite du stress economique
des menages marocains a partir de leurs comportements digitaux.

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
   dans les features serait du data leakage (triche statistique).

Formule BESI Hybrid (V3 — test H2)
------------------------------------

.. math::

   BESI_{hybrid}(t) = f(BESI_{pure}(t),\ FAO\_fpi\_yoy(t),\ fx\_yoy(t))

Utilise pour tester H2. **Resultat : H2 rejetee** (le signal macro degrade
la detection sur le Bloc B, Recall chute de 1.00 a 0.375).

Technique du chunking ancre (Google Trends)
--------------------------------------------

Google Trends normalise chaque serie independamment entre 0 et 100
(max = 100 pour le mois le plus recherche). Cela rend les series
incomparables si elles sont collectees dans des batchs separes.

**Solution — l'ancre :**

Chaque batch de requetes inclut le meme mot-cle ancre
("inflation maroc") :

.. code-block:: text

   Batch 1 : ["inflation maroc", "prix huile", "hausse prix"]
   Batch 2 : ["inflation maroc", "ghla lprix", "التضخم في المغرب"]
   Batch 3 : ["inflation maroc", "prix carburant", "subvention"]

Le ratio entre les valeurs de l'ancre dans chaque batch permet de
recalibrer toutes les series sur une echelle commune.

Normalisation du BESI
----------------------

Apres agregation ponderee, le BESI est normalise en MinMaxScaler
**sur le jeu d'entrainement uniquement** pour eviter le leakage de
normalisation vers les donnees de test.

.. code-block:: python

   scaler = MinMaxScaler(feature_range=(0, 1))
   scaler.fit(besi_train)          # FIT sur train uniquement
   besi_norm = scaler.transform(besi_all)  # TRANSFORM sur tout

Interpretation des valeurs
---------------------------

+---------------------+--------------------+-----------------------------------+
| Valeur BESI         | Etat               | Interpretation                    |
+=====================+====================+===================================+
| 0.00 a 0.35         | Normal             | Pas de stress economique detecte  |
+---------------------+--------------------+-----------------------------------+
| 0.35 a 0.65         | Warning            | Tension emergente, surveillance   |
+---------------------+--------------------+-----------------------------------+
| 0.65 a 1.00         | High Stress        | Stress economique eleve           |
+---------------------+--------------------+-----------------------------------+

Correlation BESI — Inflation
-----------------------------

+------------------+-------------+------------+------------+
| Periode          | Lag optimal | r Pearson  | p-value    |
+==================+=============+============+============+
| Periode complete | lag=0       | +0.535     | < 0.001    |
+------------------+-------------+------------+------------+
| Pre-2022         | lag=0       | +0.201     | 0.161 (ns) |
+------------------+-------------+------------+------------+
| Post-2022        | lag=5       | -0.303     | 0.110 (ns) |
+------------------+-------------+------------+------------+

.. note::

   La relation BESI-inflation est **non-lineaire**. Le test de causalite
   de Granger n'est pas significatif (p > 0.62 a tous les lags 1-4),
   ce qui confirme que le BESI agit comme **detecteur de regime**
   plutot que comme predicteur causal lineaire.

Pourquoi le BESI perd de la puissance apres 2022
-------------------------------------------------

Avant 2022, le BESI precedait bien l'inflation car les hausses de prix
etaient temporaires — les menages cherchaient sur Google parce que la
situation etait inhabituelle.

Apres 2022, l'inflation est devenue **structurelle et persistante**.
Les menages ont integre cette realite et ne cherchent plus autant
"hausse des prix" — c'est devenu la normale. Le signal comportemental
s'affaiblit meme si l'inflation reste elevee.

C'est pour cela que le coefficient BESI dans le modele SARIMAX chute
de -60% apres la rupture de 2022 (confirme par le test de Chow,
F=198.5, p < 0.001).
