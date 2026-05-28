Modeles Statistiques
=====================

SARIMA — Modele de reference
------------------------------

**SARIMA** = Seasonal AutoRegressive Integrated Moving Average

Notation complete :

.. math::

   SARIMA(p, d, q) \times (P, D, Q)[m]

Parametres retenus apres grille AIC
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   SARIMA(1, 1, 1) x (1, 0, 1)[12]   AIC Train A = 64.85

Signification de chaque parametre :

+-------------+---------+-----------------------------------------+
| Parametre   | Valeur  | Role                                    |
+=============+=========+=========================================+
| p = 1       | AR(1)   | 1 valeur passee influence la valeur     |
+-------------+---------+-----------------------------------------+
| d = 1       | I(1)    | 1 difference pour stationnariser        |
+-------------+---------+-----------------------------------------+
| q = 1       | MA(1)   | 1 erreur passee influence la valeur     |
+-------------+---------+-----------------------------------------+
| P = 1       | SAR(1)  | AR saisonnier — meme mois annee prec.   |
+-------------+---------+-----------------------------------------+
| D = 0       | SI(0)   | Pas de difference saisonniere           |
+-------------+---------+-----------------------------------------+
| Q = 1       | SMA(1)  | MA saisonnier                           |
+-------------+---------+-----------------------------------------+
| m = 12      | ---     | Periodicite mensuelle (12 mois)         |
+-------------+---------+-----------------------------------------+

Identification des ordres (ACF/PACF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

La grille de recherche AIC teste plusieurs combinaisons. L'identification
initiale par ACF/PACF donne :

- ACF lag 1 significatif → q=1 (MA partie)
- PACF lags 1-2 significatifs → p=1-2 (AR partie)
- ACF lag 12 significatif → Q=1 (MA saisonnier)

Tests de stationnarite
~~~~~~~~~~~~~~~~~~~~~~~

+------------------------+------------+------------+-------------------+
| Serie                  | ADF p-val  | KPSS p-val | Decision          |
+========================+============+============+===================+
| IPC en niveau          | 0.363      | < 0.05     | Non stationnaire  |
+------------------------+------------+------------+-------------------+
| IPC diff(1)            | 0.021      | > 0.05     | Stationnaire      |
+------------------------+------------+------------+-------------------+
| BESI niveau            | 0.089      | > 0.05     | Ambigu            |
+------------------------+------------+------------+-------------------+
| BESI diff(1)           | 0.003      | > 0.05     | Stationnaire      |
+------------------------+------------+------------+-------------------+

SARIMAX — Modele avec BESI exogene
------------------------------------

**SARIMAX** = SARIMA + variables eXogenes

.. math::

   IPC_t = SARIMA(IPC_{t-k}) + \beta \cdot BESI_{t} + \varepsilon_t

Ou :math:`\beta` est le coefficient du BESI estime par maximum de vraisemblance.

Resultats :

- AIC SARIMAX + BESI behavioral = **57.09** (vs SARIMA = 64.85)
- Delta AIC = **-7.77** — amelioration statistiquement significative
- Coefficient beta BESI : significatif (p < 0.05) sur la periode complete
- Coefficient beta BESI post-2022 : -60% (rupture structurelle confirmee)

Validation Walk-Forward
------------------------

**Principe :** simulation d'un usage en temps reel, mois par mois.

.. code-block:: text

   Pour chaque mois t dans la periode de test :
     1. Entrainer le modele sur toutes les donnees jusqu'a t-1
     2. Predire le mois t (h=1, un pas en avant)
     3. Comparer avec la vraie valeur
     4. Passer au mois suivant

C'est la validation la plus rigoureuse pour les series temporelles
car elle respecte strictement l'ordre temporel des donnees.

Metriques de performance
~~~~~~~~~~~~~~~~~~~~~~~~~

+-----------------------------+---------+---------+-------+-----------+
| Modele                      | RMSE    | MAE     | MAPE  | AIC       |
+=============================+=========+=========+=======+===========+
| Naif (persistance)          | 1.609   | 1.200   | 1.06% | ---       |
+-----------------------------+---------+---------+-------+-----------+
| SARIMA(1,1,1)(1,0,1)[12]    | 1.923   | 1.537   | 1.38% | 64.85     |
+-----------------------------+---------+---------+-------+-----------+
| **SARIMAX + BESI behav.**   | **1.891** | **1.522** | **1.36%** | **57.09** |
+-----------------------------+---------+---------+-------+-----------+
| SARIMAX + Hybrid macro      | 1.997   | 1.576   | 1.42% | ---       |
+-----------------------------+---------+---------+-------+-----------+

Test de Chow — Rupture structurelle 2022
-----------------------------------------

Le test de Chow verifie si les coefficients de regression sont
**stables avant et apres janvier 2022**.

**Principe mathematique :**

.. math::

   F = \frac{(RSS_c - RSS_{nc}) / k}{RSS_{nc} / (n - 2k)}

Ou :

- :math:`RSS_c` = erreurs du modele contraint (toute la periode)
- :math:`RSS_{nc}` = erreurs pre-2022 + erreurs post-2022
- k = nombre de parametres, n = nombre d'observations

**Resultats :**

+------------------------------+-------------------+
| Statistique                  | Valeur            |
+==============================+===================+
| F-statistique                | **198.5**         |
+------------------------------+-------------------+
| p-value                      | **< 0.001**       |
+------------------------------+-------------------+
| Rupture                      | **CONFIRMEE**     |
+------------------------------+-------------------+
| Inflation pre-2022 (moyenne) | +0.74% YoY        |
+------------------------------+-------------------+
| Inflation post-2022 (moyenne)| +8.53% YoY        |
+------------------------------+-------------------+
| Facteur multiplicatif        | **x11.6**         |
+------------------------------+-------------------+

CUSUM — Confirmation graphique
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Le CUSUM (Brown-Durbin-Evans, 1975) montre visuellement quand les
residus recursifs sortent des bandes de confiance. Dans ce projet,
la sortie des bandes se produit exactement autour de janvier 2022,
confirmant graphiquement le test de Chow.

Causalite de Granger
----------------------

**H0 :** Le BESI ne cause pas l'inflation YoY au sens de Granger.

Test bidirectionnel sur lags 1-4 :

- BESI → Inflation : **non significatif** (p > 0.62 a tous les lags)
- Inflation → BESI : **non significatif**

**Interpretation :** La relation est non-lineaire. Le BESI agit comme
detecteur de regime, pas comme predicteur causal lineaire. C'est
precisement pourquoi un modele LSTM pourrait theoriquement apporter
de la valeur sur les periodes stables.
