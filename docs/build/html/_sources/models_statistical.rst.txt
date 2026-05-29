Modeles Statistiques
=====================

SARIMA — Modele de reference
------------------------------

**SARIMA** = Seasonal AutoRegressive Integrated Moving Average

.. math::

   SARIMA(p, d, q) \times (P, D, Q)[m]

Parametres retenus apres grille AIC
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   SARIMA(1, 1, 1) x (1, 0, 1)[12]   AIC Train A = 64.85

+-------------+---------+-----------------------------------------+
| Parametre   | Valeur  | Role                                    |
+=============+=========+=========================================+
| p = 1       | AR(1)   | 1 valeur passee influence la valeur     |
+-------------+---------+-----------------------------------------+
| d = 1       | I(1)    | 1 difference pour stationnariser        |
+-------------+---------+-----------------------------------------+
| q = 1       | MA(1)   | 1 erreur passee influence la valeur     |
+-------------+---------+-----------------------------------------+
| P = 1       | SAR(1)  | AR saisonnier                           |
+-------------+---------+-----------------------------------------+
| D = 0       | SI(0)   | Pas de difference saisonniere           |
+-------------+---------+-----------------------------------------+
| Q = 1       | SMA(1)  | MA saisonnier                           |
+-------------+---------+-----------------------------------------+
| m = 12      | ---     | Periodicite mensuelle                   |
+-------------+---------+-----------------------------------------+

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

.. math::

   IPC_t = SARIMA(IPC_{t-k}) + \beta \cdot BESI_{t} + \varepsilon_t

Resultats :

- AIC SARIMAX + BESI behavioral = **57.09** (vs SARIMA = 64.85)
- Delta AIC = **-7.77** — amelioration statistiquement significative
- Coefficient BESI post-2022 : -60% (rupture structurelle confirmee)

Resultats honnetes — le naif gagne en RMSE
-------------------------------------------

.. warning::

   Le modele **naif (persistance) obtient le meilleur RMSE global (1.609)**.
   Ce resultat est documente et assume. L'IPC marocain est tres persistant
   (proche d'une marche aleatoire) — predire "le mois prochain = ce mois"
   fonctionne bien en precision point-par-point.

+-----------------------------+--------+-------+-------+-------+
| Modele                      | RMSE   | MAE   | MAPE  | AIC   |
+=============================+========+=======+=======+=======+
| Naif (persistance) MEILLEUR | 1.609  | 1.200 | 1.06% | ---   |
+-----------------------------+--------+-------+-------+-------+
| SARIMA(1,1,1)(1,0,1)[12]    | 1.923  | 1.537 | 1.38% | 64.85 |
+-----------------------------+--------+-------+-------+-------+
| SARIMAX + BESI behavioral   | 1.891  | 1.522 | 1.36% | 57.09 |
+-----------------------------+--------+-------+-------+-------+
| SARIMAX + Hybrid macro      | 1.997  | 1.576 | 1.42% | ---   |
+-----------------------------+--------+-------+-------+-------+

La valeur de SARIMAX + BESI est dans la **detection de regime**
(Recall = 1.00 sur Bloc B, AIC meilleur) et non dans la RMSE.

Test de Chow — Rupture structurelle 2022
-----------------------------------------

.. math::

   F = \frac{(RSS_c - RSS_{nc}) / k}{RSS_{nc} / (n - 2k)}

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

Causalite de Granger
----------------------

Test bidirectionnel sur lags 1-4 :

- BESI -> Inflation : **non significatif** (p > 0.62 a tous les lags)
- Inflation -> BESI : **non significatif**

**Interpretation :** La relation est non-lineaire. Le BESI agit comme
detecteur de regime d'inflation, pas comme predicteur causal lineaire.
C'est pour cela qu'un modele LSTM peut theoriquement apporter de la
valeur sur les periodes stables (confirme sur Bloc A).
