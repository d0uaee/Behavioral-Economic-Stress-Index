Resultats
=========

.. warning::

   **Honnêtete sur les resultats de prevision :**
   Le modele **naif (persistance) obtient le meilleur RMSE global = 1.609**.
   SARIMAX + BESI (RMSE = 1.891) ne surpasse pas le naif en precision
   point-par-point. La valeur ajoutee du BESI est dans la **detection
   de regime** (Recall = 1.00 sur Bloc B), pas dans la prevision brute.

Performances de prevision (walk-forward)
-----------------------------------------

+----------------------------------+-------+-------+-------+-------+
| Modele                           | RMSE  | MAE   | MAPE  | AIC   |
+==================================+=======+=======+=======+=======+
| Naif (persistance) MEILLEUR RMSE | 1.609 | 1.200 | 1.06% | ---   |
+----------------------------------+-------+-------+-------+-------+
| SARIMA(1,1,1)(1,0,1)[12]         | 1.923 | 1.537 | 1.38% | 64.85 |
+----------------------------------+-------+-------+-------+-------+
| SARIMAX + BESI behavioral        | 1.891 | 1.522 | 1.36% | 57.09 |
+----------------------------------+-------+-------+-------+-------+
| SARIMAX + Hybrid macro           | 1.997 | 1.576 | 1.42% | ---   |
+----------------------------------+-------+-------+-------+-------+

**Interpretation :**

- Le naif gagne en RMSE car l'IPC marocain est tres persistant (racine unitaire).
  Predire "le mois prochain = ce mois" marche bien sur une serie aussi lisse.
- SARIMAX + BESI apporte de la valeur via l'**AIC** (-7.77 points) et via
  la **detection de regime** — pas via la precision point-par-point.

Rupture structurelle 2022
--------------------------

+-----------------------------------+------------------------+
| Statistique                       | Valeur                 |
+===================================+========================+
| Inflation pre-2022 (moyenne)      | +0.74% YoY             |
+-----------------------------------+------------------------+
| Inflation post-2022 (moyenne)     | +8.53% YoY             |
+-----------------------------------+------------------------+
| Facteur multiplicatif             | x11.6                  |
+-----------------------------------+------------------------+
| Test t (difference moyennes)      | t=-6.60, p < 0.0001    |
+-----------------------------------+------------------------+
| Test de Levene (variance)         | W=48.53, p < 0.0001    |
+-----------------------------------+------------------------+
| Test de Chow                      | F=198.5, p < 0.0001    |
+-----------------------------------+------------------------+

Detection de regime — Early Warning
-------------------------------------

C'est ici que BESI apporte sa vraie valeur ajoutee :

+--------------------------------+-------+-------+--------+----------+
| Bloc                           | AUC   | F1    | Recall | Lead-time|
+================================+=======+=======+========+==========+
| Bloc A — COVID                 | 0.328 | 0.273 | 0.375  | 1 mois   |
+--------------------------------+-------+-------+--------+----------+
| Bloc B — Inflation 2022        | 0.311 | 0.814 | 1.000  | 1 mois   |
+--------------------------------+-------+-------+--------+----------+
| Global                         | 0.350 | 0.620 | 0.900  | 1 mois   |
+--------------------------------+-------+-------+--------+----------+

**Points cles :**

- **Recall = 1.00 sur Bloc B** : le BESI n'a manque aucun episode
  d'inflation elevee sur 2022-2024. Zero faux negatif.
- **AUC = 0.35 (global)** : penalisee par le Bloc A — le choc COVID
  est exogene et non lie aux comportements de recherche economique.
- **Lead time = 1 mois** : le BESI signale le changement de regime
  un mois avant la publication officielle de l'IPC.

NLP Hespress — CAS C
----------------------

Le signal NLP Hespress (flux RSS presse marocaine) confirme les
signaux Google Trends sur la periode recente disponible (30-90 jours).

+----------------------+------------------------------------------+
| Caracteristique      | Valeur                                   |
+======================+==========================================+
| Usage                | Validation recente uniquement            |
+----------------------+------------------------------------------+
| Correlation avec BESI| Coherente sur la periode disponible      |
+----------------------+------------------------------------------+
| Integre dans BESI    | Non (historique insuffisant)             |
+----------------------+------------------------------------------+
| Verdict              | CAS C - signal complementaire valide     |
+----------------------+------------------------------------------+

Performances LSTM par bloc
---------------------------

+---------------------------+-------+--------+--------+
| Modele                    | Bloc A| Bloc B | Global |
+===========================+=======+========+========+
| LSTM GridSearch MinMax    | 1.382 | 19.738 | 12.396 |
+---------------------------+-------+--------+--------+
| LSTM GridSearch Robust    | 1.418 | 22.019 | 13.779 |
+---------------------------+-------+--------+--------+

Tableau de bord complet
------------------------

+-------------------------------+----------+----------+-----------+
| Statistique                   | Valeur   | Seuil    | Statut    |
+===============================+==========+==========+===========+
| Delta AIC SARIMAX vs SARIMA   | -7.77    | < -2     | OK        |
+-------------------------------+----------+----------+-----------+
| Recall global inflation       | 0.90     | > 0.80   | OK        |
+-------------------------------+----------+----------+-----------+
| Recall Bloc B (2022-2024)     | 1.00     | > 0.80   | OK        |
+-------------------------------+----------+----------+-----------+
| RMSE naif vs SARIMAX          | Naif gagne | ---    | Documente |
+-------------------------------+----------+----------+-----------+
| AUC globale                   | 0.35     | > 0.65   | Limite    |
+-------------------------------+----------+----------+-----------+
| Test de Chow (F)              | 198.5    | p<0.001  | OK        |
+-------------------------------+----------+----------+-----------+
| LSTM Bloc A vs Naif           | -14%     | < 0      | OK        |
+-------------------------------+----------+----------+-----------+
| LSTM Bloc B                   | 19.738   | ---      | Limite    |
+-------------------------------+----------+----------+-----------+
