Validation des Hypotheses
==========================

H1 — BESI et detection des regimes d'inflation
------------------------------------------------

**Enonce reformule :**

   *"Le BESI comportemental apporte une information utile pour la
   detection des regimes d'inflation au Maroc."*

Criteres de validation
~~~~~~~~~~~~~~~~~~~~~~~~

+------------------------------------------+----------+--------+------------+
| Critere                                  | Valeur   | Seuil  | Statut     |
+==========================================+==========+========+============+
| Delta AIC SARIMAX vs SARIMA              | -7.77    | < -2   | VALIDE     |
+------------------------------------------+----------+--------+------------+
| Recall global episodes inflation         | 0.90     | > 0.80 | VALIDE     |
+------------------------------------------+----------+--------+------------+
| Recall Bloc B (inflation 2022-2024)      | 1.00     | > 0.80 | VALIDE     |
+------------------------------------------+----------+--------+------------+
| AUC globale                              | 0.35     | > 0.65 | NON ATTEINT|
+------------------------------------------+----------+--------+------------+
| RMSE SARIMAX vs Naif                     | +0.282   | < 0    | NON ATTEINT|
+------------------------------------------+----------+--------+------------+

.. warning::

   **Point d'honnêtete important :** le modele naif (persistance) obtient
   le meilleur RMSE global (1.609 vs 1.891 pour SARIMAX + BESI). H1 est
   validee sur la **detection de regime** (Recall, AIC) mais pas sur la
   **prevision point-par-point** (RMSE).

**Verdict H1 : PARTIELLEMENT VALIDEE**

Le BESI apporte une information utile pour la detection des regimes
d'inflation — Recall = 1.00 sur la periode 2022-2024 (zero episode
manque), AIC ameliore de -7.77 points. Mais il ne surpasse pas le
modele naif en precision de prevision mensuelle brute.

Phrase de conclusion pour l'oral
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   *"Le BESI comportemental n'ameliore pas la precision de prevision
   point-par-point — le modele naif gagne en RMSE. Mais il apporte
   une information qualitativement differente : la detection des
   changements de regime d'inflation avec un recall parfait sur
   2022-2024 et un lead time d'un mois sur les statistiques officielles.
   C'est une contribution de nature differente, coherente avec la
   non-linearite de la relation confirmee par Granger."*

H2 — Variables macroeconomiques et detection des regimes
---------------------------------------------------------

**Enonce reformule :**

   *"L'ajout de variables macroeconomiques au BESI comportemental
   ameliore la detection des regimes d'inflation au Maroc."*

Criteres de validation
~~~~~~~~~~~~~~~~~~~~~~~~

+------------------+------------+--------+---------+------------------+
| Critere          | Behavioral | Hybrid | Delta   | Statut           |
+==================+============+========+=========+==================+
| AUC globale      | 0.352      | 0.451  | +0.099  | Favorable hybrid |
+------------------+------------+--------+---------+------------------+
| Recall global    | 0.900      | 0.625  | -0.275  | DEFAVORABLE      |
+------------------+------------+--------+---------+------------------+
| Recall Bloc B    | 1.000      | 0.375  | -0.625  | REJETE           |
+------------------+------------+--------+---------+------------------+
| RMSE global      | 1.891      | 1.997  | +0.106  | DEFAVORABLE      |
+------------------+------------+--------+---------+------------------+

**Verdict H2 : REJETEE**

L'ajout des variables macro (FAO + MAD/EUR) degrade la detection des
regimes d'inflation. Sur la periode cle 2022-2024, le modele hybride
chute de Recall=1.00 a Recall=0.375 — 4 episodes sur 8 completement
manques. Les indices FAO sont des prix mondiaux qui ne capturent pas
la specificite locale marocaine.

Synthese generale
-----------------

+------+----------------------------------------------+--------------------+
| H    | Enonce                                       | Verdict            |
+======+==============================================+====================+
| H1   | BESI utile pour detecter les regimes         | PARTIELLEMENT OUI  |
+------+----------------------------------------------+--------------------+
| H2   | Macro ameliore la detection des regimes      | NON — REJETEE      |
+------+----------------------------------------------+--------------------+

**Contribution originale du projet :**

   *"Premiere tentative de construction d'un indice comportemental
   pour la detection des regimes d'inflation au Maroc base sur Google
   Trends et la presse Hespress. Validation empirique que les signaux
   digitaux comportementaux detectent les changements de regime avec
   un recall parfait sur 2022-2024 et un lead time d'un mois,
   malgre l'absence d'amelioration de la precision point-par-point."*
