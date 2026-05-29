Donnees
=======

Sources de donnees
------------------

+------------------------------+------------------------------------+-----------+----------------------------+
| Source                       | Variable                           | Periode   | Statut                     |
+==============================+====================================+===========+============================+
| HCP Maroc (manuel)           | IPC mensuel base 2017=100          | 2017-2024 | OK                         |
+------------------------------+------------------------------------+-----------+----------------------------+
| Google Trends (pytrends)     | 7 mots-cles, geo=MA                | 2017-2024 | OK                         |
+------------------------------+------------------------------------+-----------+----------------------------+
| FAO Food Price Index         | 6 sous-indices alimentaires        | 2017-2024 | OK                         |
+------------------------------+------------------------------------+-----------+----------------------------+
| ECB / interpolation lineaire | Taux MAD/EUR mensuel               | 2017-2024 | OK (interpolation lineaire)|
+------------------------------+------------------------------------+-----------+----------------------------+
| Hespress / Le360 / Medias24  | NLP presse marocaine (flux RSS)    | 30-90 j   | CAS C — validation recente |
+------------------------------+------------------------------------+-----------+----------------------------+
| Reddit r/Morocco             | NLP inflation                      | ---       | Absent (API refusee)       |
+------------------------------+------------------------------------+-----------+----------------------------+
| YouTube                      | Commentaires economiques           | ---       | Absent (quota depasse)     |
+------------------------------+------------------------------------+-----------+----------------------------+

NLP Hespress — pourquoi et comment
-------------------------------------

Reddit et YouTube n'ont pas pu etre collectes :

- **Reddit** : demande d'acces API refusee
- **YouTube** : quota API depasse en quelques minutes

Ils ont ete remplaces par la **presse marocaine via flux RSS** :
Hespress, Le360, Medias24. Ces sources sont plus representatives
des menages marocains que Reddit (surtout jeunes urbains francophones).

**Limite principale :** les flux RSS ne conservent que 30 a 90 jours
d'historique. Le signal NLP Hespress ne couvre donc pas toute la
periode 2017-2024 — il est utilise comme **validation recente**
(CAS C) et non comme composante du BESI principal.

**Fichier :** ``data/silver/press_signal_monthly.csv``

Architecture Bronze -> Silver -> Gold
---------------------------------------

Bronze — Donnees brutes
~~~~~~~~~~~~~~~~~~~~~~~~

Jamais modifiees. Servent de reference.

.. code-block:: text

   data/bronze/
   |-- cpi_hcp_monthly_raw.csv      IPC HCP Maroc 2017-2024
   |-- fao_food_price_raw.csv       FAO Food Price Index
   `-- bam_fx_raw.csv               MAD/EUR (ECB + interpolation lineaire)

Silver — Donnees nettoyees
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Standardisees, normalisees, sans valeurs aberrantes.

.. code-block:: text

   data/silver/
   |-- cpi_monthly.csv              IPC + inflation_yoy + inflation_mom
   |-- google_trends_monthly.csv    Sous-indices Trends normalises 0-1
   |-- behavioral_index_pure.csv    BESI comportemental (Trends seul)
   |-- macro_signals_monthly.csv    FAO + FX normalises
   `-- press_signal_monthly.csv     NLP Hespress (CAS C - recent)

Gold — Dataset final
~~~~~~~~~~~~~~~~~~~~~

Dataset pret pour la modelisation. **C'est ce fichier qui est utilise
par tous les modeles.**

.. code-block:: text

   data/gold/
   `-- model_dataset_monthly.csv    96 mois x 45 colonnes

Colonnes principales du Gold dataset
--------------------------------------

+-------------------------------+------------------------------------------+
| Colonne                       | Description                              |
+===============================+==========================================+
| ``ipc_level``                 | Variable cible — IPC mensuel (base 2017) |
+-------------------------------+------------------------------------------+
| ``inflation_yoy``             | Inflation annuelle (%) — cible secondaire|
+-------------------------------+------------------------------------------+
| ``behavioral_index_pure``     | BESI comportemental (Google Trends)      |
+-------------------------------+------------------------------------------+
| ``behavioral_index_pure_lag1``| BESI decale d'1 mois (feature principale)|
+-------------------------------+------------------------------------------+
| ``hybrid_macro_index``        | BESI + FAO + taux de change              |
+-------------------------------+------------------------------------------+
| ``trends_prix_alim``          | Sous-indice Trends prix alimentaires     |
+-------------------------------+------------------------------------------+
| ``trends_inflation``          | Sous-indice Trends inflation             |
+-------------------------------+------------------------------------------+
| ``fao_food_yoy``              | FAO Food Price Index (var. annuelle)     |
+-------------------------------+------------------------------------------+
| ``fao_oils_yoy``              | FAO Oils Price Index (var. annuelle)     |
+-------------------------------+------------------------------------------+
| ``fx_yoy``                    | Variation annuelle MAD/EUR               |
+-------------------------------+------------------------------------------+
| ``split_label``               | train_A / test_A / train_B / test_B      |
+-------------------------------+------------------------------------------+

Blocs d'evaluation
-------------------

.. code-block:: text

   BLOC A — Periode COVID
     Train : janvier 2017 -> decembre 2019  (36 mois)
     Test  : janvier 2020 -> decembre 2021  (24 mois)
     Contexte : choc exogene COVID-19, IPC relativement stable

   BLOC B — Periode Inflation
     Train : janvier 2017 -> decembre 2021  (60 mois)
     Test  : janvier 2022 -> decembre 2024  (36 mois)
     Contexte : choc inflationniste majeur — changement de regime x11.6

Note importante sur le data leakage
-------------------------------------

.. warning::

   ``ipc_level``, ``inflation_yoy`` et ``ipc_change`` ne sont **jamais**
   utilises comme features d'entree dans les modeles. Ce sont les variables
   cibles. Les inclure serait du **data leakage**.

   La formule BESI V3 n'inclut plus de composante IPC, contrairement
   a la version V1/V2. Les poids sont calibres par **LassoCV** sur le
   jeu d'entrainement uniquement.
