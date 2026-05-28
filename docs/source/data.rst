Donnees
=======

Sources de donnees
------------------

+------------------------------+------------------------------------+-----------+----------------------------+
| Source                       | Variable                           | Periode   | Statut                     |
+==============================+====================================+===========+============================+
| HCP Maroc (manuel)           | IPC mensuel base 2017=100          | 2017-2024 | OK                         |
+------------------------------+------------------------------------+-----------+----------------------------+
| Google Trends (pytrends)     | 7 mots-cles, geo=MA                | 2010-2024 | OK                         |
+------------------------------+------------------------------------+-----------+----------------------------+
| FAO Food Price Index         | 6 sous-indices alimentaires        | 2010-2024 | OK                         |
+------------------------------+------------------------------------+-----------+----------------------------+
| ECB / interpolation lineaire | Taux MAD/EUR mensuel               | 2010-2024 | OK (interpolation)         |
+------------------------------+------------------------------------+-----------+----------------------------+
| Reddit r/Morocco             | NLP inflation                      | ---       | Absent (API refusee)       |
+------------------------------+------------------------------------+-----------+----------------------------+
| YouTube                      | Commentaires economiques           | ---       | Absent (quota depasse)     |
+------------------------------+------------------------------------+-----------+----------------------------+

Architecture Bronze → Silver → Gold
-------------------------------------

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
   `-- macro_signals_monthly.csv    FAO + FX normalises

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
| ``trends_carburant``          | Sous-indice Trends carburant             |
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

Le dataset est divise en deux blocs temporels distincts :

.. code-block:: text

   BLOC A — Periode COVID
     Train : janvier 2017 → decembre 2019  (36 mois)
     Test  : janvier 2020 → decembre 2021  (24 mois)
     Contexte : choc exogene COVID-19, IPC relativement stable

   BLOC B — Periode Inflation
     Train : janvier 2017 → decembre 2021  (60 mois)
     Test  : janvier 2022 → decembre 2024  (36 mois)
     Contexte : choc inflationniste majeur (guerre Ukraine, prix mondiaux)

Sélection automatique des keywords Google Trends
-------------------------------------------------

La selection des mots-cles Google Trends suit un protocole en 4 etapes
pour repondre a la critique "pourquoi ces keywords et pas d'autres ?" :

1. **Generation automatique** : ``pytrends.related_queries()`` genere 50+
   candidats depuis 5 seeds economiques.

2. **Decomposition STL** : la saisonnalite (Ramadan, rentree scolaire)
   est supprimee avant correlation pour eviter les biais saisonniers.

   .. note::
      La correlation brute BESI-inflation est r=0.535, mais apres
      decomposition STL elle tombe a r=0.228. Les 0.307 points
      restants venaient uniquement du Ramadan.

3. **Filtrage** : seuls les keywords avec r > 0.25 sur les residus
   STL sont conserves.

4. **Clustering K-Means** : elimination des keywords redondants,
   7 groupes thematiques finaux retenus.

Note importante sur le data leakage
-------------------------------------

.. warning::

   ``ipc_level``, ``inflation_yoy`` et ``ipc_change`` ne sont **jamais**
   utilises comme features d'entree dans les modeles. Ce sont les variables
   cibles. Les inclure serait du **data leakage** (triche).

   La formule BESI V3 n'inclut plus de composante IPC, contrairement
   a la version V1/V2. Les poids sont calibres par **LassoCV** sur le
   jeu d'entrainement uniquement.
