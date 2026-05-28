Demarrage rapide
================

Lancer le pipeline complet V3
------------------------------

.. code-block:: bash

   python run_v3.py --skip-ingest --start-date 2017-01-01

Cette commande execute dans l'ordre :

1. Assemblage du Gold dataset
2. Backtest walk-forward (SARIMA vs SARIMAX vs Naif)
3. Calcul des metriques d'alerte precoce
4. Generation du dashboard final

Lancer etape par etape
-----------------------

.. code-block:: bash

   # Etape 1 : Assembler le Gold dataset
   python run_v3.py --step gold

   # Etape 2 : Backtest walk-forward
   python run_v3.py --step backtest

   # Etape 3 : Metriques alerte precoce
   python run_v3.py --step warnings

   # Etape 4 : Dashboard final
   python make_dashboard.py

Lancer le Deep Learning (GridSearch LSTM)
-----------------------------------------

.. code-block:: bash

   python src/deep_learning.py

.. warning::

   Le GridSearch LSTM teste 96 combinaisons d'hyperparametres sur 2 blocs.
   Duree estimee : **20 a 40 minutes** selon votre machine.
   Les resultats sont sauvegardes dans ``outputs/reports/lstm_results.csv``.

Executer les notebooks
-----------------------

.. code-block:: bash

   jupyter notebook

Puis ouvrir dans l'ordre :

1. ``notebooks/01_exploration_v3.ipynb`` — Analyse descriptive
2. ``notebooks/02_modeling_v3.ipynb``    — SARIMA / SARIMAX
3. ``notebooks/03_analysis_v3.ipynb``    — Rupture 2022, Granger, Early Warning
4. ``notebooks/04_results_v3.ipynb``     — Synthese H1 / H2

Resultats attendus
------------------

Apres execution complete, vous trouverez :

- **Figures** dans ``outputs/figures/`` (17 figures PNG)
- **CSV de resultats** dans ``outputs/reports/``
- **Modele LSTM sauvegarde** dans ``outputs/models/lstm_best.keras``

.. code-block:: text

   === COMPARAISON FINALE (exemple de sortie) ===
   Modele                  RMSE    MAE    MAPE
   Naif (persistance)     1.609  1.200   1.06%
   SARIMA(1,1,1)          1.923  1.537   1.38%
   SARIMAX + BESI         1.891  1.522   1.36%    <- meilleur statistique
   LSTM GridSearch Bloc A 1.382  1.213   1.19%    <- meilleur deep learning
