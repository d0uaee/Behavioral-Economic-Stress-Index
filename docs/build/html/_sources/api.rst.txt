Reference API
=============

Modules principaux
-------------------

src/deep_learning.py
~~~~~~~~~~~~~~~~~~~~~

Module de Deep Learning du projet.

.. code-block:: python

   from src.deep_learning import build_lstm, run_gridsearch_blocs

**Fonctions principales :**

``build_lstm(series, exog=None, train_end="2021-12-01", save_fig=True)``

   Lance le GridSearch LSTM sur une serie temporelle.

   :param series: pd.Series — IPC mensuel
   :param exog: pd.Series ou None — variable exogene (BESI)
   :param train_end: str — date de coupure train/test
   :param save_fig: bool — sauvegarder les figures
   :returns: dict avec rmse, mae, mape, best_params, gridsearch_df

``run_gridsearch_blocs(df, ipc_col="ipc_level", exog_cols=None)``

   Lance le GridSearch LSTM sur les deux blocs (A et B) du protocole
   d'evaluation defini par la binome.

   :param df: pd.DataFrame — Gold dataset
   :param ipc_col: str — nom de la colonne cible
   :param exog_cols: list — colonnes features exogenes
   :returns: pd.DataFrame avec colonnes bloc, model, rmse, mae, mape, n_test

``plot_gridsearch_results(build_lstm_result, series, train_end)``

   Genere les 3 figures de resultats du GridSearch.

   :param build_lstm_result: dict retourne par build_lstm
   :param series: pd.Series — IPC original
   :param train_end: str — date de coupure

src/evaluation/backtest.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from src.evaluation.backtest import run_backtest

``run_backtest(df, models, blocs)``

   Lance le backtest walk-forward pour les modeles statistiques.

   :param df: pd.DataFrame — Gold dataset
   :param models: list — liste des modeles a evaluer
   :param blocs: dict — definition des blocs A et B
   :returns: pd.DataFrame avec resultats par bloc et modele

src/evaluation/warning_metrics.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from src.evaluation.warning_metrics import compute_warning_metrics

``compute_warning_metrics(df, signal_col, target_col, threshold_signal, threshold_target)``

   Calcule les metriques d'alerte precoce (AUC, F1, Recall, lead-time).

   :param df: pd.DataFrame — donnees avec signal et cible
   :param signal_col: str — colonne du signal d'alerte (ex: "behavioral_index_pure")
   :param target_col: str — colonne de la cible (ex: "inflation_yoy")
   :param threshold_signal: float — seuil d'alerte signal (defaut: 0.35)
   :param threshold_target: float — seuil stress reel (defaut: 2.0)
   :returns: dict avec auc, f1, recall, precision, lead_time

src/features/indexes.py
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from src.features.indexes import build_besi_pure, build_besi_hybrid

``build_besi_pure(trends_df, train_end)``

   Construit le BESI comportemental par LassoCV.

   :param trends_df: pd.DataFrame — sous-indices Google Trends normalises
   :param train_end: str — date fin entrainement (pour calibrer les poids)
   :returns: pd.Series — BESI comportemental (0-1)

``build_besi_hybrid(besi_pure, macro_df, train_end)``

   Construit le BESI hybride (BESI pur + signaux macro).

   :param besi_pure: pd.Series — BESI comportemental
   :param macro_df: pd.DataFrame — signaux macro (FAO, FX)
   :param train_end: str — date fin entrainement
   :returns: pd.Series — BESI hybride (0-1)

Fichiers de sortie
-------------------

Rapports CSV (outputs/reports/)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+-------------------------------------+----------------------------------------+
| Fichier                             | Description                            |
+=====================================+========================================+
| backtest_v3_results.csv             | RMSE/MAE/MAPE par modele et bloc       |
+-------------------------------------+----------------------------------------+
| warning_metrics_v3.csv              | AUC/F1/Recall par bloc et signal       |
+-------------------------------------+----------------------------------------+
| granger_besi_v3.csv                 | Test de Granger (lags 1-4)             |
+-------------------------------------+----------------------------------------+
| besi_v3_behavioral_weights.csv      | Poids LassoCV composantes BESI         |
+-------------------------------------+----------------------------------------+
| lstm_results.csv                    | Resultats LSTM par bloc                |
+-------------------------------------+----------------------------------------+
| gridsearch_lstm_blocA.csv           | Top combinaisons GridSearch Bloc A     |
+-------------------------------------+----------------------------------------+
| gridsearch_lstm_blocB.csv           | Top combinaisons GridSearch Bloc B     |
+-------------------------------------+----------------------------------------+
| lstm_best_params.json               | Meilleurs hyperparametres JSON         |
+-------------------------------------+----------------------------------------+
| lstm_scaler_comparison.csv          | MinMaxScaler vs RobustScaler           |
+-------------------------------------+----------------------------------------+
