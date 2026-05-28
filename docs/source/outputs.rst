Fichiers de sortie
==================

Figures (outputs/figures/)
---------------------------

+---------------------------------------+------------------------------------------+
| Fichier                               | Description                              |
+=======================================+==========================================+
| dashboard_besi_v3_final.png           | Dashboard 6 panneaux — synthese complete |
+---------------------------------------+------------------------------------------+
| structural_break_v3.png               | Rupture structurelle janvier 2022        |
+---------------------------------------+------------------------------------------+
| acf_pacf_v3.png                       | ACF/PACF identification SARIMA           |
+---------------------------------------+------------------------------------------+
| residuals_sarima_v3.png               | Diagnostics residus SARIMA Train A       |
+---------------------------------------+------------------------------------------+
| backtest_v3_bar_comparison.png        | RMSE/MAE/MAPE par modele et bloc         |
+---------------------------------------+------------------------------------------+
| backtest_v3_predictions.png           | Predictions walk-forward vs IPC reel     |
+---------------------------------------+------------------------------------------+
| cross_corr_besi_v3.png                | Correlation croisee BESI-Inflation       |
+---------------------------------------+------------------------------------------+
| early_warning_v3.png                  | Signal BESI vs episodes inflation        |
+---------------------------------------+------------------------------------------+
| roc_curves_v3.png                     | Courbes ROC alerte precoce               |
+---------------------------------------+------------------------------------------+
| precision_recall_v3.png               | Courbes Precision-Recall                 |
+---------------------------------------+------------------------------------------+
| threshold_analysis_v3.png             | Analyse des seuils d'alerte              |
+---------------------------------------+------------------------------------------+
| lstm_gridsearch_blocs.png             | LSTM GridSearch — Bloc A vs Bloc B       |
+---------------------------------------+------------------------------------------+
| lstm_gridsearch_best.png              | Predictions meilleur modele LSTM         |
+---------------------------------------+------------------------------------------+
| gridsearch_rmse_distribution.png      | Distribution RMSE — 96 combinaisons      |
+---------------------------------------+------------------------------------------+
| gridsearch_heatmap.png                | Heatmap look_back vs lstm_units          |
+---------------------------------------+------------------------------------------+

Rapports CSV (outputs/reports/)
--------------------------------

+-------------------------------------+----------------------------------------+
| Fichier                             | Description                            |
+=====================================+========================================+
| backtest_v3_results.csv             | RMSE/MAE/MAPE par modele et bloc       |
+-------------------------------------+----------------------------------------+
| backtest_v3_summary.csv             | Moyenne globale des metriques          |
+-------------------------------------+----------------------------------------+
| warning_metrics_v3.csv              | AUC/F1/Recall par bloc et signal       |
+-------------------------------------+----------------------------------------+
| granger_besi_v3.csv                 | Test de Granger (lags 1-4)             |
+-------------------------------------+----------------------------------------+
| besi_v3_behavioral_weights.csv      | Poids LassoCV composantes BESI         |
+-------------------------------------+----------------------------------------+
| results_v3_final.md                 | Rapport complet H1/H2 et conclusions   |
+-------------------------------------+----------------------------------------+
| lstm_results.csv                    | Resultats LSTM par bloc (format binome)|
+-------------------------------------+----------------------------------------+
| gridsearch_lstm_blocA.csv           | Top combinaisons GridSearch Bloc A     |
+-------------------------------------+----------------------------------------+
| gridsearch_lstm_blocB.csv           | Top combinaisons GridSearch Bloc B     |
+-------------------------------------+----------------------------------------+
| lstm_best_params.json               | Meilleurs hyperparametres JSON         |
+-------------------------------------+----------------------------------------+
| lstm_scaler_comparison.csv          | Comparaison MinMaxScaler vs RobustScaler|
+-------------------------------------+----------------------------------------+

Modeles sauvegardes (outputs/models/)
---------------------------------------

+------------------------+-------------------------------------------+
| Fichier                | Description                               |
+========================+===========================================+
| lstm_best.keras        | Meilleur modele LSTM (GridSearch)         |
+------------------------+-------------------------------------------+
