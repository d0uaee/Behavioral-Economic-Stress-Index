# Prompts Claude Code — à utiliser session par session
# Copie-colle chaque prompt directement dans ton terminal Claude Code

# ════════════════════════════════════════════════════════
# SEMAINE 1 — DATA PIPELINE
# ════════════════════════════════════════════════════════

## Session 1 — Google Trends
"""
Lis CLAUDE.md pour comprendre le projet.
Crée src/data_pipeline.py avec une fonction fetch_google_trends()
qui utilise pytrends pour récupérer les données mensuelles pour le Maroc
(geo='MA') pour les keywords suivants :
["inflation maroc", "prix huile", "hausse prix", "credit consommation", "chomage maroc"]
Période : 2010-01-01 à 2024-12-01
Resample en mensuel (mean), normalise 0-1 chaque série,
crée un composite pondéré (moyenne simple pour l'instant),
sauvegarde dans data/processed/trends_monthly.csv
Inclus une fonction mock si pytrends échoue (rate limit).
"""

## Session 2 — Reddit + YouTube
"""
Dans src/data_pipeline.py, ajoute :
1. fetch_reddit_data() — utilise praw pour scraper r/Morocco
   avec les keywords ["inflation", "prix", "cherté", "économie"]
   Agrège en mensuel : volume de posts + score moyen
   Normalise 0-1, sauvegarde dans data/processed/reddit_monthly.csv

2. fetch_youtube_data() — utilise YouTube Data API v3
   Queries : ["inflation maroc", "hausse prix maroc"]
   Compte les vidéos publiées par mois
   Normalise 0-1, sauvegarde dans data/processed/youtube_monthly.csv

Ajoute une version mock réaliste pour les deux si les API keys 
ne sont pas configurées.
"""

## Session 3 — IPC + BESI
"""
Dans src/data_pipeline.py, ajoute :
1. load_ipc_data(filepath='data/ipc_maroc.csv') 
   Si le fichier n'existe pas, génère des données simulées 
   réalistes basées sur la trajectoire connue de l'IPC marocain
   (inflation ~1-2% avant 2022, pic à 8%+ en 2022, redescente ensuite)
   Calcule : taux YoY, variation mensuelle MoM
   
2. build_besi_index(trends_df, reddit_df, youtube_df, ipc_df)
   BESI = 0.40*trends + 0.30*reddit + 0.20*youtube + 0.10*ipc_change
   Tout normalisé 0-1
   Ajoute colonnes : stress_level (Normal/Warning/High Stress)
   stress_level basé sur : <0.35 Normal, 0.35-0.65 Warning, >0.65 High Stress
   Sauvegarde master dataset dans data/processed/master_dataset.csv
"""

## Session 4 — Feature Selection
"""
Crée src/features.py avec :
1. lag_correlation_analysis(besi_series, ipc_series, max_lag=6)
   Calcule corrélation de Pearson pour chaque lag (0 à 6 mois)
   Retourne un DataFrame avec lag, correlation, p-value
   Plot le résultat

2. granger_feature_selection(features_df, target_series, max_lag=4)
   Teste la causalité de Granger pour chaque feature vs IPC
   Garde uniquement les features avec p-value < 0.05
   Retourne liste des features significatives

3. compute_feature_importance(features_df, target_series)
   Corrélation de Spearman + Pearson pour chaque feature
   Retourne DataFrame trié par importance décroissante
"""


# ════════════════════════════════════════════════════════
# SEMAINE 2 — MODÉLISATION SARIMA/SARIMAX  
# ════════════════════════════════════════════════════════

## Session 5 — Stationnarité
"""
Crée src/models.py avec :
1. stationarity_analysis(series, name='IPC')
   - Test ADF (Augmented Dickey-Fuller)
   - Test KPSS
   - Décomposition STL (tendance, saisonnalité, résidu)
   - Affiche résultats clairement avec interprétation en français
   - Retourne : is_stationary (bool), n_diffs_needed (int)
   
2. prepare_series(series)
   - Applique les différenciations nécessaires selon stationarity_analysis
   - Plot série originale vs transformée
   - Plot ACF et PACF pour identifier ordres p,q,P,Q
"""

## Session 6 — SARIMA Baseline
"""
Dans src/models.py, ajoute :
1. fit_sarima_baseline(series, train_end='2021-12-01')
   - Utilise pmdarima.auto_arima pour trouver les meilleurs ordres
   - Fit sur données d'entraînement (2010-2021)
   - Affiche résumé du modèle
   - Diagnostics des résidus (test Ljung-Box, normalité)
   - Retourne model + ordres (p,d,q)(P,D,Q)s

2. walk_forward_validation(series, model_func, n_test=36)
   - Rolling window : entraîne sur t mois, prédit t+1, t+2, t+3
   - Calcule RMSE, MAE, MAPE à chaque pas
   - Plot prédictions vs valeurs réelles
   - Retourne dict avec métriques moyennes
"""

## Session 7 — SARIMAX
"""
Dans src/models.py, ajoute :
1. fit_sarimax(series, exog, orders, train_end='2021-12-01')
   - Fit SARIMAX avec variables exogènes
   - Exog options : (1) google_only, (2) besi_composite, (3) all_features
   - Diagnostics des résidus
   - Retourne model

2. compare_models(series, exog_variants, n_test=36)
   - Compare SARIMA vs SARIMAX(google) vs SARIMAX(BESI) vs SARIMAX(all)
   - Tableau de comparaison : AIC, BIC, RMSE, MAE, MAPE
   - Plot toutes les prédictions sur le même graphique
   - Affiche quel modèle est meilleur et de combien (% amélioration)
"""


# ════════════════════════════════════════════════════════
# SEMAINE 3 — ANALYSE STRUCTURELLE & EARLY WARNING
# ════════════════════════════════════════════════════════

## Session 8 — Rupture structurelle
"""
Crée src/analysis.py avec :
1. chow_test(series, exog, breakpoint='2022-01-01')
   - Test de Chow pour détecter la rupture structurelle
   - Compare les paramètres du modèle avant vs après 2022
   - Affiche résultats avec interprétation claire
   - Plot les deux régimes (avant/après)

2. period_performance(series, exog, periods)
   periods = {'pre_covid': ('2010','2019'),
              'shock': ('2020','2022'),
              'post_shock': ('2022','2024')}
   - Calcule RMSE/MAE pour SARIMA vs SARIMAX dans chaque période
   - Montre quand BESI aide le plus
"""

## Session 9 — Early Warning
"""
Dans src/analysis.py, ajoute :
1. early_warning_analysis(besi_series, ipc_series)
   - Calcule lag optimal entre BESI et transitions de stress IPC
   - Détermine le lead time moyen en mois
   - Calcule precision/recall des alertes BESI
   - Réponse à la question clé : 
     "BESI détecte le stress X mois avant l'IPC officiel"

2. stress_transition_matrix(stress_levels)
   - Matrice de transition Normal→Warning→High Stress
   - Fréquence et durée moyenne dans chaque état
   - Plot heatmap des transitions
"""


# ════════════════════════════════════════════════════════
# SEMAINE 4 — DEEP LEARNING & OUTPUTS FINAUX
# ════════════════════════════════════════════════════════

## Session 10 — LSTM (optionnel)
"""
Crée src/deep_learning.py avec :
1. build_lstm(series, exog, look_back=12)
   - Architecture simple : 2 couches LSTM + Dense
   - Même split train/test que les modèles statistiques
   - Même métriques : RMSE, MAE, MAPE
   - IMPORTANT : pas d'optimisation excessive — le but est la comparaison
   
2. compare_all_models(results_dict)
   - Tableau final : SARIMA vs SARIMAX vs LSTM
   - Ajoute colonne "modèle naïf" (prédiction = valeur précédente)
   - Ajoute colonne "interprétabilité" (Haute/Moyenne/Faible)
   - Ajoute colonne "temps d'entraînement"
"""

## Session 11 — Dashboard final
"""
Crée src/visualization.py avec une fonction generate_dashboard()
qui produit 6 graphiques de qualité publication :

1. IPC mensuel + taux d'inflation YoY (double axe)
2. BESI over time avec zones de stress colorées (vert/orange/rouge)
3. Signaux individuels normalisés (Trends, Reddit, YouTube)  
4. Comparaison SARIMA vs SARIMAX — prédictions vs réel
5. Analyse lag : corrélation BESI → IPC par lag (barplot)
6. Performance par période (avant/pendant/après 2022) — grouped bar

Style : fond blanc, police académique, tout en français,
labels clairs, légende propre.
Sauvegarde chaque figure dans outputs/figures/ en 300 DPI.
"""

## Session 12 — Rapport automatique
"""
Crée un script generate_report.py qui produit automatiquement
un résumé de résultats dans outputs/reports/results_summary.md avec :

1. Statistiques descriptives de toutes les séries
2. Résultats des tests de stationnarité (ADF, KPSS)
3. Ordres SARIMA sélectionnés et justification
4. Tableau de comparaison des modèles (RMSE, MAE, MAPE)
5. Résultats du test de Chow (rupture 2022)
6. Lead time quantifié : "BESI détecte le stress X mois avant l'IPC"
7. Réponse à H1 : les signaux Google améliorent-ils le modèle ? De combien ?
"""


# ════════════════════════════════════════════════════════
# PROMPT UNIVERSEL — À UTILISER APRÈS CHAQUE SESSION
# ════════════════════════════════════════════════════════

"""
Explique-moi ce code comme si je devais le présenter à mon professeur.
Pour chaque fonction importante :
1. Que fait-elle exactement ?
2. Pourquoi ce choix méthodologique ?
3. Quelle question de recherche est-ce que ça répond ?
4. Comment je l'interpréterais dans mon rapport ?
"""


# ════════════════════════════════════════════════════════
# PROMPT DE DÉBOGAGE — si quelque chose ne marche pas
# ════════════════════════════════════════════════════════

"""
J'ai cette erreur : [colle l'erreur ici]
Dans ce contexte : [décris ce que tu essayais de faire]
Corrige-la sans changer la logique du code déjà écrit.
Explique pourquoi ça ne marchait pas.
"""
