# NLP_RESULTS

## 1. Couverture des données

year
2017     397
2018     442
2019     424
2020     558
2021     704
2022    1027
2023    1048
2024    1188

- Mois imputés (coverage_flag=0) : 0
- Source primaire du signal : post_excerpt / RSS fallback si nécessaire
- Nature du signal : presse editoriale (pas commentaires lecteurs)

## 2. Composition du lexique

- Poids Lasso : alpha=1.000, beta=0.000

Top 10 termes les plus fréquents :
- ارتفع: 312
- استقر: 229
- غلاء: 209
- انخفض: 89
- زاد: 48
- غالي: 37
- رخيص: 19
- هبط: 10
- مغال: 2
- cher: 1

## 3. Performance comparative BESI v1 vs BESI v2

```text
                     model            signal_col       aic  rmse_bloc_b  recall_bloc_b  precision_bloc_b  f1_bloc_b  ap_bloc_b  threshold_signal  tp  fp  fn  aic_delta_vs_v1
SARIMAX + BESI v1 (Trends) behavioral_index_pure 2077.3972       1.9761          1.000            0.6857     0.8136     0.5685            0.2401  24  11   0           0.0000
 SARIMAX + BESI v2a (Fixe)        besi_v2a_fixed  648.9701       1.9859          1.000            0.6857     0.8136     0.6889            0.2418  24  11   0       -1428.4271
SARIMAX + BESI v2b (Lasso)        besi_v2b_lasso 2077.3972       1.9761          1.000            0.6857     0.8136     0.5685            0.2401  24  11   0           0.0000
        SARIMAX + NLP seul     besi_v2c_nlp_only   73.8735       1.9046          0.875            0.6774     0.7636     0.6951            0.2460  21  10   3       -2003.5237
```

## 4. Verdict honnête

Le Lasso assigne un poids nul au signal NLP. Le signal presse n'apporte pas d'information conditionnelle supplémentaire au-delà des Trends.

## 5. Phrase pour l'oral

Le signal presse editorial a ete teste proprement, mais le Lasso lui attribue un poids nul face aux Trends.
