Notebooks
=========

Le projet contient 4 notebooks Jupyter commentes, a executer dans l'ordre.

01_exploration_v3.ipynb
------------------------

**Objectif :** Analyse descriptive et exploration des donnees.

Contenu :

- Chargement du Gold dataset (96 mois x 45 colonnes)
- Statistiques descriptives IPC et signaux comportementaux
- Tests de stationnarite ADF et KPSS
- Decomposition STL (Tendance / Saisonnalite / Residu)
- ACF et PACF pour identification des ordres SARIMA
- Visualisation des composantes BESI
- Matrice de correlation IPC / BESI / Trends / FAO / FX

02_modeling_v3.ipynb
---------------------

**Objectif :** Identification et ajustement des modeles statistiques.

Contenu :

- Grille de recherche AIC (6 specifications SARIMA)
- Ajustement SARIMA(1,1,1)(1,0,1)[12]
- Diagnostic des residus (QQ-plot, Ljung-Box, ACF residus)
- Modele SARIMAX avec BESI comme variable exogene
- Walk-forward validation Bloc A et Bloc B
- Comparaison SARIMA vs SARIMAX vs Naif

03_analysis_v3.ipynb
---------------------

**Objectif :** Analyses avancees sur la rupture et l'alerte precoce.

Contenu :

- Test de Chow (rupture structurelle janvier 2022)
- CUSUM recursif (Brown-Durbin-Evans, 1975)
- Causalite de Granger bidirectionnelle (lags 1-4)
- Cross-Correlation Function BESI → Inflation
- Early Warning System (Recall, Precision, F1, AUC-ROC)
- Analyse des seuils d'alerte
- Matrice de confusion et courbes ROC

04_results_v3.ipynb
--------------------

**Objectif :** Synthese finale et validation des hypotheses.

Contenu :

- Tableau de bord complet des resultats
- Reponse formelle a H1 (partiellement validee)
- Reponse formelle a H2 (rejetee)
- Dashboard final 6 panneaux
- Comparaison modeles statistiques vs deep learning
- Limites et perspectives
- Phrase de conclusion

Executer les notebooks
-----------------------

.. code-block:: bash

   # Methode 1 : Jupyter classique
   jupyter notebook
   # Ouvrir dans l'ordre : 01, 02, 03, 04

   # Methode 2 : Execution en ligne de commande
   jupyter nbconvert --to notebook --execute notebooks/01_exploration_v3.ipynb --inplace
   jupyter nbconvert --to notebook --execute notebooks/02_modeling_v3.ipynb    --inplace
   jupyter nbconvert --to notebook --execute notebooks/03_analysis_v3.ipynb    --inplace
   jupyter nbconvert --to notebook --execute notebooks/04_results_v3.ipynb     --inplace

   # Methode 3 : Avec un kernel specifique
   jupyter nbconvert --to notebook --execute --inplace \
     --ExecutePreprocessor.kernel_name=besi_v3 \
     notebooks/02_modeling_v3.ipynb

.. warning::

   Les notebooks doivent etre executes dans l'ordre. Le notebook 02
   depend des sorties du notebook 01, et ainsi de suite.
