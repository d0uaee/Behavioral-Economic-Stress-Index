Modeles Deep Learning (partie Adama Basse)
==========================================

Introduction
------------

Cette section documente la partie Deep Learning du projet. L'objectif est de tester si un modele LSTM optimise
peut capturer les non-linearites de la relation BESI-IPC que les modeles
statistiques lineaires (SARIMA/SARIMAX) ne peuvent pas detecter.

**Motivation issue du test de Granger :** La causalite de Granger n'est
pas significative lineairement (p > 0.62). Cela sugere une relation
non-lineaire — terrain naturel pour les reseaux de neurones recurrents.

Architecture LSTM
-----------------

Le modele LSTM (*Long Short-Term Memory*) est un reseau de neurones
recurrent concu pour apprendre les dependances a long terme dans les
series temporelles.

Architecture retenue apres GridSearch :

.. code-block:: python

   model = Sequential([
       Input(shape=(look_back, n_features)),
       LSTM(lstm_units_1, return_sequences=True),
       Dropout(dropout_rate),
       LSTM(lstm_units_2),
       Dense(1)
   ])
   model.compile(optimizer=Adam(learning_rate=lr), loss="mse")

Preparaton des features
------------------------

En suivant les conseils de la binome, les features d'entree sont :

.. code-block:: python

   features = [
       "ipc_level",                       # serie cible (valeurs passees)
       "behavioral_index_pure_lag1",      # BESI decale d'1 mois
       "trends_prix_alim",                # sous-indice Trends alimentaire
       "fao_oils_yoy",                    # prix huiles FAO (var. annuelle)
       "fx_yoy",                          # variation MAD/EUR
       "month_sin",                       # encodage cyclique mois
       "month_cos",                       # encodage cyclique mois
   ]
   # Variable cible : ipc_level au pas t+1

.. warning::

   ``inflation_yoy``, ``ipc_change`` et toute autre derivee de l'IPC
   sont **exclues** des features. Les inclure constituerait du
   **data leakage** (la cible ne peut pas etre une feature).

Encodage cyclique du mois
~~~~~~~~~~~~~~~~~~~~~~~~~~

Pour eviter le biais du Ramadan et des saisonnalites, le mois est encode
par deux variables sinusoidales :

.. math::

   month\_sin = \sin\left(\frac{2\pi \cdot mois}{12}\right)

   month\_cos = \cos\left(\frac{2\pi \cdot mois}{12}\right)

Cela permet au modele d'apprendre lui-meme la saisonnalite sans etre
biaise par les pics de recherche du Ramadan.

Normalisation sans data leakage
---------------------------------

.. code-block:: python

   # FIT sur le train uniquement — jamais sur le test
   scaler = MinMaxScaler(feature_range=(0, 1))
   scaler.fit(data_arr[train_idx])          # train seulement
   data_norm = scaler.transform(data_arr)   # applique sur tout

.. note::

   Deux scalers ont ete testes : ``MinMaxScaler`` et ``RobustScaler``.
   Le RobustScaler (base sur la mediane et l'IQR) est theoriquement
   plus robuste aux valeurs extremes hors distribution. En pratique,
   les deux donnent des resultats similaires sur le Bloc B.

GridSearch des hyperparametres
--------------------------------

Un GridSearch exhaustif de **96 combinaisons** a ete realise
sur les deux blocs d'evaluation.

Grille de recherche :

+------------------+----------------------+
| Hyperparametre   | Valeurs testees      |
+==================+======================+
| look_back        | [6, 12, 18, 24]      |
+------------------+----------------------+
| lstm_units_1     | [32, 64, 128]        |
+------------------+----------------------+
| lstm_units_2     | [16, 32, 64]         |
+------------------+----------------------+
| dropout          | [0.1, 0.2, 0.3]      |
+------------------+----------------------+
| learning_rate    | [0.001, 0.0005]      |
+------------------+----------------------+
| batch_size       | [8, 16, 32]          |
+------------------+----------------------+

Parametres fixes :

- **Epochs max :** 150
- **EarlyStopping :** patience=10, restore_best_weights=True
- **Validation split :** 15% du train
- **shuffle :** False (obligatoire pour les series temporelles)

Protocole d'evaluation — meme que SARIMA
-----------------------------------------

Pour permettre une comparaison juste avec les modeles statistiques,
le meme protocole walk-forward sur deux blocs est utilise :

.. code-block:: text

   BLOC A — Periode COVID
     Train : janvier 2017 → decembre 2019  (36 mois)
     Test  : janvier 2020 → decembre 2021  (24 pas)

   BLOC B — Periode Inflation
     Train : janvier 2017 → decembre 2021  (60 mois)
     Test  : janvier 2022 → decembre 2024  (36 pas)

Resultats du GridSearch LSTM
-------------------------------

+-----------------------------+---------+---------+-------+
| Modele / Bloc               | RMSE    | MAE     | MAPE  |
+=============================+=========+=========+=======+
| LSTM GridSearch — Bloc A    | **1.382** | 1.213 | 1.19% |
+-----------------------------+---------+---------+-------+
| LSTM GridSearch — Bloc B    | 19.738  | 18.544  | 14.67%|
+-----------------------------+---------+---------+-------+
| LSTM GridSearch — Global    | 12.396  | 11.611  | 9.28% |
+-----------------------------+---------+---------+-------+

Comparaison avec les baselines :

+-----------------------------+---------+---------------------------+
| Modele                      | RMSE    | Verdict                   |
+=============================+=========+===========================+
| Naif (persistance)          | 1.609   | Baseline                  |
+-----------------------------+---------+---------------------------+
| SARIMAX + BESI              | 1.891   | Meilleur statistique      |
+-----------------------------+---------+---------------------------+
| **LSTM — Bloc A**           | **1.382** | **Bat le naif (+14%)** |
+-----------------------------+---------+---------------------------+
| LSTM — Bloc B               | 19.738  | Effondrement              |
+-----------------------------+---------+---------------------------+

Analyse du comportement asymetrique
-------------------------------------

**Bloc A (COVID 2020-2021) — LSTM performant**

RMSE = 1.382 < Naif = 1.609. Le LSTM bat le modele naif de **14%**.
Sur une periode de variation moderee, le LSTM capture des dynamiques
non-lineaires invisibles pour SARIMA.

**Bloc B (Inflation 2022-2024) — LSTM en echec**

RMSE = 19.738, soit **12x plus eleve que SARIMAX**. Pourquoi ?

1. **Rupture structurelle hors distribution :**
   Le modele est entraine sur 2017-2021 ou l'IPC monte de ~0.5%/an.
   En 2022 l'IPC monte de 6-8%/an. C'est 12x plus rapide — le LSTM
   n'a jamais vu ca.

2. **Probleme du scaler :**
   Avec MinMaxScaler calibre sur 2017-2021, les valeurs de 2022-2024
   sortent de la plage [0, 1] → extrapolation catastrophique.
   Le passage au RobustScaler n'ameliore pas significativement
   (RMSE Bloc B = 22.02 avec RobustScaler).

3. **Taille du dataset insuffisante :**
   60 observations d'entrainement sont structurellement insuffisantes
   pour un reseau LSTM. La litterature recommande plusieurs milliers
   de sequences.

Autres modeles testes
----------------------

Avant le GridSearch LSTM, d'autres approches ont ete evaluees :

+--------------------------------+-------+-------------------------------+
| Modele                         | RMSE  | Raison d'echec                |
+================================+=======+===============================+
| LSTM simple (window=12)        | 0.021 | Non optimise                  |
+--------------------------------+-------+-------------------------------+
| LSTM sliding window (3-18 mois)| 0.049 | Pire que LSTM original        |
+--------------------------------+-------+-------------------------------+
| Prophet (Facebook)             | 0.061 | Serie trop reguliere          |
+--------------------------------+-------+-------------------------------+

.. note::

   Ces RMSE sont en echelle normalisee (IPC entre 0 et 1), differente
   des RMSE du GridSearch final (IPC en valeur absolue, base 2017=100).

Conclusion de la partie Deep Learning
---------------------------------------

Le GridSearch LSTM sur 96 combinaisons revele un comportement
**asymetrique et instructif** :

   *"Le LSTM apporte de la valeur sur les periodes de relative stabilite
   (Bloc A : RMSE=1.38, meilleur que le naif de 14%), mais s'effondre
   sur les periodes de rupture structurelle majeure (Bloc B : RMSE=19.74).
   Ce resultat confirme que 60 observations avec une rupture structurelle
   de magnitude x11.6 rendent le deep learning inapplicable hors
   distribution. SARIMAX reste superieur sur les series macroeconomiques
   courtes avec choc structurel — les modeles statistiques dominent
   car ils modelisent explicitement la structure saisonniere sans
   supposer de stationnarite de la distribution."*

Fichiers generes
-----------------

+----------------------------------------------+-------------------------------------+
| Fichier                                      | Description                         |
+==============================================+=====================================+
| ``outputs/reports/lstm_results.csv``         | Resultats par bloc (format binome)  |
+----------------------------------------------+-------------------------------------+
| ``outputs/reports/gridsearch_lstm_blocA.csv``| Top combinaisons Bloc A             |
+----------------------------------------------+-------------------------------------+
| ``outputs/reports/gridsearch_lstm_blocB.csv``| Top combinaisons Bloc B             |
+----------------------------------------------+-------------------------------------+
| ``outputs/reports/lstm_results_minmax.csv``  | Resultats avec MinMaxScaler         |
+----------------------------------------------+-------------------------------------+
| ``outputs/reports/lstm_scaler_comparison.csv``| Comparaison MinMax vs Robust       |
+----------------------------------------------+-------------------------------------+
| ``outputs/reports/lstm_best_params.json``    | Meilleurs hyperparametres JSON      |
+----------------------------------------------+-------------------------------------+
| ``outputs/figures/lstm_gridsearch_blocs.png``| Figure comparative 2 blocs          |
+----------------------------------------------+-------------------------------------+
| ``outputs/models/lstm_best.keras``           | Modele LSTM sauvegarde              |
+----------------------------------------------+-------------------------------------+
