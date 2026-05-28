Changelog
=========

Version 3.0 — Mai 2026 (actuelle)
-----------------------------------

**Donnees**

- Architecture Bronze → Silver → Gold implementee
- Gold dataset : 96 mois x 45 colonnes (zero simulation)
- FAO Food Price Index integre (6 sous-indices)
- Taux MAD/EUR via ECB + interpolation lineaire
- Reddit/YouTube documentees comme absentes (limites)

**BESI**

- Suppression de la composante IPC (data leakage corrige)
- Poids calibres par LassoCV (remplacement des poids fixes)
- Selection automatique des keywords en 4 etapes
- Correction biais Ramadan par decomposition STL

**Modelisation statistique**

- Protocole walk-forward sur deux blocs (A:COVID, B:Inflation)
- Test de Granger bidirectionnel (vs unidirectionnel en V2)
- Correlation BESI-inflation sur residus STL (vs brute en V2)
- H2 testee et rejetee (signal macro FAO+FX non utile)

**Deep Learning **

- GridSearch LSTM : 96 combinaisons d'hyperparametres
- Encodage cyclique du mois (month_sin / month_cos)
- Normalisation strictement sur le train (RobustScaler teste)
- Protocole walk-forward identique aux modeles statistiques
- Resultat asymetrique : Bloc A RMSE=1.38 vs Bloc B RMSE=19.74

Version 2.0 — Mars 2026
------------------------

- Ajout de Google Trends avec chunking ancre
- Construction BESI V2 (avec composante IPC — corrige en V3)
- Premiers modeles SARIMA/SARIMAX
- Notebooks 01-04 premiers jets

Version 1.0 — Fevrier 2026
---------------------------

- Initialisation du projet
- Collecte manuelle IPC HCP Maroc 2017-2024
- Exploration preliminaire des donnees
