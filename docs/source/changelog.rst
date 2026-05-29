Changelog
=========

Version 3.0 — Mai 2026 (actuelle)
-----------------------------------

**Donnees**

- Architecture Bronze -> Silver -> Gold implementee
- Gold dataset : 96 mois x 45 colonnes (zero simulation)
- FAO Food Price Index integre (6 sous-indices)
- Taux MAD/EUR via ECB + interpolation lineaire
- NLP Hespress via flux RSS (CAS C — validation recente)
- Reddit/YouTube documentees comme absentes (limites)

**BESI**

- Suppression de la composante IPC (data leakage corrige)
- Poids calibres par LassoCV (remplacement des poids fixes)
- Selection automatique des keywords en 4 etapes
- Correction biais Ramadan par decomposition STL

**Hypotheses reformulees**

- H1 : "Le BESI comportemental apporte une information utile pour
  la detection des regimes d'inflation au Maroc"
- H2 : "L'ajout de variables macroeconomiques au BESI comportemental
  ameliore la detection des regimes d'inflation au Maroc"

**Resultats honnetes documentes**

- Le modele naif gagne en RMSE global (1.609) — assume et documente
- SARIMAX + BESI apporte de la valeur via la detection de regime
  (Recall = 1.00 sur Bloc B) et non via la RMSE

**Modelisation statistique**

- Protocole walk-forward sur deux blocs (A:COVID, B:Inflation)
- Test de Granger bidirectionnel
- H2 testee et rejetee (signal macro FAO+FX degrade le Recall)

**Deep Learning**

- GridSearch LSTM : 96 combinaisons d'hyperparametres
- Encodage cyclique du mois (month_sin / month_cos)
- Normalisation strictement sur le train
- MinMaxScaler et RobustScaler testes et compares
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
