Vue d'ensemble du projet
========================

Contexte et motivation
-----------------------

Les statistiques officielles de l'inflation au Maroc (IPC) sont publiees
mensuellement par le **Haut-Commissariat au Plan (HCP)** avec un delai de
3 a 6 semaines. Ce delai empeche une detection precoce du stress economique
des menages.

**Observation cle :** Quand les prix commencent a monter, les menages
reagissent *avant* la publication officielle — ils cherchent sur Google
"prix huile", "hausse prix", postent sur les reseaux sociaux, consultent
des articles economiques. Ces comportements digitaux sont des **signaux avances**.

Question de recherche
---------------------

   *"Les signaux comportementaux digitaux (Google Trends) permettent-ils
   de detecter et predire le stress economique des menages marocains
   avant les statistiques officielles ?"*

Hypotheses testees
------------------

**H1 :** Le BESI comportemental (base Google Trends) ameliore la detection
des episodes d'inflation elevee par rapport a un modele SARIMA seul.

**H2 :** L'ajout de signaux macro-economiques (FAO + taux de change MAD/EUR)
ameliore encore la detection par rapport au BESI comportemental seul.

Architecture du projet
-----------------------

Le projet suit une architecture **Bronze → Silver → Gold** :

.. code-block:: text

   Donnees brutes (Bronze)
       |
       v
   Donnees nettoyees (Silver)
       |
       v
   Dataset modelisation (Gold)   96 mois x 45 colonnes
       |
       +---> Modeles statistiques (SARIMA / SARIMAX)
       |         Walk-forward, Bloc A et Bloc B
       |
       +---> Deep Learning (GridSearch LSTM)
       |         96 combinaisons d'hyperparametres
       |
       +---> Analyses
                 Rupture structurelle, Granger, Early Warning

Contributions originales
------------------------

1. **Construction du BESI** : indice composite comportemental multi-sources,
   calibre par LassoCV sans data leakage (IPC retire des features).

2. **Protocole d'evaluation rigoureux** : deux blocs temporels distincts
   (COVID 2020-2021 et Inflation 2022-2024) pour mesurer la robustesse.

3. **GridSearch LSTM** : 96 combinaisons d'hyperparametres testees avec
   encodage cyclique du mois et normalisation strictement limitee au train.

4. **Resultat asymetrique du deep learning** : le LSTM bat le modele naif
   sur le Bloc A (RMSE=1.38 vs 1.609) mais s'effondre sur le Bloc B
   (RMSE=22 vs 1.609) a cause de la rupture structurelle de 2022.

Equipe et cadre academique
---------------------------

+------------------+-------------------------------------------+
| Element          | Detail                                    |
+==================+===========================================+
| Auteurs          | Douae Ahadji & Adama Basse                |
+------------------+-------------------------------------------+
| Etablissement    | ENSAM Meknes (Maroc)                      |
+------------------+-------------------------------------------+
| Cours            | Series Temporelles                        |
+------------------+-------------------------------------------+
| Duree            | 8 semaines                                |
+------------------+-------------------------------------------+
| Annee            | 2025-2026                                 |
+------------------+-------------------------------------------+
| Langage          | Python 3.10                               |
+------------------+-------------------------------------------+
| Depot GitHub     | d0uaee/Behavioral-Economic-Stress-Index   |
+------------------+-------------------------------------------+
