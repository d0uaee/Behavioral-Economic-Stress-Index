Vue d'ensemble du projet
========================

Contexte et motivation
-----------------------

Les statistiques officielles de l'inflation au Maroc (IPC) sont publiees
mensuellement par le **Haut-Commissariat au Plan (HCP)** avec un delai de
3 a 6 semaines. Ce delai empeche une detection precoce des changements
de regime inflationniste.

**Observation cle :** Quand les prix commencent a monter, les menages
reagissent *avant* la publication officielle — ils cherchent sur Google
"prix huile", "hausse prix", lisent des articles economiques sur Hespress.
Ces comportements digitaux sont des **signaux avances de changement de regime**.

Question de recherche
---------------------

   *"Les signaux comportementaux digitaux (Google Trends, presse Hespress)
   permettent-ils de detecter les changements de regime d'inflation au Maroc
   avant les statistiques officielles ?"*

Hypotheses testees
------------------

**H1 :** Le BESI comportemental apporte une information utile pour la
detection des regimes d'inflation au Maroc.

**H2 :** L'ajout de variables macroeconomiques au BESI comportemental
ameliore la detection des regimes d'inflation au Maroc.

Architecture du projet
-----------------------

Le projet suit une architecture **Bronze -> Silver -> Gold** :

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
       +---> NLP Hespress (CAS C)
       |         Signal de validation recent
       |
       +---> Deep Learning (GridSearch LSTM)
                 96 combinaisons d'hyperparametres

Contributions originales
------------------------

1. **Construction du BESI comportemental** : indice composite base sur
   Google Trends, calibre par LassoCV sans data leakage.

2. **NLP Hespress (CAS C)** : signal textuel issu de la presse marocaine
   (Hespress, Le360, Medias24) via flux RSS. Utilise comme validation
   recente du signal comportemental.

3. **Honnêtete sur les resultats de prevision** : le modele naif
   (persistance) obtient le meilleur RMSE global (1.609). SARIMAX + BESI
   apporte de la valeur via la detection de regime (Recall = 1.00 sur Bloc B)
   mais pas via la precision point-par-point.

4. **GridSearch LSTM** : 96 combinaisons d'hyperparametres testees avec
   encodage cyclique du mois et normalisation limitee au train.

5. **Resultat asymetrique du deep learning** : le LSTM bat le modele naif
   sur le Bloc A (RMSE=1.38) mais s'effondre sur le Bloc B (RMSE=19.74)
   a cause de la rupture structurelle de 2022.

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
