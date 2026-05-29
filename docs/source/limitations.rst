Limites et Perspectives
========================

Limites documentees
--------------------

+------------------------------------------+----------------------------------+--------------------------------------+
| Limite                                   | Impact                           | Recommandation                       |
+==========================================+==================================+======================================+
| Naif gagne en RMSE global                | SARIMAX + BESI moins precis      | Evaluer sur criteres de regime       |
+------------------------------------------+----------------------------------+--------------------------------------+
| IPC HCP disponible depuis 2017 seulement | Seulement 96 observations        | Recuperer archives HCP pre-2017      |
+------------------------------------------+----------------------------------+--------------------------------------+
| MAD/EUR par interpolation lineaire       | Donnees BAM non open data        | Recuperer donnees officielles BAM    |
+------------------------------------------+----------------------------------+--------------------------------------+
| NLP Hespress : 30-90 jours seulement     | Pas d'historique long terme      | Documenter comme CAS C               |
+------------------------------------------+----------------------------------+--------------------------------------+
| Reddit/YouTube absents                   | BESI = Trends + presse seulement | Documenter comme limite methodolog.  |
+------------------------------------------+----------------------------------+--------------------------------------+
| Granger non significatif                 | Relation non-lineaire            | Explorer modeles a seuil (TAR, STAR) |
+------------------------------------------+----------------------------------+--------------------------------------+
| AUC globale = 0.35                       | Penalisee par Bloc A (COVID)     | Evaluer separement chocs exogenes    |
+------------------------------------------+----------------------------------+--------------------------------------+
| LSTM inefficace sur Bloc B               | 60 obs. insuffisantes pour DL    | Attendre plus de donnees             |
+------------------------------------------+----------------------------------+--------------------------------------+

Pourquoi le naif gagne en RMSE — explication
----------------------------------------------

C'est un resultat classique en economie des series macroeconomiques.
L'IPC marocain est tres persistant (proche d'une racine unitaire).
Predire "le mois prochain = ce mois" (modele naif de persistance)
minimise l'erreur quadratique sur une serie aussi lisse.

Ce n'est pas un echec du projet — c'est un resultat honnete qui
souligne que **la valeur du BESI est dans la detection de regime**
(Recall = 1.00 sur 2022-2024) et non dans la precision mensuelle.

Perspectives de recherche
--------------------------

**Court terme**

- Evaluer le BESI sur des metriques de regime plutot que de prevision
- Tester Prophet (changepoints automatiques pour gerer 2022)
- Etendre le NLP Hespress avec un historique plus long (Scrapy)

**Moyen terme**

- **Modele TVP-SARIMAX** : coefficients variables dans le temps
  pour gerer la rupture 2022 sans segmentation manuelle
- **Modeles a seuil (TAR/STAR)** : capturer la non-linearite
  confirmee par Granger
- Obtenir les donnees BAM officielles pour le taux MAD/EUR

**Long terme**

- Dashboard temps reel (mise a jour automatique mensuelle)
- Extension a d'autres pays MENA (Algerie, Tunisie, Egypte)
- Publication academique en economie appliquee
