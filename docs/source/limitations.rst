Limites et Perspectives
========================

Limites documentees
--------------------

+------------------------------------------+----------------------------------+--------------------------------------+
| Limite                                   | Impact                           | Recommandation                       |
+==========================================+==================================+======================================+
| IPC HCP disponible depuis 2017 seulement | Seulement 96 observations        | Recuperer archives HCP pre-2017      |
+------------------------------------------+----------------------------------+--------------------------------------+
| MAD/EUR par interpolation lineaire       | Donnees BAM non open data        | Recuperer donnees officielles BAM    |
+------------------------------------------+----------------------------------+--------------------------------------+
| Reddit/YouTube absents                   | BESI = Trends seul               | Documenter comme limite methodolog.  |
+------------------------------------------+----------------------------------+--------------------------------------+
| Granger non significatif                 | Relation non-lineaire non captee | Explorer modeles a seuil (TAR, STAR) |
+------------------------------------------+----------------------------------+--------------------------------------+
| AUC globale = 0.35                       | Penalisee par Bloc A (COVID)     | Evaluer separement les types de choc |
+------------------------------------------+----------------------------------+--------------------------------------+
| LSTM inefficace sur Bloc B               | 60 obs. insuffisantes pour DL    | Attendre plus de donnees disponibles |
+------------------------------------------+----------------------------------+--------------------------------------+
| trends_carburant = 100% NaN              | Feature ignoree dans le LSTM     | Recollecte de cette serie            |
+------------------------------------------+----------------------------------+--------------------------------------+

Limites specifiques au Deep Learning
--------------------------------------

**Taille du dataset :**
96 observations mensuelles dont 60 pour l'entrainement sont
structurellement insuffisantes pour un reseau LSTM. La litterature
recommande plusieurs milliers de sequences pour un apprentissage robuste.

**Rupture hors distribution :**
Le choc inflationniste de 2022 (x11.6) est si eloigne de la
distribution d'entrainement 2017-2021 que ni MinMaxScaler ni
RobustScaler ne peuvent corriger l'extrapolation catastrophique.

**Non-linearite limitee :**
L'IPC marocain est tres lisse et saisonnier. Il n'y a pas assez de
non-linearites complexes pour justifier un reseau de neurones —
SARIMA les capture avec 5 parametres seulement.

Perspectives de recherche
--------------------------

**Court terme**

- Optimiser les poids BESI par validation croisee temporelle
- Tester Prophet (changepoints automatiques pour gerer 2022)
- Ajouter donnees HCP haute frequence (hebdomadaires si disponibles)
- Recuperer la serie ``trends_carburant`` manquante

**Moyen terme**

- **NLP sur Darija :** analyse de sentiment sur Reddit r/Morocco
  en langue darija pour enrichir le BESI
- **Modele TVP-SARIMAX :** coefficients variables dans le temps
  pour gerer la rupture 2022 sans segmentation manuelle
- **Modeles a seuil (TAR/STAR) :** capturer la non-linearite
  confirmee par le test de Granger bidirectionnel
- Obtenir les donnees BAM officielles pour le taux MAD/EUR

**Long terme**

- Dashboard temps reel (mise a jour automatique mensuelle)
- Extension a d'autres pays MENA (Algerie, Tunisie, Egypte)
- Publication academique dans une revue d'economie appliquee
- Comparaison avec l'indice de confiance des consommateurs
  publie par le HCP (si disponible)

Note sur le choix de SARIMA vs Deep Learning
---------------------------------------------

Ce projet demontre empiriquement que sur des series macroeconomiques
courtes (< 100 observations) avec rupture structurelle, les modeles
statistiques classiques dominent le deep learning. Ce resultat est
coherent avec la litterature sur les series temporelles economiques
courtes.

Le deep learning apporte de la valeur **uniquement** sur le Bloc A
(periode stable COVID, RMSE=1.38 vs Naif=1.609). Ce resultat partiel
montre le potentiel du LSTM pour les periodes sans rupture structurelle,
mais confirme ses limites fondamentales sur les chocs economiques
majeurs hors distribution.
