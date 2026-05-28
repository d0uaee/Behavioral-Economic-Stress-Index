Evaluation et Metriques
========================

Metriques de prevision
-----------------------

Trois metriques sont utilisees pour evaluer les modeles de prevision :

**RMSE (Root Mean Square Error)**

.. math::

   RMSE = \sqrt{\frac{1}{n} \sum_{t=1}^{n} (y_t - \hat{y}_t)^2}

Penalise fortement les grosses erreurs. Sensible aux outliers.
En points IPC (base 2017=100).

**MAE (Mean Absolute Error)**

.. math::

   MAE = \frac{1}{n} \sum_{t=1}^{n} |y_t - \hat{y}_t|

Erreur absolue moyenne. Plus intuitive que le RMSE.

**MAPE (Mean Absolute Percentage Error)**

.. math::

   MAPE = \frac{100}{n} \sum_{t=1}^{n} \left|\frac{y_t - \hat{y}_t}{y_t}\right|

Erreur en pourcentage. Comparable entre series de differentes echelles.

Metriques d'alerte precoce
---------------------------

Pour evaluer le systeme d'early warning BESI, les metriques de
classification binaire sont utilisees :

+------------+------+---------------------------------------------------+
| Metrique   | Formule | Interpretation                               |
+============+======+===================================================+
| Recall     | TP/(TP+FN) | % d'episodes de stress correctement detectes |
+------------+------+---------------------------------------------------+
| Precision  | TP/(TP+FP) | % des alertes qui correspondent a un vrai stress |
+------------+------+---------------------------------------------------+
| F1-score   | 2xPxR/(P+R) | Equilibre precision/recall                   |
+------------+------+---------------------------------------------------+
| AUC-ROC    | ---  | Capacite de discrimination globale            |
+------------+------+---------------------------------------------------+

Ou :

- **TP** = BESI en alerte ET stress IPC reel (detection correcte)
- **FP** = BESI en alerte MAIS pas de stress IPC (fausse alerte)
- **FN** = BESI calme MAIS stress IPC reel (episode manque)

.. note::

   Dans ce projet, le **Recall est prioritaire sur la Precision**.
   En politique economique, il vaut mieux une fausse alerte qu'un
   episode de stress manque (asymetrie des couts).

Seuils d'alerte utilises
~~~~~~~~~~~~~~~~~~~~~~~~~

- **Seuil BESI** : 0.35 (valeur au-dessus = alerte declenchee)
- **Seuil Inflation** : 2% YoY (au-dessus = episode de stress reel)

Protocole walk-forward
-----------------------

Pour garantir l'absence de data leakage temporel, un protocole
**walk-forward strict** est applique :

.. code-block:: text

   Pour t = debut_test, ..., fin_test :
       1. Entrainer sur toutes les donnees [debut, t-1]
       2. Predire uniquement t (h=1, un pas en avant)
       3. Enregistrer l'erreur
       4. Avancer d'un mois

Ce protocole simule exactement les conditions d'un usage en temps reel
et evite tout regard vers le futur dans les donnees d'entrainement.
