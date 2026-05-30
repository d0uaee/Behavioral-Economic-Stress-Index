BESI — Behavioral Economic Stress Index
========================================

**Detection Precoce des Regimes d'Inflation au Maroc**

| **Auteurs :** Douae Ahadji & Adama Basse
| **Cours :** Series Temporelles — ENSAM Meknes
| **Annee :** 2025-2026
| **GitHub :** https://github.com/d0uaee/Behavioral-Economic-Stress-Index

----

**Resume du projet**

Ce projet construit **BESI** (*Behavioral Economic Stress Index*), un indice composite
fonde sur les signaux digitaux comportementaux marocains (Google Trends + presse Hespress),
et l'integre dans un modele **SARIMAX** pour detecter les regimes d'inflation de l'IPC
mensuel du Maroc (2017-2024).

**Resultats principaux (V3) :**

- Le modele **naif (persistance) obtient le meilleur RMSE global (1.609)** — resultat honnete documente
- SARIMAX + BESI ameliore l'AIC de **-7.77 points** vs SARIMA pur
- SARIMAX + BESI detecte **100% des mois a inflation elevee** en 2022-2024 (Recall = 1.00)
- NLP Hespress (CAS C) : signal de validation recent, non integre dans le BESI principal
- Rupture structurelle 2022 confirmee : inflation x11.6 (p < 0.0001)
- H1 **partiellement validee** — H2 **rejetee**
- GridSearch LSTM (96 combinaisons) : RMSE=1.38 sur Bloc A (COVID)

----

.. toctree::
   :maxdepth: 2
   :caption: Demarrage rapide

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Le Projet

   overview
   data
   besi_index

.. toctree::
   :maxdepth: 2
   :caption: Modelisation

   models_statistical
   models_deep_learning
   evaluation

.. toctree::
   :maxdepth: 2
   :caption: Resultats et Analyses

   results
   hypotheses
   limitations

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
   notebooks
   outputs

.. toctree::
   :maxdepth: 1
   :caption: A propos

   changelog
