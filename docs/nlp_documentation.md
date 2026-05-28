# Documentation Technique — Extension NLP Presse Marocaine

Projet concerné : `BESI V3`  
Module documenté : extension NLP exploratoire basée sur la presse marocaine  
Période couverte : `2017-01-01` → `2024-12-01`  
Fréquence : mensuelle (`MS`)

---

## 1. Objectif

Cette extension NLP a pour objectif de tester si un **signal textuel local marocain**
peut compléter le `Behavioral Economic Stress Index (BESI)` existant, déjà construit
à partir de Google Trends.

L’idée n’est pas de remplacer le BESI principal, mais de répondre à la question suivante :

> Un signal texte issu de la presse marocaine apporte-t-il une information
> conditionnelle supplémentaire au-delà des Google Trends pour la détection
> des régimes d’inflation ?

Cette extension a été conçue comme un module **exploratoire**, documenté honnêtement,
avec une règle stricte : **ne pas intégrer le signal au BESI principal si les données
ou les résultats ne le justifient pas**.

---

## 2. Positionnement Méthodologique

Le signal NLP implémenté ici n’est **pas** un vrai signal de commentaires lecteurs.

Dans la version effectivement exécutée :
- la source principale est **Hespress via son API WordPress JSON**
- le texte utilisé est formé de `title + excerpt`
- le signal doit donc être interprété comme un **signal presse éditorial**
- il reflète surtout :
  - la pression médiatique économique locale
  - le cadrage éditorial de l’actualité économique
  - la fréquence d’apparition de termes liés à la hausse des prix

Il ne faut donc pas le sur-vendre comme :
- “sentiment pur des ménages”
- ou “opinion directe des lecteurs marocains”

Le bon nom conceptuel est :
- `press_stress_signal`
- ou `signal presse économique marocain`

---

## 3. Architecture NLP

Le pipeline NLP a été implémenté sans modifier le pipeline V3 principal.

### Fichiers sources

| Fichier | Rôle |
|---|---|
| `src/nlp/scraper_hespress.py` | collecte historique Hespress |
| `src/nlp/preprocess_darija.py` | nettoyage léger du texte |
| `src/nlp/sentiment_scorer.py` | scoring lexical mensuel |
| `src/nlp/besi_v2.py` | intégration au BESI et évaluation comparative |

### Principe de pipeline

1. collecte du corpus brut
2. nettoyage / normalisation légère
3. scoring lexical par texte
4. agrégation mensuelle
5. merge avec le Gold dataset existant
6. test de variantes BESI v2
7. verdict honnête

---

## 4. Source de Données

### 4.1 Source principale

Source utilisée :
- `https://www.hespress.com/wp-json/wp/v2/posts`

Catégories exploitées :
- `economie`
- `hausse`
- `baisse`

La collecte se fait sur la période :
- `2017-01-01` → `2024-12-31`

### 4.2 Nature des textes

Chaque observation correspond à un texte exploitable composé de :
- `title`
- `excerpt`

Autrement dit :
- le module collecte des **articles/extraits éditoriaux**
- pas des commentaires utilisateurs
- pas un corpus annoté
- pas un flux social direct

### 4.3 Fallback prévu

Le scraper contient un fallback RSS si l’API échoue.

Dans la version exécutée, le fallback n’a pas été nécessaire, car :
- l’API a fonctionné
- la couverture historique a été suffisante

---

## 5. Couverture Observée

La couverture réelle est bien meilleure que ce qui était anticipé au départ.

### Couverture annuelle

| Année | Nombre de textes |
|---|---:|
| 2017 | 397 |
| 2018 | 442 |
| 2019 | 424 |
| 2020 | 558 |
| 2021 | 704 |
| 2022 | 1027 |
| 2023 | 1048 |
| 2024 | 1188 |

### Résumé

- total textes collectés : **5 788**
- couverture mensuelle réelle : **96/96 mois**
- mois imputés : **0**

Conclusion :
- la contrainte “couverture minimale 2019-2024” est satisfaite
- la couverture 2017-2019 existe aussi, sans zone creuse majeure

Références :
- `outputs/reports/hespress_coverage_by_year.csv`
- `outputs/reports/hespress_text_coverage_by_year.csv`

---

## 6. Schéma Exact des Fichiers Produits

### 6.1 Bronze — `data/bronze/hespress_raw.csv`

Un enregistrement = un texte brut issu de l’API Hespress ou du fallback RSS.

| Nom | Type | Description |
|---|---|---|
| `date` | `date` | mois de référence au format `YYYY-MM-01` |
| `published_at` | `datetime` | date/heure de publication de l’article |
| `title` | `str` | titre de l’article |
| `text` | `str` | texte brut utilisé pour le NLP |
| `section` | `str` | catégorie de collecte (`economie`, `hausse`, `baisse`, etc.) |
| `url` | `str` | lien source |
| `source_type` | `str` | type de texte (`post_excerpt`, `rss`) |
| `article_id` | `str` | identifiant article si disponible |
| `comment_count` | `float`/`NaN` | non utilisé dans cette version |
| `collection_method` | `str` | `wp_json_api` ou `rss_fallback` |

### 6.2 Silver texte — `data/silver/hespress_clean.csv`

Un enregistrement = un texte nettoyé.

| Nom | Type | Description |
|---|---|---|
| `date` | `date` | mois de référence |
| `article_id` | `str` | identifiant article |
| `source_type` | `str` | source NLP |
| `title` | `str` | titre original |
| `url` | `str` | lien source |
| `raw_text` | `str` | texte avant nettoyage |
| `clean_text` | `str` | texte après nettoyage |
| `lang_profile` | `str` | profil linguistique heuristique |
| `token_count` | `int` | nombre de tokens après nettoyage |
| `has_arabizi` | `0/1` | présence de chiffres Arabizi détectée |
| `coverage_flag` | `0/1` | ici `1` car texte réel |
| `processing_notes` | `str` | note éventuelle |

### 6.3 Silver mensuel — `data/silver/sentiment_monthly.csv`

Un enregistrement = un mois.

| Nom | Type | Description |
|---|---|---|
| `date` | `date` | mois au format `YYYY-MM-01` |
| `score_lexical_raw` | `float` | score mensuel moyen avant normalisation |
| `n_texts` | `int` | nombre de textes dans le mois |
| `source_type_dominant` | `str` | source dominante du mois |
| `n_negative_hits` | `int` | nombre total de matches négatifs |
| `n_positive_hits` | `int` | nombre total de matches positifs |
| `coverage_flag` | `0/1` | `1` si données réelles, `0` si imputation |
| `imputation_reason` | `str` | raison d’imputation si besoin |
| `score_lexical_norm` | `float` | score normalisé sur `[0,1]` |
| `notes` | `str` | champ libre |

### 6.4 Variantes BESI — `data/silver/besi_v2_variants_monthly.csv`

| Nom | Type | Description |
|---|---|---|
| `month` | `date` | mois |
| `behavioral_index_pure` | `float` | BESI Trends existant |
| `score_lexical_norm` | `float` | signal NLP mensuel |
| `besi_v2a_fixed` | `float` | combinaison fixe 70/30 |
| `besi_v2b_lasso` | `float` | combinaison pondérée par Lasso |
| `besi_v2c_nlp_only` | `float` | signal NLP seul |

---

## 7. Prétraitement du Texte

Le nettoyage a volontairement été gardé **simple et robuste**.

### Étapes appliquées

- passage en minuscules
- suppression URLs
- suppression `@mentions` et `#hashtags`
- suppression emojis
- normalisation Arabizi minimale
- tokenisation simple
- suppression de stopwords darija/français fréquents

### Arabizi minimal

Mappings utilisés :
- `3 -> ع`
- `7 -> ح`
- `9 -> ق`
- `5 -> خ`
- `2 -> ء`

Important :
- il ne s’agit pas d’une translittération complète
- le but était de nettoyer légèrement sans introduire une complexité fragile

### Détection linguistique

Le champ `lang_profile` est heuristique :
- `darija`
- `fr`
- `mixed`
- `unknown`

Dans l’exécution observée, la quasi-totalité des textes est classée `darija`,
ce qui reflète surtout la présence dominante d’arabe dans les extraits Hespress.

---

## 8. Méthode de Scoring

### 8.1 Approche retenue

Seule la **baseline lexicale** a été retenue.

Le choix est volontaire :
- faible coût
- interprétabilité forte
- robustesse sur petit corpus
- pas de dépendance à un corpus annoté

### 8.2 Lexique

Trois groupes de termes sont utilisés :

- termes négatifs de stress économique
- termes positifs de stabilité / détente
- intensificateurs

Exemples de termes négatifs :
- `hausse`
- `flambée`
- `augmentation`
- `غلاء`
- `ارتفع`
- `زاد`

Exemples de termes positifs :
- `baisse`
- `normal`
- `استقر`
- `انخفض`
- `رخيص`

### 8.3 Score par texte

Formule simplifiée :

`score = (nombre_hits_negatifs - nombre_hits_positifs) * intensificateur`

où :
- l’intensificateur vaut `1.5` si certains termes sont présents
- sinon `1.0`

### 8.4 Agrégation mensuelle

Pour chaque mois :
- moyenne des scores des textes
- comptage des hits positifs / négatifs
- normalisation du score sur `[0,1]`

### 8.5 Normalisation

La normalisation est faite **sur train uniquement** :
- train jusqu’à `2021-12-01`

Cela respecte la contrainte :
- **pas de leakage**

---

## 9. Intégration au BESI

Le merge se fait avec :
- `data/gold/model_dataset_monthly.csv`

Trois variantes sont construites :

### `BESI_v2a_fixed`

Combinaison fixe :

`0.7 * behavioral_index_pure + 0.3 * score_lexical_norm`

Objectif :
- comparaison simple
- test manuel de contribution

### `BESI_v2b_lasso`

Combinaison optimisée :

- features :
  - `behavioral_index_pure`
  - `score_lexical_norm`
- cible :
  - `target_inflation_yoy_t1`
- période train :
  - `2017-01` → `2021-12`

Objectif :
- mesurer honnêtement si le NLP apporte une information conditionnelle

### `BESI_v2c_nlp_only`

Signal NLP seul, utilisé comme diagnostic.

Objectif :
- mesurer sa contribution brute sans Trends

---

## 10. Évaluation

Les variantes sont comparées à `BESI v1 (Trends)` avec :

- `AIC`
- `RMSE Bloc B`
- `Recall Bloc B`
- `Precision Bloc B`
- `F1 Bloc B`
- `AP Bloc B`

Bloc B correspond à :
- `2022-01` → `2024-12`

Il s’agit du bloc inflationniste le plus important pour l’analyse.

---

## 11. Résultats Observés

### Poids Lasso

| Feature | Poids |
|---|---:|
| `behavioral_index_pure` | 1.000 |
| `score_lexical_norm` | 0.000 |

Conclusion immédiate :
- le Lasso donne un poids nul au signal NLP

### Comparatif principal

| Modèle | RMSE Bloc B | Recall Bloc B | F1 Bloc B |
|---|---:|---:|---:|
| `SARIMAX + BESI v1 (Trends)` | 1.9761 | 1.000 | 0.8136 |
| `SARIMAX + BESI v2a (Fixe)` | 1.9859 | 1.000 | 0.8136 |
| `SARIMAX + BESI v2b (Lasso)` | 1.9761 | 1.000 | 0.8136 |
| `SARIMAX + NLP seul` | 1.9046 | 0.875 | 0.7636 |

Lecture correcte :
- `v2b_lasso` retombe exactement sur `v1`
- `v2a_fixed` n’améliore pas le BESI principal
- `NLP seul` est intéressant comme signal exploratoire
- mais il ne justifie pas son intégration conditionnelle au BESI existant

---

## 12. Verdict Honnête

Verdict final :
- **CAS C**

Interprétation :

> Le Lasso assigne un poids nul au signal NLP, ce qui indique que ce signal
> n’apporte pas d’information conditionnelle supplémentaire au-delà des
> Google Trends sur cet échantillon.

Ce résultat est scientifiquement valide.

Il ne faut pas le présenter comme :
- un échec
- ou un bug du pipeline

Il signifie simplement que :
- le signal presse éditorial est probablement redondant avec Trends
- ou qu’il est moins informatif conditionnellement au BESI comportemental

---

## 13. Limites

### Limite 1 — nature du signal

Le signal n’est pas un vrai sentiment lecteur.

Il s’agit d’un signal :
- éditorial
- médiatique
- local

Donc :
- bon pour enrichir l’interprétation
- moins fort comme proxy direct du ressenti des ménages

### Limite 2 — absence de commentaires réels

Le pipeline final n’utilise pas :
- commentaires utilisateurs
- likes
- réponses de lecteurs

### Limite 3 — scoring lexical simple

Le scoring lexical est volontairement simple :
- robuste
- transparent
- mais limité

Il ne capture pas bien :
- ironie
- négation complexe
- contexte
- sentiment implicite

### Limite 4 — petit échantillon temporel

La période reste courte :
- 96 mois

Cela limite :
- la stabilité des pondérations
- la robustesse des conclusions hors échantillon

---

## 14. Reproductibilité

Le pipeline peut être rejoué dans cet ordre :

1. `python -m src.nlp.scraper_hespress`
2. `python -m src.nlp.preprocess_darija`
3. `python -m src.nlp.sentiment_scorer`
4. `python -m src.nlp.besi_v2`

Sorties attendues :
- `data/bronze/hespress_raw.csv`
- `data/silver/hespress_clean.csv`
- `data/silver/sentiment_monthly.csv`
- `data/silver/besi_v2_variants_monthly.csv`
- `outputs/reports/nlp_besi_comparison.csv`
- `outputs/reports/nlp_lasso_weights.csv`
- `results/NLP_RESULTS.md`

---

## 15. Phrase Recommandée pour l’Oral

> Nous avons testé une extension NLP locale à partir de la presse marocaine,
> en construisant un signal mensuel basé sur Hespress.
> La couverture des données est complète sur 2017-2024, sans imputation.
> En revanche, le Lasso attribue un poids nul à ce signal face aux Google Trends.
> Cela signifie que le signal presse éditorial n’apporte pas d’information
> conditionnelle supplémentaire au BESI principal sur cet échantillon.
> C’est un résultat négatif, mais méthodologiquement propre et scientifiquement valide.

---

## 16. Fichiers de Référence

| Fichier | Rôle |
|---|---|
| `src/nlp/scraper_hespress.py` | collecte historique |
| `src/nlp/preprocess_darija.py` | nettoyage texte |
| `src/nlp/sentiment_scorer.py` | scoring lexical mensuel |
| `src/nlp/besi_v2.py` | intégration BESI + évaluation |
| `data/bronze/hespress_raw.csv` | corpus brut |
| `data/silver/hespress_clean.csv` | corpus nettoyé |
| `data/silver/sentiment_monthly.csv` | signal mensuel |
| `outputs/reports/nlp_besi_comparison.csv` | métriques comparatives |
| `outputs/reports/nlp_lasso_weights.csv` | poids finaux |
| `results/NLP_RESULTS.md` | verdict synthétique |
