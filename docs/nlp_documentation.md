# Documentation NLP - Extension presse marocaine

**Projet :** BESI V3  
**Module :** extension NLP exploratoire  
**Periode :** 2017-01 a 2024-12

## 1. Objectif

Cette extension teste si un signal texte local marocain peut completer le BESI
construit a partir de Google Trends.

La bonne interpretation n'est pas :

- "sentiment pur des menages"

La bonne interpretation est :

- **signal presse editorial local**

## 2. Source utilisee

Source principale :

- Hespress via API WordPress

Texte utilise :

- `title + excerpt`

Donc :

- ce n'est pas un vrai corpus de commentaires lecteurs
- c'est un indicateur de pression mediatique economique locale

## 3. Pipeline

| Etape | Fichier | Role |
|---|---|---|
| Collecte | `src/nlp/scraper_hespress.py` | extraction Hespress |
| Nettoyage | `src/nlp/preprocess_darija.py` | nettoyage texte |
| Scoring | `src/nlp/sentiment_scorer.py` | score lexical mensuel |
| Integration | `src/nlp/besi_v2.py` | merge et comparaison v1/v2 |

## 4. Sorties produites

| Fichier | Description |
|---|---|
| `data/bronze/hespress_raw.csv` | corpus brut |
| `data/silver/hespress_clean.csv` | corpus nettoye |
| `data/silver/sentiment_monthly.csv` | score mensuel |
| `data/silver/besi_v2_variants_monthly.csv` | variantes BESI v2 |
| `results/NLP_RESULTS.md` | rapport final NLP |

## 5. Couverture reelle

| Indicateur | Valeur |
|---|---|
| Corpus total | 5 788 textes |
| Couverture mensuelle | 96/96 mois |
| Imputation | 0 mois |

## 6. Methode de scoring

Le scoring retenu est volontairement simple :

- lexique economique negatif / positif
- agregation mensuelle
- normalisation sur train

Le projet n'utilise pas de modele BERT dans la version finale.

## 7. Resultat final

Le Lasso attribue :

- `alpha = 1.0` au BESI Trends
- `beta = 0.0` au signal NLP

Verdict :

- **CAS C**

Interpretation :

- le signal NLP n'ajoute pas d'information conditionnelle supplementaire
- il enrichit l'analyse qualitative
- il n'est pas retenu dans le BESI principal

## 8. Message a retenir

Cette extension est utile pour montrer que le projet a teste une piste NLP
locale de maniere honnete, meme si le resultat final est negatif.
