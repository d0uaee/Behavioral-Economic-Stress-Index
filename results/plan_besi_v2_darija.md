# BESI v2 — Extension Sentiment Darija Marocain
## Plan de Travail pour la Section « Perspectives »

> **Statut** : Perspectives futures — non implémenté dans ce mémoire  
> **Horizon** : 6-8 mois de travail post-mémoire  
> **Référence** : Tetlock (2007), Baker & Wurgler (2006), Gaanoun et al. (2023), Abdul-Mageed et al. (2021)

---

## 1. Motivation

Le BESI actuel repose sur des **requêtes Google Trends** — des signaux d'*intention de recherche*. Il ne capture pas le **sentiment** exprimé dans les discussions en ligne : commentaires sur Hespress, Facebook marocain, ou forums économiques.

La Darija (arabe marocain dialectal) est la langue dominante de ces conversations mais reste **sous-représentée dans les modèles NLP arabes standard**. Les modèles dédiés (DarijaBERT, MarBERT) ont émergé depuis 2021 et ouvrent une voie réaliste.

**Hypothèse H2** (non testée dans ce mémoire) :  
> *Un indice de sentiment en Darija, agrégé mensuellement depuis les sources digitales marocaines, améliore la détection précoce du stress économique au-delà du BESI actuel.*

---

## 2. Sources de données

### 2.1 Sources prioritaires
| Source | Volume estimé | Langue | Méthode d'accès |
|---|---|---|---|
| Commentaires Hespress | ~50 000 art./an | Darija + arabe | Scraping BeautifulSoup |
| Facebook public (groupes éco.) | Variable | Darija | Graph API (accès limité) |
| YouTube comments (chaînes éco.) | ~10 000/mois | Darija + français | YouTube Data API v3 |

### 2.2 Filtre thématique
Keywords en Darija : `{"غلاء المعيشة", "ثمن", "مزال غالي", "الحال صعيبة", "سعر"}` + translittération latine (`"ghla", "taman", "cher"`)

---

## 3. Pipeline NLP

### 3.1 Prétraitement Darija
```python
# Normalisation spécifique Darija
def normalize_darija(text):
    text = re.sub(r'[0-9]+', 'NUM', text)
    # Diacritiques et caractères spéciaux arabes
    text = re.sub(r'[ً-ٟ]', '', text)
    # Code-switching latin → arabe (heuristique)
    # "cher" → "غالي", "prix" → "ثمن"
    return text
```

### 3.2 Modèles de sentiment recommandés
| Modèle | Taille | Darija | Citation |
|---|---|---|---|
| **DarijaBERT** | 110M | Natif | Gaanoun et al. (2023) |
| **MarBERT** | 163M | Partiel | Abdul-Mageed et al. (2021) |
| CAMeLBERT-mix | 163M | Partiel | Inoue et al. (2021) |

**Recommandation** : DarijaBERT comme modèle principal, MarBERT comme robustesse.

### 3.3 Absence d'annotations → stratégie zero-shot
```python
from transformers import pipeline

# Option 1 : Zero-shot avec DarijaBERT fine-tuné sur SA arabe
sentiment_pipeline = pipeline("text-classification",
                               model="SI2M-Lab/DarijaBERT",
                               tokenizer="SI2M-Lab/DarijaBERT")

# Option 2 : Lexique manuel de 200 mots économiques Darija
# (construit manuellement ou via GPT-4 avec validation native)
LEXIQUE_POSITIF = {"باغي", "زوين", "رخيص", ...}
LEXIQUE_NEGATIF = {"غالي", "صعيب", "ما كاينش", ...}
```

---

## 4. Construction du BESI v2

### 4.1 Agrégation mensuelle
```python
def aggregate_monthly_sentiment(comments_df):
    """
    Entrée  : DataFrame avec colonnes [date, score_sentiment, source]
    Sortie  : Série mensuelle BESI_sentiment (moyenne pondérée)
    """
    monthly = (comments_df
               .set_index('date')
               .resample('MS')['score_sentiment']
               .agg(['mean', 'std', 'count']))
    
    # Pondération par volume (plus de commentaires = signal plus fiable)
    monthly['weighted_score'] = monthly['mean'] * np.log1p(monthly['count'])
    return monthly['weighted_score'].rename('besi_sentiment')
```

### 4.2 Formule BESI v2 (proposition)
```
BESI_v2(t) = α × BESI_trends(t) + β × BESI_sentiment(t)
```

Où α et β sont estimés par **LassoCV** sur la période d'entraînement (Bloc A, 2017-2021) :
```python
from sklearn.linear_model import LassoCV
lasso = LassoCV(cv=5).fit(X_train[['besi_trends', 'besi_sentiment']], y_train)
alpha_opt, beta_opt = lasso.coef_
```

---

## 5. Protocole de Validation

### 5.1 Architecture de test
Reproduire exactement l'architecture du BESI actuel :
- Même découpage Bloc A (2020-2021) / Bloc B (2022-2024)
- Même walk-forward 1-step-ahead
- Mêmes métriques : RMSE, Recall, F1, Delta AIC

### 5.2 Critères de succès

| Métrique | Seuil minimal (BESI v2 doit dépasser) | Source de comparaison |
|---|---|---|
| Delta AIC (BESI v2 vs SARIMA) | < −2.72 | BESI actuel (METRIQUES_OFFICIELLES.md) |
| Recall Bloc B | ≥ 1.000 | BESI actuel |
| F1 Bloc B | > 0.814 | BESI actuel |
| RMSE Bloc B | < 1.976 | BESI actuel |
| MC placebo p-value | < 0.098 | BESI actuel |

> **Note importante sur le Chow test** : On s'attend à ce que BESI v2 présente **également une rupture structurelle en 2022** (Chow p < 0.05), confirmant que le signal sentiment réagit au même choc. Ceci est une *question de recherche* (est-ce que le sentiment Darija détecte le même régime de crise ?), pas un critère de qualité. Une absence de rupture dans BESI v2 serait au contraire intéressante : elle indiquerait une plus grande stabilité du signal sentiment.

### 5.3 Test de complémentarité
```python
# H2 : BESI_v2 apporte une information orthogonale à BESI_trends
from statsmodels.stats.stattools import durbin_watson
corr = np.corrcoef(besi_trends, besi_sentiment)[0,1]
# Si |corr| < 0.7 : les deux signaux sont complémentaires
# Si |corr| > 0.7 : le sentiment double l'information des Trends
```

---

## 6. Limites Anticipées

| Limite | Impact | Mitigation |
|---|---|---|
| Accès aux données (Hespress API non publique) | Haut | Scraping + rate limiting |
| Annotations ground-truth inexistantes | Moyen | Zero-shot + validation manuelle d'un échantillon de 200 commentaires |
| Code-switching Darija/français/arabe | Moyen | Prétraitement multilingue (langdetect + normalisation) |
| Volume faible pré-2020 | Moyen | Interpolation ou pondération décroissante |
| Biais de sélection (utilisateurs Hespress ≠ ménages modestes) | Haut | Triangulation avec données GSM ou enquêtes HCP |

---

## 7. Calendrier Indicatif (6 mois)

| Mois | Tâche |
|---|---|
| 1-2 | Collecte données Hespress + YouTube (backfill 2020-2024) |
| 2-3 | Fine-tuning DarijaBERT sur 500 commentaires annotés manuellement |
| 3-4 | Agrégation mensuelle + construction BESI_sentiment |
| 4-5 | Estimation LassoCV des poids α, β + validation croisée |
| 5-6 | Évaluation complète BESI v2 vs BESI v1 + rapport |

---

## 8. Références

- Tetlock, P.C. (2007). Giving content to investor sentiment: The role of media in the stock market. *Journal of Finance*, 62(3), 1139-1168.
- Baker, M., & Wurgler, J. (2006). Investor sentiment and the cross-section of stock returns. *Journal of Finance*, 61(4), 1645-1680.
- Gaanoun, K., Naira, M., Allak, A., & Boumahdi, F. (2023). DarijaBERT: The Moroccan Dialect Version of BERT. *arXiv:2307.07078*.
- Abdul-Mageed, M., Elmadany, A., & Nagoudi, E.M.B. (2021). ARBERT & MARBERT: Deep bidirectional transformers for Arabic. *ACL 2021*, 7088-7105.
- Inoue, G., Jiang, B., Daher, N., & Habash, N. (2021). CAMeLBERT: A collection of pre-trained models for Arabic NLP. *EMNLP 2021*.
