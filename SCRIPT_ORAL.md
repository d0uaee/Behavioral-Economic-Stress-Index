# SCRIPT ORAL — SOUTENANCE
## BESI Maroc — Détection Précoce du Stress Économique des Ménages
### Douae Ahadji & Adama Basse — ENSAM Meknès — Mai 2026

---

> **DURÉE CIBLE : 20-25 minutes**  
> Format conseillé : Douae présente les parties 1-3 (données, SARIMA, analyse)  
> Adama présente les parties 4-5 (Deep Learning, conclusion)

---

---

## SLIDE 1 — TITRE *(~30 secondes)*

**[DOUAE]**

Bonjour à tous.

Je m'appelle Douae Ahadji, et je présente ce projet avec Adama Basse.

Notre travail s'intitule *"Détection Précoce du Stress Économique des Ménages au Maroc"*, et repose sur un indice que nous avons construit : le **BESI**, pour *Behavioral Economic Stress Index*.

En quelques mots : nous avons cherché à détecter le stress économique des Marocains **avant** qu'il apparaisse dans les statistiques officielles, en exploitant les signaux comportementaux laissés sur Internet.

---

---

## SLIDE 2 — ÉTAT D'AVANCEMENT *(~1 minute)*

**[DOUAE]**

Avant d'entrer dans le vif du sujet, voici l'état d'avancement complet du projet.

Nous avons réalisé **8 modules** sur 8 semaines, tous terminés :
- Le pipeline de données,
- La modélisation SARIMA et SARIMAX,
- L'analyse statistique complète,
- Le Deep Learning avec LSTM et Prophet,
- Un pipeline NLP multilingue sur la presse marocaine,
- Quatre notebooks Jupyter avec outputs,
- Et tout le rapport final.

Je vais maintenant vous présenter les résultats clés.

---

---

## SLIDE 3 — CONTEXTE & HYPOTHÈSE H1 *(~2 minutes)*

**[DOUAE]**

### Pourquoi ce sujet ?

L'IPC, l'Indice des Prix à la Consommation, est publié chaque mois par le HCP avec un délai d'environ un mois. Ce délai est acceptable en temps normal, mais il pose un **problème critique en période de crise** comme celle de 2022 : l'inflation a atteint **6.57% en mars 2022** — un record depuis 30 ans au Maroc — causé principalement par la guerre en Ukraine.

Notre question est simple : *est-ce qu'on pouvait le voir venir ?*

Et notre hypothèse de départ, H1, disait :

> *"Les signaux comportementaux digitaux — Google Trends, Reddit, YouTube — permettent de détecter le stress économique 1 à 2 mois avant l'IPC."*

Spoiler : la réalité est plus nuancée, et c'est ce que je vais vous montrer.

Notre contribution originale est **BESI** : le premier indice composite marocain basé sur des signaux digitaux en Darija, en Arabe et en Français.

---

---

## SLIDE 4 — DONNÉES & BESI *(~2 minutes)*

**[DOUAE]**

### Les données

Notre variable cible est l'IPC mensuel du HCP, de **2010 à 2024**, soit **180 observations** à fréquence mensuelle.

Pour les signaux comportementaux, nous avons collecté quatre sources :
- **Google Trends** via pytrends, avec 7 mots-clés ciblés : "inflation maroc", "prix huile", "hausse prix", "crédit consommation"... sur la zone géographique Maroc.
- **Reddit r/Morocco** via l'API praw, avec les posts parlant de prix et de cherté.
- **YouTube** via la Data API v3, sur 4 grandes chaînes d'information marocaines.
- Et une **presse marocaine** scrappée : Hespress, Le360, Medias24 et L'Économiste.

### Les trois versions de BESI

Nous avons construit trois versions de l'indice :

La première, **BESI composite**, utilise les quatre sources avec des poids 40-30-20-10.

La deuxième, **BESI_trends**, n'utilise que Google Trends et les variations de l'IPC. C'est la version que nous avons utilisée en modélisation, parce que c'est la seule qui repose entièrement sur des **données réelles**, sans simulation.

La troisième, **BESI_enrichi**, intègre en plus notre pipeline NLP sur la presse marocaine.

Tous les signaux sont normalisés entre 0 et 1 avant d'être pondérés.

---

---

## SLIDE 5 — STATIONNARITÉ & SARIMA *(~2 minutes 30)*

**[DOUAE]**

### Analyse de stationnarité

Avant toute modélisation, nous avons vérifié la stationnarité avec les tests ADF et KPSS.

L'IPC en niveau n'est **pas stationnaire** : il a une tendance haussière claire sur 14 ans. Après une différenciation simple — d=1 — et une différenciation saisonnière — D=1 sur 12 mois — la série devient pleinement stationnaire.

C'est ce qui justifie l'utilisation d'un modèle SARIMA avec d=1, D=1, et une saisonnalité s=12.

### Identification du modèle

Pour identifier les ordres, nous avons lu les graphiques ACF et PACF :
- L'ACF montre une décroissance lente avec des pics aux lags 12 et 24 → saisonnalité s=12, D=1.
- Le PACF montre une coupure nette au lag 2 → ordre AR p=2.
- L'ACF résiduelle n'a qu'un pic significatif au lag 1 → MA q=1.

Nous avons ensuite testé une grille de 6 modèles et comparé leurs AIC.

Le modèle retenu est :

**SARIMA(2,1,1) × (0,1,1)[12]**

avec un AIC de **-502.56** et un BIC de **-492.34**.

Le diagnostic des résidus confirme qu'il s'agit d'un **bruit blanc** — le test de Ljung-Box est non significatif.

---

---

## SLIDE 6 — COMPARAISON DES MODÈLES *(~3 minutes)*

**[DOUAE]**

### Protocole de validation

Nous avons utilisé une **validation walk-forward** à horizon h=1 mois.

Le principe : on entraîne le modèle jusqu'à un certain point, on fait une prédiction, on avance d'un mois, on réentraîne, etc. C'est la méthode la plus réaliste pour une application opérationnelle.

**Période d'entraînement :** janvier 2015 à décembre 2021.  
**Période de test :** janvier 2022 à décembre 2024, soit **36 mois**.

### Résultats

Regardons ce tableau.

Le **modèle Naïf** — qui prédit simplement la valeur précédente — obtient un RMSE de 0.00409.

**SARIMA** est clairement meilleur avec un RMSE de **0.00272** — soit 33% de gain sur le Naïf.

**SARIMAX avec BESI_trends** a un RMSE de 0.00304, légèrement moins bon que SARIMA. Mais regardez l'AIC : **-545.88**, c'est le meilleur de tous les modèles. Cela veut dire que BESI_trends apporte une **information statistique réelle**, même si elle ne se traduit pas immédiatement en gain de RMSE.

**LSTM** : RMSE de 0.019 — **7 fois pire** que SARIMA.  
**Prophet** : RMSE de 0.061 — **22 fois pire** que SARIMA.

**Conclusion : SARIMA reste le roi sur cette série.**

---

---

## SLIDE 7 — PERFORMANCES PAR SOUS-PÉRIODE *(~2 minutes)*

**[DOUAE]**

### Pourquoi décomposer la période de test ?

La période 2022-2024 n'est pas homogène. Il y a deux phases très différentes :
- Le **choc 2022** : 12 mois d'inflation record, conditions inédites.
- La **période post-choc 2023-2024** : retour progressif à la normale.

### Ce que révèle la décomposition

Pendant le **choc 2022**, SARIMA obtient un RMSE de **0.00197** — c'est remarquable. SARIMAX_BT fait 0.00362. Pourquoi ? Parce que la variable exogène BESI, qui a capturé la montée du stress, **perturbe** le modèle sur un choc aussi brutal et imprévisible.

**Après 2022**, c'est différent. SARIMAX_BT descend à **0.00271**, légèrement meilleur que SARIMA (0.00303). Une fois le régime stabilisé, BESI commence à apporter une valeur réelle.

L'interprétation est claire : BESI est un **signal de tendance lourde**, pas de court terme. Il est utile pour anticiper l'évolution structurelle, pas les chocs soudains.

---

---

## SLIDE 8 — RUPTURE STRUCTURELLE 2022 *(~2 minutes)*

**[DOUAE]**

### Test de Chow

Nous avons testé la présence d'une rupture structurelle au 1er janvier 2022.

Le test de Chow **confirme la rupture** avec p < 0.05. Les coefficients du modèle ont changé significativement entre avant et après 2022 :
- La constante augmente de 10%,
- La tendance augmente de 60%,
- Et le coefficient de BESI augmente de **67%** — BESI joue un rôle bien plus fort en période de crise.

### Matrice de Markov

Nous avons aussi modélisé la dynamique des états de stress de BESI avec une chaîne de Markov.

Ce que dit la matrice : quand on est en état "High Stress", la probabilité de **rester** en High Stress au mois suivant est de **90.9%**.

Cela confirme que le choc de 2022 n'est pas un épisode temporaire — c'est un **changement de régime durable**. Une fois entré en crise économique, il est très difficile d'en sortir rapidement.

---

---

## SLIDE 9 — GRANGER & EARLY WARNING *(~2 minutes)*

**[DOUAE]**

### Causalité de Granger

Le test de Granger répond à la question : est-ce que BESI *précède* l'IPC dans le temps ?

Les résultats sont clairs : **7 variables sur 7** sont significatives au seuil de 5%. BESI Granger-cause l'IPC. L'information comportementale que nous capturons est **prédictive**, pas spurieuse.

### Early Warning

C'est le résultat le plus parlant du projet.

En analysant la corrélation croisée entre BESI et l'IPC, nous trouvons que BESI est le plus corrélé avec l'IPC **décalé de 12 mois**.

Plus concrètement : notre système a détecté un signal d'alerte en **mai 2021**, soit **12 mois avant** le pic d'inflation de mars 2022.

Les métriques du système d'alerte :
- **Rappel = 100%** : aucune crise n'a été ratée, zéro faux négatif.
- **F1-Score ≈ 0.82** : très bonne performance globale.

H1 disait 1-2 mois. La réalité dit 12 mois. Ce n'est pas un échec — c'est une **nuance importante** : BESI est un outil de **politique macroéconomique**, pas d'intervention tactique.

---

---

## SLIDE 10 — LSTM *(~1 minute 30)* — ADAMA

**[ADAMA]**

J'ai travaillé sur la partie Deep Learning.

Nous avons testé des réseaux de neurones LSTM avec **8 configurations différentes** : 4 tailles de fenêtre — 6, 12, 18 et 24 mois — avec et sans BESI comme variable exogène.

L'architecture de base est : une couche LSTM de 64 neurones, un Dropout à 10%, puis une seconde couche LSTM de 32 neurones.

Les résultats sont clairs :
- La meilleure configuration LSTM est la fenêtre de **24 mois sans exogène**, avec un RMSE de 0.04627.
- C'est quand même **17 fois pire** que SARIMA.
- Et fait intéressant : ajouter BESI **dégrade** systématiquement les performances LSTM.

Pourquoi ? Parce que le LSTM travaille sur un horizon de quelques mois. Or BESI est un signal qui prédit à **12 mois d'avance** — les deux horizons sont incompatibles.

Ce résultat confirme que sur des séries mensuelless courtes avec une rupture structurelle, **SARIMA reste supérieur au Deep Learning**.

---

---

## SLIDE 11 — PROPHET *(~1 minute)* — ADAMA

**[ADAMA]**

Nous avons aussi testé Prophet, le modèle Bayésien développé par Meta.

La configuration : saisonnalité annuelle multiplicative, train sur 2010-2021, test sur 2022-2024.

Les résultats : RMSE de **0.06082**, soit **22 fois pire** que SARIMA.

Ce résultat n'est pas surprenant. Prophet est conçu pour des séries longues, journalières ou hebdomadaires, avec une saisonnalité forte et régulière. L'IPC mensuel marocain, avec sa rupture structurelle de 2022, est le contre-exemple parfait.

C'est un résultat scientifiquement valide : il confirme que la rupture structurelle est la principale difficulté de cette série, et que les méthodes classiques bien spécifiées — comme SARIMA — restent supérieures.

---

---

## SLIDE 12 — NLP *(~1 minute 30)* — DOUAE ou ADAMA

**[DOUAE ou ADAMA selon répartition]**

En parallèle de la modélisation, nous avons construit un pipeline NLP pour analyser les commentaires de la presse marocaine.

Le pipeline comporte 6 modules :
1. Scraping des 4 grands sites d'information marocains.
2. Collecte des commentaires YouTube sur 4 chaînes.
3. Scoring NLP avec un dictionnaire de **80+ mots-clés** en Darija, Arabe et Français.
4. Agrégation mensuelle pondérée par l'engagement.
5. Calcul de BESI_enrichi et mise à jour du dataset.
6. Visualisation dual-axis NLP vs IPC.

La formule de scoring est :

> score = (0.6 × keyword_score + 0.4 × intensité) × engagement_weight

L'intensité capte les signaux forts : majuscules, emojis de colère, répétitions.

BESI_enrichi intègre 35% de Google Trends et 25% de NLP presse marocaine comme principal signal comportemental.

---

---

## SLIDE 13 — VERDICT H1 & CONCLUSION *(~2 minutes)*

**[DOUAE]**

### Verdict sur H1

Récapitulons.

H1 est **partiellement rejetée**, mais avec des nuances importantes.

Ce qui est **rejeté** :
- Le lead time de 1-2 mois n'est pas confirmé. Le lead time réel est de **12 mois**.
- SARIMAX+BESI n'améliore pas le RMSE par rapport à SARIMA.

Ce qui est **confirmé** :
- BESI précède l'IPC de 12 mois — corrélation croisée significative.
- 7 variables sur 7 avec causalité de Granger — l'information comportementale est réelle.
- BESI_trends améliore l'AIC de -502 à -546 — information statistique confirmée.
- **Rappel = 100%** sur le système d'alerte précoce — aucune crise ratée.
- Persistance du stress : 90.9% de rester en "High Stress" une fois en crise.

### Implication

L'horizon de BESI n'est pas tactique — ce n'est pas un outil pour réagir en urgence dans 2 mois. C'est un **outil de politique macroéconomique** pour anticiper les tensions à 12 mois et ajuster les politiques de crédit, de subventions alimentaires et d'aide sociale.

Les décideurs publics marocains — HCP, Ministère de l'Économie — pourraient utiliser BESI comme **indicateur avancé de tendance lourde**.

---

---

## SLIDE 14 — PHRASE FINALE *(~30 secondes)*

**[DOUAE]**

Pour conclure en une phrase :

> *"Je reste dans le cadre SARIMA/SARIMAX du cours, mais j'introduis une dimension comportementale multi-sources pour tester la stabilité structurelle après 2022 et quantifier la capacité d'alerte précoce des signaux digitaux.*
>
> *SARIMA reste le meilleur modèle en validation walk-forward (RMSE=0.00272), mais BESI_trends améliore l'AIC de -502 à -546. BESI détecte le stress économique 12 mois d'avance — offrant un signal macroéconomique robuste pour les politiques contracycliques au Maroc."*

Merci. Nous sommes disponibles pour vos questions.

---

---

# QUESTIONS FRÉQUENTES — RÉPONSES PRÉPARÉES

---

### Q1 : Pourquoi le RMSE de SARIMAX est-il pire que SARIMA si BESI est utile ?

**Réponse :**
C'est la distinction entre **performance prédictive** et **information statistique**. L'AIC de SARIMAX_BT est de -545 contre -502 pour SARIMA — c'est le meilleur AIC. L'AIC mesure la qualité du modèle en tenant compte de la complexité. BESI apporte de l'information, mais en période de choc brutal comme 2022, une variable exogène peut introduire du bruit dans la prédiction à court terme. Post-2022, SARIMAX_BT est légèrement meilleur (RMSE 0.00271 vs 0.00303).

---

### Q2 : Pourquoi H1 parle de 1-2 mois et vous trouvez 12 mois ?

**Réponse :**
Notre hypothèse de départ était basée sur l'idée que les gens cherchent des informations sur les prix quand ils en ressentent le besoin — réaction immédiate. En réalité, les comportements digitaux reflètent une **anxiété diffuse** qui précède de loin les tensions officielles. Google Trends pour "hausse prix" ou "pouvoir achat" commence à augmenter bien avant que l'IPC ne bouge. C'est d'ailleurs une **découverte intéressante** : BESI est un indicateur de tendance lourde, plus utile pour la planification de moyen terme que pour l'alerte à très court terme.

---

### Q3 : Les poids de BESI sont arbitraires, comment les justifiez-vous ?

**Réponse :**
C'est une limite que nous reconnaissons explicitement dans les limites de l'étude. Les poids 40-30-20-10 ont été fixés sur la base de la qualité des données disponibles : Google Trends est la donnée la plus complète et fiable (données réelles, 14 ans), d'où le poids le plus élevé. Reddit et YouTube sont partiellement simulés, d'où des poids plus faibles. Une optimisation par validation croisée serait une perspective d'amélioration.

---

### Q4 : Pourquoi LSTM est-il si mauvais comparé à SARIMA ?

**Réponse :**
Plusieurs raisons : premièrement, **180 observations** c'est très peu pour un réseau de neurones — le DL brille sur des milliers de points. Deuxièmement, la **rupture structurelle de 2022** crée une distribution non-stationnaire que le LSTM ne peut pas capturer sans données post-rupture en entraînement. Troisièmement, SARIMA est **spécifiquement conçu** pour les séries temporelles mensuelles avec saisonnalité — c'est son domaine d'excellence. Ce résultat est connu dans la littérature : sur des séries courtes macroéconomiques, les modèles classiques restent supérieurs.

---

### Q5 : Pourquoi Prophet échoue ?

**Réponse :**
Prophet est optimisé pour des séries longues, journalières ou hebdomadaires, avec une saisonnalité stable et prévisible — pensez aux données de ventes e-commerce ou de trafic web. L'IPC mensuel marocain a une saisonnalité faible et une rupture structurelle majeure en 2022 — exactement le type de données pour lequel Prophet n'est pas adapté. C'est un résultat scientifiquement valide qui renforce notre conclusion : **SARIMA est le bon outil ici**.

---

### Q6 : La causalité de Granger prouve-t-elle la vraie causalité ?

**Réponse :**
Non, et c'est une limite que nous mentionnons. La causalité de Granger teste l'**ordre temporel** — est-ce que X aide à prédire Y ? Ce n'est pas de la causalité au sens philosophique. Il pourrait exister une troisième variable cachée qui cause les deux. Cependant, dans notre contexte, l'interprétation économique est cohérente : quand les gens cherchent "inflation maroc" sur Google, c'est souvent en réaction à des tensions qu'ils ressentent avant que l'IPC officiel ne les mesure. C'est plausible mécaniquement.

---

### Q7 : Qu'est-ce que vous feriez si vous aviez plus de temps ?

**Réponse :**
Quatre axes : premièrement, **optimiser les poids de BESI** par validation croisée. Deuxièmement, tester un **modèle VAR** pour capturer les effets rétroactifs IPC→BESI. Troisièmement, **stratifier par région** — les données Google Trends au niveau wilaya permettraient une analyse géographique. Quatrièmement, étendre le pipeline NLP à **Twitter/X en temps réel** pour construire un vrai système d'alerte précoce opérationnel.

---

*Douae Ahadji & Adama Basse — ENSAM Meknès — Séries Temporelles — Mai 2026*
