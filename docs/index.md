# BESI V3

**Detection precoce des regimes d'inflation au Maroc via signaux comportementaux digitaux**

Bienvenue dans la documentation technique et academique de **BESI V3**.

Cette documentation est concue pour accompagner un depot GitHub et une
publication Read the Docs. Elle documente a la fois :

- le **cadrage scientifique** du projet ;
- la **pipeline de donnees** `bronze -> silver -> gold` ;
- les **choix methodologiques** de forecasting et d'early warning ;
- les **resultats finaux** et leur interpretation honnete ;
- l'**extension NLP** exploree a partir de la presse marocaine.

## En une phrase

> **Problematique finale** : dans quelle mesure des signaux comportementaux
> digitaux marocains peuvent-ils aider a detecter des regimes d'inflation avant
> ou autour de la publication officielle de l'IPC ?

## Ce que le projet montre

- Le **BESI behavioral**, construit a partir de Google Trends, apporte une
  information utile pour la **detection de regime**.
- Le projet est **plus fort en early warning** qu'en prevision point par point.
- La baseline **naive** reste la meilleure en RMSE global.
- L'ajout du **macro** n'ameliore pas la detection dans cette version.
- L'extension **NLP Hespress** est methodologiquement propre, mais n'ajoute pas
  d'information conditionnelle supplementaire.

## Ce que le projet ne pretend pas

- mesurer completement le stress economique des menages ;
- prouver une causalite forte entre Google Trends et inflation ;
- battre toutes les baselines sur tous les criteres ;
- fournir un systeme de production deja operationnel.

## Points d'entree recommandes

- Si vous voulez **installer et executer** rapidement le projet :
  commencez par [](installation.md) puis [](quickstart.md).
- Si vous voulez **comprendre la logique technique** :
  lisez [](architecture.md), [](data.md) et [](methodology.md).
- Si vous voulez **comprendre la valeur scientifique** du projet :
  allez directement a [](results.md) puis [](nlp_documentation.md).

```{toctree}
:maxdepth: 2
:caption: Demarrage

installation
quickstart
scripts
```

```{toctree}
:maxdepth: 2
:caption: Comprendre le projet

architecture
data
methodology
results
nlp_documentation
limitations
perspectives
faq
references
```

```{toctree}
:maxdepth: 2
:caption: Annexes

v3_problem_statement
data_dictionary_v3
repo_guide
appendices
```

