# Installation

Cette page decrit l'installation minimale necessaire pour lire, executer et
reproduire la version finale **BESI V3**.

## Prerequis

- Python 3.13 recommande
- Un environnement virtuel dedie
- Un acces local au depot GitHub

## Recuperer le projet

```bash
git clone https://github.com/d0uaee/Behavioral-Economic-Stress-Index.git
cd Behavioral-Economic-Stress-Index
```

## Creer un environnement virtuel

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

## Installer les dependances principales

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Installer les dependances de documentation

Si vous souhaitez construire la documentation localement :

```bash
python -m pip install -r docs/requirements.txt
```

## Structure attendue apres installation

- `run_v3.py` : point d'entree principal
- `src/` : logique de pipeline et d'evaluation
- `data/` : donnees bronze, silver et gold
- `outputs/` : rapports et figures finales
- `docs/` : documentation Read the Docs

## Verifications rapides

Pour verifier que l'environnement est fonctionnel :

```bash
python --version
python run_v3.py --help
```

## Questions auxquelles cette page repond

- Comment installer le projet proprement ?
- Quelles dependances sont necessaires ?
- Comment separer les dependances projet et documentation ?

