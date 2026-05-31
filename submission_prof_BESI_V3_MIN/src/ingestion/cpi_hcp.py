"""
src/ingestion/cpi_hcp.py — Ingestion IPC HCP Maroc (Bronze)

Source officielle : Haut-Commissariat au Plan (hcp.ma)
URL : https://www.hcp.ma/Indice-des-prix-a-la-consommation_a633.html

Format attendu du fichier téléchargé manuellement :
  - Colonnes : date (YYYY-MM-DD ou YYYY-MM) + ipc (valeur numérique)
  - Fréquence : mensuelle
  - Base : 2017=100 (la plus récente du HCP)
  - Plage recommandée : 2010-01 à 2024-12

Usage :
    from src.ingestion.cpi_hcp import ingest_cpi_hcp
    df = ingest_cpi_hcp("data/bronze/ipc_hcp_raw.csv")
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parent.parent.parent
BRONZE_DIR = ROOT / "data" / "bronze"
BRONZE_DIR.mkdir(parents=True, exist_ok=True)

# Colonnes attendues dans le fichier HCP brut (flexibilité sur les noms)
_DATE_ALIASES = ["date", "mois", "month", "periode", "Date", "Mois"]
_IPC_ALIASES  = ["ipc", "indice", "cpi", "valeur", "IPC", "Indice", "CPI", "index"]


def ingest_cpi_hcp(
    filepath: str | Path | None = None,
    output_path: str | Path | None = None,
    start_date: str = "2010-01-01",
    end_date:   str = "2024-12-01",
) -> pd.DataFrame:
    """
    Charge le fichier IPC HCP brut, standardise les colonnes,
    valide la qualité et sauvegarde en bronze/.

    Paramètres
    ----------
    filepath    : chemin vers le CSV HCP téléchargé manuellement.
                  Par défaut : data/bronze/ipc_hcp_raw.csv
    output_path : où sauvegarder le bronze nettoyé.
                  Par défaut : data/bronze/cpi_hcp_monthly_raw.csv
    start_date  : date de début de la plage d'analyse
    end_date    : date de fin de la plage d'analyse

    Retourne
    --------
    pd.DataFrame avec colonnes :
        date         : DatetimeIndex MS
        ipc_level    : indice brut HCP
        source_file  : nom du fichier source
        ingested_at  : timestamp d'ingestion UTC
    """
    if filepath is None:
        filepath = BRONZE_DIR / "ipc_hcp_raw.csv"
    filepath = Path(filepath)

    if output_path is None:
        output_path = BRONZE_DIR / "cpi_hcp_monthly_raw.csv"
    output_path = Path(output_path)

    if not filepath.exists():
        raise FileNotFoundError(
            f"Fichier IPC HCP introuvable : {filepath}\n\n"
            "  Télécharger depuis : https://www.hcp.ma/Indice-des-prix-a-la-consommation_a633.html\n"
            "  Choisir la série mensuelle IPC base 2017=100\n"
            "  Sauvegarder sous : data/bronze/ipc_hcp_raw.csv\n"
            "  Format colonnes : date | ipc  (valeurs numériques)"
        )

    logger.info(f"IPC HCP — chargement depuis {filepath} ...")

    # Lecture flexible (CSV, Excel, séparateur variable)
    suffix = filepath.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        raw = pd.read_excel(filepath)
    else:
        # Essayer plusieurs séparateurs
        for sep in [",", ";", "\t"]:
            try:
                raw = pd.read_csv(filepath, sep=sep)
                if len(raw.columns) >= 2:
                    break
            except Exception:
                continue

    raw.columns = [str(c).strip() for c in raw.columns]

    # Trouver la colonne date
    date_col = next(
        (c for c in raw.columns if c.lower() in [a.lower() for a in _DATE_ALIASES]),
        None
    )
    if date_col is None:
        raise ValueError(
            f"Colonne date introuvable. Colonnes trouvées : {list(raw.columns)}\n"
            f"Colonnes attendues : {_DATE_ALIASES}"
        )

    # Trouver la colonne IPC
    ipc_col = next(
        (c for c in raw.columns if c.lower() in [a.lower() for a in _IPC_ALIASES]),
        None
    )
    if ipc_col is None:
        # Fallback : première colonne numérique qui n'est pas la date
        num_cols = raw.drop(columns=[date_col]).select_dtypes(include="number").columns
        if len(num_cols) == 0:
            raise ValueError(
                f"Aucune colonne IPC numérique trouvée. Colonnes : {list(raw.columns)}"
            )
        ipc_col = num_cols[0]
        logger.warning(f"Colonne IPC non reconnue — utilisation de '{ipc_col}'")

    # Construction du DataFrame bronze
    df = pd.DataFrame({
        "date":      pd.to_datetime(raw[date_col], dayfirst=True, errors="coerce"),
        "ipc_level": pd.to_numeric(raw[ipc_col], errors="coerce"),
    }).dropna(subset=["date", "ipc_level"])

    # Normaliser à la fréquence mensuelle (début de mois)
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp("MS")
    df = df.sort_values("date").drop_duplicates("date").set_index("date")
    df.index = pd.DatetimeIndex(df.index, freq="MS")

    # Filtrer sur la plage d'analyse
    df = df.loc[start_date:end_date]

    # Validation qualité
    n_total    = len(df)
    n_expected = len(pd.date_range(start_date, end_date, freq="MS"))
    n_missing  = n_expected - n_total
    n_nan      = df["ipc_level"].isna().sum()

    logger.info(f"  Mois chargés   : {n_total}")
    logger.info(f"  Mois attendus  : {n_expected}")
    logger.info(f"  Mois manquants : {n_missing}")
    logger.info(f"  NaN ipc_level  : {n_nan}")
    logger.info(f"  Plage          : {df.index[0].date()} → {df.index[-1].date()}")
    logger.info(f"  IPC min={df['ipc_level'].min():.2f}  "
                f"max={df['ipc_level'].max():.2f}  "
                f"mean={df['ipc_level'].mean():.2f}")

    if n_missing > 6:
        logger.warning(
            f"  ATTENTION : {n_missing} mois manquants — vérifier le fichier HCP."
        )
    if n_nan > 0:
        logger.warning(f"  ATTENTION : {n_nan} valeurs NaN dans ipc_level.")

    # Métadonnées de provenance
    df["source_file"]  = filepath.name
    df["ingested_at"]  = datetime.now(tz=timezone.utc).isoformat()
    df["is_official"]  = True   # source HCP = donnée officielle

    df.index.name = "date"
    df.to_csv(output_path, index=True)
    logger.info(f"  Sauvegardé bronze : {output_path}")

    return df[["ipc_level", "source_file", "ingested_at", "is_official"]]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = ingest_cpi_hcp()
    print(df.head(10))
    print(f"\nShape : {df.shape}")
