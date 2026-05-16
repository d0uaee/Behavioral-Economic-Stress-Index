"""
src/transforms/trends.py — Transform Bronze → Silver : Google Trends v3

Input  : data/bronze/google_trends_raw_v3.csv
Output : data/silver/google_trends_monthly.csv

Sous-indices thématiques (normalisés 0-1 indépendamment) :
    trends_prix_alim    : prix alimentaires (huile, légumes, hausse prix)
    trends_inflation    : ancrage inflation ("inflation maroc" + arabe)
    trends_carburant    : carburant (prix carburant maroc + hausse arabe)
    trends_subvention   : politique subventions
    trends_composite    : moyenne simple des 4 sous-indices (baseline)
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parent.parent.parent
BRONZE_DIR = ROOT / "data" / "bronze"
SILVER_DIR = ROOT / "data" / "silver"
SILVER_DIR.mkdir(parents=True, exist_ok=True)

# Mapping keyword → sous-indice thématique
# Flexible : si un keyword manque dans le CSV, il est ignoré silencieusement
THEME_MAP = {
    "trends_prix_alim": [
        "prix huile",
        "hausse prix",
        "prix legumes maroc",
        "أسعار المواد الغذائية",
        "غلاء المعيشة",
    ],
    "trends_inflation": [
        "inflation maroc",
        "التضخم في المغرب",
    ],
    "trends_carburant": [
        "prix carburant maroc",
        "ارتفاع الأسعار",
    ],
    "trends_subvention": [
        "subvention maroc",
    ],
}


def _normalise_0_1(s: pd.Series) -> pd.Series:
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(0.0, index=s.index, name=s.name)
    return (s - mn) / (mx - mn)


def transform_trends(
    input_path:  str | Path | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Construit les sous-indices thématiques Google Trends normalisés 0-1.

    Chaque sous-indice = moyenne des keywords du thème disponibles dans le CSV.
    Les keywords absents sont ignorés (avec warning).

    Retourne
    --------
    pd.DataFrame silver avec colonnes :
        month                : DatetimeIndex MS
        trends_prix_alim     : 0-1
        trends_inflation     : 0-1
        trends_carburant     : 0-1
        trends_subvention    : 0-1 (souvent faible avant 2020)
        trends_composite     : moyenne des 4 sous-indices
        n_keywords_used      : nombre de keywords effectivement disponibles
    """
    if input_path is None:
        # Essayer v3 d'abord, puis legacy
        for candidate in [
            BRONZE_DIR / "google_trends_raw_v3.csv",
            ROOT / "data" / "processed" / "trends_monthly.csv",
        ]:
            if candidate.exists():
                input_path = candidate
                break

    if output_path is None:
        output_path = SILVER_DIR / "google_trends_monthly.csv"

    input_path  = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Bronze Trends introuvable : {input_path}\n"
            "Lancer d'abord : from src.ingestion.google_trends_v3 import ingest_google_trends_v3"
        )

    df = pd.read_csv(input_path, parse_dates=True, index_col=0)
    try:
        df.index = pd.DatetimeIndex(df.index, freq="MS")
    except Exception:
        df.index = pd.DatetimeIndex(df.index)
        df = df.asfreq("MS")

    # Supprimer colonnes méta
    df = df.drop(columns=[c for c in df.columns if c in ["pulled_at", "isPartial"]], errors="ignore")
    df = df.select_dtypes(include="number")

    result = pd.DataFrame(index=df.index)
    result.index.name = "month"
    total_keywords_used = 0

    for theme, keywords in THEME_MAP.items():
        available = [k for k in keywords if k in df.columns]
        missing   = [k for k in keywords if k not in df.columns]

        if missing:
            logger.debug(f"  {theme} : keywords absents → {missing}")

        if not available:
            logger.warning(f"  {theme} : aucun keyword disponible → NaN")
            result[theme] = np.nan
            continue

        raw_mean = df[available].mean(axis=1)
        result[theme] = _normalise_0_1(raw_mean)
        total_keywords_used += len(available)
        logger.info(f"  {theme:<25} : {len(available)} keywords  "
                    f"mean={result[theme].mean():.3f}")

    # Composite : moyenne des sous-indices disponibles
    sub_cols = [c for c in result.columns if c.startswith("trends_") and c != "trends_composite"]
    result["trends_composite"] = result[sub_cols].mean(axis=1, skipna=True)
    result["n_keywords_used"]  = total_keywords_used

    logger.info(f"\nTrends Silver :")
    logger.info(f"  Mois total         : {len(result)}")
    logger.info(f"  Keywords utilisés  : {total_keywords_used}")
    logger.info(f"  trends_composite   : mean={result['trends_composite'].mean():.3f}  "
                f"std={result['trends_composite'].std():.3f}")

    result.to_csv(output_path, index=True)
    logger.info(f"  Sauvegardé silver : {output_path}")

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = transform_trends()
    print(df.tail(12))
