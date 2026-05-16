"""
src/transforms/cpi.py — Transform Bronze → Silver : IPC

Input  : data/bronze/cpi_hcp_monthly_raw.csv
Output : data/silver/cpi_monthly.csv

Colonnes produites :
    month               : DatetimeIndex MS
    ipc_level           : indice brut HCP
    inflation_mom       : variation mensuelle %  (MoM)
    inflation_yoy       : variation annuelle %   (YoY)
    publication_date    : date estimée de publication HCP (~J+20 du mois)
    is_official         : True (source HCP) / False (interpolé)
    inflation_regime    : 1 si inflation_yoy >= 2.0%, sinon 0
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

INFLATION_REGIME_THRESHOLD = 2.0   # % YoY — seuil Bank Al-Maghrib stabilité des prix


def transform_cpi(
    input_path:  str | Path | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Transforme le bronze IPC HCP en silver avec toutes les features temporelles.

    Règle as-of-date :
        - ipc_level(t), inflation_mom(t), inflation_yoy(t) → connus après publication
        - publication_date(t) ~ 20 jours après la fin du mois t
        - Pour prédire IPC(t), les features IPC disponibles sont uniquement IPC(t-1) et avant

    Retourne
    --------
    pd.DataFrame silver avec colonnes documentées ci-dessus.
    """
    if input_path is None:
        input_path = BRONZE_DIR / "cpi_hcp_monthly_raw.csv"
    if output_path is None:
        output_path = SILVER_DIR / "cpi_monthly.csv"

    input_path  = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Bronze IPC introuvable : {input_path}\n"
            "Lancer d'abord : from src.ingestion.cpi_hcp import ingest_cpi_hcp"
        )

    df = pd.read_csv(input_path, parse_dates=["date"], index_col="date")
    try:
        df.index = pd.DatetimeIndex(df.index, freq="MS")
    except Exception:
        df.index = pd.DatetimeIndex(df.index)
        df = df.asfreq("MS")

    df = df.sort_index()

    result = pd.DataFrame(index=df.index)
    result.index.name = "month"

    # Série principale
    result["ipc_level"]   = df["ipc_level"]

    # Variations
    result["inflation_mom"] = df["ipc_level"].pct_change(1)  * 100   # %
    result["inflation_yoy"] = df["ipc_level"].pct_change(12) * 100   # %

    # Date de publication estimée (~20 jours après fin de mois)
    result["publication_date"] = result.index.to_series().apply(
        lambda d: (d + pd.offsets.MonthEnd(0) + pd.Timedelta(days=20)).date()
    )

    result["is_official"] = df.get("is_official", True)

    # Régime inflation (cible binaire)
    result["inflation_regime"] = (result["inflation_yoy"] >= INFLATION_REGIME_THRESHOLD).astype(int)

    # Rapport qualité
    n_regime_1 = result["inflation_regime"].sum()
    n_total    = result["inflation_regime"].notna().sum()
    logger.info(f"CPI Silver :")
    logger.info(f"  Mois          : {len(result)}")
    logger.info(f"  Plage         : {result.index[0].date()} → {result.index[-1].date()}")
    logger.info(f"  inflation_yoy : min={result['inflation_yoy'].min():.2f}%  "
                f"max={result['inflation_yoy'].max():.2f}%  "
                f"mean={result['inflation_yoy'].mean():.2f}%")
    logger.info(f"  Régime haute inflation (yoy >= {INFLATION_REGIME_THRESHOLD}%) : "
                f"{n_regime_1}/{n_total} mois ({100*n_regime_1/n_total:.1f}%)")

    result.to_csv(output_path, index=True)
    logger.info(f"  Sauvegardé silver : {output_path}")

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = transform_cpi()
    print(df.tail(24))
