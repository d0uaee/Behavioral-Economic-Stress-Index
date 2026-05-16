"""
src/transforms/macro.py — Transform Bronze → Silver : FAO + BAM FX

Input  : data/bronze/fao_food_price_raw.csv
         data/bronze/bam_fx_raw.csv
Output : data/silver/macro_signals_monthly.csv

Colonnes produites :
    fao_food_index      : indice FAO global (base 2014-2016=100)
    fao_cereals_index   : sous-indice céréales FAO
    fao_oils_index      : sous-indice huiles FAO
    fao_food_yoy        : variation annuelle FAO global
    fao_oils_yoy        : variation annuelle huiles (pertinent pour Maroc)
    mad_eur             : taux de change MAD/EUR
    fx_yoy              : dépréciation annuelle MAD/EUR en %

Justification économique :
    - Maroc importe ~60% de ses céréales → fao_cereals_index driver direct IPC
    - Huile d'olive + tournesol dans panier HCP → fao_oils_index
    - Zone Euro = principal partenaire commercial → MAD/EUR pression importations
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


def _normalise_0_1(s: pd.Series) -> pd.Series:
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(0.0, index=s.index, name=s.name)
    return (s - mn) / (mx - mn)


def transform_macro(
    fao_path:    str | Path | None = None,
    fx_path:     str | Path | None = None,
    output_path: str | Path | None = None,
    start_date:  str = "2010-01-01",
    end_date:    str = "2024-12-01",
) -> pd.DataFrame:
    """
    Fusionne FAO FPI et taux de change BAM en silver mensuel.

    Retourne
    --------
    pd.DataFrame silver prêt pour le Gold dataset.
    """
    if fao_path    is None: fao_path    = BRONZE_DIR / "fao_food_price_raw.csv"
    if fx_path     is None: fx_path     = BRONZE_DIR / "bam_fx_raw.csv"
    if output_path is None: output_path = SILVER_DIR / "macro_signals_monthly.csv"

    fao_path    = Path(fao_path)
    fx_path     = Path(fx_path)
    output_path = Path(output_path)

    frames = []

    # ── FAO ──────────────────────────────────────────────────────────────────
    if fao_path.exists():
        fao = pd.read_csv(fao_path, parse_dates=["date"], index_col="date")
        try:
            fao.index = pd.DatetimeIndex(fao.index, freq="MS")
        except Exception:
            fao.index = pd.DatetimeIndex(fao.index)
            fao = fao.asfreq("MS")

        fao = fao.drop(columns=[c for c in fao.columns if c in ["pulled_at"]], errors="ignore")
        fao = fao.select_dtypes(include="number")
        fao = fao.loc[start_date:end_date]

        # Variations annuelles
        fao_silver = pd.DataFrame(index=fao.index)
        for col in ["fao_food_index", "fao_cereals_index", "fao_oils_index"]:
            if col in fao.columns:
                fao_silver[col]                  = fao[col]
                fao_silver[col.replace("_index", "_yoy")] = fao[col].pct_change(12) * 100

        frames.append(fao_silver)
        logger.info(f"Macro Silver — FAO : {len(fao_silver)} mois chargés")
    else:
        logger.warning(f"FAO bronze introuvable : {fao_path} — colonnes FAO absentes du silver")

    # ── BAM FX ───────────────────────────────────────────────────────────────
    if fx_path.exists():
        fx = pd.read_csv(fx_path, parse_dates=["date"], index_col="date")
        try:
            fx.index = pd.DatetimeIndex(fx.index, freq="MS")
        except Exception:
            fx.index = pd.DatetimeIndex(fx.index)
            fx = fx.asfreq("MS")

        fx = fx.drop(columns=[c for c in fx.columns if c in ["pulled_at"]], errors="ignore")
        fx = fx.select_dtypes(include="number")
        fx = fx.loc[start_date:end_date]

        fx_silver = pd.DataFrame(index=fx.index)
        if "mad_eur" in fx.columns:
            fx_silver["mad_eur"] = fx["mad_eur"]
            fx_silver["fx_yoy"]  = fx["mad_eur"].pct_change(12) * 100
        if "mad_usd" in fx.columns:
            fx_silver["mad_usd"] = fx["mad_usd"]

        frames.append(fx_silver)
        logger.info(f"Macro Silver — BAM FX : {len(fx_silver)} mois chargés")
    else:
        logger.warning(f"BAM FX bronze introuvable : {fx_path} — colonnes FX absentes du silver")

    if not frames:
        raise RuntimeError(
            "Aucune source macro disponible. "
            "Lancer d'abord src/ingestion/fao.py et src/ingestion/bam_fx.py"
        )

    # Fusion sur index mensuel
    full_idx = pd.date_range(start_date, end_date, freq="MS")
    result = frames[0].reindex(full_idx)
    for f in frames[1:]:
        result = result.join(f.reindex(full_idx), how="outer")

    result.index.name = "month"

    # Log qualité
    for col in result.columns:
        n_nan = result[col].isna().sum()
        if n_nan > 0:
            logger.warning(f"  {col:<25} : {n_nan} NaN ({100*n_nan/len(result):.0f}%)")

    result.to_csv(output_path, index=True)
    logger.info(f"  Sauvegardé silver : {output_path}  ({len(result)} mois, {len(result.columns)} colonnes)")

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = transform_macro()
    print(df.tail(12))
    print(f"\nShape : {df.shape}")
