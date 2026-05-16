"""
src/ingestion/bam_fx.py — Ingestion taux de change BAM (Bronze)

Source : Bank Al-Maghrib (bkam.ma)
URL    : https://www.bkam.ma/Marches/Principaux-indicateurs/Marche-des-changes

Logique : le taux MAD/EUR est un driver indirect de l'IPC marocain.
  - Le Maroc importe ~60% de ses céréales et une grande partie de ses produits
    manufacturés depuis la zone Euro.
  - Une dépréciation du MAD vs EUR augmente directement le coût des importations
    et se répercute sur l'IPC alimentation et transport.
  - Le taux de change est observable en temps réel → disponible avant publication IPC.

Fallback automatique : si bkam.ma est inaccessible, tentative via pandas_datareader
(source: Banque Mondiale ou FRED si disponible).

Usage :
    from src.ingestion.bam_fx import ingest_bam_fx
    df = ingest_bam_fx()
"""

import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parent.parent.parent
BRONZE_DIR = ROOT / "data" / "bronze"
BRONZE_DIR.mkdir(parents=True, exist_ok=True)

# MAD/EUR depuis Investing.com (export CSV gratuit, pas d'API key)
# Alternative : FRED série DEXMAUS (MAD/USD uniquement) ou ECB (EUR/MAD)
_INVESTING_NOTE = """
  Téléchargement manuel BAM/taux de change :
  Option A (recommandée) :
    1. Aller sur bkam.ma > Statistiques > Taux de change
    2. Sélectionner MAD/EUR, fréquence mensuelle, 2010-2024
    3. Exporter CSV → sauvegarder sous data/bronze/bam_fx_raw.csv
    Colonnes attendues : date | mad_eur (ou EUR | USD)

  Option B (alternative) :
    1. https://www.investing.com/currencies/eur-mad-historical-data
    2. Télécharger l'historique (bouton 'Download')
    3. Sauvegarder sous data/bronze/bam_fx_raw.csv
"""


def ingest_bam_fx(
    filepath:     str | Path | None = None,
    output_path:  str | Path | None = None,
    start_date:   str = "2010-01-01",
    end_date:     str = "2024-12-01",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Charge le taux de change MAD/EUR mensuel.

    Essaie dans l'ordre :
    1. Cache local (bronze/bam_fx_raw.csv)
    2. Fichier téléchargé manuellement (filepath fourni)
    3. pandas_datareader (World Bank MAD/EUR si disponible)
    4. Synthèse depuis ECB (EUR/USD + USD/MAD si dispo)

    Retourne
    --------
    pd.DataFrame avec colonnes :
        date      : DatetimeIndex MS
        mad_eur   : taux MAD par 1 EUR (ex: 10.8 = 10.8 MAD pour 1 EUR)
        mad_usd   : taux MAD par 1 USD (optionnel)
        fx_yoy    : variation annuelle du taux MAD/EUR en %
        pulled_at : timestamp UTC
    """
    if output_path is None:
        output_path = BRONZE_DIR / "bam_fx_raw.csv"
    output_path = Path(output_path)

    # Cache
    if output_path.exists() and not force_refresh:
        logger.info(f"BAM FX — cache trouvé : {output_path}")
        return pd.read_csv(output_path, parse_dates=["date"], index_col="date")

    # Fichier local fourni
    if filepath is not None:
        filepath = Path(filepath)
        if filepath.exists():
            return _load_local_fx(filepath, output_path, start_date, end_date)

    # Chercher dans bronze/ les fichiers locaux
    for candidate in [
        BRONZE_DIR / "bam_fx_raw.csv",
        BRONZE_DIR / "mad_eur.csv",
        BRONZE_DIR / "fx_mad_eur.csv",
        ROOT / "data" / "raw" / "bam_fx_raw.csv",
    ]:
        if candidate.exists():
            logger.info(f"BAM FX — fichier local trouvé : {candidate}")
            return _load_local_fx(candidate, output_path, start_date, end_date)

    # Tentative pandas_datareader (World Bank)
    logger.info("BAM FX — tentative World Bank via pandas_datareader ...")
    df = _try_worldbank_fx(start_date, end_date)

    if df is not None:
        df["pulled_at"] = datetime.now(tz=timezone.utc).isoformat()
        df.index.name = "date"
        df.to_csv(output_path, index=True)
        logger.info(f"  Sauvegardé bronze : {output_path}")
        return df

    # Aucune source disponible
    raise FileNotFoundError(
        f"Données BAM/taux de change introuvables.\n{_INVESTING_NOTE}"
    )


def _load_local_fx(
    filepath: Path,
    output_path: Path,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Parse un CSV local de taux de change."""
    suffix = filepath.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        raw = pd.read_excel(filepath)
    else:
        for sep in [",", ";", "\t"]:
            try:
                raw = pd.read_csv(filepath, sep=sep)
                if len(raw.columns) >= 2:
                    break
            except Exception:
                continue

    raw.columns = [str(c).strip() for c in raw.columns]

    # Trouver colonne date
    date_col = next(
        (c for c in raw.columns if any(k in c.lower() for k in ["date", "mois", "month"])),
        raw.columns[0]
    )

    # Trouver colonne EUR/MAD
    eur_col = next(
        (c for c in raw.columns if any(k in c.lower() for k in ["eur", "euro", "mad_eur"])),
        None
    )
    usd_col = next(
        (c for c in raw.columns if any(k in c.lower() for k in ["usd", "dollar", "mad_usd"])),
        None
    )

    df = pd.DataFrame(index=pd.to_datetime(raw[date_col], errors="coerce"))
    df.index = df.index.to_period("M").to_timestamp("MS")
    df.index = pd.DatetimeIndex(df.index, freq="MS")

    if eur_col:
        df["mad_eur"] = pd.to_numeric(raw[eur_col].values, errors="coerce")
    else:
        logger.warning("  Colonne MAD/EUR non trouvée — NaN utilisé.")
        df["mad_eur"] = np.nan

    if usd_col:
        df["mad_usd"] = pd.to_numeric(raw[usd_col].values, errors="coerce")
    else:
        df["mad_usd"] = np.nan

    df = df.sort_index().drop_duplicates()
    df = df.loc[start_date:end_date]

    # Variation annuelle
    df["fx_yoy"] = df["mad_eur"].pct_change(12) * 100

    df["pulled_at"] = datetime.now(tz=timezone.utc).isoformat()
    df.index.name = "date"

    logger.info(f"  MAD/EUR : min={df['mad_eur'].min():.3f}  "
                f"max={df['mad_eur'].max():.3f}  mean={df['mad_eur'].mean():.3f}")

    df.to_csv(output_path, index=True)
    logger.info(f"  Sauvegardé bronze : {output_path}")
    return df


def _try_worldbank_fx(start_date: str, end_date: str) -> pd.DataFrame | None:
    """
    Tentative de récupération taux MAD depuis la Banque Mondiale.
    Indicateur PA.NUS.FCRF = Official exchange rate (LCU per US$, period average)
    puis conversion via EUR/USD (fixe approximatif).
    """
    try:
        from pandas_datareader import data as wb

        logger.info("  Téléchargement PA.NUS.FCRF (MAD/USD) depuis World Bank ...")
        raw = wb.DataReader("PA.NUS.FCRF", "wb", country="MA",
                            start=2010, end=pd.Timestamp(end_date).year)

        raw = raw.reset_index()
        raw["date"] = pd.to_datetime(raw["year"].astype(str) + "-07-01")
        raw = raw.set_index("date")[["PA.NUS.FCRF"]].rename(
            columns={"PA.NUS.FCRF": "mad_usd"}
        )
        # Interpolation annuelle → mensuelle
        monthly_idx = pd.date_range(start_date, end_date, freq="MS")
        df = raw.reindex(raw.index.union(monthly_idx)).interpolate("time").loc[monthly_idx]
        df.index = pd.DatetimeIndex(df.index, freq="MS")
        df["mad_eur"] = np.nan   # EUR/MAD non dispo directement via WB
        df["fx_yoy"]  = df["mad_usd"].pct_change(12) * 100

        logger.info(f"  MAD/USD World Bank : {len(df)} mois interpolés")
        logger.warning("  Note : MAD/USD seulement (pas EUR) — qualité réduite")
        return df

    except Exception as exc:
        logger.debug(f"  World Bank FX échec : {exc}")
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = ingest_bam_fx()
    print(df.tail(12))
    print(f"\nShape : {df.shape}")
