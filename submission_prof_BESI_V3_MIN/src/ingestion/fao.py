"""
src/ingestion/fao.py — Ingestion FAO Food Price Index (Bronze)

Source : Food and Agriculture Organization of the United Nations
URL CSV : https://www.fao.org/giews/food-prices/food-inflation/src/data/FoodInflation.xlsx
URL alt : https://www.fao.org/worldfoodsituation/foodpricesindex/en/

Le FAO FPI est publié chaque début de mois pour le mois précédent.
=> Disponible AVANT la publication de l'IPC HCP du même mois.
=> Signal avancé économiquement justifié (40% de l'IPC marocain = alimentation).

Composantes disponibles :
    Food Price Index    : indice global alimentaire
    Cereals Index       : céréales (blé → pain, farine — poids élevé Maroc)
    Oils Index          : huiles végétales (huile d'olive, soja, tournesol)
    Dairy Index         : produits laitiers
    Meat Index          : viandes
    Sugar Index         : sucre

Usage :
    from src.ingestion.fao import ingest_fao_fpi
    df = ingest_fao_fpi()
"""

import io
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

# URL officielle FAO Food Price Index (données historiques mensuelles)
FAO_CSV_URLS = [
    # URL principale Excel
    "https://www.fao.org/fileadmin/templates/worldfood/Reports_and_docs/Food_price_indices_data_jul25.xlsx",
    # Fallback CSV
    "https://www.fao.org/giews/food-prices/food-inflation/src/data/FoodInflation.csv",
]

# Mapping des noms de colonnes FAO → noms standardisés silver
_COL_MAP = {
    "Food Price Index": "fao_food_index",
    "Cereals":         "fao_cereals_index",
    "Oils":            "fao_oils_index",
    "Dairy":           "fao_dairy_index",
    "Meat":            "fao_meat_index",
    "Sugar":           "fao_sugar_index",
}


def ingest_fao_fpi(
    output_path: str | Path | None = None,
    start_date:  str = "2010-01-01",
    end_date:    str = "2024-12-01",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Télécharge et sauvegarde le FAO Food Price Index mensuel en bronze.

    Logique cache-first : si bronze/fao_food_price_raw.csv existe, lecture locale.
    Supprimer le fichier pour forcer un re-téléchargement.

    Retourne
    --------
    pd.DataFrame avec colonnes :
        date               : DatetimeIndex MS
        fao_food_index     : indice global FAO (base 2014-2016=100)
        fao_cereals_index  : sous-indice céréales
        fao_oils_index     : sous-indice huiles
        fao_dairy_index    : sous-indice laitiers
        fao_meat_index     : sous-indice viandes
        fao_sugar_index    : sous-indice sucre
        pulled_at          : timestamp de téléchargement UTC
    """
    if output_path is None:
        output_path = BRONZE_DIR / "fao_food_price_raw.csv"
    output_path = Path(output_path)

    if output_path.exists() and not force_refresh:
        logger.info(f"FAO FPI — cache trouvé : {output_path}")
        df = pd.read_csv(output_path, parse_dates=["date"], index_col="date")
        return df

    logger.info("FAO FPI — téléchargement depuis fao.org ...")

    df = None
    last_error = None

    for url in FAO_CSV_URLS:
        try:
            logger.info(f"  Essai URL : {url}")

            if url.endswith(".xlsx"):
                df = _parse_fao_excel(url)
            else:
                df = _parse_fao_csv(url)

            if df is not None and len(df) > 0:
                logger.info(f"  Succès : {len(df)} lignes téléchargées")
                break

        except Exception as exc:
            logger.warning(f"  Échec URL {url} : {exc}")
            last_error = exc
            time.sleep(2)

    if df is None or len(df) == 0:
        # Fallback : instructions manuelles
        raise RuntimeError(
            f"Impossible de télécharger le FAO FPI ({last_error}).\n\n"
            "  Téléchargement manuel :\n"
            "  1. Aller sur : https://www.fao.org/worldfoodsituation/foodpricesindex/en/\n"
            "  2. Cliquer 'Download data' → télécharger le CSV/Excel\n"
            "  3. Sauvegarder sous : data/bronze/fao_food_price_raw.csv\n"
            "  4. Colonnes nécessaires : date, Food Price Index, Cereals, Oils"
        )

    # Filtrer sur la plage d'analyse
    df = df.loc[start_date:end_date]
    df["pulled_at"] = datetime.now(tz=timezone.utc).isoformat()

    # Validation
    logger.info(f"  Plage       : {df.index[0].date()} → {df.index[-1].date()}")
    logger.info(f"  Colonnes    : {list(df.columns)}")
    logger.info(f"  FAO Food idx: min={df['fao_food_index'].min():.1f}  "
                f"max={df['fao_food_index'].max():.1f}")

    df.index.name = "date"
    df.to_csv(output_path, index=True)
    logger.info(f"  Sauvegardé bronze : {output_path}")

    return df


def _parse_fao_excel(url: str) -> pd.DataFrame:
    """Parse le fichier Excel FAO (format officiel)."""
    try:
        import requests
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        raw_excel = pd.read_excel(io.BytesIO(resp.content), sheet_name=None)
    except ImportError:
        raise ImportError("requests requis : pip install requests")

    # Le FAO Excel a plusieurs sheets — chercher la sheet "Food Price Indices"
    target_sheet = None
    for name, sheet in raw_excel.items():
        if any(kw in str(name).lower() for kw in ["price", "food", "index"]):
            target_sheet = sheet
            break
    if target_sheet is None:
        target_sheet = list(raw_excel.values())[0]

    return _standardize_fao_df(target_sheet)


def _parse_fao_csv(url: str) -> pd.DataFrame:
    """Parse le CSV FAO."""
    try:
        import requests
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        raw = pd.read_csv(io.StringIO(resp.text))
    except ImportError:
        raise ImportError("requests requis : pip install requests")

    return _standardize_fao_df(raw)


def _standardize_fao_df(raw: pd.DataFrame) -> pd.DataFrame:
    """Standardise un DataFrame FAO brut en format silver."""
    raw = raw.copy()
    raw.columns = [str(c).strip() for c in raw.columns]

    # Trouver la colonne date
    date_col = None
    for c in raw.columns:
        if any(kw in c.lower() for kw in ["date", "month", "year", "mois", "année"]):
            date_col = c
            break
    if date_col is None:
        date_col = raw.columns[0]

    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    raw = raw.dropna(subset=[date_col])
    raw[date_col] = raw[date_col].dt.to_period("M").dt.to_timestamp("MS")
    raw = raw.sort_values(date_col).drop_duplicates(date_col).set_index(date_col)
    raw.index = pd.DatetimeIndex(raw.index, freq="MS")

    # Mapper les colonnes
    result = pd.DataFrame(index=raw.index)
    for src_col, dst_col in _COL_MAP.items():
        # Recherche insensible à la casse + partielle
        match = next(
            (c for c in raw.columns
             if src_col.lower() in c.lower() or c.lower() in src_col.lower()),
            None
        )
        if match:
            result[dst_col] = pd.to_numeric(raw[match], errors="coerce")
        else:
            result[dst_col] = np.nan
            logger.debug(f"  Colonne FAO '{src_col}' non trouvée → NaN")

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = ingest_fao_fpi(force_refresh=False)
    print(df.tail(12))
    print(f"\nShape : {df.shape}")
    print(f"Colonnes : {list(df.columns)}")
