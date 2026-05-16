"""
src/ingestion/google_trends_v3.py — Ingestion Google Trends v3 (Bronze)

Différences vs data_pipeline.py original :
  - Keywords nettoyés : suppression "chomage maroc" (non corrélé à l'inflation directe)
  - Ajout : "prix carburant maroc", "prix legumes maroc", "subvention maroc"
  - 3 sous-indices thématiques dès l'ingestion (pas un composite unique)
  - Sauvegarde brute en bronze/ avant normalisation

Keywords v3 regroupés par thème :
  PRIX_ALIM  : "prix huile", "hausse prix", "prix legumes maroc", أسعار المواد الغذائية
  INFLATION  : "inflation maroc" (anchor), التضخم في المغرب
  CARBURANT  : "prix carburant maroc", ارتفاع الأسعار
  SUBVENTION : "subvention maroc" (lié directement à la politique prix HCP)

Usage :
    from src.ingestion.google_trends_v3 import ingest_google_trends_v3
    df = ingest_google_trends_v3()
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

START_DATE = "2010-01-01"
END_DATE   = "2024-12-01"
GEO        = "MA"
TIMEFRAME  = f"{START_DATE[:7]} {END_DATE[:7]}"

# Keywords v3 — économiquement justifiés, sans "chomage"
KEYWORDS_V3 = {
    # Anchor (obligatoire dans chaque chunk pour la comparabilité)
    "anchor": "inflation maroc",

    # Thème : Prix alimentaires directs (40% panier IPC)
    "prix_alim": [
        "prix huile",
        "hausse prix",
        "prix legumes maroc",
        "أسعار المواد الغذائية",   # prix alimentaires (arabe)
        "غلاء المعيشة",             # cherté de la vie (arabe)
    ],

    # Thème : Inflation générale
    "inflation": [
        "inflation maroc",          # anchor
        "التضخم في المغرب",         # inflation au Maroc (arabe)
    ],

    # Thème : Carburant & énergie (transport = 9% panier IPC)
    "carburant": [
        "prix carburant maroc",
        "ارتفاع الأسعار",           # hausse des prix (arabe)
    ],

    # Thème : Politique de subventions (impact direct sur prix officiels)
    "subvention": [
        "subvention maroc",
    ],
}


def ingest_google_trends_v3(
    output_path:   str | Path | None = None,
    force_refresh: bool = False,
    sleep_between: float = 60.0,
    retries:       int = 3,
) -> pd.DataFrame:
    """
    Récupère Google Trends v3 pour le Maroc — keywords économiquement justifiés.

    Retourne
    --------
    pd.DataFrame (index MS) avec une colonne par keyword + pulled_at
    Sauvegarde : data/bronze/google_trends_raw_v3.csv
    """
    if output_path is None:
        output_path = BRONZE_DIR / "google_trends_raw_v3.csv"
    output_path = Path(output_path)

    if output_path.exists() and not force_refresh:
        logger.info(f"Trends v3 — cache trouvé : {output_path}")
        return pd.read_csv(output_path, parse_dates=["date"], index_col="date")

    try:
        from pytrends.request import TrendReq
    except ImportError:
        raise ImportError("pytrends requis : pip install pytrends")

    anchor = KEYWORDS_V3["anchor"]
    all_keywords = [anchor] + [
        kw
        for theme, kws in KEYWORDS_V3.items()
        if theme != "anchor"
        for kw in kws
        if kw != anchor
    ]
    # Dédupliquer en gardant l'ordre
    seen = set()
    all_keywords_dedup = []
    for kw in all_keywords:
        if kw not in seen:
            seen.add(kw)
            all_keywords_dedup.append(kw)

    logger.info(f"Trends v3 — {len(all_keywords_dedup)} keywords, geo={GEO}")
    logger.info(f"  Keywords : {all_keywords_dedup}")

    # Chunking avec anchor pour comparabilité cross-chunk (max 5 par requête)
    chunks = [
        [anchor] + all_keywords_dedup[i:i+4]
        for i in range(0, len(all_keywords_dedup), 4)
        if all_keywords_dedup[i:i+4]
    ]
    # Retirer le duplicate anchor du premier chunk
    if chunks and chunks[0][0] == anchor and anchor in chunks[0][1:]:
        chunks[0] = [anchor] + [k for k in chunks[0][1:] if k != anchor]

    frames: list[pd.DataFrame] = []
    anchor_ref: pd.Series | None = None

    for attempt in range(1, retries + 1):
        try:
            pt = TrendReq(hl="fr-MA", tz=0, timeout=(10, 30), retries=2)
            frames = []
            anchor_ref = None

            for i, chunk in enumerate(chunks):
                logger.info(f"  Chunk {i+1}/{len(chunks)} : {chunk}")
                pt.build_payload(chunk, timeframe=TIMEFRAME, geo=GEO)
                raw = pt.interest_over_time().drop(columns=["isPartial"], errors="ignore")

                if i == 0:
                    anchor_ref = raw[anchor].replace(0, np.nan)
                    frames.append(raw)
                else:
                    anchor_this = raw[anchor].replace(0, np.nan)
                    scale = (anchor_ref / anchor_this).ffill().bfill()
                    cols_to_scale = [c for c in raw.columns if c != anchor]
                    frames.append(raw[cols_to_scale].multiply(scale, axis=0))

                if i < len(chunks) - 1:
                    time.sleep(5)

            break  # succès

        except Exception as exc:
            logger.warning(f"  Tentative {attempt}/{retries} échouée : {exc}")
            if attempt < retries:
                logger.info(f"  Attente {sleep_between}s ...")
                time.sleep(sleep_between)
            else:
                raise RuntimeError(
                    f"Google Trends indisponible après {retries} tentatives.\n"
                    f"Vérifier la connexion réseau. Erreur : {exc}"
                ) from exc

    if not frames:
        raise RuntimeError("Aucune donnée Trends récupérée.")

    df = pd.concat(frames, axis=1)
    df.index = pd.DatetimeIndex(df.index)
    df = df.resample("MS").mean().loc[START_DATE:END_DATE]

    # Métadonnées
    df["pulled_at"] = datetime.now(tz=timezone.utc).isoformat()
    df.index.name = "date"
    df.to_csv(output_path, index=True)

    logger.info(f"  Sauvegardé bronze : {output_path}  ({len(df)} mois, {len(df.columns)} colonnes)")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = ingest_google_trends_v3(force_refresh=False)
    print(df.tail(6))
    print(f"\nShape : {df.shape}")
    print(f"Colonnes : {list(df.columns)}")
