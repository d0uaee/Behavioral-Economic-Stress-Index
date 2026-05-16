"""
Pipeline de collecte et nettoyage des données — Projet BESI Maroc
Collecte : Google Trends (FR + Arabe + Darija), Reddit, YouTube, IPC HCP/World Bank

Variables d'environnement nécessaires pour les APIs :
  REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
  YOUTUBE_API_KEY

Logique cache-first : si le CSV traité existe dans data/processed/, il est relu
directement sans appeler l'API. Supprimer le CSV pour forcer un re-téléchargement.
"""

import os
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

np.random.seed(42)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─── Chemins ──────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).resolve().parent.parent
DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

# ─── Plage temporelle dynamique ───────────────────────────────────────────────
START_DATE = "2010-01-01"
# Date de fin figee par defaut pour garder des resultats reproductibles.
# On peut la surcharger via BESI_END_DATE=YYYY-MM-DD si necessaire.
END_DATE   = os.getenv("BESI_END_DATE", "2024-12-01")
GEO        = "MA"
TIMEFRAME  = f"{START_DATE[:7]} {END_DATE[:7]}"

# ─── Keywords multilingues : FR + Arabe + Darija ──────────────────────────────
# "inflation maroc" est l'ANCHOR — inclus dans chaque chunk pytrends pour
# garantir la comparabilité inter-chunks (technique standard, cf. Choi & Varian 2012).
KEYWORDS = [
    # Français
    "inflation maroc",           # anchor
    "prix huile",
    "hausse prix",
    "credit consommation",
    "chomage maroc",
    # Arabe
    "أسعار المواد الغذائية",    # prix alimentaires
    "غلاء المعيشة",              # cherté de la vie
    "التضخم في المغرب",          # inflation au Maroc
    "ارتفاع الأسعار",            # hausse des prix
    # Darija translittéré
    "ghla lprix",
    "inflation lmaroc",
]

REDDIT_KEYWORDS  = ["inflation", "prix", "cherté", "économie"]
REDDIT_SUBREDDIT = "Morocco"
YOUTUBE_QUERIES  = ["inflation maroc", "hausse prix maroc"]


# ─── Normalisation 0-1 ────────────────────────────────────────────────────────
def _normalise_0_1(series: pd.Series) -> pd.Series:
    """Normalise une série entre 0 et 1 (min-max). Renvoie 0 si série constante."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.0, index=series.index, name=series.name)
    return (series - mn) / (mx - mn)


# ─── Pytrends : chunking avec ancrage inter-chunks ────────────────────────────
def _fetch_trends_chunked(
    pt,
    keywords: list[str],
    timeframe: str,
    geo: str,
    chunk_sleep: float = 10.0,
) -> pd.DataFrame:
    """
    Interroge pytrends par chunks de 5 avec ancrage pour comparabilité cross-chunk.

    Google Trends normalise chaque batch indépendamment (max = 100).
    En répétant l'anchor dans chaque chunk, on peut rescaler les valeurs :
      scale = anchor_chunk0 / anchor_chunkN  (élément par élément)
    Cela rend les séries comparables malgré des normalizations distinctes.
    """
    anchor = keywords[0]
    others = [k for k in keywords if k != anchor]
    # Chunks : anchor + 4 autres par lot (max 5 par requête pytrends)
    chunks = [[anchor] + others[i:i+4] for i in range(0, len(others), 4)]

    frames: list[pd.DataFrame] = []
    anchor_ref: pd.Series | None = None

    for i, chunk in enumerate(chunks):
        logger.info(f"  Chunk {i+1}/{len(chunks)} : {chunk}")
        pt.build_payload(chunk, timeframe=timeframe, geo=geo)
        raw = pt.interest_over_time().drop(columns=["isPartial"], errors="ignore")

        if i == 0:
            anchor_ref = raw[anchor].replace(0, np.nan)
            frames.append(raw)
        else:
            # Rescaler par le ratio de l'anchor pour comparer au chunk 0
            anchor_this = raw[anchor].replace(0, np.nan)
            scale = (anchor_ref / anchor_this).ffill().bfill()
            cols  = [c for c in raw.columns if c != anchor]
            frames.append(raw[cols].multiply(scale, axis=0))

        if i < len(chunks) - 1:
            time.sleep(chunk_sleep)

    return pd.concat(frames, axis=1)


# ═══════════════════════════════════════════════════════════════════════════════
# GOOGLE TRENDS
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_google_trends(
    keywords: list[str] | None = None,
    retries: int = 3,
    sleep_between: float = 60.0,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Récupère les données Google Trends mensuelles pour le Maroc (geo='MA').
    Couvre le français, l'arabe et le darija translittéré (11 keywords).

    Logique
    -------
    1. Cache local (trends_monthly.csv) → lecture directe si présent
    2. Sinon → pytrends avec chunking ancré → sauvegarde → retour

    Pour mettre à jour les données : supprimer data/processed/trends_monthly.csv

    Colonnes produites
    ------------------
    Une colonne par keyword (normalisée 0-1) + 'trends_composite' (moyenne).
    Sauvegarde : data/processed/trends_monthly.csv
    """
    if keywords is None:
        keywords = KEYWORDS

    cache = DATA_PROCESSED / "trends_monthly.csv"
    if cache.exists() and not force_refresh:
        logger.info("Trends — cache trouvé, lecture locale.")
        return pd.read_csv(cache, parse_dates=["date"], index_col="date")

    try:
        from pytrends.request import TrendReq
    except ImportError:
        raise ImportError("pytrends requis — pip install pytrends")

    for attempt in range(1, retries + 1):
        try:
            logger.info(f"Google Trends — tentative {attempt}/{retries} …")
            pt     = TrendReq(hl="fr-MA", tz=0, timeout=(10, 25), retries=2)
            raw_df = _fetch_trends_chunked(pt, keywords, TIMEFRAME, GEO)
            logger.info("Google Trends récupéré avec succès.")
            break
        except Exception as exc:
            logger.warning(f"Erreur tentative {attempt} : {exc}")
            if attempt < retries:
                logger.info(f"Attente {sleep_between}s …")
                time.sleep(sleep_between)
            else:
                raise RuntimeError(
                    f"Google Trends indisponible après {retries} tentatives. "
                    "Vérifier la connexion réseau et les quotas."
                ) from exc

    # Resampling mensuel + restriction temporelle
    raw_df.index = pd.DatetimeIndex(raw_df.index)
    monthly = raw_df.resample("MS").mean().loc[START_DATE:END_DATE]

    cols = [k for k in keywords if k in monthly.columns]
    monthly = monthly[cols].copy()
    for col in cols:
        monthly[col] = _normalise_0_1(monthly[col])
    monthly["trends_composite"] = monthly[cols].mean(axis=1)

    monthly.index.name = "date"
    monthly.to_csv(cache, index=True)
    logger.info(f"Sauvegardé : {cache}  ({len(monthly)} lignes, {len(monthly.columns)} colonnes)")
    return monthly


# ═══════════════════════════════════════════════════════════════════════════════
# REDDIT
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_reddit_data(
    keywords: list[str] | None = None,
    subreddit: str = REDDIT_SUBREDDIT,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Scrape r/Morocco via praw et agrège en mensuel.

    Nécessite les variables d'environnement :
      REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET
      (créer une app sur reddit.com/prefs/apps, type 'script')

    Logique cache-first — supprimer reddit_monthly.csv pour forcer un refresh.

    Colonnes produites
    ------------------
    post_volume, avg_score (normalisés 0-1), reddit_composite.
    Sauvegarde : data/processed/reddit_monthly.csv
    """
    if keywords is None:
        keywords = REDDIT_KEYWORDS

    cache = DATA_PROCESSED / "reddit_monthly.csv"
    if cache.exists() and not force_refresh:
        logger.info("Reddit — cache trouvé, lecture locale.")
        return pd.read_csv(cache, parse_dates=["date"], index_col="date")

    client_id     = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent    = os.getenv("REDDIT_USER_AGENT", "BESI-Morocco/1.0")

    if not client_id or not client_secret:
        raise EnvironmentError(
            "Variables manquantes : REDDIT_CLIENT_ID et REDDIT_CLIENT_SECRET\n"
            "  1. Aller sur https://www.reddit.com/prefs/apps\n"
            "  2. Créer une app 'script'\n"
            "  3. export REDDIT_CLIENT_ID=...\n"
            "     export REDDIT_CLIENT_SECRET=..."
        )

    try:
        import praw
    except ImportError:
        raise ImportError("praw requis — pip install praw")

    logger.info(f"Reddit — connexion à r/{subreddit} …")
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        read_only=True,
    )
    sub = reddit.subreddit(subreddit)

    raw_records: list[dict] = []
    for kw in keywords:
        logger.info(f"  Recherche : '{kw}' …")
        for post in sub.search(kw, sort="new", limit=1000, time_filter="all"):
            raw_records.append({
                "date": datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
                        .replace(day=1, hour=0, minute=0, second=0, microsecond=0, tzinfo=None),
                "score": post.score,
            })
        time.sleep(1)

    logger.info(f"Reddit — {len(raw_records)} posts récupérés.")

    df_raw = pd.DataFrame(raw_records)
    df_raw["date"] = pd.to_datetime(df_raw["date"])
    df_raw = df_raw[(df_raw["date"] >= START_DATE) & (df_raw["date"] <= END_DATE)]

    monthly = df_raw.groupby("date").agg(
        post_volume=("score", "count"),
        avg_score=("score", "mean"),
    )
    monthly.index = pd.DatetimeIndex(monthly.index)
    full_idx = pd.date_range(start=START_DATE, end=END_DATE, freq="MS")
    monthly  = monthly.reindex(full_idx, fill_value=0)

    monthly["post_volume"] = _normalise_0_1(monthly["post_volume"])
    monthly["avg_score"]   = _normalise_0_1(monthly["avg_score"])
    monthly["reddit_composite"] = monthly[["post_volume", "avg_score"]].mean(axis=1)

    monthly.index.name = "date"
    monthly.to_csv(cache, index=True)
    logger.info(f"Sauvegardé : {cache}  ({len(monthly)} lignes, {len(monthly.columns)} colonnes)")
    return monthly


# ═══════════════════════════════════════════════════════════════════════════════
# YOUTUBE
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_youtube_data(
    queries: list[str] | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Compte les vidéos publiées chaque mois via YouTube Data API v3.

    Nécessite : YOUTUBE_API_KEY dans l'environnement.
    Quota API : ~10 000 unités/jour. L'API n'est appelée qu'une seule fois ;
    le résultat est mis en cache — les appels suivants lisent le CSV local.

    Logique cache-first — supprimer youtube_monthly.csv pour forcer un refresh.

    Colonnes produites
    ------------------
    video_count (normalisé 0-1), youtube_composite.
    Sauvegarde : data/processed/youtube_monthly.csv
    """
    if queries is None:
        queries = YOUTUBE_QUERIES

    cache = DATA_PROCESSED / "youtube_monthly.csv"
    if cache.exists() and not force_refresh:
        logger.info("YouTube — cache trouvé, lecture locale.")
        return pd.read_csv(cache, parse_dates=["date"], index_col="date")

    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Variable manquante : YOUTUBE_API_KEY\n"
            "  1. Aller sur https://console.cloud.google.com\n"
            "  2. Activer 'YouTube Data API v3'\n"
            "  3. export YOUTUBE_API_KEY=..."
        )

    try:
        from googleapiclient.discovery import build
    except ImportError:
        raise ImportError("google-api-python-client requis — pip install google-api-python-client")

    logger.info("YouTube — connexion à l'API v3 …")
    yt = build("youtube", "v3", developerKey=api_key)

    records: list[dict] = []
    month_range = pd.date_range(start=START_DATE, end=END_DATE, freq="MS")
    for month_start in month_range:
        month_end = (month_start + pd.offsets.MonthEnd(0)).replace(hour=23, minute=59, second=59)
        count = 0
        for query in queries:
            next_page = None
            while True:
                resp = yt.search().list(
                    q=query, part="id", type="video",
                    relevanceLanguage="fr", regionCode="MA",
                    publishedAfter=month_start.strftime("%Y-%m-%dT00:00:00Z"),
                    publishedBefore=month_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    maxResults=50, pageToken=next_page,
                ).execute()
                count     += len(resp.get("items", []))
                next_page  = resp.get("nextPageToken")
                if not next_page:
                    break
                time.sleep(0.5)
        records.append({"date": month_start, "video_count": count})
        time.sleep(0.2)  # respecter le quota journalier

    logger.info(f"YouTube — {sum(r['video_count'] for r in records)} vidéos recensées.")

    monthly = pd.DataFrame(records).set_index("date")
    monthly.index = pd.DatetimeIndex(monthly.index)
    monthly["video_count"]       = _normalise_0_1(monthly["video_count"])
    monthly["youtube_composite"] = monthly["video_count"]

    monthly.index.name = "date"
    monthly.to_csv(cache, index=True)
    logger.info(f"Sauvegardé : {cache}  ({len(monthly)} lignes, {len(monthly.columns)} colonnes)")
    return monthly


# ═══════════════════════════════════════════════════════════════════════════════
# IPC MAROC
# ═══════════════════════════════════════════════════════════════════════════════

def load_ipc_data(filepath: str | Path | None = None) -> pd.DataFrame:
    """
    Charge l'IPC mensuel Maroc.

    Ordre de priorité
    -----------------
    1. data/processed/ipc_processed.csv       (cache traité — lecture directe)
    2. data/raw/ipc_maroc.csv ou filepath     (CSV HCP brut téléchargé manuellement)
    3. World Bank via pandas_datareader       (FP.CPI.TOTL, annuel → interpolé mensuel)
    → RuntimeError avec instructions si aucune source disponible.

    Format CSV attendu (option 2)
    -----------------------------
    Colonnes 'date' + 'ipc' (ou toute colonne numérique si 'ipc' absent).
    Source : hcp.ma  ou  data.worldbank.org/indicator/FP.CPI.TOTL?locations=MA

    Colonnes produites
    ------------------
    ipc        : indice brut
    ipc_yoy    : variation annuelle en % (Year-over-Year)
    ipc_mom    : variation mensuelle en % (Month-over-Month)
    ipc_change : |ipc_yoy| normalisé 0-1 — signal BESI
    Sauvegarde : data/processed/ipc_processed.csv
    """
    cache = DATA_PROCESSED / "ipc_processed.csv"
    if cache.exists():
        logger.info("IPC — cache traité trouvé, lecture locale.")
        return pd.read_csv(cache, parse_dates=["date"], index_col="date")

    # ── Tentative 1 : fichier local brut ─────────────────────────────────────
    raw_path = Path(filepath) if filepath else DATA_RAW / "ipc_maroc.csv"
    df: pd.DataFrame | None = None

    if raw_path.exists():
        logger.info(f"IPC — chargement depuis {raw_path} …")
        try:
            raw = pd.read_csv(raw_path, parse_dates=["date"], index_col="date")
            raw.columns = [c.lower().strip() for c in raw.columns]
            if "ipc" not in raw.columns:
                num_cols = raw.select_dtypes(include=np.number).columns
                if len(num_cols) == 0:
                    raise ValueError("Aucune colonne numérique dans le CSV IPC.")
                raw = raw[[num_cols[0]]].rename(columns={num_cols[0]: "ipc"})
            df = raw[["ipc"]].copy()
            df.index = pd.DatetimeIndex(df.index)
            df = df.resample("MS").mean().loc[START_DATE:END_DATE]
            logger.info(f"IPC local chargé : {len(df)} mois.")
        except Exception as exc:
            logger.warning(f"Erreur lecture CSV local ({exc}) — tentative World Bank.")
            df = None

    # ── Tentative 2 : World Bank via pandas_datareader ────────────────────────
    if df is None:
        logger.info("IPC — téléchargement World Bank (FP.CPI.TOTL, base 2010=100) …")
        try:
            from pandas_datareader import data as wb
            raw_wb = wb.DataReader(
                "FP.CPI.TOTL", "wb",
                country="MA",
                start=2010,
                end=pd.Timestamp(END_DATE).year,
            )
            # pandas_datareader retourne MultiIndex (country, year) → simplifier
            raw_wb = raw_wb.reset_index()
            raw_wb["date"] = pd.to_datetime(raw_wb["year"].astype(str) + "-01-01")
            raw_wb = raw_wb.set_index("date")[["FP.CPI.TOTL"]].rename(
                columns={"FP.CPI.TOTL": "ipc"}
            ).sort_index()

            # Interpolation linéaire : annuel → mensuel
            monthly_idx = pd.date_range(start=START_DATE, end=END_DATE, freq="MS")
            df = raw_wb.reindex(raw_wb.index.union(monthly_idx)).interpolate("time")
            df = df.loc[monthly_idx]
            logger.info(f"IPC World Bank téléchargé et interpolé : {len(df)} mois.")
        except Exception as exc:
            raise RuntimeError(
                f"Impossible de charger l'IPC Maroc ({exc}).\n\n"
                "Solutions :\n"
                "  Option A — Téléchargement manuel HCP :\n"
                "    1. Aller sur hcp.ma > Publications > IPC\n"
                "    2. Sauvegarder en data/raw/ipc_maroc.csv (colonnes: date, ipc)\n\n"
                "  Option B — World Bank automatique :\n"
                "    pip install pandas-datareader\n"
                "    et relancer load_ipc_data()"
            ) from exc

    # ─── Dérivées temporelles ─────────────────────────────────────────────────
    df["ipc_yoy"]    = df["ipc"].pct_change(12) * 100   # variation annuelle en %
    df["ipc_mom"]    = df["ipc"].pct_change(1)  * 100   # variation mensuelle en %
    df["ipc_change"] = _normalise_0_1(df["ipc_yoy"].abs())

    df.index.name = "date"
    df.to_csv(cache, index=True)
    logger.info(f"Sauvegardé : {cache}  ({len(df)} lignes, {len(df.columns)} colonnes)")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# BESI — Behavioral Economic Stress Index
# ═══════════════════════════════════════════════════════════════════════════════

# Poids BESI composite (legacy v1/v2 — conservé pour compatibilité run_all)
# NOTE v3 : ipc_change retiré du BESI pur → voir src/features/indexes.py
_W_TRENDS  = 0.50   # rééquilibré : plus de composante IPC
_W_REDDIT  = 0.30
_W_YOUTUBE = 0.20
# _W_IPC supprimé — ipc_change ne doit pas être dans un indice comportemental

_THRESH_NORMAL  = 0.35   # < 0.35        → Normal
_THRESH_WARNING = 0.65   # 0.35 – 0.65  → Warning  /  > 0.65 → High Stress


def build_besi_index(
    trends_df:  pd.DataFrame,
    reddit_df:  pd.DataFrame,
    youtube_df: pd.DataFrame,
    ipc_df:     pd.DataFrame,
) -> pd.DataFrame:
    """
    Construit le BESI (Behavioral Economic Stress Index) composite.

    BESI = 0.40*trends + 0.30*reddit + 0.20*youtube + 0.10*ipc_change

    Toutes les composantes sont renormalisées 0-1 avant agrégation.
    L'alignement se fait sur l'intersection temporelle des quatre sources.

    stress_level : 'Normal' (<0.35) / 'Warning' (0.35-0.65) / 'High Stress' (>0.65)
    Sauvegarde : data/processed/master_dataset.csv
    """
    idx = (
        trends_df.index
        .intersection(reddit_df.index)
        .intersection(youtube_df.index)
        .intersection(ipc_df.index)
    )
    if len(idx) == 0:
        raise ValueError("Aucun mois commun entre les quatre sources — vérifier les plages de dates.")

    master = pd.DataFrame(index=idx)
    master.index.name = "date"

    master["trends_composite"]  = _normalise_0_1(trends_df.loc[idx, "trends_composite"])
    master["reddit_composite"]  = _normalise_0_1(reddit_df.loc[idx, "reddit_composite"])
    master["youtube_composite"] = _normalise_0_1(youtube_df.loc[idx, "youtube_composite"])
    master["ipc_change"]        = _normalise_0_1(ipc_df.loc[idx, "ipc_change"])

    for col in ["ipc", "ipc_yoy", "ipc_mom"]:
        if col in ipc_df.columns:
            master[col] = ipc_df.loc[idx, col]

    # BESI composite — sans ipc_change (évite le data leakage cible → feature)
    master["besi"] = (
        _W_TRENDS  * master["trends_composite"]
        + _W_REDDIT  * master["reddit_composite"]
        + _W_YOUTUBE * master["youtube_composite"]
    )

    def _label(v) -> str:
        """Labellise uniquement les valeurs non-nulles."""
        if pd.isna(v):
            return "Unknown"   # bug fix : NaN ne doit pas devenir "High Stress"
        if v < _THRESH_NORMAL:  return "Normal"
        if v < _THRESH_WARNING: return "Warning"
        return "High Stress"

    master["stress_level"] = master["besi"].apply(_label)

    counts = master["stress_level"].value_counts()
    logger.info(
        f"BESI construit sur {len(master)} mois | "
        f"Normal={counts.get('Normal', 0)}, "
        f"Warning={counts.get('Warning', 0)}, "
        f"High Stress={counts.get('High Stress', 0)}"
    )
    logger.info(
        f"BESI min={master['besi'].min():.3f}  "
        f"mean={master['besi'].mean():.3f}  "
        f"max={master['besi'].max():.3f}"
    )

    out_path = DATA_PROCESSED / "master_dataset.csv"
    master.to_csv(out_path, index=True)
    logger.info(f"Sauvegardé : {out_path}  ({len(master)} lignes, {len(master.columns)} colonnes)")
    return master


# ─── Point d'entrée ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    df_trends = fetch_google_trends()
    df_reddit = fetch_reddit_data()
    df_yt     = fetch_youtube_data()
    df_ipc    = load_ipc_data()
    df_besi   = build_besi_index(df_trends, df_reddit, df_yt, df_ipc)

    print(df_besi[["besi", "stress_level", "ipc_yoy"]].tail(24).to_string())
    print("\n--- Resume ---")
    for name, df in [
        ("Trends",  df_trends),
        ("Reddit",  df_reddit),
        ("YouTube", df_yt),
        ("IPC",     df_ipc),
        ("BESI",    df_besi),
    ]:
        print(
            f"{name:8s} : {df.shape[0]} lignes x {df.shape[1]} col  | "
            f"{df.index[0].date()} -> {df.index[-1].date()}"
        )
