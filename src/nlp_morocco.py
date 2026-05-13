"""
nlp_morocco.py — Enrichissement NLP des signaux comportementaux marocains
Projet BESI Maroc | ENSAM Meknes | Douae & Adama

Modules
-------
1. Scraping medias marocains (Hespress, le360, h24info, alyaoum24)
2. Commentaires YouTube (chaines media marocaines)
3. Scoring NLP Darija / Arabe / Francais
4. Agregation mensuelle du signal NLP
5. BESI enrichi (Google Trends + NLP + YouTube + Reddit + IPC)
6. Visualisation : morocco_nlp_vs_ipc.png

Fallback : si scraping/API bloque -> donnees simulees realistes
           + journal dans outputs/reports/data_sources.txt
"""

import os
import re
import time
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

np.random.seed(42)
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─── Chemins ──────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
RAW_DIR  = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
FIG_DIR  = ROOT / "outputs" / "figures"
REP_DIR  = ROOT / "outputs" / "reports"

for _d in (RAW_DIR, PROC_DIR, FIG_DIR, REP_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ─── Configuration globale ────────────────────────────────────────────────────
CONFIG = {
    "YOUTUBE_API_KEY":         os.getenv("YOUTUBE_API_KEY", ""),
    "REQUEST_DELAY":           2.0,       # secondes entre requetes web
    "MAX_ARTICLES":            100,       # articles par site
    "MAX_COMMENTS_ARTICLE":    30,        # commentaires par article
    "MAX_COMMENTS_VIDEO":      100,       # commentaires par video YouTube
    "SIM_START":               "2018-01-01",   # debut simulation fallback
    "USER_AGENT": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

# ─── Chaines YouTube cibles ───────────────────────────────────────────────────
# IDs verifies ; si un ID est invalide la fonction _search_channel() prend le relais
YOUTUBE_CHANNELS = {
    "2M Maroc":    "UCR8bCffDzPQNZ7FNAhDDHdw",
    "Medi1TV":     "UCW3Ai4B-8oSt18BsP1VfK_g",
    "Hespress TV": "UCvgfHhIwPkh3yjqZdGxvTkA",
    "Medias24":    "UCb9HN3PikUk_fPwFXBrNv1A",
}

# Mots-cles de recherche video (FR + Arabe)
YT_KEYWORDS = [
    "prix", "inflation", "economie", "maroc", "carburant",
    "الأسعار",   # al-as3ar (les prix)
    "تضخم",                       # tadakhkhum (inflation)
    "الاقتصاد",  # al-iqtisad (economie)
]


# ═══════════════════════════════════════════════════════════════════════════════
# DICTIONNAIRE NLP — DARIJA / ARABE / FRANCAIS
# ═══════════════════════════════════════════════════════════════════════════════

STRESS_KEYWORDS: dict[str, list[str]] = {
    # Cherté / hausse des prix
    "prix_eleves": [
        "ghali", "ghla", "cher", "tghla", "zid fl prix", "prix monte",
        "trop cher", "hors de prix", "flambee",
        "غالي",          # ghali (arabe)
        "غلاء",           # ghla2
        "ارتفاع الأسعار",  # irtifa3 al-as3ar
        "غلاء المعيشة",  # ghla2 al-ma3ischa
    ],
    # Manque d'argent / pauvrete
    "manque_argent": [
        "ma b9ach", "ma 3ndich", "flouss", "mskine", "ma kayn",
        "pas d argent", "fin de mois", "fin du mois", "serre",
        "dette", "endette",
        "فلوس",           # flouss
        "ما عنديش",  # ma 3ndich
        "فقر",                 # fa9r (pauvrete)
    ],
    # Frustration / colere
    "frustration": [
        "hshuma", "3ib", "crise", "khasara", "wakha",
        "scandale", "honte", "inacceptable", "inadmissible",
        "msikin blad", "blad mahasna", "probleme",
        "عيب",                 # 3ayb
        "حشومة",     # hchouma
        "أزمة",           # azma (crise)
    ],
    # Produits de base
    "produits_base": [
        "zit", "sokkar", "dqiq", "carburant", "essence", "gasoil",
        "gaz", "huile", "farine", "sucre", "pain",
        "زيت",                 # zit (huile)
        "سكر",                 # sokkar (sucre)
        "دقيق",           # dqiq (farine)
        "وقود",           # waqoud (carburant)
        "غاز",                 # ghaz (gaz)
    ],
    # Sentiment positif (ponderation negative dans le scoring)
    "positif": [
        "mzyan", "bahi", "hamdullah", "zwina", "bien", "super",
        "amelioration", "baisse", "bonne nouvelle",
        "الحمد لله",  # hamdoullah
        "مزيان",    # mzyan (bien)
    ],
}

# Emojis negatifs (stress)
_NEG_EMOJIS = ["\U0001F621", "\U0001F624", "\U0001F62D",
               "\U0001F92C", "\U0001F622"]  # 😡 😤 😭 🤬 😢


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS INTERNES
# ═══════════════════════════════════════════════════════════════════════════════

def _normalise_0_1(s: pd.Series) -> pd.Series:
    """Normalisation min-max 0-1. Retourne 0 si serie constante."""
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(0.0, index=s.index, name=s.name)
    return (s - mn) / (mx - mn)


def _log_source(msg: str, mode: str = "a") -> None:
    """Ajoute une ligne au journal data_sources.txt."""
    with open(REP_DIR / "data_sources.txt", mode, encoding="utf-8") as fh:
        fh.write(msg + "\n")


def _http_get(url: str, timeout: int = 15) -> Optional[object]:
    """
    GET HTTP avec headers navigateur. Retourne l'objet Response ou None si erreur.
    Respecte CONFIG['REQUEST_DELAY'] avant chaque requete.
    """
    try:
        import requests
        time.sleep(CONFIG["REQUEST_DELAY"])
        resp = requests.get(
            url,
            headers={"User-Agent": CONFIG["USER_AGENT"],
                     "Accept-Language": "fr-MA,fr;q=0.9,ar;q=0.8"},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp
    except Exception as exc:
        logger.debug("HTTP GET echec : %s  (%s)", url, exc)
        return None


def _make_soup(html: str):
    """Parse le HTML avec BeautifulSoup (lxml ou html.parser)."""
    try:
        from bs4 import BeautifulSoup
        try:
            return BeautifulSoup(html, "lxml")
        except Exception:
            return BeautifulSoup(html, "html.parser")
    except ImportError:
        raise ImportError("beautifulsoup4 requis : pip install beautifulsoup4 lxml")


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION FALLBACK
# ═══════════════════════════════════════════════════════════════════════════════

def _simulate_comments(
    source: str,
    n_months: int = 84,
    articles_per_month: int = 25,
    comments_per_article: int = 15,
) -> pd.DataFrame:
    """
    Genere des commentaires simules realistes si le scraping est impossible.

    Le niveau de stress suit le profil IPC marocain :
    - 2018-2021 : stress modere (base ~0.28)
    - 2022      : pic inflationniste (+0.35)
    - 2023-2024 : niveau eleve mais en baisse (~0.45)
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range(CONFIG["SIM_START"], periods=n_months, freq="MS")

    # Profil de stress mensuel realiste (cale sur l'IPC marocain)
    t = np.linspace(0, 1, n_months)
    base_stress = (
        0.25
        + 0.10 * t                                             # tendance longue
        + 0.35 * np.exp(-((np.arange(n_months) - 48) ** 2) / (2 * 6 ** 2))  # pic 2022
        + rng.normal(0, 0.04, n_months)                        # bruit
    )
    base_stress = np.clip(base_stress, 0.05, 0.95)

    rows = []
    for i, dt in enumerate(dates):
        n_art = rng.integers(max(5, articles_per_month - 10),
                             articles_per_month + 10)
        for _ in range(int(n_art)):
            n_cmt = rng.integers(3, comments_per_article + 1)
            rows.append({
                "date":          dt,
                "source":        source,
                "article_title": f"[simule] Article economie {dt.strftime('%Y-%m')}",
                "comment_text":  "[simule]",
                "comment_count": int(n_cmt),
                "stress_score":  float(
                    np.clip(rng.normal(base_stress[i], 0.08), 0, 1)
                ),
                "keyword_score":    0.0,
                "intensity_score":  0.0,
                "engagement_weight": 1.0,
            })

    return pd.DataFrame(rows)


def _simulate_yt_comments(
    channel: str,
    n_months: int = 84,
    videos_per_month: int = 8,
    comments_per_video: int = 20,
) -> pd.DataFrame:
    """Genere des commentaires YouTube simules si l'API est indisponible."""
    rng = np.random.default_rng(42)
    dates = pd.date_range(CONFIG["SIM_START"], periods=n_months, freq="MS")

    t = np.linspace(0, 1, n_months)
    base_stress = np.clip(
        0.28 + 0.12 * t
        + 0.30 * np.exp(-((np.arange(n_months) - 48) ** 2) / (2 * 8 ** 2))
        + rng.normal(0, 0.05, n_months),
        0.05, 0.95,
    )
    rows = []
    for i, dt in enumerate(dates):
        n_vid = rng.integers(max(2, videos_per_month - 3), videos_per_month + 3)
        for _ in range(int(n_vid)):
            views = int(rng.integers(1000, 200000))
            likes = int(views * rng.uniform(0.01, 0.08))
            n_cmt = rng.integers(5, comments_per_video + 1)
            for _ in range(int(n_cmt)):
                rows.append({
                    "date":          dt,
                    "channel":       channel,
                    "video_title":   f"[simule] Video economie {dt.strftime('%Y-%m')}",
                    "views":         views,
                    "likes":         likes,
                    "comment_text":  "[simule]",
                    "video_id":      f"sim_{channel[:4]}_{i}",
                    "stress_score":  float(
                        np.clip(rng.normal(base_stress[i], 0.09), 0, 1)
                    ),
                    "keyword_score":    0.0,
                    "intensity_score":  0.0,
                    "engagement_weight": float(np.log1p(likes)),
                })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — SCRAPING MEDIAS MAROCAINS
# ═══════════════════════════════════════════════════════════════════════════════

# Configuration des sites : URL economie + selecteurs CSS specifiques
# Plusieurs selecteurs par categorie : le premier qui trouve quelque chose est utilise
_MEDIA_CONFIG: dict[str, dict] = {
    "hespress": {
        "url": "https://www.hespress.com/economie",
        "article_selectors":  [
            ("article", "h2 a"),
            (".block-list article", "h2 a"),
            (".post-title a", None),
        ],
        "comment_selectors": [
            ".kmt-text",
            ".comment-body p",
            ".comment_content",
            ".the-comment",
        ],
        "date_selectors": [".date-post", "time", ".post-date"],
        "next_page_param": "?page={n}",
    },
    "le360": {
        "url": "https://le360.ma/economie",
        "article_selectors": [
            (".article-item h3 a", None),
            (".article-list .title a", None),
            ("article h2 a", None),
            (".entry-title a", None),
        ],
        "comment_selectors": [
            ".comment-text",
            ".comment-body",
            ".commentaire",
            ".user-comment",
        ],
        "date_selectors": [".article-date", "time[datetime]", ".date"],
        "next_page_param": "?page={n}",
    },
    "h24info": {
        "url": "https://h24info.ma/economie",
        "article_selectors": [
            ("article h2 a", None),
            (".post-title a", None),
            (".entry-title a", None),
        ],
        "comment_selectors": [
            ".comment-content p",
            ".comment-body",
            ".comment-text",
        ],
        "date_selectors": ["time", ".entry-date", ".post-date"],
        "next_page_param": "/page/{n}/",
    },
    "alyaoum24": {
        "url": "https://www.alyaoum24.com/economie",
        "article_selectors": [
            (".post-title a", None),
            ("article h2 a", None),
            (".entry-title a", None),
            ("h3.title a", None),
        ],
        "comment_selectors": [
            ".comment-content",
            ".kmt-content",
            ".commentBody",
        ],
        "date_selectors": [".post-date", "time", ".date-post"],
        "next_page_param": "/page/{n}",
    },
}


def _scrape_one_site(
    site_key: str,
    cfg: dict,
    max_articles: int = 100,
    max_comments: int = 30,
) -> list[dict]:
    """
    Tente de scraper un site media marocain.
    Retourne une liste de dicts {date, source, article_title, comment_text, comment_count}.
    Retourne [] si le site bloque ou si la structure HTML ne correspond pas.
    """
    rows: list[dict] = []
    base_url = cfg["url"]

    resp = _http_get(base_url)
    if resp is None:
        logger.info("  [%s] Inaccessible — passe au suivant.", site_key)
        return []

    try:
        soup = _make_soup(resp.text)
    except ImportError as exc:
        logger.warning("BeautifulSoup manquant : %s", exc)
        return []

    # --- Extraction des URLs d'articles ---
    article_links: list[tuple[str, str]] = []  # (url, titre)
    for sel_tuple in cfg["article_selectors"]:
        if isinstance(sel_tuple, tuple):
            container_sel, link_sel = sel_tuple
        else:
            container_sel, link_sel = sel_tuple, None

        if link_sel:
            items = soup.select(f"{container_sel} {link_sel}")
        else:
            items = soup.select(container_sel)

        if items:
            for item in items[:max_articles]:
                href  = item.get("href", "")
                title = item.get_text(strip=True)
                if href and title:
                    if not href.startswith("http"):
                        from urllib.parse import urljoin
                        href = urljoin(base_url, href)
                    article_links.append((href, title))
            if article_links:
                break

    if not article_links:
        logger.info("  [%s] Aucun article trouve — structure HTML inconnue.", site_key)
        return []

    logger.info("  [%s] %d articles trouves.", site_key, len(article_links))

    # --- Pour chaque article : date + commentaires ---
    for art_url, art_title in article_links[:max_articles]:
        art_resp = _http_get(art_url)
        if art_resp is None:
            continue

        try:
            art_soup = _make_soup(art_resp.text)
        except Exception:
            continue

        # Date de l'article
        art_date = None
        for d_sel in cfg["date_selectors"]:
            tag = art_soup.select_one(d_sel)
            if tag:
                dt_str = tag.get("datetime") or tag.get_text(strip=True)
                try:
                    art_date = pd.to_datetime(dt_str, errors="coerce")
                    if pd.notna(art_date):
                        art_date = art_date.to_period("M").to_timestamp()
                        break
                except Exception:
                    pass
        if art_date is None or pd.isna(art_date):
            art_date = pd.Timestamp.now().to_period("M").to_timestamp()

        # Commentaires
        comments: list[str] = []
        for c_sel in cfg["comment_selectors"]:
            found = art_soup.select(c_sel)
            if found:
                comments = [c.get_text(strip=True)
                            for c in found[:max_comments]
                            if len(c.get_text(strip=True)) > 5]
                if comments:
                    break

        # Meme si 0 commentaire, on garde l'article (titre utile)
        if not comments:
            comments = [""]

        for cmt in comments:
            rows.append({
                "date":          art_date,
                "source":        site_key,
                "article_title": art_title[:200],
                "comment_text":  cmt[:500],
                "comment_count": len(comments),
            })

    return rows


def scrape_media_comments(
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    MODULE 1 — Scraping des commentaires des medias marocains.

    Sites cibles : Hespress, le360, h24info, alyaoum24 (section economie).
    Pour chaque site : 100 articles max, 30 commentaires max par article.
    Delai de 2 secondes entre chaque requete (CONFIG['REQUEST_DELAY']).

    Si un site bloque ou renvoie une structure inconnue :
      - passage silencieux au site suivant
      - simulation realiste pour ce site

    Sauvegarde : data/raw/media_comments.csv
    Retourne   : DataFrame brut (avant scoring NLP)
    """
    cache = RAW_DIR / "media_comments.csv"
    if cache.exists() and not force_refresh:
        logger.info("MODULE 1 — Cache media_comments.csv trouve. Lecture locale.")
        return pd.read_csv(cache, parse_dates=["date"])

    logger.info("MODULE 1 — Scraping medias marocains ...")
    _log_source("\n=== MODULE 1 : Scraping medias marocains ===", mode="a")

    all_rows: list[dict] = []
    sim_sources: list[str] = []

    for site_key, cfg in _MEDIA_CONFIG.items():
        logger.info("  Scraping %s ...", site_key)
        rows = _scrape_one_site(
            site_key, cfg,
            max_articles=CONFIG["MAX_ARTICLES"],
            max_comments=CONFIG["MAX_COMMENTS_ARTICLE"],
        )

        if rows:
            all_rows.extend(rows)
            _log_source(f"  {site_key} : {len(rows)} lignes scrapees (reel)")
        else:
            # Fallback : simulation realiste
            logger.info("  [%s] Scraping echec — donnees simulees.", site_key)
            sim_df = _simulate_comments(source=site_key)
            all_rows.extend(sim_df.to_dict("records"))
            sim_sources.append(site_key)
            _log_source(f"  {site_key} : simule (scraping impossible)")

    if not all_rows:
        logger.warning("MODULE 1 — Aucune donnee reelle ni simulee. Fallback global.")
        for sk in _MEDIA_CONFIG:
            all_rows.extend(_simulate_comments(source=sk).to_dict("records"))

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Colonnes manquantes (si venant du scraping reel, sans scoring encore)
    for col in ["stress_score", "keyword_score", "intensity_score", "engagement_weight"]:
        if col not in df.columns:
            df[col] = 0.0

    df.to_csv(cache, index=False, encoding="utf-8-sig")
    logger.info("MODULE 1 — Sauvegarde : %s  (%d lignes)", cache, len(df))

    if sim_sources:
        _log_source(f"  Sites simules : {sim_sources}")
        _log_source("  Raison : site bloque ou structure HTML inconnue")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — COMMENTAIRES YOUTUBE (chaines media marocaines)
# ═══════════════════════════════════════════════════════════════════════════════

def _search_channel_id(yt, channel_name: str) -> Optional[str]:
    """
    Cherche l'ID d'une chaine YouTube par son nom via l'API.
    Retourne l'ID ou None si non trouve.
    """
    try:
        resp = yt.search().list(
            q=channel_name, part="snippet", type="channel", maxResults=1
        ).execute()
        items = resp.get("items", [])
        if items:
            return items[0]["snippet"]["channelId"]
    except Exception as exc:
        logger.debug("Recherche chaine %s : %s", channel_name, exc)
    return None


def _get_channel_videos(
    yt,
    channel_id: str,
    keywords: list[str],
    since_year: int = 2018,
    max_videos: int = 200,
) -> list[dict]:
    """
    Recupere les videos d'une chaine contenant des mots-cles economiques.
    Retourne une liste de {video_id, title, published_at, views, likes, comment_count}.
    """
    published_after = f"{since_year}-01-01T00:00:00Z"
    videos: list[dict] = []
    seen_ids: set = set()

    for kw in keywords[:5]:   # limiter la consommation de quota
        try:
            next_page = None
            while len(videos) < max_videos:
                resp = yt.search().list(
                    q=kw,
                    channelId=channel_id,
                    part="id,snippet",
                    type="video",
                    publishedAfter=published_after,
                    maxResults=50,
                    pageToken=next_page,
                ).execute()

                for item in resp.get("items", []):
                    vid_id = item["id"].get("videoId")
                    if not vid_id or vid_id in seen_ids:
                        continue
                    seen_ids.add(vid_id)
                    title = item["snippet"].get("title", "")
                    pub   = item["snippet"].get("publishedAt", "")
                    videos.append({
                        "video_id":     vid_id,
                        "title":        title,
                        "published_at": pub,
                        "views":        0,
                        "likes":        0,
                        "comment_count": 0,
                    })

                next_page = resp.get("nextPageToken")
                if not next_page:
                    break
                time.sleep(0.3)
        except Exception as exc:
            logger.debug("Videos chaine %s kw=%s : %s", channel_id, kw, exc)

    # Enrichir avec les statistiques video
    if videos:
        ids_batch = [v["video_id"] for v in videos[:50]]
        try:
            stats_resp = yt.videos().list(
                id=",".join(ids_batch), part="statistics"
            ).execute()
            stats_map = {
                it["id"]: it.get("statistics", {})
                for it in stats_resp.get("items", [])
            }
            for v in videos:
                st = stats_map.get(v["video_id"], {})
                v["views"]         = int(st.get("viewCount",    0))
                v["likes"]         = int(st.get("likeCount",    0))
                v["comment_count"] = int(st.get("commentCount", 0))
        except Exception:
            pass

    return videos


def _get_video_comments(yt, video_id: str, max_comments: int = 100) -> list[str]:
    """Recupere les commentaires texte d'une video YouTube (top-level)."""
    comments: list[str] = []
    try:
        next_page = None
        while len(comments) < max_comments:
            resp = yt.commentThreads().list(
                videoId=video_id,
                part="snippet",
                maxResults=min(100, max_comments - len(comments)),
                order="relevance",
                pageToken=next_page,
            ).execute()
            for item in resp.get("items", []):
                txt = (item["snippet"]["topLevelComment"]["snippet"]
                       .get("textDisplay", ""))
                if txt:
                    comments.append(txt[:500])
            next_page = resp.get("nextPageToken")
            if not next_page:
                break
            time.sleep(0.2)
    except Exception as exc:
        logger.debug("Commentaires video %s : %s", video_id, exc)
    return comments


def fetch_youtube_comments(
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    MODULE 2 — Recuperation des commentaires YouTube (chaines media marocaines).

    Pour chaque chaine :
    1. Recupere les videos depuis 2018 contenant des mots-cles economiques
    2. Pour chaque video : stats (views, likes) + 100 premiers commentaires
    3. Si l'ID de chaine est invalide : recherche par nom via l'API

    Si l'API est indisponible ou le quota depasse :
      -> simulation realiste + log dans data_sources.txt

    Sauvegarde : data/raw/youtube_comments.csv
    Retourne   : DataFrame brut (avant scoring NLP)
    """
    cache = RAW_DIR / "youtube_comments.csv"
    if cache.exists() and not force_refresh:
        logger.info("MODULE 2 — Cache youtube_comments.csv trouve. Lecture locale.")
        return pd.read_csv(cache, parse_dates=["date"])

    logger.info("MODULE 2 — YouTube commentaires ...")
    _log_source("\n=== MODULE 2 : YouTube commentaires ===", mode="a")

    api_key = CONFIG["YOUTUBE_API_KEY"]
    all_rows: list[dict] = []
    simulated: list[str] = []

    if not api_key:
        logger.warning("MODULE 2 — YOUTUBE_API_KEY manquante. Simulation complete.")
        _log_source("  Erreur : YOUTUBE_API_KEY non definie -> simulation complete")
        for ch_name in YOUTUBE_CHANNELS:
            df_sim = _simulate_yt_comments(channel=ch_name)
            all_rows.extend(df_sim.to_dict("records"))
            simulated.append(ch_name)
    else:
        try:
            from googleapiclient.discovery import build
            yt = build("youtube", "v3", developerKey=api_key)
            logger.info("  Connexion YouTube API v3 OK.")
        except ImportError:
            logger.warning("google-api-python-client manquant. Simulation complete.")
            _log_source("  Erreur : google-api-python-client non installe -> simulation")
            for ch_name in YOUTUBE_CHANNELS:
                df_sim = _simulate_yt_comments(channel=ch_name)
                all_rows.extend(df_sim.to_dict("records"))
                simulated.append(ch_name)
            yt = None

        if yt is not None:
            for ch_name, ch_id in YOUTUBE_CHANNELS.items():
                logger.info("  Chaine : %s (ID=%s)", ch_name, ch_id)
                try:
                    # Verifier / trouver l'ID de chaine
                    videos = _get_channel_videos(
                        yt, ch_id, YT_KEYWORDS, since_year=2018
                    )
                    if not videos:
                        logger.info(
                            "  ID invalide pour %s — recherche par nom ...", ch_name
                        )
                        found_id = _search_channel_id(yt, ch_name)
                        if found_id:
                            videos = _get_channel_videos(
                                yt, found_id, YT_KEYWORDS, since_year=2018
                            )
                        else:
                            logger.info(
                                "  Chaine %s introuvable — simulation.", ch_name
                            )

                    if not videos:
                        raise ValueError("Aucune video trouvee")

                    logger.info(
                        "  %s : %d videos trouvees.", ch_name, len(videos)
                    )
                    _log_source(
                        f"  {ch_name} : {len(videos)} videos (reel)"
                    )

                    for vid in videos[:50]:   # limiter le quota
                        pub_date = pd.to_datetime(
                            vid["published_at"], errors="coerce"
                        )
                        if pd.isna(pub_date):
                            continue
                        pub_month = pub_date.to_period("M").to_timestamp()

                        comments = _get_video_comments(
                            yt, vid["video_id"],
                            max_comments=CONFIG["MAX_COMMENTS_VIDEO"],
                        )
                        if not comments:
                            comments = [""]

                        for cmt in comments:
                            all_rows.append({
                                "date":        pub_month,
                                "channel":     ch_name,
                                "video_title": vid["title"][:200],
                                "views":       vid["views"],
                                "likes":       vid["likes"],
                                "comment_text": cmt,
                                "video_id":    vid["video_id"],
                            })
                        time.sleep(0.5)

                except Exception as exc:
                    logger.info(
                        "  [%s] Erreur API : %s — simulation.", ch_name, exc
                    )
                    _log_source(
                        f"  {ch_name} : simule (erreur API : {exc})"
                    )
                    df_sim = _simulate_yt_comments(channel=ch_name)
                    all_rows.extend(df_sim.to_dict("records"))
                    simulated.append(ch_name)

    if not all_rows:
        logger.warning("MODULE 2 — Aucune donnee. Fallback simulation globale.")
        for ch_name in YOUTUBE_CHANNELS:
            all_rows.extend(
                _simulate_yt_comments(channel=ch_name).to_dict("records")
            )

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    for col in ["stress_score", "keyword_score", "intensity_score", "engagement_weight"]:
        if col not in df.columns:
            df[col] = 0.0
    for col in ["views", "likes"]:
        if col not in df.columns:
            df[col] = 0

    df.to_csv(cache, index=False, encoding="utf-8-sig")
    logger.info("MODULE 2 — Sauvegarde : %s  (%d lignes)", cache, len(df))

    if simulated:
        _log_source(f"  Chaines simulees : {simulated}")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — SCORING NLP DARIJA / ARABE / FRANCAIS
# ═══════════════════════════════════════════════════════════════════════════════

def score_comment(
    text: str,
    likes: int = 0,
) -> dict[str, float]:
    """
    Calcule le score de stress economique d'un commentaire.

    Formule
    -------
    a) keyword_score   = mots_stress / max(nb_mots, 1)
    b) intensity_score = emojis_negatifs*0.3 + majuscules>30%*0.2
                         + repetitions*0.1 - mots_positifs*0.2
    c) engagement_weight = log(1 + likes)  [ou 1.0 si likes=0]
    d) stress_score = (0.6*keyword + 0.4*intensity) * engagement
                      -> normalise 0-1 (clip)

    Parametres
    ----------
    text  : texte du commentaire (FR / AR / Darija)
    likes : nombre de likes (pour la ponderation engagement)

    Retourne
    --------
    dict {keyword_score, intensity_score, engagement_weight, stress_score_raw}
    """
    if not text or text == "[simule]":
        return {
            "keyword_score":    0.0,
            "intensity_score":  0.0,
            "engagement_weight": float(np.log1p(likes)) if likes > 0 else 1.0,
            "stress_score_raw": 0.0,
        }

    text_lower = text.lower()
    words      = text_lower.split()
    n_words    = max(len(words), 1)

    # a) Keyword score
    stress_count   = 0
    positive_count = 0

    for category, kw_list in STRESS_KEYWORDS.items():
        for kw in kw_list:
            if kw.lower() in text_lower:
                if category == "positif":
                    positive_count += 1
                else:
                    stress_count += 1

    keyword_score = min(stress_count / n_words, 1.0)

    # b) Intensity score
    # Emojis negatifs
    emoji_hits   = sum(text.count(e) for e in _NEG_EMOJIS)
    emoji_score  = min(emoji_hits * 0.3, 0.9)

    # Majuscules > 30 % du texte non-vide
    non_space    = [c for c in text if not c.isspace()]
    caps_ratio   = sum(1 for c in non_space if c.isupper()) / max(len(non_space), 1)
    caps_score   = 0.2 if caps_ratio > 0.30 else 0.0

    # Repetitions de mots (au moins 1 mot repete)
    rep_score    = 0.1 if len(words) > len(set(words)) else 0.0

    # Mots positifs (reduisent le score)
    pos_score    = -0.2 * min(positive_count, 1)

    intensity_score = max(
        float(emoji_score + caps_score + rep_score + pos_score),
        0.0,
    )

    # c) Engagement weight
    engagement_weight = float(np.log1p(likes)) if likes > 0 else 1.0

    # d) Score brut
    raw = (0.6 * keyword_score + 0.4 * intensity_score) * engagement_weight

    return {
        "keyword_score":     round(keyword_score,   4),
        "intensity_score":   round(intensity_score, 4),
        "engagement_weight": round(engagement_weight, 4),
        "stress_score_raw":  round(raw, 6),
    }


def score_dataframe(df: pd.DataFrame, text_col: str = "comment_text",
                    likes_col: Optional[str] = "likes") -> pd.DataFrame:
    """
    MODULE 3 — Applique score_comment() a tout un DataFrame de commentaires.

    Colonnes ajoutees : keyword_score, intensity_score, engagement_weight,
                        stress_score_raw, stress_score (normalise 0-1).

    Parametres
    ----------
    df        : DataFrame avec une colonne texte et optionnellement une colonne likes
    text_col  : nom de la colonne texte (defaut : 'comment_text')
    likes_col : nom de la colonne likes (None si indisponible)

    Retourne
    --------
    DataFrame enrichi avec les scores NLP
    """
    logger.info("MODULE 3 — Scoring NLP sur %d commentaires ...", len(df))

    results = []
    for _, row in df.iterrows():
        txt   = str(row.get(text_col, ""))
        likes = int(row.get(likes_col, 0)) if likes_col and likes_col in df.columns else 0
        results.append(score_comment(txt, likes))

    scores_df = pd.DataFrame(results)

    df = df.copy()
    df["keyword_score"]    = scores_df["keyword_score"].values
    df["intensity_score"]  = scores_df["intensity_score"].values
    df["engagement_weight"] = scores_df["engagement_weight"].values

    raw = scores_df["stress_score_raw"].values.astype(float)

    # Normalisation 0-1 sur le score brut (clip sur [0, 1])
    r_min, r_max = raw.min(), raw.max()
    if r_max > r_min:
        stress_norm = (raw - r_min) / (r_max - r_min)
    else:
        stress_norm = np.zeros_like(raw)

    # Pour les commentaires simules, le champ stress_score est deja rempli
    # -> on le garde tel quel si score_raw == 0 (texte "[simule]")
    if "stress_score" in df.columns:
        existing = df["stress_score"].values.astype(float)
        mask_real = raw > 0
        existing[mask_real] = stress_norm[mask_real]
        df["stress_score"] = existing
    else:
        df["stress_score"] = stress_norm

    logger.info("MODULE 3 — Score moyen : %.3f  |  std : %.3f",
                df["stress_score"].mean(), df["stress_score"].std())
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — AGREGATION MENSUELLE
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_monthly_nlp(
    df_media: pd.DataFrame,
    df_youtube: pd.DataFrame,
) -> pd.DataFrame:
    """
    MODULE 4 — Agregation mensuelle des signaux NLP.

    Formule de ponderation
    ----------------------
    Pour chaque mois m :
      weighted_stress(m) = sum(stress_score * engagement_weight) /
                           sum(engagement_weight)

    Le signal final morocco_nlp_signal est la moyenne ponderee de toutes
    les sources (medias + YouTube), normalisee 0-1.

    Colonnes produites
    ------------------
    morocco_nlp_signal, media_signal, youtube_signal,
    total_comments, total_articles, total_videos

    Sauvegarde : data/processed/morocco_nlp_monthly.csv
    """
    logger.info("MODULE 4 — Agregation mensuelle NLP ...")

    def _weighted_agg(df: pd.DataFrame, label: str) -> pd.DataFrame:
        """Aggregation mensuelle ponderee pour un DataFrame de commentaires."""
        if df.empty:
            return pd.DataFrame()

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()

        # Ponderation : stress * engagement
        df["ws"] = df["stress_score"] * df["engagement_weight"]

        grp = df.groupby("date").agg(
            total_ws      = ("ws",               "sum"),
            total_weight  = ("engagement_weight","sum"),
            n_comments    = ("comment_text",     "count"),
        )
        grp[f"{label}_signal"] = (
            grp["total_ws"] / grp["total_weight"].replace(0, 1)
        )
        return grp[[f"{label}_signal", "n_comments"]]

    agg_media = _weighted_agg(df_media,   "media")
    agg_yt    = _weighted_agg(df_youtube, "youtube")

    # Jointure sur l'index de dates
    monthly = pd.DataFrame(index=pd.date_range(
        CONFIG["SIM_START"], periods=84, freq="MS"
    ))
    monthly.index.name = "date"

    if not agg_media.empty:
        monthly = monthly.join(agg_media, how="left")
    if not agg_yt.empty:
        monthly = monthly.join(
            agg_yt.rename(columns={"n_comments": "n_yt_comments"}),
            how="left"
        )

    # Colonnes par defaut si absentes
    for col in ["media_signal", "youtube_signal"]:
        if col not in monthly.columns:
            monthly[col] = np.nan
    for col in ["n_comments", "n_yt_comments"]:
        if col not in monthly.columns:
            monthly[col] = 0

    monthly = monthly.fillna(0.0)

    # Signal NLP global = moyenne des deux sources (si les deux disponibles)
    has_media = (monthly["media_signal"]   > 0).any()
    has_yt    = (monthly["youtube_signal"] > 0).any()

    if has_media and has_yt:
        monthly["morocco_nlp_signal"] = (
            0.60 * monthly["media_signal"] + 0.40 * monthly["youtube_signal"]
        )
    elif has_media:
        monthly["morocco_nlp_signal"] = monthly["media_signal"]
    elif has_yt:
        monthly["morocco_nlp_signal"] = monthly["youtube_signal"]
    else:
        monthly["morocco_nlp_signal"] = 0.0

    # Normalisation 0-1
    monthly["media_signal"]        = _normalise_0_1(monthly["media_signal"])
    monthly["youtube_signal"]      = _normalise_0_1(monthly["youtube_signal"])
    monthly["morocco_nlp_signal"]  = _normalise_0_1(monthly["morocco_nlp_signal"])

    monthly["total_comments"] = (
        monthly.get("n_comments", 0) + monthly.get("n_yt_comments", 0)
    )

    path = PROC_DIR / "morocco_nlp_monthly.csv"
    monthly.to_csv(path, encoding="utf-8-sig")
    logger.info("MODULE 4 — Sauvegarde : %s  (%d mois)", path, len(monthly))

    logger.info(
        "MODULE 4 — NLP signal : mean=%.3f  std=%.3f  max=%.3f",
        monthly["morocco_nlp_signal"].mean(),
        monthly["morocco_nlp_signal"].std(),
        monthly["morocco_nlp_signal"].max(),
    )
    return monthly


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 5 — BESI ENRICHI
# ═══════════════════════════════════════════════════════════════════════════════

def compute_besi_enrichi(
    monthly_nlp: Optional[pd.DataFrame] = None,
    min_months: int = 12,
) -> pd.DataFrame:
    """
    MODULE 5 — Calcule BESI_enrichi et l'ajoute au master_dataset.csv.

    Formule (si NLP disponible sur > min_months mois)
    -------------------------------------------------
    BESI_enrichi = 0.35 * google_composite
                 + 0.25 * morocco_nlp_signal
                 + 0.20 * youtube_composite
                 + 0.10 * reddit_composite
                 + 0.10 * ipc_change

    Sinon : BESI_enrichi = BESI original (pas de changement).

    Parametres
    ----------
    monthly_nlp : DataFrame produit par aggregate_monthly_nlp()
                  Si None, tente de lire morocco_nlp_monthly.csv
    min_months  : seuil minimum de mois NLP pour activer la formule enrichie

    Retourne
    --------
    master_df mis a jour (avec colonne besi_enrichi)
    """
    logger.info("MODULE 5 — Calcul BESI enrichi ...")

    # Charger le master dataset
    master_path = PROC_DIR / "master_dataset.csv"
    if not master_path.exists():
        raise FileNotFoundError(
            f"{master_path} introuvable — lancer data_pipeline.py d'abord."
        )
    master = pd.read_csv(master_path, index_col=0, parse_dates=True)
    try:
        master.index = pd.DatetimeIndex(master.index, freq="MS")
    except Exception:
        master.index = pd.DatetimeIndex(master.index)
        master = master.asfreq("MS")

    # Charger le signal NLP
    if monthly_nlp is None:
        nlp_path = PROC_DIR / "morocco_nlp_monthly.csv"
        if nlp_path.exists():
            monthly_nlp = pd.read_csv(
                nlp_path, index_col=0, parse_dates=True
            )
        else:
            logger.warning(
                "MODULE 5 — morocco_nlp_monthly.csv absent. BESI_enrichi = BESI."
            )
            master["besi_enrichi"] = master.get("besi", 0.0)
            master.to_csv(master_path, encoding="utf-8-sig")
            return master

    # Aligner sur l'index du master
    nlp_aligned = monthly_nlp.reindex(master.index)

    nlp_col = "morocco_nlp_signal"
    n_nlp   = int((nlp_aligned[nlp_col] > 0).sum()) if nlp_col in nlp_aligned.columns \
              else 0

    if n_nlp >= min_months:
        logger.info(
            "MODULE 5 — NLP disponible (%d mois >= %d). Formule enrichie activee.",
            n_nlp, min_months,
        )
        # Recuperer les composantes du master (avec fallback 0)
        def _get_col(df, col):
            s = df.get(col, pd.Series(0.0, index=df.index))
            return s.fillna(0.0)

        g  = _get_col(master, "trends_composite")
        n  = nlp_aligned[nlp_col].fillna(0.0)
        y  = _get_col(master, "youtube_composite")
        r  = _get_col(master, "reddit_composite")
        ic = _get_col(master, "ipc_change").abs()
        # Renormaliser ipc_change 0-1
        ic_min, ic_max = ic.min(), ic.max()
        ic_n = (ic - ic_min) / (ic_max - ic_min) if ic_max > ic_min else ic * 0.0

        besi_enrichi = (
            0.35 * g
            + 0.25 * n
            + 0.20 * y
            + 0.10 * r
            + 0.10 * ic_n
        )
        # Normalisation finale 0-1
        master["besi_enrichi"] = _normalise_0_1(besi_enrichi).values

        logger.info(
            "MODULE 5 — BESI_enrichi : mean=%.3f  std=%.3f",
            master["besi_enrichi"].mean(),
            master["besi_enrichi"].std(),
        )
    else:
        logger.info(
            "MODULE 5 — NLP insuffisant (%d mois < %d). BESI_enrichi = BESI.",
            n_nlp, min_months,
        )
        master["besi_enrichi"] = master.get("besi", pd.Series(0.0, index=master.index))

    master.to_csv(master_path, encoding="utf-8-sig")
    logger.info("MODULE 5 — master_dataset.csv mis a jour avec besi_enrichi.")
    return master


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 6 — VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_nlp_vs_ipc(
    monthly_nlp: Optional[pd.DataFrame] = None,
    master_df: Optional[pd.DataFrame] = None,
    dpi: int = 300,
    save: bool = True,
) -> plt.Figure:
    """
    MODULE 6 — Graphique double-axe : signaux NLP vs IPC YoY.

    Contenu
    -------
    - Axe gauche  : morocco_nlp_signal (bleu), media_signal (orange),
                    youtube_signal (vert) — signaux comportementaux
    - Axe droit   : IPC YoY % (rouge pointille) — variable cible
    - Zones de couleur : stress Normal / Warning / High Stress
    - Ligne rouge verticale : janvier 2022 (rupture structurelle)
    - Titre : "Signaux NLP Medias Marocains vs Inflation — 2018-2026"

    Sauvegarde : outputs/figures/morocco_nlp_vs_ipc.png (300 DPI)
    """
    logger.info("MODULE 6 — Graphique NLP vs IPC ...")

    # Charger les donnees si non fournies
    if monthly_nlp is None:
        nlp_path = PROC_DIR / "morocco_nlp_monthly.csv"
        if nlp_path.exists():
            monthly_nlp = pd.read_csv(nlp_path, index_col=0, parse_dates=True)
        else:
            logger.warning("MODULE 6 — morocco_nlp_monthly.csv absent.")
            monthly_nlp = pd.DataFrame()

    if master_df is None:
        master_path = PROC_DIR / "master_dataset.csv"
        if master_path.exists():
            master_df = pd.read_csv(master_path, index_col=0, parse_dates=True)
        else:
            master_df = pd.DataFrame()

    # ── Style academique ──────────────────────────────────────────────────────
    _STYLE = {
        "font.family":      "DejaVu Serif",
        "font.size":        10,
        "axes.labelsize":   10,
        "axes.titlesize":   11,
        "xtick.labelsize":  9,
        "ytick.labelsize":  9,
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.grid":        True,
        "grid.alpha":       0.3,
        "grid.linestyle":   "--",
    }
    plt.rcParams.update(_STYLE)

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()

    colors = {
        "nlp":     "#2C5F8A",   # bleu principal
        "media":   "#E07B39",   # orange medias
        "youtube": "#2CA02C",   # vert YouTube
        "ipc":     "#D62728",   # rouge IPC
    }

    # ── Zones de stress (axe gauche 0-1) ─────────────────────────────────────
    ax1.axhspan(0.00, 0.35, alpha=0.06, color="#2CA02C", label="_nolegend_")
    ax1.axhspan(0.35, 0.65, alpha=0.06, color="#FFA500", label="_nolegend_")
    ax1.axhspan(0.65, 1.00, alpha=0.06, color="#D62728", label="_nolegend_")

    # Annotations des zones
    for y_mid, txt in [(0.175, "Normal"), (0.50, "Warning"), (0.825, "High Stress")]:
        ax1.text(
            pd.Timestamp("2018-03-01"), y_mid, txt,
            fontsize=7, color="gray", va="center", alpha=0.7,
        )

    # ── Signaux NLP ───────────────────────────────────────────────────────────
    if not monthly_nlp.empty and "morocco_nlp_signal" in monthly_nlp.columns:
        nlp_idx = monthly_nlp.index
        ax1.plot(
            nlp_idx, monthly_nlp["morocco_nlp_signal"],
            color=colors["nlp"], lw=2.2, label="Signal NLP global (medias+YT)",
            zorder=4,
        )
        if "media_signal" in monthly_nlp.columns:
            ax1.plot(
                nlp_idx, monthly_nlp["media_signal"],
                color=colors["media"], lw=1.3, ls="--", alpha=0.8,
                label="Medias (Hespress, le360, h24, alyaoum24)", zorder=3,
            )
        if "youtube_signal" in monthly_nlp.columns:
            ax1.plot(
                nlp_idx, monthly_nlp["youtube_signal"],
                color=colors["youtube"], lw=1.3, ls=":", alpha=0.8,
                label="YouTube (2M, Medi1, Hespress TV, Medias24)", zorder=3,
            )
    else:
        ax1.text(
            0.5, 0.5, "Donnees NLP non disponibles\n(lancer scrape_media_comments())",
            ha="center", va="center", transform=ax1.transAxes,
            fontsize=10, color="gray",
        )

    # ── BESI enrichi si disponible ────────────────────────────────────────────
    if master_df is not None and not master_df.empty \
            and "besi_enrichi" in master_df.columns:
        ax1.fill_between(
            master_df.index, master_df["besi_enrichi"],
            alpha=0.12, color=colors["nlp"], label="BESI enrichi (aire)",
        )

    # ── IPC YoY sur axe droit ─────────────────────────────────────────────────
    if master_df is not None and not master_df.empty \
            and "ipc_yoy" in master_df.columns:
        ipc_yoy = master_df["ipc_yoy"].dropna() * 100
        # Filtrer sur la meme plage que le NLP
        if not monthly_nlp.empty:
            ipc_yoy = ipc_yoy.loc[
                ipc_yoy.index >= pd.Timestamp(CONFIG["SIM_START"])
            ]
        ax2.plot(
            ipc_yoy.index, ipc_yoy.values,
            color=colors["ipc"], lw=1.8, ls="-.",
            label="IPC YoY (%)", zorder=5,
        )
        ax2.axhline(2.0, color=colors["ipc"], lw=0.7, ls=":", alpha=0.5)
        ax2.set_ylabel("IPC Variation annuelle (%)", color=colors["ipc"], fontsize=10)
        ax2.tick_params(axis="y", labelcolor=colors["ipc"])
        ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))

    # ── Ligne rupture 2022 ────────────────────────────────────────────────────
    ax1.axvline(
        pd.Timestamp("2022-01-01"),
        color="red", lw=1.5, ls="--", alpha=0.85, zorder=6,
        label="Rupture 2022",
    )
    ax1.text(
        pd.Timestamp("2022-02-01"), 0.94,
        "Choc\ninflationniste", fontsize=7.5, color="red", va="top",
    )

    # ── Mise en forme ─────────────────────────────────────────────────────────
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Signal NLP (normalise 0-1)", fontsize=10)
    ax1.set_xlabel("Date", fontsize=10)

    ax1.set_title(
        "Signaux NLP Medias Marocains vs Inflation — 2018-2026\n"
        "Sources : Hespress, le360, h24info, alyaoum24, YouTube (2M, Medi1TV, ...)",
        fontsize=11, fontweight="bold",
    )

    # Legende combinee
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, labs1 + labs2,
        fontsize=8, loc="upper left", ncol=2,
        framealpha=0.9,
    )

    # Supprimer les bordures superieures/droites gauche
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    plt.tight_layout()

    if save:
        out = FIG_DIR / "morocco_nlp_vs_ipc.png"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        logger.info("MODULE 6 — Figure sauvegardee : %s", out)

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE COMPLET
# ═══════════════════════════════════════════════════════════════════════════════

def run_nlp_pipeline(
    force_refresh: bool = False,
    dpi: int = 300,
) -> dict:
    """
    Lance tous les modules NLP en sequence.

    Ordre d'execution
    -----------------
    1. scrape_media_comments()       -> data/raw/media_comments.csv
    2. fetch_youtube_comments()      -> data/raw/youtube_comments.csv
    3. score_dataframe() sur medias  -> scores NLP medias
    4. score_dataframe() sur YouTube -> scores NLP YouTube
    5. aggregate_monthly_nlp()       -> data/processed/morocco_nlp_monthly.csv
    6. compute_besi_enrichi()        -> master_dataset.csv mis a jour
    7. plot_nlp_vs_ipc()             -> outputs/figures/morocco_nlp_vs_ipc.png

    Parametres
    ----------
    force_refresh : forcer le re-scraping / re-download meme si cache present
    dpi           : resolution des figures (defaut 300)

    Retourne
    --------
    dict {monthly_nlp, master_df, fig} — pour utilisation en notebook
    """
    t0 = time.time()

    _log_source(
        f"\n{'='*60}\n"
        f"Pipeline NLP Maroc — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"{'='*60}",
        mode="a",
    )

    print("\n" + "="*60)
    print("  PIPELINE NLP MAROC")
    print("="*60)

    # --- 1 + 2 : Collecte des donnees ---
    print("\n[1/7] Scraping medias marocains ...")
    df_media = scrape_media_comments(force_refresh=force_refresh)
    print(f"      -> {len(df_media)} lignes")

    print("\n[2/7] Commentaires YouTube ...")
    df_youtube = fetch_youtube_comments(force_refresh=force_refresh)
    print(f"      -> {len(df_youtube)} lignes")

    # --- 3 + 4 : Scoring NLP ---
    print("\n[3/7] Scoring NLP medias ...")
    df_media = score_dataframe(df_media, text_col="comment_text", likes_col=None)
    print(f"      -> Score moyen : {df_media['stress_score'].mean():.3f}")

    print("\n[4/7] Scoring NLP YouTube ...")
    df_youtube = score_dataframe(df_youtube, text_col="comment_text", likes_col="likes")
    print(f"      -> Score moyen : {df_youtube['stress_score'].mean():.3f}")

    # Sauvegarder les versions scorees
    df_media.to_csv(RAW_DIR / "media_comments_scored.csv",
                    index=False, encoding="utf-8-sig")
    df_youtube.to_csv(RAW_DIR / "youtube_comments_scored.csv",
                      index=False, encoding="utf-8-sig")

    # --- 5 : Agregation mensuelle ---
    print("\n[5/7] Agregation mensuelle NLP ...")
    monthly_nlp = aggregate_monthly_nlp(df_media, df_youtube)
    print(f"      -> {len(monthly_nlp)} mois")

    # --- 6 : BESI enrichi ---
    print("\n[6/7] Calcul BESI enrichi ...")
    master_df = compute_besi_enrichi(monthly_nlp)
    has_enrichi = "besi_enrichi" in master_df.columns
    if has_enrichi:
        print(f"      -> BESI_enrichi : mean={master_df['besi_enrichi'].mean():.3f}")
    else:
        print("      -> BESI original conserve")

    # --- 7 : Visualisation ---
    print("\n[7/7] Visualisation NLP vs IPC ...")
    fig = plot_nlp_vs_ipc(monthly_nlp, master_df, dpi=dpi, save=True)
    plt.close(fig)
    print("      -> outputs/figures/morocco_nlp_vs_ipc.png")

    elapsed = time.time() - t0
    print(f"\n  Pipeline termine en {elapsed:.1f}s")
    print("="*60)

    _log_source(f"\nPipeline complete en {elapsed:.1f}s")

    return {
        "monthly_nlp": monthly_nlp,
        "master_df":   master_df,
        "fig":         fig,
        "df_media":    df_media,
        "df_youtube":  df_youtube,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTREE — TEST INDEPENDANT PAR MODULE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pipeline NLP Maroc — BESI enrichi avec medias marocains"
    )
    parser.add_argument(
        "--module", type=int, default=0,
        help=(
            "0 = pipeline complet (defaut)\n"
            "1 = scraping medias seulement\n"
            "2 = YouTube seulement\n"
            "3 = scoring NLP seulement (sur media_comments.csv)\n"
            "4 = agregation mensuelle (necessite modules 1+2+3)\n"
            "5 = BESI enrichi seulement\n"
            "6 = visualisation seulement"
        ),
    )
    parser.add_argument("--refresh", action="store_true",
                        help="Forcer le re-scraping meme si cache present")
    args = parser.parse_args()

    print("\n  BESI Maroc | nlp_morocco.py")
    print(f"  Module selectionne : {args.module or 'COMPLET'}")
    print(f"  Force refresh      : {args.refresh}")

    if args.module == 0:
        # Pipeline complet
        results = run_nlp_pipeline(force_refresh=args.refresh)

    elif args.module == 1:
        print("\n--- MODULE 1 : Scraping medias ---")
        df = scrape_media_comments(force_refresh=args.refresh)
        print(f"Resultat : {len(df)} lignes")
        print(df[["date", "source", "article_title"]].head(5).to_string())

    elif args.module == 2:
        print("\n--- MODULE 2 : YouTube commentaires ---")
        df = fetch_youtube_comments(force_refresh=args.refresh)
        print(f"Resultat : {len(df)} lignes")
        print(df[["date", "channel", "video_title"]].head(5).to_string())

    elif args.module == 3:
        print("\n--- MODULE 3 : Scoring NLP ---")
        cache = RAW_DIR / "media_comments.csv"
        if cache.exists():
            df = pd.read_csv(cache, parse_dates=["date"])
        else:
            print("  media_comments.csv absent — creation depuis simulation ...")
            df = _simulate_comments("test", n_months=12)
            # Injecter quelques vrais textes pour tester
            test_texts = [
                "ghali bzaf les prix makaynach flouss",
                "hamdullah bahi",
                "hshuma 3la had lgouvernoment ghla lprix",
                "ma b9ach nchouf chwiya  PRIX TGHLAAAA",
                "mzyan had lkhabar bahi",
            ]
            test_likes = [15, 3, 42, 88, 5]
            for i, (txt, lk) in enumerate(zip(test_texts, test_likes)):
                res = score_comment(txt, lk)
                print(
                    f"  [{i+1}] text='{txt[:45]}...'\n"
                    f"       kw={res['keyword_score']:.3f}  "
                    f"int={res['intensity_score']:.3f}  "
                    f"eng={res['engagement_weight']:.3f}  "
                    f"raw={res['stress_score_raw']:.4f}"
                )

        df_scored = score_dataframe(df)
        print(f"\nScore moyen  : {df_scored['stress_score'].mean():.4f}")
        print(f"Score median : {df_scored['stress_score'].median():.4f}")
        print(f"Score std    : {df_scored['stress_score'].std():.4f}")

    elif args.module == 4:
        print("\n--- MODULE 4 : Agregation mensuelle ---")
        c_media = RAW_DIR / "media_comments_scored.csv"
        c_yt    = RAW_DIR / "youtube_comments_scored.csv"
        df_m = pd.read_csv(c_media,  parse_dates=["date"]) if c_media.exists() \
               else _simulate_comments("hespress")
        df_y = pd.read_csv(c_yt,     parse_dates=["date"]) if c_yt.exists()    \
               else _simulate_yt_comments("2M Maroc")
        if "stress_score" not in df_m.columns:
            df_m = score_dataframe(df_m)
        if "stress_score" not in df_y.columns:
            df_y = score_dataframe(df_y)
        monthly = aggregate_monthly_nlp(df_m, df_y)
        print(monthly[["morocco_nlp_signal", "media_signal", "youtube_signal"]].head(10))

    elif args.module == 5:
        print("\n--- MODULE 5 : BESI enrichi ---")
        master = compute_besi_enrichi()
        cols = [c for c in ["besi", "besi_trends", "besi_enrichi"] if c in master.columns]
        print(master[cols].describe())

    elif args.module == 6:
        print("\n--- MODULE 6 : Visualisation ---")
        fig = plot_nlp_vs_ipc(dpi=150)
        plt.close(fig)
        print("Figure sauvegardee : outputs/figures/morocco_nlp_vs_ipc.png")

    else:
        print(f"Module {args.module} inconnu. Utilisez --module 0..6")
