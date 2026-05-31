"""
src/ingestion/moroccan_press.py — Signal presse marocaine via flux RSS

JUSTIFICATION (remplacement de Reddit/YouTube) :
    Reddit   → API refusée + utilisateurs non représentatifs des ménages marocains
    YouTube  → Quota API insuffisant + signal biaisé par algorithme recommandation
    PRESSE   → Flux RSS publics + couverture nationale + articles datés précisément
               + reflet de l'agenda économique marocain (pas de biais urbain/jeune)

SOURCES :
    Hespress   (ar/fr) : premier site d'info marocain en volume
    Le360      (fr)    : économie et politique marocaine
    Medias24   (fr)    : spécialité économie et finance
    L'Economiste (fr)  : presse économique de référence

SIGNAL PRODUIT :
    press_economic_volume : nombre d'articles économiques par mois
    (normalisé 0-1 sur la période 2017-2024)

LIMITE DOCUMENTÉE :
    Les flux RSS ne donnent que les 30-90 derniers jours.
    Pour l'historique, on utilise le Common Crawl index (si disponible)
    ou on restreint la période de validation à la disponibilité réelle.
    Cette limite est documentée explicitement dans le rapport.

Output :
    data/bronze/moroccan_press_raw.csv
    data/silver/press_signal_monthly.csv
"""

import logging
import time
import warnings
import re
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

ROOT       = Path(__file__).resolve().parent.parent.parent
BRONZE_DIR = ROOT / "data" / "bronze"
SILVER_DIR = ROOT / "data" / "silver"
BRONZE_DIR.mkdir(parents=True, exist_ok=True)
SILVER_DIR.mkdir(parents=True, exist_ok=True)

# ── Sources RSS ───────────────────────────────────────────────────────────────
RSS_SOURCES = {
    "hespress_fr":   "https://fr.hespress.com/feed",
    "hespress_ar":   "https://hespress.com/feed",
    "le360":         "https://fr.le360.ma/rss.xml",
    "medias24":      "https://medias24.com/feed",
    "leconomiste":   "https://www.leconomiste.com/rss.xml",
}

# Mots-clés économiques pour filtrer les articles pertinents
# Organisés par thème pour transparence méthodologique
KEYWORDS_ECO = {
    "prix":         ["prix", "tarif", "coût", "tarification", "cher", "cherté",
                     "hausse", "augmentation", "flambée", "renchérissement",
                     "أسعار", "ارتفاع", "غلاء"],
    "inflation":    ["inflation", "deflation", "IPC", "indice des prix",
                     "pouvoir d'achat", "التضخم", "القوة الشرائية"],
    "alimentaire":  ["huile", "farine", "sucre", "légumes", "viande", "lait",
                     "produits alimentaires", "panier", "ONSSA",
                     "الزيت", "الدقيق", "السكر", "الخضر"],
    "énergie":      ["carburant", "essence", "gasoil", "gaz", "électricité",
                     "ONEE", "الوقود", "الغاز", "البنزين"],
    "politique_eco": ["subvention", "compensation", "caisse compensation",
                      "HCP", "BAM", "Bank Al-Maghrib", "taux directeur",
                      "budget", "دعم", "صندوق المقاصة"],
}

# Tous les mots-clés en une liste plate pour le filtrage rapide
ALL_KEYWORDS = [kw for kws in KEYWORDS_ECO.values() for kw in kws]


def _is_economic_article(text: str) -> bool:
    """
    Détermine si un article est économiquement pertinent.
    Retourne True si au moins un mot-clé économique est présent.
    """
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in ALL_KEYWORDS)


def _classify_article_themes(text: str) -> List[str]:
    """
    Identifie les thèmes économiques présents dans un article.
    Utile pour une analyse thématique plus fine.
    """
    text_lower = text.lower()
    return [
        theme for theme, kws in KEYWORDS_ECO.items()
        if any(kw.lower() in text_lower for kw in kws)
    ]


def _parse_date(entry) -> Optional[datetime]:
    """
    Extrait la date de publication d'une entrée RSS.
    Gère les différents formats de date RSS.
    """
    for attr in ["published_parsed", "updated_parsed", "created_parsed"]:
        val = getattr(entry, attr, None)
        if val:
            try:
                return datetime(*val[:6], tzinfo=timezone.utc)
            except Exception:
                pass
    # Fallback : parser le champ published comme string
    for attr in ["published", "updated"]:
        val = getattr(entry, attr, None)
        if val:
            try:
                return pd.to_datetime(val, utc=True).to_pydatetime()
            except Exception:
                pass
    return None


def fetch_rss_feed(source_name: str, url: str) -> pd.DataFrame:
    """
    Télécharge et parse un flux RSS.
    Retourne un DataFrame avec les articles économiques.
    """
    try:
        import feedparser
    except ImportError:
        logger.error("feedparser non installé — pip install feedparser")
        return pd.DataFrame()

    try:
        logger.info(f"  Téléchargement RSS : {source_name} ({url})")
        feed = feedparser.parse(url)

        if feed.bozo:
            logger.warning(f"  RSS mal formé pour {source_name} : {feed.bozo_exception}")

        records = []
        for entry in feed.entries:
            # Extraire le texte (titre + résumé)
            title   = getattr(entry, "title", "")
            summary = getattr(entry, "summary", "")
            text    = f"{title} {summary}"

            # Nettoyer le HTML
            text = re.sub(r"<[^>]+>", " ", text)

            # Filtrer sur les mots-clés économiques
            if not _is_economic_article(text):
                continue

            date = _parse_date(entry)
            if date is None:
                continue

            themes = _classify_article_themes(text)

            records.append({
                "date":    date,
                "source":  source_name,
                "title":   title[:200],
                "themes":  "|".join(themes),
                "n_themes": len(themes),
            })

        logger.info(f"    -> {len(records)} articles économiques trouvés")
        return pd.DataFrame(records)

    except Exception as e:
        logger.warning(f"  Erreur pour {source_name} : {e}")
        return pd.DataFrame()


def ingest_moroccan_press(
    out_bronze: Path = None,
    out_silver: Path = None,
) -> pd.DataFrame:
    """
    Pipeline principal : collecte les articles de presse marocaine,
    filtre sur les thèmes économiques, agrège en fréquence mensuelle.

    IMPORTANT — limite documentée :
    Les flux RSS couvrent seulement les 30-90 derniers jours.
    Le signal historique 2017-2024 n'est PAS disponible via RSS.
    On utilise ce signal pour :
    (a) valider que nos Trends correlent bien avec les pics de presse récents
    (b) enrichir le BESI pour les mois disponibles
    Cette limite est explicitement documentée dans les rapports.

    Retourne un DataFrame mensuel avec le volume d'articles économiques.
    """
    if out_bronze is None:
        out_bronze = BRONZE_DIR / "moroccan_press_raw.csv"
    if out_silver is None:
        out_silver = SILVER_DIR / "press_signal_monthly.csv"

    # Collecte de toutes les sources
    all_articles = []
    for source_name, url in RSS_SOURCES.items():
        df = fetch_rss_feed(source_name, url)
        if not df.empty:
            all_articles.append(df)
        time.sleep(2)

    if not all_articles:
        logger.error("Aucun article collecté — vérifier la connexion et les URLs RSS")
        return pd.DataFrame()

    # Consolider
    df_all = pd.concat(all_articles, ignore_index=True)
    df_all["date"] = pd.to_datetime(df_all["date"], utc=True)
    df_all.sort_values("date", inplace=True)

    # Sauvegarder le bronze brut
    df_all.to_csv(out_bronze, index=False, encoding="utf-8")
    logger.info(f"  Bronze sauvegardé : {len(df_all)} articles -> {out_bronze.name}")

    # Agréger par mois
    df_all["month"] = df_all["date"].dt.to_period("M").dt.to_timestamp()

    # Volume total d'articles économiques par mois
    monthly = df_all.groupby("month").agg(
        press_volume   = ("title", "count"),
        n_sources      = ("source", "nunique"),
        themes_prix    = ("themes", lambda x: x.str.contains("prix").sum()),
        themes_inflation = ("themes", lambda x: x.str.contains("inflation").sum()),
        themes_alim    = ("themes", lambda x: x.str.contains("alimentaire").sum()),
    ).reset_index()

    monthly = monthly.set_index("month")
    monthly.index.name = "month"

    # Normalisation 0-1 pour intégration dans BESI
    for col in ["press_volume", "themes_prix", "themes_inflation", "themes_alim"]:
        mn, mx = monthly[col].min(), monthly[col].max()
        if mx > mn:
            monthly[f"{col}_norm"] = (monthly[col] - mn) / (mx - mn)
        else:
            monthly[f"{col}_norm"] = 0.0

    # Index composite presse (moyenne des thèmes normalisés)
    monthly["press_index"] = monthly[
        ["themes_prix_norm", "themes_inflation_norm", "themes_alim_norm"]
    ].mean(axis=1)

    monthly.to_csv(out_silver, encoding="utf-8")
    logger.info(f"  Silver sauvegardé : {monthly.shape} -> {out_silver.name}")
    logger.info(f"  Période couverte : {monthly.index.min()} à {monthly.index.max()}")
    logger.warning(
        "  [LIMITE] Signal presse limité aux 30-90 derniers jours via RSS. "
        "Historique 2017-2024 non disponible sans scraping d'archives. "
        "Ce signal est utilisé en validation, pas comme composante BESI principale."
    )

    return monthly


def validate_press_vs_trends(
    press_monthly: pd.DataFrame,
    trends_monthly: pd.DataFrame,
    inflation_monthly: pd.Series,
) -> pd.DataFrame:
    """
    Valide que le signal presse est cohérent avec les Trends et l'inflation.
    Si press_volume corrèle avec Trends → les deux sources capturent le même phénomène.
    Si press_volume corrèle avec inflation → signal économique confirmé.

    Cette validation remplace l'argument "faites confiance à nos keywords manuels".
    """
    from scipy.stats import pearsonr

    common_idx = press_monthly.index.intersection(trends_monthly.index)
    common_idx = common_idx.intersection(inflation_monthly.index)

    if len(common_idx) < 6:
        logger.warning(f"Seulement {len(common_idx)} mois communs pour la validation")
        return pd.DataFrame()

    press_vol = press_monthly.loc[common_idx, "press_volume"]
    infl      = inflation_monthly.reindex(common_idx)

    # Corrélations presse ~ inflation
    records = []
    for lag in range(0, 3):
        infl_lagged = infl.shift(-lag)
        mask = press_vol.notna() & infl_lagged.notna()
        if mask.sum() < 6:
            continue
        r, p = pearsonr(press_vol[mask], infl_lagged[mask])
        records.append({
            "signal": "press_volume",
            "lag": lag,
            "r_pearson": round(r, 3),
            "p_value": round(p, 4),
            "significant": p < 0.05,
        })

    # Corrélations presse ~ trends composites (si disponible)
    if "trends_composite" in trends_monthly.columns:
        trends_comp = trends_monthly.loc[common_idx, "trends_composite"]
        mask = press_vol.notna() & trends_comp.notna()
        if mask.sum() >= 6:
            r, p = pearsonr(press_vol[mask], trends_comp[mask])
            records.append({
                "signal": "press_vs_trends",
                "lag": 0,
                "r_pearson": round(r, 3),
                "p_value": round(p, 4),
                "significant": p < 0.05,
            })

    df_val = pd.DataFrame(records)
    logger.info("[Validation presse]")
    for _, row in df_val.iterrows():
        sig = "(*)" if row["significant"] else "(ns)"
        logger.info(f"  {row['signal']} ~ inflation lag={row['lag']} : "
                    f"r={row['r_pearson']:.3f} p={row['p_value']:.4f} {sig}")

    return df_val
