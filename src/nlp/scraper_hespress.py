"""
Collecte historique Hespress via l'API WordPress JSON.

Stratégie :
- source principale : API posts Hespress (historique 2017-2024)
- source secondaire : RSS récent si l'API échoue

Le corpus produit n'est pas un corpus de commentaires pur. Par défaut, il
utilise les titres + extraits des articles, ce qui en fait un signal texte
éditorial/presse. Ce point doit être documenté honnêtement dans le rapport.
"""

from __future__ import annotations

import html
import logging
import re
import time
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

import pandas as pd
import requests

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
BRONZE_DIR = ROOT / "data" / "bronze"
REPORTS_DIR = ROOT / "outputs" / "reports"
BRONZE_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

POSTS_API = "https://www.hespress.com/wp-json/wp/v2/posts"
RSS_FEEDS = [
    "https://www.hespress.com/feed",
    "https://fr.hespress.com/feed",
]

# Catégories Hespress confirmées via wp-json.
CATEGORY_IDS = {
    "economie": 5,
    "hausse": 23,
    "baisse": 24,
}

ECONOMIC_KEYWORDS = [
    "prix", "inflation", "economie", "économie", "hausse", "baisse",
    "pouvoir d'achat", "carburant", "essence", "gasoil", "gaz", "sucre",
    "farine", "huile", "coût", "cout", "budget", "subvention",
    "الأسعار", "التضخم", "اقتصاد", "غلاء", "ارتفاع", "الوقود", "الزيت",
]


def _clean_html_text(value: str) -> str:
    value = html.unescape(value or "")
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _is_economic_text(text: str) -> bool:
    text_l = (text or "").lower()
    return any(keyword.lower() in text_l for keyword in ECONOMIC_KEYWORDS)


def _request_json(url: str, params: dict | None = None) -> tuple[list, requests.Response]:
    resp = requests.get(
        url,
        params=params,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    resp.raise_for_status()
    return resp.json(), resp


def _fetch_posts_for_category(
    category_id: int,
    start_date: str,
    end_date: str,
) -> list[dict]:
    records: list[dict] = []
    page = 1
    per_page = 100

    while True:
        items, resp = _request_json(
            POSTS_API,
            params={
                "categories": category_id,
                "after": f"{start_date}T00:00:00",
                "before": f"{end_date}T23:59:59",
                "per_page": per_page,
                "page": page,
                "_fields": "id,date,link,title,excerpt,categories,comment_status",
            },
        )
        if not items:
            break

        for item in items:
            title = _clean_html_text(item.get("title", {}).get("rendered", ""))
            excerpt = _clean_html_text(item.get("excerpt", {}).get("rendered", ""))
            text = f"{title}. {excerpt}".strip()
            if not _is_economic_text(text):
                continue

            dt = pd.to_datetime(item["date"], errors="coerce")
            if pd.isna(dt):
                continue
            month = dt.to_period("M").to_timestamp()

            records.append({
                "date": month.strftime("%Y-%m-%d"),
                "published_at": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "title": title,
                "text": text,
                "section": str(category_id),
                "url": item.get("link", ""),
                "source_type": "post_excerpt",
                "article_id": str(item.get("id", "")),
                "comment_count": pd.NA,
                "collection_method": "wp_json_api",
            })

        total_pages = int(resp.headers.get("X-WP-TotalPages", "1"))
        if page >= total_pages:
            break
        page += 1
        if page % 20 == 0:
            logger.info("  catégorie %s : page %s/%s", category_id, page, total_pages)
        time.sleep(0.15)

    return records


def _fetch_rss_fallback() -> list[dict]:
    records: list[dict] = []
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    for feed_url in RSS_FEEDS:
        try:
            resp = requests.get(feed_url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
        except Exception as exc:
            logger.warning("RSS indisponible %s : %s", feed_url, exc)
            continue

        channel = root.find("channel")
        items = channel.findall("item") if channel is not None else []
        for item in items:
            title = _clean_html_text(item.findtext("title", default=""))
            desc = _clean_html_text(item.findtext("description", default=""))
            text = f"{title}. {desc}".strip()
            if not _is_economic_text(text):
                continue

            pub_date = item.findtext("pubDate", default="")
            dt = pd.to_datetime(pub_date, errors="coerce", utc=True)
            if pd.isna(dt):
                continue
            month = dt.to_period("M").to_timestamp()

            records.append({
                "date": month.strftime("%Y-%m-%d"),
                "published_at": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "title": title,
                "text": text,
                "section": "rss",
                "url": item.findtext("link", default=""),
                "source_type": "rss",
                "article_id": "",
                "comment_count": pd.NA,
                "collection_method": "rss_fallback",
            })

    return records


def _coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    yearly = (
        df.assign(year=pd.to_datetime(df["date"]).dt.year)
        .groupby("year")
        .agg(n_texts=("text", "count"), n_articles=("article_id", "nunique"))
        .reset_index()
    )
    return yearly


def scrape_hespress_history(
    start_date: str = "2017-01-01",
    end_date: str = "2024-12-31",
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    if output_path is None:
        output_path = BRONZE_DIR / "hespress_raw.csv"
    output_path = Path(output_path)

    logger.info("Début collecte Hespress historique %s -> %s", start_date, end_date)
    records: list[dict] = []

    for section, category_id in CATEGORY_IDS.items():
        try:
            section_records = _fetch_posts_for_category(category_id, start_date, end_date)
            for rec in section_records:
                rec["section"] = section
            records.extend(section_records)
            logger.info("  section %s : %s textes", section, len(section_records))
        except Exception as exc:
            logger.warning("  échec API catégorie %s : %s", section, exc)

    if len(records) < 100:
        logger.warning("Moins de 100 textes via API historique -> fallback RSS explicite")
        records.extend(_fetch_rss_fallback())

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("Aucune donnée Hespress récupérée.")

    df = (
        df.drop_duplicates(subset=["date", "title", "url"])
        .sort_values(["date", "published_at", "title"])
        .reset_index(drop=True)
    )
    df.to_csv(output_path, index=False, encoding="utf-8")

    coverage = _coverage_report(df)
    coverage_path = REPORTS_DIR / "hespress_coverage_by_year.csv"
    coverage.to_csv(coverage_path, index=False, encoding="utf-8")

    logger.info("Bronze sauvegardé : %s", output_path)
    logger.info("Couverture annuelle :\n%s", coverage.to_string(index=False))

    pre2020 = coverage[coverage["year"] < 2020]
    if not pre2020.empty and (pre2020["n_articles"] < 50).any():
        logger.warning("DONNÉES INSUFFISANTES 2017-2019 : au moins une année < 50 articles.")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    scrape_hespress_history()
