"""
Scoring lexical mensuel du signal presse marocain.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
SILVER_DIR = ROOT / "data" / "silver"
REPORTS_DIR = ROOT / "outputs" / "reports"
SILVER_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

NEGATIVE_TERMS = [
    "ghali", "mghal", "3la", "ma imkinch", "l3icha s3iba", "ma بقاش", "ma b9ach",
    "3ziz", "zad fl prix", "tla3 lprix", "zid", "ma 3ndnach", "hchouma", "3ar",
    "skhoun", "ghalya", "ghlaw", "mghla", "cher", "cherte", "cherté",
    "hausse", "flambee", "flambée", "augmentation", "غالي", "مغال", "زاد", "ارتفع", "غلاء",
]

POSITIVE_TERMS = [
    "hbat", "nq9s", "rkhis", "mezyan", "walo mushkil", "b5ir",
    "baisse", "stabilite", "stabilité", "normal", "هبط", "رخيص", "انخفض", "استقر",
]

INTENSIFIERS = [
    "bzzaf", "ktir", "3zim", "jddan", "vraiment", "tres", "très", "trop", "جداً", "كثير", "عظيم",
]


def _score_single_text(text: str) -> tuple[float, Counter]:
    text_l = (text or "").lower()
    intensifier_multiplier = 1.5 if any(term in text_l for term in INTENSIFIERS) else 1.0
    neg_hits = Counter({term: text_l.count(term) for term in NEGATIVE_TERMS if term in text_l})
    pos_hits = Counter({term: text_l.count(term) for term in POSITIVE_TERMS if term in text_l})

    # Score de stress : plus le score est élevé, plus le texte reflète un stress économique.
    raw = (sum(neg_hits.values()) - sum(pos_hits.values())) * intensifier_multiplier
    combined = neg_hits + pos_hits
    return float(raw), combined


def _normalise_train_only(series: pd.Series, train_mask: pd.Series) -> pd.Series:
    train_values = series[train_mask & series.notna()]
    if train_values.empty:
        return pd.Series(0.0, index=series.index)
    mn, mx = train_values.min(), train_values.max()
    if mx == mn:
        return pd.Series(0.0, index=series.index)
    return ((series - mn) / (mx - mn)).clip(lower=0.0, upper=1.0)


def build_sentiment_monthly(
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    start_date: str = "2017-01-01",
    end_date: str = "2024-12-01",
    train_end: str = "2021-12-01",
) -> pd.DataFrame:
    if input_path is None:
        input_path = SILVER_DIR / "hespress_clean.csv"
    if output_path is None:
        output_path = SILVER_DIR / "sentiment_monthly.csv"
    input_path = Path(input_path)
    output_path = Path(output_path)

    df = pd.read_csv(input_path, parse_dates=["date"])
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()

    scores = df["clean_text"].fillna("").apply(_score_single_text)
    df["score_lexical_raw_item"] = scores.apply(lambda x: x[0])

    token_hits = Counter()
    for _, counts in scores:
        token_hits.update(counts)

    monthly = (
        df.groupby("date")
        .agg(
            score_lexical_raw=("score_lexical_raw_item", "mean"),
            n_texts=("clean_text", "count"),
            source_type_dominant=("source_type", lambda s: s.mode().iloc[0] if not s.mode().empty else "unknown"),
        )
        .sort_index()
    )

    monthly["n_negative_hits"] = 0
    monthly["n_positive_hits"] = 0
    for dt, sub in df.groupby("date"):
        text_l = " ".join(sub["clean_text"].fillna("").tolist()).lower()
        monthly.loc[dt, "n_negative_hits"] = sum(text_l.count(term) for term in NEGATIVE_TERMS)
        monthly.loc[dt, "n_positive_hits"] = sum(text_l.count(term) for term in POSITIVE_TERMS)

    full_idx = pd.date_range(start_date, end_date, freq="MS")
    monthly = monthly.reindex(full_idx)
    monthly.index.name = "date"
    monthly["coverage_flag"] = (monthly["n_texts"].fillna(0) > 0).astype(int)
    monthly["imputation_reason"] = ""
    monthly.loc[monthly["coverage_flag"] == 0, "imputation_reason"] = "no_data_neutral_imputation"
    monthly["score_lexical_raw"] = monthly["score_lexical_raw"].fillna(0.0)
    monthly["n_texts"] = monthly["n_texts"].fillna(0).astype(int)
    monthly["n_negative_hits"] = monthly["n_negative_hits"].fillna(0).astype(int)
    monthly["n_positive_hits"] = monthly["n_positive_hits"].fillna(0).astype(int)
    monthly["source_type_dominant"] = monthly["source_type_dominant"].fillna("none")

    train_mask = monthly.index <= pd.Timestamp(train_end)
    monthly["score_lexical_norm"] = _normalise_train_only(monthly["score_lexical_raw"], train_mask)
    monthly["notes"] = ""

    top_terms = pd.DataFrame(token_hits.most_common(20), columns=["term", "count"])
    top_terms.to_csv(REPORTS_DIR / "hespress_top_lexical_terms.csv", index=False, encoding="utf-8")

    yearly = (
        monthly.groupby(monthly.index.year)["n_texts"]
        .sum()
        .rename_axis("year")
        .reset_index(name="n_texts")
    )
    yearly.to_csv(REPORTS_DIR / "hespress_text_coverage_by_year.csv", index=False, encoding="utf-8")
    if ((yearly["year"] < 2020) & (yearly["n_texts"] < 50)).any():
        logger.warning("DONNÉES INSUFFISANTES 2017-2019 : au moins une année < 50 textes.")

    monthly.reset_index().to_csv(output_path, index=False, encoding="utf-8")
    logger.info("Signal mensuel sauvegardé : %s", output_path)
    logger.info("Couverture mensuelle réelle : %s/%s mois", int(monthly["coverage_flag"].sum()), len(monthly))
    return monthly.reset_index()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    build_sentiment_monthly()
