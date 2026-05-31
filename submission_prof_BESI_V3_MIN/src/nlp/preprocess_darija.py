"""
Nettoyage léger des textes Hespress pour un scoring lexical robuste.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
BRONZE_DIR = ROOT / "data" / "bronze"
SILVER_DIR = ROOT / "data" / "silver"
SILVER_DIR.mkdir(parents=True, exist_ok=True)

STOPWORDS = {
    "wa", "li", "fi", "3la", "dyal", "hada", "hadchi", "mashi",
    "walo", "bzzaf", "chwiya", "de", "la", "le", "les", "des",
    "du", "un", "une", "et", "en", "au", "aux", "sur",
}

ARABIZI_MAP = {
    "3": "ع",
    "7": "ح",
    "9": "ق",
    "5": "خ",
    "2": "ء",
}


def _strip_urls_mentions(text: str) -> str:
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[@#]\w+", " ", text)
    return text


def _normalise_arabizi(text: str) -> tuple[str, int]:
    has_arabizi = 0
    for digit, repl in ARABIZI_MAP.items():
        if digit in text:
            has_arabizi = 1
            text = text.replace(digit, repl)
    return text, has_arabizi


def _remove_emojis(text: str) -> str:
    return re.sub(r"[\U00010000-\U0010ffff]", " ", text)


def _tokenise(text: str) -> list[str]:
    return [tok for tok in re.split(r"[^\w\u0600-\u06FF]+", text) if tok]


def _detect_lang_profile(tokens: list[str]) -> str:
    if not tokens:
        return "unknown"
    arabic_tokens = sum(bool(re.search(r"[\u0600-\u06FF]", tok)) for tok in tokens)
    latin_tokens = sum(bool(re.search(r"[a-zA-Z]", tok)) for tok in tokens)
    total = max(len(tokens), 1)
    if arabic_tokens / total >= 0.6:
        return "darija"
    if latin_tokens / total >= 0.6:
        return "fr"
    if arabic_tokens and latin_tokens:
        return "mixed"
    return "unknown"


def _clean_text(raw_text: str) -> tuple[str, str, int, int]:
    text = (raw_text or "").lower()
    text = _strip_urls_mentions(text)
    text = _remove_emojis(text)
    text, has_arabizi = _normalise_arabizi(text)
    tokens = _tokenise(text)
    tokens = [tok for tok in tokens if tok not in STOPWORDS]
    lang_profile = _detect_lang_profile(tokens)
    clean_text = " ".join(tokens)
    return clean_text, lang_profile, len(tokens), has_arabizi


def preprocess_hespress_raw(
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    if input_path is None:
        input_path = BRONZE_DIR / "hespress_raw.csv"
    if output_path is None:
        output_path = SILVER_DIR / "hespress_clean.csv"
    input_path = Path(input_path)
    output_path = Path(output_path)

    df = pd.read_csv(input_path)
    out = df.copy()
    out["raw_text"] = out["text"].fillna("")

    cleaned = out["raw_text"].apply(_clean_text)
    out["clean_text"] = cleaned.apply(lambda x: x[0])
    out["lang_profile"] = cleaned.apply(lambda x: x[1])
    out["token_count"] = cleaned.apply(lambda x: x[2])
    out["has_arabizi"] = cleaned.apply(lambda x: x[3])
    out["coverage_flag"] = 1
    out["processing_notes"] = ""

    cols = [
        "date", "article_id", "source_type", "title", "url",
        "raw_text", "clean_text", "lang_profile", "token_count",
        "has_arabizi", "coverage_flag", "processing_notes",
    ]
    out = out[[c for c in cols if c in out.columns]]
    out.to_csv(output_path, index=False, encoding="utf-8")

    logger.info("Silver texte sauvegardé : %s (%s lignes)", output_path, len(out))
    logger.info("Répartition langues :\n%s", out["lang_profile"].value_counts(dropna=False).to_string())
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    preprocess_hespress_raw()

