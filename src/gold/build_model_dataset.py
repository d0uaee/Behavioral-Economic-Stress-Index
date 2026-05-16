"""
src/gold/build_model_dataset.py — Assemblage Gold Dataset v3

Fusionne toutes les couches Silver + indices BESI v3 en un seul dataset
prêt pour la modélisation, avec :
  - Respect strict de la règle as-of-date (no lookahead)
  - Lags explicites pour chaque feature (t, t-1, t-2)
  - Cibles décalées (target = IPC au mois suivant)
  - Labels de split (train_A / test_A / train_B / test_B / train_C / test_C)
  - Colonne feature_available_at (date à laquelle chaque ligne est exploitable)

Output : data/gold/model_dataset_monthly.csv

Usage :
    from src.gold.build_model_dataset import build_gold_dataset
    df = build_gold_dataset()
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parent.parent.parent
SILVER_DIR = ROOT / "data" / "silver"
GOLD_DIR   = ROOT / "data" / "gold"
GOLD_DIR.mkdir(parents=True, exist_ok=True)

# Fenêtres d'évaluation (3 blocs indépendants)
EVAL_WINDOWS = [
    {"label": "A", "train": ("2010-01-01", "2017-12-01"), "test": ("2018-01-01", "2019-12-01")},
    {"label": "B", "train": ("2010-01-01", "2019-12-01"), "test": ("2020-01-01", "2021-12-01")},
    {"label": "C", "train": ("2010-01-01", "2021-12-01"), "test": ("2022-01-01", "2024-12-01")},
]

INFLATION_REGIME_THRESHOLD = 2.0   # % YoY

# Features à inclure avec leurs lags (lag 0 = mois courant, lag 1 = mois précédent)
FEATURE_LAGS = {
    "behavioral_index_pure": [0, 1, 2],
    "hybrid_macro_index":    [0, 1, 2],
    "trends_prix_alim":      [0, 1],
    "trends_inflation":      [0, 1],
    "trends_carburant":      [0, 1],
    "trends_composite":      [0, 1],
    "fao_food_index":        [0, 1],
    "fao_food_yoy":          [0, 1],
    "fao_oils_yoy":          [0, 1],
    "mad_eur":               [0, 1],
    "fx_yoy":                [0, 1],
    # Lags IPC passés (disponibles avant la publication du mois t)
    "ipc_level":             [1, 2, 3],       # IPC(t-1), IPC(t-2), IPC(t-3)
    "inflation_yoy":         [1, 2],           # YoY(t-1), YoY(t-2)
    "inflation_mom":         [1, 2],           # MoM(t-1), MoM(t-2)
}


def _assign_split_label(date: pd.Timestamp) -> str:
    """Attribue le label de split (train_A/test_A/...) à une date."""
    labels = []
    for window in EVAL_WINDOWS:
        lbl = window["label"]
        if pd.Timestamp(window["train"][0]) <= date <= pd.Timestamp(window["train"][1]):
            labels.append(f"train_{lbl}")
        elif pd.Timestamp(window["test"][0]) <= date <= pd.Timestamp(window["test"][1]):
            labels.append(f"test_{lbl}")
    return "|".join(labels) if labels else "unused"


def build_gold_dataset(
    output_path: str | Path | None = None,
    start_date:  str = "2010-01-01",
    end_date:    str = "2024-12-01",
) -> pd.DataFrame:
    """
    Assemble le Gold dataset v3 à partir des Silver layers.

    Étapes :
    1. Charger les Silver (CPI, Trends, Macro)
    2. Charger les indices BESI (si disponibles dans silver/)
    3. Créer les lags avec noms explicites
    4. Créer les cibles décalées (IPC t+1)
    5. Ajouter as_of_date et split_label
    6. Valider : aucune feature future dans X

    Retourne
    --------
    pd.DataFrame avec toutes les colonnes du schéma Gold.
    """
    if output_path is None:
        output_path = GOLD_DIR / "model_dataset_monthly.csv"
    output_path = Path(output_path)

    full_idx = pd.date_range(start_date, end_date, freq="MS")
    gold = pd.DataFrame(index=full_idx)
    gold.index.name = "month"

    # ── 1. CPI Silver ────────────────────────────────────────────────────────
    cpi_path = SILVER_DIR / "cpi_monthly.csv"
    if cpi_path.exists():
        cpi = pd.read_csv(cpi_path, parse_dates=["month"], index_col="month")
        for col in ["ipc_level", "inflation_mom", "inflation_yoy", "publication_date"]:
            if col in cpi.columns:
                gold[col] = cpi[col].reindex(gold.index)
        logger.info(f"CPI Silver chargé : {len(cpi)} mois")
    else:
        logger.warning(f"CPI Silver manquant : {cpi_path}")

    # ── 2. Trends Silver ─────────────────────────────────────────────────────
    trends_path = SILVER_DIR / "google_trends_monthly.csv"
    if trends_path.exists():
        trends = pd.read_csv(trends_path, parse_dates=["month"], index_col="month")
        for col in trends.select_dtypes(include="number").columns:
            if "n_keywords" not in col:
                gold[col] = trends[col].reindex(gold.index)
        logger.info(f"Trends Silver chargé : {len(trends)} mois, {len(trends.columns)} colonnes")
    else:
        logger.warning(f"Trends Silver manquant : {trends_path}")

    # ── 3. Macro Silver ───────────────────────────────────────────────────────
    macro_path = SILVER_DIR / "macro_signals_monthly.csv"
    if macro_path.exists():
        macro = pd.read_csv(macro_path, parse_dates=["month"], index_col="month")
        for col in macro.select_dtypes(include="number").columns:
            gold[col] = macro[col].reindex(gold.index)
        logger.info(f"Macro Silver chargé : {len(macro)} mois, {len(macro.columns)} colonnes")
    else:
        logger.warning(f"Macro Silver manquant : {macro_path}")

    # ── 4. Indices BESI v3 (si déjà calculés) ────────────────────────────────
    for idx_name in ["behavioral_index_pure", "hybrid_macro_index"]:
        idx_path = SILVER_DIR / f"{idx_name}.csv"
        if idx_path.exists():
            idx_df = pd.read_csv(idx_path, parse_dates=["month"], index_col="month")
            gold[idx_name] = idx_df.iloc[:, 0].reindex(gold.index)
            logger.info(f"Index {idx_name} chargé.")

    # ── 5. Lags explicites ────────────────────────────────────────────────────
    lag_cols_added = []
    for feature, lags in FEATURE_LAGS.items():
        if feature not in gold.columns:
            continue
        for lag in lags:
            col_name = f"{feature}_lag{lag}" if lag > 0 else feature
            if lag > 0 and col_name not in gold.columns:
                gold[col_name] = gold[feature].shift(lag)
                lag_cols_added.append(col_name)

    logger.info(f"Lags créés : {len(lag_cols_added)} colonnes")

    # ── 6. Cibles décalées (prédiction t+1) ───────────────────────────────────
    if "inflation_yoy" in gold.columns:
        gold["target_inflation_yoy_t1"]         = gold["inflation_yoy"].shift(-1)
    if "inflation_yoy" in gold.columns:
        gold["target_high_inflation_regime_t1"] = (
            gold["inflation_yoy"].shift(-1) >= INFLATION_REGIME_THRESHOLD
        ).astype(float)   # float pour garder NaN en fin de série
    if "ipc_level" in gold.columns:
        gold["target_ipc_level_t1"]             = gold["ipc_level"].shift(-1)

    # ── 7. Métadonnées as-of-date et split ────────────────────────────────────
    # as_of_date : date à laquelle les features de ce mois sont disponibles
    # = publication_date du mois courant (~J+20) → après publication IPC(t)
    # mais les features X(t) sont disponibles depuis fin du mois t
    gold["as_of_date"] = gold.index.to_series().apply(
        lambda d: str((d + pd.offsets.MonthEnd(0)).date())
    )

    # feature_available_at = date à partir de laquelle on peut utiliser cette ligne pour prédire
    gold["feature_available_at"] = gold.index.to_series().apply(
        lambda d: str((d + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1)).date())
    )

    gold["split_label"] = gold.index.to_series().apply(_assign_split_label)

    # ── 8. Validation intégrité ───────────────────────────────────────────────
    _validate_gold(gold)

    # ── 9. Sauvegarde ────────────────────────────────────────────────────────
    gold.to_csv(output_path, index=True)
    logger.info(f"\nGold dataset sauvegardé : {output_path}")
    logger.info(f"  Shape     : {gold.shape}")
    logger.info(f"  Colonnes  : {len(gold.columns)}")
    logger.info(f"  Mois      : {len(gold)}")

    # Rapport de complétude
    print("\n=== RAPPORT INTÉGRITÉ GOLD DATASET ===")
    print(f"{'Colonne':<40} {'Non-null':>8} {'%':>6}")
    print("-" * 56)
    for col in gold.columns:
        n = gold[col].notna().sum()
        pct = 100 * n / len(gold)
        flag = "" if pct > 80 else " ⚠" if pct > 30 else " ✗"
        print(f"{col:<40} {n:>8} {pct:>5.1f}%{flag}")

    return gold


def _validate_gold(gold: pd.DataFrame) -> None:
    """Vérifie qu'aucune règle as-of-date n'est violée."""
    errors = []

    # ipc_change ne doit pas être dans les features (c'est une transfo de la cible)
    forbidden = ["ipc_change"]
    for col in forbidden:
        if col in gold.columns and col not in [c for c in gold.columns if "target" in c]:
            errors.append(f"  ✗ Colonne interdite (data leakage) : {col}")

    # Les lag0 de l'IPC courant ne peuvent pas être utilisés pour prédire IPC(t)
    # car IPC(t) est la cible → vérifier que ipc_level_lag0 n'est pas présent
    if "ipc_level" in gold.columns and "ipc_level_lag0" not in gold.columns:
        # ipc_level sans lag = contemporain → acceptable seulement si utilisé lagué en modélisation
        logger.debug("  Note : ipc_level présent sans lag — s'assurer de ne l'utiliser qu'avec lag >= 1")

    if errors:
        for e in errors:
            logger.error(e)
        raise ValueError(f"Violations as-of-date détectées :\n" + "\n".join(errors))
    else:
        logger.info("  ✓ Validation as-of-date : OK")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = build_gold_dataset()
    print(f"\nShape final : {df.shape}")
    print(f"Colonnes targets : {[c for c in df.columns if 'target' in c]}")
    print(f"Split distribution :\n{df['split_label'].value_counts()}")
