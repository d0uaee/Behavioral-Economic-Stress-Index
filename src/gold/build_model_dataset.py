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

# Fenêtres d'évaluation — version COMPLÈTE (données 2010-2024)
EVAL_WINDOWS = [
    {"label": "A", "train": ("2010-01-01", "2017-12-01"), "test": ("2018-01-01", "2019-12-01")},
    {"label": "B", "train": ("2010-01-01", "2019-12-01"), "test": ("2020-01-01", "2021-12-01")},
    {"label": "C", "train": ("2010-01-01", "2021-12-01"), "test": ("2022-01-01", "2024-12-01")},
]

# Fenêtres d'évaluation — version COURTE (données 2017-2024, HCP uniquement)
# Bloc A : période COVID (2020-2021) — test de robustesse choc exogène
# Bloc B : choc inflationniste (2022-2024) — test principal H1/H2
SHORT_EVAL_WINDOWS = [
    {"label": "A", "train": ("2017-01-01", "2019-12-01"), "test": ("2020-01-01", "2021-12-01")},
    {"label": "B", "train": ("2017-01-01", "2021-12-01"), "test": ("2022-01-01", "2024-12-01")},
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


def _select_eval_windows(start_date: str) -> list:
    """
    Sélectionne automatiquement les fenêtres d'évaluation selon la date de début.
    - Données depuis 2010 → EVAL_WINDOWS complet (3 blocs)
    - Données depuis 2017 → SHORT_EVAL_WINDOWS (2 blocs COVID + inflation)
    """
    if pd.Timestamp(start_date) <= pd.Timestamp("2012-01-01"):
        return EVAL_WINDOWS
    return SHORT_EVAL_WINDOWS


def _assign_split_label(date: pd.Timestamp, windows: list) -> str:
    """Attribue le label de split (train_A/test_A/...) à une date."""
    labels = []
    for window in windows:
        lbl = window["label"]
        if pd.Timestamp(window["train"][0]) <= date <= pd.Timestamp(window["train"][1]):
            labels.append(f"train_{lbl}")
        elif pd.Timestamp(window["test"][0]) <= date <= pd.Timestamp(window["test"][1]):
            labels.append(f"test_{lbl}")
    return "|".join(labels) if labels else "unused"


def build_gold_dataset(
    output_path:  str | Path | None = None,
    start_date:   str = "2010-01-01",
    end_date:     str = "2024-12-01",
    eval_windows: list | None = None,
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

    Paramètres
    ----------
    start_date   : début de la plage (auto-adapté selon données disponibles)
    eval_windows : fenêtres d'évaluation — si None, sélectionné automatiquement
                   selon start_date (FULL si 2010, SHORT si 2017)

    Retourne
    --------
    pd.DataFrame avec toutes les colonnes du schéma Gold.
    """
    if output_path is None:
        output_path = GOLD_DIR / "model_dataset_monthly.csv"
    output_path = Path(output_path)

    # Sélection automatique des fenêtres si non forcées
    if eval_windows is None:
        eval_windows = _select_eval_windows(start_date)
    logger.info(f"Fenêtres d'évaluation : {[w['label'] for w in eval_windows]} "
                f"({'FULL' if len(eval_windows) == 3 else 'SHORT'})")

    # Auto-détection du start_date réel à partir des données CPI disponibles
    cpi_path_probe = SILVER_DIR / "cpi_monthly.csv"
    if cpi_path_probe.exists():
        _probe = pd.read_csv(cpi_path_probe, parse_dates=["month"], index_col="month")
        actual_start = _probe.index.min()
        if actual_start > pd.Timestamp(start_date):
            logger.warning(
                f"Données CPI disponibles depuis {actual_start.date()} "
                f"(demandé : {start_date}) — start_date ajusté automatiquement."
            )
            start_date = str(actual_start.date())
            if eval_windows is EVAL_WINDOWS:
                eval_windows = _select_eval_windows(start_date)

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
        # Décaler d'abord, PUIS tester — ainsi le dernier mois reste NaN
        # (la comparaison booléenne transformerait NaN→False→0.0 sans ce garde)
        _shifted_yoy = gold["inflation_yoy"].shift(-1)
        gold["target_high_inflation_regime_t1"] = np.where(
            _shifted_yoy.isna(),
            np.nan,
            (_shifted_yoy >= INFLATION_REGIME_THRESHOLD).astype(float),
        )
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

    gold["split_label"] = gold.index.to_series().apply(
        lambda d: _assign_split_label(d, eval_windows)
    )

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
    errors   = []
    warnings = []

    # ── Colonnes strictement interdites (data leakage direct) ─────────────────
    # ipc_change = dérivée de la cible → ne doit jamais être une feature
    forbidden = ["ipc_change"]
    for col in forbidden:
        if col in gold.columns:
            errors.append(f"  ✗ Colonne interdite (data leakage) : {col}")

    # ── Colonnes contemporaines dangereuses ────────────────────────────────────
    # Ces colonnes sont présentes dans le Gold (pour référence et calcul des lags),
    # mais NE DOIVENT PAS être utilisées directement comme features de modélisation.
    # Seules leurs versions laggées (_lag1, _lag2, …) sont légitimes.
    contemporaneous_danger = [
        "ipc_level",        # = quasi-cible pour prédire ipc_level_t1
        "inflation_yoy",    # = quasi-cible pour prédire target_inflation_yoy_t1
        "inflation_mom",    # publication ~J+20, non dispo au moment de la prédiction
        "inflation_regime", # dérivé de inflation_yoy courant
    ]
    found_danger = [c for c in contemporaneous_danger if c in gold.columns]
    if found_danger:
        warnings.append(
            f"  ⚠ Colonnes contemporaines présentes (usage en features INTERDIT) : "
            f"{found_danger}\n"
            "    → Utiliser uniquement les versions _lag1, _lag2, _lag3"
        )

    # ── Vérifier que les lags existent bien ───────────────────────────────────
    for col in ["ipc_level", "inflation_yoy"]:
        if col in gold.columns and f"{col}_lag1" not in gold.columns:
            warnings.append(f"  ⚠ '{col}_lag1' absent alors que '{col}' est présent")

    # ── Vérifier que les targets sont bien décalées ───────────────────────────
    for target_col, source_col in [
        ("target_inflation_yoy_t1", "inflation_yoy"),
        ("target_ipc_level_t1",     "ipc_level"),
    ]:
        if target_col in gold.columns and source_col in gold.columns:
            # Le dernier mois doit être NaN dans la target
            if gold[target_col].iloc[-1] is not None and not pd.isna(gold[target_col].iloc[-1]):
                errors.append(
                    f"  ✗ '{target_col}' : dernier mois non-NaN "
                    f"({gold[target_col].iloc[-1]}) — vérifier le shift(-1)"
                )

    for w in warnings:
        logger.warning(w)

    if errors:
        for e in errors:
            logger.error(e)
        raise ValueError("Violations as-of-date détectées :\n" + "\n".join(errors))
    else:
        logger.info("  ✓ Validation as-of-date : OK")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = build_gold_dataset()
    print(f"\nShape final : {df.shape}")
    print(f"Colonnes targets : {[c for c in df.columns if 'target' in c]}")
    print(f"Split distribution :\n{df['split_label'].value_counts()}")
