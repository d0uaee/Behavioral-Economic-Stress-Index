"""
Métriques par période — Projet BESI Maroc

Objectif
--------
Construire un tableau exhaustif par modèle et par période pour montrer
que la performance varie fortement entre Bloc A (COVID), Bloc B
(inflation 2022-2024), Pre-2022, Post-2022 et Global.

Le script calcule :
    - RMSE, MAE, MAPE sur les prévisions IPC
    - Recall, Precision, F1, AUC, AP sur un score de stress dérivé
      des prévisions de l'IPC en yoy
    - n_obs, % de mois en stress, inflation moyenne

Sorties
-------
    results/metrics_by_period.csv
    results/figures/metrics_by_period.png

Notes méthodologiques
---------------------
Le seuil de stress élevé est figé sur l'entraînement pré-2022 uniquement
(75e percentile de l'inflation YoY) et appliqué tel quel sur toutes les
périodes, sans recalibration test.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.backtest import _mae, _mape, _rmse, _sarima_fit, _safe_forecast

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - dépend de l'environnement
    matplotlib = None
    plt = None

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
GOLD_PATH = ROOT / "data" / "gold" / "model_dataset_monthly.csv"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

MODEL_CONFIG = {
    "naif": {
        "label": "Naif",
        "exog": None,
        "kind": "naif",
        "color": "#7f8c8d",
    },
    "sarima": {
        "label": "SARIMA",
        "exog": None,
        "kind": "sarima",
        "color": "#2C5F8A",
    },
    "sarimax_behavioral": {
        "label": "SARIMAX+BESI",
        "exog": "behavioral_index_pure_lag1",
        "kind": "sarimax",
        "color": "#4CAF50",
    },
    "sarimax_hybrid": {
        "label": "SARIMAX+Hybrid",
        "exog": "hybrid_macro_index_lag1",
        "kind": "sarimax",
        "color": "#FF9800",
    },
}

PERIOD_ORDER = ["Bloc A", "Bloc B", "Pre-2022", "Post-2022", "Global"]
PLOT_METRICS = ["RMSE", "F1", "AUC", "AP"]


def _load_gold(gold_path: Path = GOLD_PATH) -> pd.DataFrame:
    if not gold_path.exists():
        raise FileNotFoundError(
            f"Gold dataset introuvable : {gold_path}\n"
            "Lancer d'abord : python run_v3.py --step gold"
        )

    gold = pd.read_csv(gold_path, parse_dates=["month"], index_col="month")
    gold = gold.sort_index()
    return gold


def _derive_period_masks(gold: pd.DataFrame) -> dict[str, pd.Series]:
    """Construit les masques de période sur l'index mensuel."""
    index = gold.index
    masks = {
        "Bloc A": gold["split_label"].str.contains("test_A", na=False),
        "Bloc B": gold["split_label"].str.contains("test_B", na=False),
        "Pre-2022": index < pd.Timestamp("2022-01-01"),
        "Post-2022": index >= pd.Timestamp("2022-01-01"),
        "Global": pd.Series(True, index=index),
    }
    return masks


def _stress_threshold(gold: pd.DataFrame) -> float:
    """Seuil de stress élevé appris uniquement sur le train pré-2022."""
    if "inflation_yoy" not in gold.columns:
        raise KeyError("La colonne 'inflation_yoy' est absente du Gold dataset.")

    train_mask = gold.index < pd.Timestamp("2022-01-01")
    train_yoy = gold.loc[train_mask, "inflation_yoy"].dropna().astype(float)
    if train_yoy.empty:
        raise ValueError("Impossible de calculer le seuil de stress : train vide.")
    return float(np.percentile(train_yoy.values, 75))


def _walk_forward_ipc_predictions(
    gold: pd.DataFrame,
    model_key: str,
    min_train_months: int = 24,
) -> pd.Series:
    """Génère une prévision 1-pas-à-l'avance de l'IPC pour chaque mois."""
    cfg = MODEL_CONFIG[model_key]
    target_col = "ipc_level"
    if target_col not in gold.columns:
        raise KeyError(f"Colonne '{target_col}' absente du Gold dataset.")

    actual = gold[target_col].dropna()
    if actual.empty:
        raise ValueError("Aucune observation IPC disponible.")

    preds: list[float] = []
    dates: list[pd.Timestamp] = []
    data_start = actual.index.min()

    for current_date in actual.index:
        cutoff = current_date - pd.offsets.MonthBegin(1)
        train_slice = gold.loc[data_start:cutoff, target_col].dropna()

        if len(train_slice) < min_train_months:
            pred = float(train_slice.iloc[-1]) if len(train_slice) > 0 else np.nan
        elif cfg["kind"] == "naif":
            pred = float(train_slice.iloc[-1])
        else:
            exog_train = None
            exog_te = None
            exog_col = cfg["exog"]

            if cfg["kind"] == "sarimax" and exog_col and exog_col in gold.columns:
                exog_series = gold[exog_col]
                exog_train_raw = exog_series.loc[data_start:cutoff].reindex(train_slice.index).ffill().bfill()
                if exog_train_raw.isna().mean() < 0.3:
                    exog_train = exog_train_raw.to_frame(exog_col)
                    te_val = exog_series.get(current_date, np.nan)
                    if pd.isna(te_val):
                        te_val = exog_series.loc[:cutoff].iloc[-1] if len(exog_series.loc[:cutoff]) > 0 else 0.0
                    exog_te = pd.DataFrame({exog_col: [te_val]}, index=[current_date])

            fit_result = _sarima_fit(train_slice, exog_train=exog_train, simple=len(train_slice) < 36)
            if fit_result is None:
                fit_result = _sarima_fit(train_slice, exog_train=exog_train, simple=True)
            if fit_result is None:
                pred = float(train_slice.iloc[-1]) if len(train_slice) > 0 else np.nan
            else:
                pred = _safe_forecast(fit_result, 1, exog_te, train_slice)

        preds.append(pred)
        dates.append(current_date)

    return pd.Series(preds, index=pd.DatetimeIndex(dates), name=model_key)


def _forecasted_yoy(actual_ipc: pd.Series, predicted_ipc: pd.Series) -> pd.Series:
    """Convertit la prévision IPC en score de stress via le YoY estimé."""
    lag12 = actual_ipc.shift(12)
    score = (predicted_ipc / lag12 - 1.0) * 100.0
    return score.replace([np.inf, -np.inf], np.nan)


def _roc_auc_ap(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Calcule AUC ROC et AP sans dépendre de sklearn."""
    if len(y_true) == 0:
        return float("nan"), float("nan")

    thresholds = np.unique(scores)[::-1]
    pos = int(y_true.sum())
    neg = len(y_true) - pos

    if pos == 0 or neg == 0:
        return 0.5, float(pos / len(y_true)) if len(y_true) > 0 else float("nan")

    tprs = [0.0]
    fprs = [0.0]
    precisions = []
    recalls = []

    for thr in thresholds:
        pred = (scores >= thr).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        tprs.append(recall)
        fprs.append(1.0 - specificity)
        precisions.append(precision)
        recalls.append(recall)

    tprs.append(1.0)
    fprs.append(1.0)

    auc = float(abs(np.trapezoid(tprs, fprs) if hasattr(np, "trapezoid") else np.trapz(tprs, fprs)))
    idx = np.argsort(recalls)
    ap = float(abs(np.trapezoid(np.array(precisions)[idx], np.array(recalls)[idx])
                   if hasattr(np, "trapezoid") else np.trapz(np.array(precisions)[idx], np.array(recalls)[idx])))
    return auc, ap


def _classification_metrics(y_true: pd.Series, scores: pd.Series, threshold: float) -> dict:
    valid = y_true.dropna().index.intersection(scores.dropna().index)
    if len(valid) == 0:
        return {
            "Recall": np.nan,
            "Precision": np.nan,
            "F1": np.nan,
            "AUC": np.nan,
            "AP": np.nan,
            "n_obs": 0,
            "stress_pct": np.nan,
            "avg_inflation_yoy": np.nan,
        }

    y = y_true.loc[valid].astype(int).values
    s = scores.loc[valid].astype(float).values
    pred = (s >= threshold).astype(int)

    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    auc, ap = _roc_auc_ap(y, s)

    return {
        "Recall": float(recall),
        "Precision": float(precision),
        "F1": float(f1),
        "AUC": float(auc),
        "AP": float(ap),
        "n_obs": int(len(valid)),
        "stress_pct": float(y.mean() * 100.0),
        "avg_inflation_yoy": float(y_true.loc[valid].mean()),
    }


def _regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    valid = y_true.dropna().index.intersection(y_pred.dropna().index)
    if len(valid) == 0:
        return {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan}

    yt = y_true.loc[valid].values
    yp = y_pred.loc[valid].values
    return {
        "RMSE": _rmse(yt, yp),
        "MAE": _mae(yt, yp),
        "MAPE": _mape(yt, yp),
    }


def _build_table(gold: pd.DataFrame) -> pd.DataFrame:
    threshold = _stress_threshold(gold)
    actual_yoy = gold["inflation_yoy"].copy()
    actual_stress = (actual_yoy >= threshold).astype(float)

    masks = _derive_period_masks(gold)
    model_predictions = {
        model_key: _walk_forward_ipc_predictions(gold, model_key)
        for model_key in MODEL_CONFIG
    }

    rows = []
    for model_key, pred_ipc in model_predictions.items():
        cfg = MODEL_CONFIG[model_key]
        model_yoy_score = _forecasted_yoy(gold["ipc_level"], pred_ipc)

        for period_name in PERIOD_ORDER:
            period_mask = masks[period_name]
            period_y_true = gold.loc[period_mask, "ipc_level"]
            period_y_pred = pred_ipc.loc[period_mask]
            reg = _regression_metrics(period_y_true, period_y_pred)

            period_scores = model_yoy_score.loc[period_mask]
            period_stress = actual_stress.loc[period_mask]
            cls = _classification_metrics(period_stress, period_scores, threshold)

            rows.append({
                "Model": cfg["label"],
                "Period": period_name,
                "RMSE": round(reg["RMSE"], 4) if pd.notna(reg["RMSE"]) else np.nan,
                "MAE": round(reg["MAE"], 4) if pd.notna(reg["MAE"]) else np.nan,
                "MAPE": round(reg["MAPE"], 4) if pd.notna(reg["MAPE"]) else np.nan,
                "Recall": round(cls["Recall"], 4) if pd.notna(cls["Recall"]) else np.nan,
                "Precision": round(cls["Precision"], 4) if pd.notna(cls["Precision"]) else np.nan,
                "F1": round(cls["F1"], 4) if pd.notna(cls["F1"]) else np.nan,
                "AUC": round(cls["AUC"], 4) if pd.notna(cls["AUC"]) else np.nan,
                "AP": round(cls["AP"], 4) if pd.notna(cls["AP"]) else np.nan,
                "n_obs": cls["n_obs"],
                "stress_pct": round(cls["stress_pct"], 2) if pd.notna(cls["stress_pct"]) else np.nan,
                "inflation_yoy_mean": round(float(actual_yoy.loc[period_mask].mean()), 4) if actual_yoy.loc[period_mask].notna().any() else np.nan,
                "stress_threshold_yoy": round(threshold, 4),
            })

    table = pd.DataFrame(rows)
    table["Period"] = pd.Categorical(table["Period"], categories=PERIOD_ORDER, ordered=True)
    table["Model"] = pd.Categorical(
        table["Model"],
        categories=[cfg["label"] for cfg in MODEL_CONFIG.values()],
        ordered=True,
    )
    table = table.sort_values(["Period", "Model"], kind="stable").reset_index(drop=True)
    return table


def _plot_grouped_bars(table: pd.DataFrame, output_path: Path) -> None:
    if plt is None:
        logger.warning("matplotlib indisponible — graphique non généré")
        return

    model_order = [cfg["label"] for cfg in MODEL_CONFIG.values()]
    metric_labels = {
        "RMSE": "RMSE",
        "F1": "F1",
        "AUC": "AUC",
        "AP": "AP",
    }
    colors = {cfg["label"]: cfg["color"] for cfg in MODEL_CONFIG.values()}

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    axes = axes.flatten()

    x = np.arange(len(PERIOD_ORDER))
    width = 0.18
    offsets = np.linspace(-(len(model_order) - 1) / 2, (len(model_order) - 1) / 2, len(model_order)) * width

    for ax, metric in zip(axes, PLOT_METRICS):
        for idx, model in enumerate(model_order):
            sub = table[table["Model"] == model].set_index("Period").reindex(PERIOD_ORDER)
            values = sub[metric].astype(float).values
            bars = ax.bar(x + offsets[idx], values, width, label=model if metric == "RMSE" else None,
                          color=colors[model], alpha=0.86, edgecolor="white", linewidth=0.4)
            for bar, val in zip(bars, values):
                if pd.notna(val):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                            f"{val:.2f}", ha="center", va="bottom", fontsize=6, rotation=90)

        ax.set_title(metric_labels[metric], fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(PERIOD_ORDER, rotation=0)
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].legend(ncol=2, fontsize=9, loc="upper right")
    fig.suptitle(
        "Metrics par période — SARIMA vs SARIMAX+BESI vs SARIMAX+Hybrid vs Naif\n"
        "Seuil stress appris sur le train pré-2022, appliqué sans recalibration",
        fontweight="bold",
        fontsize=13,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _interpret(table: pd.DataFrame) -> list[str]:
    """Génère quelques phrases de lecture automatique."""
    notes: list[str] = []

    def _val(model: str, period: str, metric: str) -> float:
        row = table[(table["Model"] == model) & (table["Period"] == period)]
        if row.empty:
            return float("nan")
        return float(row.iloc[0][metric])

    sarima_b = _val("SARIMA", "Bloc B", "RMSE")
    beh_b = _val("SARIMAX+BESI", "Bloc B", "RMSE")
    delta_b = sarima_b - beh_b if pd.notna(sarima_b) and pd.notna(beh_b) else float("nan")
    if pd.notna(delta_b):
        if delta_b > 0:
            notes.append(f"SARIMAX+BESI surperforme SARIMA pur sur Bloc B (delta RMSE = {delta_b:.4f}).")
        else:
            notes.append(f"Sur Bloc B, SARIMA pur garde un léger avantage RMSE (delta RMSE = {delta_b:.4f}).")

    bloc_a_models = table[table["Period"] == "Bloc A"][["Model", "RMSE", "F1", "AUC"]].dropna()
    if not bloc_a_models.empty:
        best_rmse = bloc_a_models.sort_values("RMSE").iloc[0]
        notes.append(
            f"Sur Bloc A (COVID), la performance reste plus faible et le meilleur RMSE est porté par {best_rmse['Model']} ({best_rmse['RMSE']:.4f})."
        )

    global_auc = table[table["Period"] == "Global"]["AUC"].mean()
    bloc_b_auc = table[table["Period"] == "Bloc B"]["AUC"].mean()
    if pd.notna(global_auc) and pd.notna(bloc_b_auc):
        notes.append(
            f"L'AUC globale ({global_auc:.4f}) masque une performance Bloc B plus forte ({bloc_b_auc:.4f})."
        )

    return notes


def main() -> pd.DataFrame:
    gold = _load_gold()
    table = _build_table(gold)

    csv_path = RESULTS_DIR / "metrics_by_period.csv"
    table.to_csv(csv_path, index=False, encoding="utf-8")
    logger.info(f"Table sauvegardee : {csv_path}")

    fig_path = FIG_DIR / "metrics_by_period.png"
    _plot_grouped_bars(table, fig_path)
    if fig_path.exists():
        logger.info(f"Figure sauvegardee : {fig_path}")

    print("\n=== INTERPRETATION AUTOMATIQUE ===")
    for note in _interpret(table):
        print(f"- {note}")

    return table


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = main()
    print(f"\nShape final : {df.shape}")