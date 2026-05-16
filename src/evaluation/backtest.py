"""
src/evaluation/backtest.py — Walk-forward Backtest v3

Évalue les modèles de prédiction sur 3 fenêtres temporelles indépendantes
(blocs A, B, C) en respectant strictement la règle as-of-date.

Modèles évalués :
    - SARIMA(2,1,1)×(0,1,1)[12]          baseline
    - SARIMAX + behavioral_index_pure
    - SARIMAX + hybrid_macro_index
    - Naïf (IPC(t) = IPC(t-1))            lower bound

Métriques :
    - RMSE, MAE, MAPE  (prévision quantitative)
    - Rapport sur target_high_inflation_regime_t1 → confié à warning_metrics.py

Output :
    outputs/reports/backtest_v3_results.csv
    outputs/reports/backtest_v3_summary.csv
    outputs/figures/backtest_v3_predictions.png
"""

import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT        = Path(__file__).resolve().parent.parent.parent
GOLD_DIR    = ROOT / "data" / "gold"
REPORTS     = ROOT / "outputs" / "reports"
FIGURES     = ROOT / "outputs" / "figures"
REPORTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# Fenêtres d'évaluation — DOIT correspondre à build_model_dataset.py
EVAL_WINDOWS = [
    {"label": "A", "train": ("2010-01-01", "2017-12-01"), "test": ("2018-01-01", "2019-12-01")},
    {"label": "B", "train": ("2010-01-01", "2019-12-01"), "test": ("2020-01-01", "2021-12-01")},
    {"label": "C", "train": ("2010-01-01", "2021-12-01"), "test": ("2022-01-01", "2024-12-01")},
]

SARIMA_ORDER        = (2, 1, 1)
SARIMA_SEASONAL     = (0, 1, 1, 12)


# ─── Métriques ────────────────────────────────────────────────────────────────

def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def _mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def _mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ─── Prédictions ──────────────────────────────────────────────────────────────

def _naive_predict(train_series: pd.Series, n_steps: int) -> np.ndarray:
    """Naïf : prévision = dernier ipc_level observé (sans drift)."""
    return np.full(n_steps, train_series.iloc[-1])


def _sarima_predict(
    train_series: pd.Series,
    n_steps:      int,
    exog_train:   "pd.Series | pd.DataFrame | None" = None,
    exog_test:    "pd.Series | pd.DataFrame | None" = None,
) -> np.ndarray:
    """
    Ajuste un SARIMA(2,1,1)×(0,1,1)[12] sur train_series,
    puis prédit n_steps mois en avant.

    Paramètres
    ----------
    train_series : pd.Series de ipc_level (fréquence MS)
    n_steps      : nombre de mois à prévoir
    exog_train   : variable(s) exogènes sur la période train (optionnel)
    exog_test    : variable(s) exogènes sur la période test  (optionnel)
    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        raise ImportError("statsmodels requis : pip install statsmodels")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        model = SARIMAX(
            train_series,
            exog             = exog_train,
            order            = SARIMA_ORDER,
            seasonal_order   = SARIMA_SEASONAL,
            enforce_stationarity  = False,
            enforce_invertibility = False,
        )
        try:
            result = model.fit(disp=False, maxiter=200)
        except Exception as e:
            logger.warning(f"SARIMA fit failed ({e}) — fallback naïf")
            return _naive_predict(train_series, n_steps)

        forecast = result.get_forecast(steps=n_steps, exog=exog_test)
        return forecast.predicted_mean.values


# ─── Backtest principal ────────────────────────────────────────────────────────

def run_backtest(
    gold_path:   str | Path | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Lance le backtest complet sur les 3 blocs A/B/C.

    Modèles testés :
        naif          — prévision constante
        sarima        — SARIMA pur
        sarimax_beh   — SARIMAX + behavioral_index_pure (lag1)
        sarimax_hyb   — SARIMAX + hybrid_macro_index    (lag1)

    Retourne
    --------
    pd.DataFrame avec une ligne par (bloc, modèle) et colonnes rmse/mae/mape.
    """
    if gold_path is None:
        gold_path = GOLD_DIR / "model_dataset_monthly.csv"
    if output_path is None:
        output_path = REPORTS / "backtest_v3_results.csv"

    gold_path   = Path(gold_path)
    output_path = Path(output_path)

    if not gold_path.exists():
        raise FileNotFoundError(
            f"Gold dataset introuvable : {gold_path}\n"
            "Lancer d'abord : from src.gold.build_model_dataset import build_gold_dataset"
        )

    gold = pd.read_csv(gold_path, parse_dates=["month"], index_col="month")
    logger.info(f"Gold dataset chargé : {gold.shape}")

    # Colonnes nécessaires
    target_col = "ipc_level"
    beh_col    = "behavioral_index_pure_lag1"   # lag1 = disponible avant publication IPC(t)
    hyb_col    = "hybrid_macro_index_lag1"

    if target_col not in gold.columns:
        raise KeyError(f"Colonne '{target_col}' absente du Gold dataset. Colonnes : {list(gold.columns)}")

    rows        = []          # résultats agrégés
    pred_frames = []          # prédictions détaillées

    for window in EVAL_WINDOWS:
        lbl        = window["label"]
        train_start = pd.Timestamp(window["train"][0])
        train_end   = pd.Timestamp(window["train"][1])
        test_start  = pd.Timestamp(window["test"][0])
        test_end    = pd.Timestamp(window["test"][1])

        train = gold.loc[train_start:train_end]
        test  = gold.loc[test_start:test_end]

        if len(train) < 24 or len(test) < 6:
            logger.warning(f"Bloc {lbl} : trop peu de données (train={len(train)}, test={len(test)}) — ignoré")
            continue

        y_train = train[target_col].dropna()
        y_test  = test[target_col].dropna()
        n       = len(y_test)

        if n == 0:
            logger.warning(f"Bloc {lbl} : aucune observation test — ignoré")
            continue

        logger.info(f"\nBloc {lbl} | train={len(y_train)} mois | test={n} mois")

        # ── Modèle 1 : Naïf ──────────────────────────────────────────────────
        pred_naif = _naive_predict(y_train, n)
        rows.append({
            "bloc": lbl, "model": "naif",
            "rmse": _rmse(y_test, pred_naif),
            "mae":  _mae(y_test,  pred_naif),
            "mape": _mape(y_test, pred_naif),
            "n_test": n,
        })
        pred_frames.append(_pred_frame(y_test, pred_naif, lbl, "naif"))

        # ── Modèle 2 : SARIMA ────────────────────────────────────────────────
        pred_sarima = _sarima_predict(y_train, n)
        rows.append({
            "bloc": lbl, "model": "sarima",
            "rmse": _rmse(y_test, pred_sarima),
            "mae":  _mae(y_test,  pred_sarima),
            "mape": _mape(y_test, pred_sarima),
            "n_test": n,
        })
        pred_frames.append(_pred_frame(y_test, pred_sarima, lbl, "sarima"))

        # ── Modèle 3 : SARIMAX + behavioral ─────────────────────────────────
        if beh_col in gold.columns:
            exog_tr = _get_exog(train, beh_col, y_train.index)
            exog_te = _get_exog(test,  beh_col, y_test.index)
            if exog_tr is not None and exog_te is not None:
                pred_beh = _sarima_predict(y_train, n, exog_train=exog_tr, exog_test=exog_te)
                rows.append({
                    "bloc": lbl, "model": "sarimax_behavioral",
                    "rmse": _rmse(y_test, pred_beh),
                    "mae":  _mae(y_test,  pred_beh),
                    "mape": _mape(y_test, pred_beh),
                    "n_test": n,
                })
                pred_frames.append(_pred_frame(y_test, pred_beh, lbl, "sarimax_behavioral"))
        else:
            logger.warning(f"  '{beh_col}' absent du Gold — SARIMAX_behavioral ignoré")

        # ── Modèle 4 : SARIMAX + hybrid ──────────────────────────────────────
        if hyb_col in gold.columns:
            exog_tr = _get_exog(train, hyb_col, y_train.index)
            exog_te = _get_exog(test,  hyb_col, y_test.index)
            if exog_tr is not None and exog_te is not None:
                pred_hyb = _sarima_predict(y_train, n, exog_train=exog_tr, exog_test=exog_te)
                rows.append({
                    "bloc": lbl, "model": "sarimax_hybrid",
                    "rmse": _rmse(y_test, pred_hyb),
                    "mae":  _mae(y_test,  pred_hyb),
                    "mape": _mape(y_test, pred_hyb),
                    "n_test": n,
                })
                pred_frames.append(_pred_frame(y_test, pred_hyb, lbl, "sarimax_hybrid"))
        else:
            logger.warning(f"  '{hyb_col}' absent du Gold — SARIMAX_hybrid ignoré")

    if not rows:
        raise RuntimeError("Aucun résultat de backtest — vérifier le Gold dataset.")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_path, index=False)
    logger.info(f"\nRésultats backtest sauvegardés : {output_path}")

    # Tableau de synthèse
    summary = results_df.groupby("model")[["rmse", "mae", "mape"]].mean().round(4)
    summary.to_csv(REPORTS / "backtest_v3_summary.csv")
    logger.info(f"Synthèse sauvegardée : {REPORTS / 'backtest_v3_summary.csv'}")

    # ── Graphique ─────────────────────────────────────────────────────────────
    if pred_frames:
        preds_df = pd.concat(pred_frames)
        _plot_predictions(preds_df, gold[target_col])

    _print_summary(results_df)
    return results_df


def _get_exog(
    df:      pd.DataFrame,
    col:     str,
    idx:     pd.DatetimeIndex,
) -> "pd.DataFrame | None":
    """Extrait et aligne une colonne exogène ; retourne None si trop de NaN."""
    s = df[col].reindex(idx)
    if s.isna().mean() > 0.3:
        logger.warning(f"  '{col}' : >{30}% NaN → exog ignorée")
        return None
    return s.ffill().bfill().to_frame(col)


def _pred_frame(
    y_true:  pd.Series,
    y_pred:  np.ndarray,
    bloc:    str,
    model:   str,
) -> pd.DataFrame:
    """Construit un DataFrame de prédictions pour le graphique."""
    return pd.DataFrame({
        "month":  y_true.index,
        "y_true": y_true.values,
        "y_pred": y_pred,
        "bloc":   bloc,
        "model":  model,
    })


def _plot_predictions(preds_df: pd.DataFrame, ipc_full: pd.Series) -> None:
    """Trace les prédictions vs réalisé pour chaque bloc × modèle."""
    blocs  = preds_df["bloc"].unique()
    models = [m for m in ["sarima", "sarimax_behavioral", "sarimax_hybrid", "naif"]
              if m in preds_df["model"].unique()]

    colors = {
        "sarima":               "#2196F3",
        "sarimax_behavioral":   "#4CAF50",
        "sarimax_hybrid":       "#FF9800",
        "naif":                 "#9E9E9E",
    }

    n_blocs = len(blocs)
    fig, axes = plt.subplots(n_blocs, 1, figsize=(14, 4 * n_blocs), squeeze=False)
    fig.suptitle("Backtest V3 — Prédictions vs Réalisé (IPC Maroc)", fontsize=14, fontweight="bold")

    for i, bloc in enumerate(sorted(blocs)):
        ax    = axes[i][0]
        sub   = preds_df[preds_df["bloc"] == bloc]

        # Fond gris : période test
        test_idx = sub[sub["model"] == models[0]]["month"].values if models else []

        # IPC réalisé (contexte élargi)
        ax.plot(ipc_full.index, ipc_full.values, color="#333333",
                linewidth=1.2, label="IPC réalisé", alpha=0.5)

        for model in models:
            m_sub = sub[sub["model"] == model]
            if m_sub.empty:
                continue
            ax.plot(m_sub["month"].values, m_sub["y_pred"].values,
                    color=colors.get(model, "black"),
                    linewidth=2.0 if model != "naif" else 1.2,
                    linestyle="--" if model == "naif" else "-",
                    label=model, alpha=0.85)

        ax.set_title(f"Bloc {bloc}", fontweight="bold")
        ax.set_ylabel("IPC (base 2017=100)")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = FIGURES / "backtest_v3_predictions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Graphique sauvegardé : {out}")


def _print_summary(results_df: pd.DataFrame) -> None:
    """Affiche un tableau récapitulatif dans la console."""
    print("\n" + "=" * 65)
    print("BACKTEST V3 — RÉSULTATS PAR BLOC")
    print("=" * 65)
    print(f"{'Bloc':<6} {'Modèle':<25} {'RMSE':>8} {'MAE':>8} {'MAPE%':>8}")
    print("-" * 65)

    for _, row in results_df.sort_values(["bloc", "rmse"]).iterrows():
        print(f"{row['bloc']:<6} {row['model']:<25} "
              f"{row['rmse']:>8.4f} {row['mae']:>8.4f} "
              f"{row['mape']:>7.2f}%")

    print("\nMOYENNE SUR TOUS LES BLOCS :")
    print(f"{'Modèle':<25} {'RMSE':>8} {'MAE':>8} {'MAPE%':>8}")
    print("-" * 45)
    for model, grp in results_df.groupby("model"):
        print(f"{model:<25} {grp['rmse'].mean():>8.4f} "
              f"{grp['mae'].mean():>8.4f} {grp['mape'].mean():>7.2f}%")
    print("=" * 65)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = run_backtest()
    print(f"\nShape résultats : {df.shape}")
