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

SARIMA_ORDER    = (2, 1, 1)
SARIMA_SEASONAL = (0, 1, 1, 12)

# Ordre simplifié utilisé quand les données sont trop courtes pour le modèle complet
SARIMA_ORDER_SIMPLE    = (1, 1, 0)
SARIMA_SEASONAL_SIMPLE = (0, 1, 0, 12)
MIN_TRAIN_MONTHS = 36   # minimum absolu pour SARIMA(2,1,1)×(0,1,1)[12]


def _derive_windows(gold: pd.DataFrame) -> list:
    """
    Dérive les fenêtres d'évaluation directement depuis la colonne split_label
    du Gold dataset. Adaptatif : fonctionne avec SHORT (A,B) ou FULL (A,B,C).
    """
    windows = []
    labels_found = set()
    for cell in gold["split_label"].dropna().unique():
        for part in str(cell).split("|"):
            if "_" in part:
                labels_found.add(part.split("_")[1])   # "A", "B", "C"

    for lbl in sorted(labels_found):
        train_mask = gold["split_label"].str.contains(f"train_{lbl}", na=False)
        test_mask  = gold["split_label"].str.contains(f"test_{lbl}",  na=False)
        if not (train_mask.any() and test_mask.any()):
            continue
        train_idx = gold.index[train_mask]
        test_idx  = gold.index[test_mask]
        windows.append({
            "label": lbl,
            "train_start": train_idx.min(),
            "train_end":   train_idx.max(),
            "test_start":  test_idx.min(),
            "test_end":    test_idx.max(),
        })

    return windows


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

def _naive_predict_one(train_series: pd.Series) -> float:
    """Naïf : prévision = dernier ipc_level observé."""
    return float(train_series.iloc[-1])


def _sarima_fit(
    train_series: pd.Series,
    exog_train:   "pd.DataFrame | None" = None,
    simple:       bool = False,
):
    """
    Ajuste un SARIMA sur train_series.
    - simple=False : SARIMA(2,1,1)×(0,1,1)[12]  — modèle complet
    - simple=True  : SARIMA(1,1,0)×(0,1,0)[12]  — modèle robuste pour peu de données
    Retourne le résultat fitté, ou None si échec.
    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        raise ImportError("statsmodels requis : pip install statsmodels")

    order    = SARIMA_ORDER_SIMPLE    if simple else SARIMA_ORDER
    seasonal = SARIMA_SEASONAL_SIMPLE if simple else SARIMA_SEASONAL

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            train_series,
            exog                  = exog_train,
            order                 = order,
            seasonal_order        = seasonal,
            enforce_stationarity  = False,
            enforce_invertibility = False,
        )
        try:
            return model.fit(disp=False, maxiter=300)
        except Exception as e:
            logger.debug(f"  SARIMA fit failed : {e}")
            return None


def _safe_forecast(fit_result, n_steps: int, exog_te, train_series: pd.Series) -> float:
    """
    Récupère la prévision 1-pas et valide qu'elle est dans une plage raisonnable.
    Si la prévision est absurde (> 5× l'écart de la série train), retourne naïf.
    """
    try:
        fc = fit_result.get_forecast(steps=n_steps, exog=exog_te)
        pred = float(fc.predicted_mean.iloc[0])

        # Sanity check : la prévision doit rester dans ±30% de la plage train
        train_mean  = float(train_series.mean())
        train_range = float(train_series.max() - train_series.min()) + 1e-6
        if abs(pred - train_mean) > 5 * train_range:
            logger.debug(f"  Prediction hors bornes ({pred:.1f}) → fallback naïf")
            return _naive_predict_one(train_series)
        return pred
    except Exception:
        return _naive_predict_one(train_series)


def _walk_forward_predict(
    gold:        pd.DataFrame,
    target_col:  str,
    train_start: pd.Timestamp,
    test_dates:  pd.DatetimeIndex,
    model_type:  str,                       # "naif" | "sarima" | "sarimax"
    exog_col:    "str | None" = None,
) -> np.ndarray:
    """
    Vrai walk-forward expanding window (1-step-ahead).

    Pour chaque date t dans test_dates :
      - Train sur [data_start, t-1]  (train_start = premier mois disponible)
      - Prédit t (un seul pas)
      - Enregistre la prédiction

    Le modèle SARIMA complet (2,1,1)×(0,1,1)[12] est utilisé si >= 48 mois.
    Sinon, modèle simplifié (1,1,0)×(0,1,0)[12] pour stabilité.

    Retourne un array de longueur len(test_dates).
    """
    preds = []
    # Vrai début des données disponibles (peut être > train_start si données partielles)
    data_start = gold[target_col].dropna().index.min()
    effective_start = max(train_start, data_start)

    for i, test_date in enumerate(test_dates):
        # Fenêtre train expansible : tout jusqu'au mois PRÉCÉDANT test_date
        cutoff      = test_date - pd.offsets.MonthBegin(1)
        train_slice = gold.loc[effective_start:cutoff, target_col].dropna()

        if len(train_slice) < 24:
            preds.append(float(train_slice.iloc[-1]) if len(train_slice) > 0 else np.nan)
            continue

        if model_type == "naif":
            preds.append(_naive_predict_one(train_slice))
            continue

        # Choisir le niveau de complexité selon la taille du train
        use_simple = len(train_slice) < MIN_TRAIN_MONTHS

        # SARIMA / SARIMAX
        exog_tr = None
        exog_te = None
        if model_type == "sarimax" and exog_col and exog_col in gold.columns:
            exog_series = gold[exog_col]
            exog_tr_raw = exog_series.loc[effective_start:cutoff].reindex(train_slice.index)
            exog_tr_raw = exog_tr_raw.ffill().bfill()
            if exog_tr_raw.isna().mean() < 0.3:
                exog_tr = exog_tr_raw.to_frame(exog_col)
                # Exog pour le pas de prévision (test_date)
                te_val = exog_series.get(test_date, np.nan)
                if pd.isna(te_val):
                    te_val = exog_series.loc[:cutoff].iloc[-1] if len(exog_series.loc[:cutoff]) > 0 else 0.0
                exog_te = pd.DataFrame({exog_col: [te_val]})

        fit_result = _sarima_fit(train_slice, exog_train=exog_tr, simple=use_simple)
        if fit_result is None:
            # Réessayer avec le modèle simple si le complet échoue
            fit_result = _sarima_fit(train_slice, exog_train=exog_tr, simple=True)
        if fit_result is None:
            preds.append(_naive_predict_one(train_slice))
            continue

        preds.append(_safe_forecast(fit_result, 1, exog_te, train_slice))

        if (i + 1) % 6 == 0:
            logger.info(f"    Walk-forward {model_type} : {i+1}/{len(test_dates)} pas effectues")

    return np.array(preds)


# ─── Backtest principal ────────────────────────────────────────────────────────

def run_backtest(
    gold_path:   str | Path | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Lance le backtest complet sur les 3 blocs A/B/C.

    Méthode : expanding-window walk-forward 1-step-ahead.
    Pour chaque mois du bloc test, le modèle est ré-ajusté sur
    [train_start, mois-1] puis prédit le mois courant.
    Ce n'est PAS un forecast multi-pas sur tout le bloc test d'un coup.

    Modèles testés :
        naif               — prévision constante (dernier IPC observé)
        sarima             — SARIMA(2,1,1)×(0,1,1)[12] pur
        sarimax_behavioral — SARIMAX + behavioral_index_pure (lag1)
        sarimax_hybrid     — SARIMAX + hybrid_macro_index    (lag1)

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
    logger.info(f"Gold dataset charge : {gold.shape}")

    # Colonnes nécessaires
    target_col = "ipc_level"
    beh_col    = "behavioral_index_pure_lag1"
    hyb_col    = "hybrid_macro_index_lag1"

    if target_col not in gold.columns:
        raise KeyError(f"Colonne '{target_col}' absente du Gold dataset.")

    # Dériver les fenêtres depuis les split_labels du Gold (adaptatif FULL/SHORT)
    windows = _derive_windows(gold)
    if not windows:
        raise RuntimeError("Aucune fenetre d'evaluation trouvee dans le Gold dataset.")
    logger.info(f"Fenetres detectees : {[w['label'] for w in windows]}")

    rows        = []
    pred_frames = []

    for window in windows:
        lbl         = window["label"]
        train_start = window["train_start"]
        test_start  = window["test_start"]
        test_end    = window["test_end"]

        test  = gold.loc[test_start:test_end]
        y_test = test[target_col].dropna()
        n      = len(y_test)

        # Vérifier qu'on a assez de données train effectives
        data_start   = gold[target_col].dropna().index.min()
        eff_start    = max(train_start, data_start)
        cutoff_first = test_start - pd.offsets.MonthBegin(1)
        n_train_eff  = len(gold.loc[eff_start:cutoff_first, target_col].dropna())

        if n_train_eff < 24 or n < 6:
            logger.warning(f"Bloc {lbl} : train effectif={n_train_eff}, test={n} — ignore")
            continue

        if n == 0:
            logger.warning(f"Bloc {lbl} : aucune observation test — ignore")
            continue

        model_label = "simple" if n_train_eff < MIN_TRAIN_MONTHS else "complet"
        logger.info(f"\nBloc {lbl} | train effectif={n_train_eff} mois ({model_label}) | test={n} mois")
        logger.info(f"  -> Walk-forward 1-step-ahead")

        test_dates = y_test.index

        # ── Modèle 1 : Naïf (walk-forward) ───────────────────────────────────
        logger.info(f"  Naif...")
        pred_naif = _walk_forward_predict(
            gold, target_col, train_start, test_dates, "naif"
        )
        rows.append({
            "bloc": lbl, "model": "naif",
            "rmse": _rmse(y_test, pred_naif),
            "mae":  _mae(y_test,  pred_naif),
            "mape": _mape(y_test, pred_naif),
            "n_test": n,
        })
        pred_frames.append(_pred_frame(y_test, pred_naif, lbl, "naif"))

        # ── Modèle 2 : SARIMA walk-forward ────────────────────────────────────
        logger.info(f"  SARIMA expanding-window (~{n} fits)...")
        pred_sarima = _walk_forward_predict(
            gold, target_col, train_start, test_dates, "sarima"
        )
        rows.append({
            "bloc": lbl, "model": "sarima",
            "rmse": _rmse(y_test, pred_sarima),
            "mae":  _mae(y_test,  pred_sarima),
            "mape": _mape(y_test, pred_sarima),
            "n_test": n,
        })
        pred_frames.append(_pred_frame(y_test, pred_sarima, lbl, "sarima"))

        # ── Modèle 3 : SARIMAX + behavioral (walk-forward) ────────────────────
        if beh_col in gold.columns:
            logger.info(f"  SARIMAX + behavioral...")
            pred_beh = _walk_forward_predict(
                gold, target_col, train_start, test_dates,
                "sarimax", exog_col=beh_col,
            )
            rows.append({
                "bloc": lbl, "model": "sarimax_behavioral",
                "rmse": _rmse(y_test, pred_beh),
                "mae":  _mae(y_test,  pred_beh),
                "mape": _mape(y_test, pred_beh),
                "n_test": n,
            })
            pred_frames.append(_pred_frame(y_test, pred_beh, lbl, "sarimax_behavioral"))
        else:
            logger.warning(f"  '{beh_col}' absent du Gold — SARIMAX_behavioral ignore")

        # ── Modèle 4 : SARIMAX + hybrid (walk-forward) ────────────────────────
        if hyb_col in gold.columns:
            logger.info(f"  SARIMAX + hybrid...")
            pred_hyb = _walk_forward_predict(
                gold, target_col, train_start, test_dates,
                "sarimax", exog_col=hyb_col,
            )
            rows.append({
                "bloc": lbl, "model": "sarimax_hybrid",
                "rmse": _rmse(y_test, pred_hyb),
                "mae":  _mae(y_test,  pred_hyb),
                "mape": _mape(y_test, pred_hyb),
                "n_test": n,
            })
            pred_frames.append(_pred_frame(y_test, pred_hyb, lbl, "sarimax_hybrid"))
        else:
            logger.warning(f"  '{hyb_col}' absent du Gold — SARIMAX_hybrid ignore")

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
