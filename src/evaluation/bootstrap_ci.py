"""
src/evaluation/bootstrap_ci.py — Intervalles de confiance Bootstrap v1

Quantifie l'incertitude sur les métriques RMSE, MAE, AUC, F1, Recall, Precision
via block bootstrap et bootstrap bayésien (Rubin 1981).

Pourquoi block bootstrap ?
    Les résidus d'un SARIMA sont autocorrélés dans le temps.
    Rééchantillonner des observations indépendantes brise cette structure et
    sous-estime la variance. En rééchantillonnant des blocs consécutifs de
    taille b=6 mois, on préserve la dépendance locale.

Stratégie :
    1. Générer les prédictions UNE SEULE FOIS par modèle (walk-forward)
    2. Stocker les paires (y_reel, y_pred) par (modèle, bloc)
    3. Appliquer le block bootstrap sur ces paires → CI RMSE, MAE
    4. Pour AUC/F1 : bootstrap sur les paires (score_besi, regime_reel)
       issues du Gold dataset

Modèles évalués :
    naif               — prévision constante (dernier IPC observé)
    sarima             — SARIMA(2,1,1)x(0,1,1)[12]
    sarimax_behavioral — SARIMAX + behavioral_index_pure_lag1
    sarimax_hybrid     — SARIMAX + hybrid_macro_index_lag1

Scopes : Bloc A, Bloc B, Global (A+B concaténés)

Output :
    outputs/reports/bootstrap_ci.csv
    outputs/reports/bootstrap_ci_overlap.csv
    outputs/figures/forest_plot.png
"""

import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    matplotlib = None
    plt = None

logger = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parent.parent.parent
GOLD_DIR = ROOT / "data" / "gold"
REPORTS  = ROOT / "outputs" / "reports"
FIGURES  = ROOT / "outputs" / "figures"
REPORTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# ─── Paramètres globaux ───────────────────────────────────────────────────────

N_BOOT      = 1000
BLOCK_SIZE  = 6        # demi-année : donne ~4 blocs pour 24m, ~6 pour 36m
SEED        = 42
ALPHA       = 0.05     # IC à 95%

SARIMA_ORDER    = (2, 1, 1)
SARIMA_SEASONAL = (0, 1, 1, 12)
SARIMA_SIMPLE   = (1, 1, 0)
SEASONAL_SIMPLE = (0, 1, 0, 12)
MIN_TRAIN       = 36

TARGET_COL  = "ipc_level"
BEH_COL     = "behavioral_index_pure_lag1"
HYB_COL     = "hybrid_macro_index_lag1"
YOY_COL     = "inflation_yoy"
REGIME_PERCENTILE = 75   # seuil stress = 75e percentile YoY sur train

MODEL_COLORS = {
    "naif":               "#95a5a6",
    "sarima":             "#3498db",
    "sarimax_behavioral": "#2ecc71",
    "sarimax_hybrid":     "#e67e22",
}

BLOC_COLORS = {
    "A":      "#3498db",
    "B":      "#e74c3c",
    "global": "#7f8c8d",
}

METRICS_FORECAST     = ["RMSE", "MAE"]
METRICS_CLASSIF      = ["AUC", "F1", "Recall", "Precision"]
METRICS_ALL          = METRICS_FORECAST + METRICS_CLASSIF


# ─── Utilitaires ──────────────────────────────────────────────────────────────

def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def _mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def _roc_auc(y_true, scores):
    """AUC ROC maison (sans sklearn)."""
    y_true  = np.asarray(y_true, dtype=int)
    scores  = np.asarray(scores, dtype=float)
    pos     = int(y_true.sum())
    neg     = len(y_true) - pos
    if pos == 0 or neg == 0:
        return 0.5
    thresholds = np.unique(scores)[::-1]
    tprs, fprs = [0.0], [0.0]
    for t in thresholds:
        pred = (scores >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        tprs.append(tp / pos)
        fprs.append(fp / neg)
    tprs.append(1.0); fprs.append(1.0)
    tprs = np.array(tprs)
    fprs = np.array(fprs)
    return float(abs(
        np.trapezoid(tprs, fprs) if hasattr(np, "trapezoid") else np.trapz(tprs, fprs)
    ))


def _best_f1_recall_precision(y_true, scores):
    """
    Retourne le meilleur F1 (et les Recall/Precision associés) balayé sur tous les seuils.
    """
    y_true  = np.asarray(y_true, dtype=int)
    scores  = np.asarray(scores, dtype=float)
    pos     = int(y_true.sum())
    if pos == 0 or len(y_true) == 0:
        return 0.0, 0.0, 0.0

    best_f1, best_r, best_p = 0.0, 0.0, 0.0
    for t in np.unique(scores):
        pred = (scores >= t).astype(int)
        tp   = int(((pred == 1) & (y_true == 1)).sum())
        fp   = int(((pred == 1) & (y_true == 0)).sum())
        fn   = int(((pred == 0) & (y_true == 1)).sum())
        p    = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_r, best_p = f1, r, p

    return best_f1, best_r, best_p


# ─── Block bootstrap ─────────────────────────────────────────────────────────

def block_bootstrap_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """
    Circular block bootstrap : génère un tableau d'indices de longueur n.

    Algorithme :
        1. Choisir aléatoirement des positions de départ de blocs
        2. Étendre chaque bloc en utilisant l'indexation circulaire (% n)
        3. Concaténer jusqu'à avoir >= n indices, puis tronquer

    Le mode circulaire garantit que les blocs en fin de série peuvent
    'boucler' sur le début, évitant les bords vides.
    """
    n_blocks_needed = int(np.ceil(n / block_size))
    starts   = rng.integers(0, n, size=n_blocks_needed)
    indices  = np.concatenate([
        np.arange(s, s + block_size) % n for s in starts
    ])
    return indices[:n]


# ─── Walk-forward predictions (une seule fois par modèle) ────────────────────

def _sarima_fit_once(train_series: pd.Series, exog_train=None, simple=False):
    """Ajuste SARIMA/SARIMAX sur train complet. Retourne fit_result ou None."""
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        raise ImportError("statsmodels requis : pip install statsmodels")

    order    = SARIMA_SIMPLE    if simple else SARIMA_ORDER
    seasonal = SEASONAL_SIMPLE  if simple else SARIMA_SEASONAL

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
            logger.debug(f"  SARIMA fit echec : {e}")
            return None


def _generate_predictions_for_bloc(
    gold:        pd.DataFrame,
    train_start: pd.Timestamp,
    test_dates:  pd.DatetimeIndex,
) -> dict:
    """
    Génère les prédictions walk-forward (expanding window, 1-step-ahead)
    pour les 4 modèles sur un bloc donné.

    Retourne un dict : {"naif": array, "sarima": array, ...}
    """
    from src.evaluation.backtest import _walk_forward_predict

    logger.info(f"  Generating walk-forward predictions for {len(test_dates)} test months...")

    preds = {}

    # Naïf
    logger.info("    Modele naif...")
    preds["naif"] = _walk_forward_predict(
        gold, TARGET_COL, train_start, test_dates, "naif"
    )

    # SARIMA pur
    logger.info("    Modele SARIMA...")
    preds["sarima"] = _walk_forward_predict(
        gold, TARGET_COL, train_start, test_dates, "sarima"
    )

    # SARIMAX + behavioral
    if BEH_COL in gold.columns:
        logger.info("    Modele SARIMAX + behavioral...")
        preds["sarimax_behavioral"] = _walk_forward_predict(
            gold, TARGET_COL, train_start, test_dates, "sarimax", exog_col=BEH_COL
        )

    # SARIMAX + hybrid
    if HYB_COL in gold.columns:
        logger.info("    Modele SARIMAX + hybrid...")
        preds["sarimax_hybrid"] = _walk_forward_predict(
            gold, TARGET_COL, train_start, test_dates, "sarimax", exog_col=HYB_COL
        )

    return preds


# ─── Régime de stress (label binaire) ─────────────────────────────────────────

def _build_regime_series(gold: pd.DataFrame, train_mask: pd.Series) -> pd.Series:
    """
    Construit la série binaire du régime de stress (t+1) en calibrant
    le seuil sur TRAIN uniquement (anti-leakage).

    Utilise l'inflation YoY décalée de -1 mois → on prédit le mois suivant.
    """
    shifted = gold[YOY_COL].shift(-1)
    train_yoy = gold.loc[train_mask, YOY_COL].dropna().astype(float)
    if train_yoy.empty:
        return pd.Series(dtype=int)
    threshold = float(np.percentile(train_yoy.values, REGIME_PERCENTILE))
    regime    = (shifted >= threshold).astype(int)
    return regime, threshold


# ─── Métriques sur un échantillon bootstrap ───────────────────────────────────

def _forecast_metrics_on_sample(y_true_arr, y_pred_arr):
    """RMSE et MAE sur un tableau."""
    return {
        "RMSE": _rmse(y_true_arr, y_pred_arr),
        "MAE":  _mae(y_true_arr, y_pred_arr),
    }


def _classif_metrics_on_sample(y_regime_arr, scores_arr):
    """AUC, meilleur F1, Recall et Precision associés."""
    auc = _roc_auc(y_regime_arr, scores_arr)
    f1, rec, pre = _best_f1_recall_precision(y_regime_arr, scores_arr)
    return {
        "AUC":       auc,
        "F1":        f1,
        "Recall":    rec,
        "Precision": pre,
    }


# ─── Bootstrap bayésien (Rubin 1981) ─────────────────────────────────────────

def _bayesian_bootstrap_rmse(y_true, y_pred, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """
    Bootstrap bayésien pour le RMSE.

    Principe (Rubin 1981) :
        Au lieu de rééchantillonner les indices, on tire des poids
        w ~ Dirichlet(1,...,1) et on calcule la version pondérée du RMSE :
            RMSE_bayes = sqrt( sum_i( w_i * (y_true_i - y_pred_i)^2 ) )

    Avantages vs bootstrap classique :
        - Distribution continue → pas de problème de répétitions exactes
        - Mêmes propriétés asymptotiques (BvM Theorem)
        - Compatible avec les petits échantillons

    Retourne un array de n_boot valeurs de RMSE.
    """
    errs_sq = (np.asarray(y_true) - np.asarray(y_pred)) ** 2
    n       = len(errs_sq)
    rmses   = np.empty(n_boot)

    for i in range(n_boot):
        # Dirichlet(1,...,1) = Uniform sur le simplexe
        w        = rng.exponential(1.0, size=n)
        w       /= w.sum()
        rmses[i] = np.sqrt(float(np.dot(w, errs_sq)))

    return rmses


def _ci_from_boot(boot_values: np.ndarray, alpha: float = ALPHA):
    """Retourne (mean, lo, hi) de l'intervalle de confiance (1-alpha) = 95%."""
    lo = float(np.percentile(boot_values, 100 * alpha / 2))
    hi = float(np.percentile(boot_values, 100 * (1 - alpha / 2)))
    mu = float(np.mean(boot_values))
    return mu, lo, hi


# ─── Boucle bootstrap principale ─────────────────────────────────────────────

def run_bootstrap_ci(
    gold_path:   "str | Path | None" = None,
    n_boot:      int  = N_BOOT,
    block_size:  int  = BLOCK_SIZE,
    seed:        int  = SEED,
) -> pd.DataFrame:
    """
    Lance le block bootstrap complet.

    1. Charge le Gold dataset
    2. Pour chaque bloc (A, B) : génère les prédictions walk-forward
    3. Block bootstrap sur (y_reel, y_pred) → IC RMSE, MAE
    4. Block bootstrap sur (score_besi, regime) → IC AUC, F1, Recall, Precision
    5. Bayesian bootstrap (bonus) → IC RMSE alternatif
    6. Construit le tableau global (+ global A+B)
    7. Sauvegarde CSV + forest plot

    Retourne
    --------
    pd.DataFrame : une ligne par (modèle, bloc, scope)
    """
    if gold_path is None:
        gold_path = GOLD_DIR / "model_dataset_monthly.csv"
    gold_path = Path(gold_path)

    if not gold_path.exists():
        raise FileNotFoundError(
            f"Gold dataset introuvable : {gold_path}\n"
            "Lancer d'abord : python run_v3.py --step gold"
        )

    gold = pd.read_csv(gold_path, parse_dates=["month"], index_col="month")
    logger.info(f"Gold dataset charge : {gold.shape}")

    if TARGET_COL not in gold.columns:
        raise KeyError(f"Colonne '{TARGET_COL}' absente du Gold dataset.")
    if YOY_COL not in gold.columns:
        raise KeyError(f"Colonne '{YOY_COL}' absente du Gold dataset.")

    rng = np.random.default_rng(seed)

    # ── Dériver les fenêtres d'évaluation ─────────────────────────────────────
    from src.evaluation.backtest import _derive_windows
    windows = _derive_windows(gold)
    if not windows:
        raise RuntimeError("Aucune fenetre d'evaluation trouvee dans le Gold dataset.")
    logger.info(f"Fenetres : {[w['label'] for w in windows]}")

    # Signaux BESI disponibles
    beh_col_available = BEH_COL if BEH_COL in gold.columns else None
    hyb_col_available = HYB_COL if HYB_COL in gold.columns else None

    # ── Collecte des résultats par bloc ──────────────────────────────────────
    # pool_forecast[model][bloc] = {"y_true": arr, "y_pred": arr}
    # pool_classif[signal][bloc] = {"y_regime": arr, "scores": arr}
    pool_forecast = {
        "naif": {}, "sarima": {}, "sarimax_behavioral": {}, "sarimax_hybrid": {}
    }
    pool_classif  = {}
    if beh_col_available:
        pool_classif["behavioral"] = {}
    if hyb_col_available:
        pool_classif["hybrid"]     = {}

    for window in windows:
        lbl         = window["label"]
        train_start = window["train_start"]
        test_start  = window["test_start"]
        test_end    = window["test_end"]

        test        = gold.loc[test_start:test_end]
        y_true      = test[TARGET_COL].dropna()
        test_dates  = y_true.index
        n_test      = len(test_dates)

        if n_test < 12:
            logger.warning(f"Bloc {lbl} : test trop court ({n_test} mois) — ignore")
            continue

        # Données train effectives
        data_start  = gold[TARGET_COL].dropna().index.min()
        eff_start   = max(train_start, data_start)
        cutoff_first = test_start - pd.offsets.MonthBegin(1)
        n_train_eff = len(gold.loc[eff_start:cutoff_first, TARGET_COL].dropna())

        if n_train_eff < 24:
            logger.warning(f"Bloc {lbl} : train trop court ({n_train_eff} mois) — ignore")
            continue

        logger.info(f"\n=== Bloc {lbl} | train={n_train_eff}m | test={n_test}m ===")
        logger.info(f"  Block bootstrap : b={block_size}, n_boot={n_boot}")
        logger.info(f"  Blocs disponibles sur test : ~{int(n_test / block_size)}")

        # ── Prédictions walk-forward ────────────────────────────────────────
        preds_bloc = _generate_predictions_for_bloc(gold, train_start, test_dates)

        y_true_arr = y_true.values
        for model_name, y_pred_arr in preds_bloc.items():
            # Aligner les longueurs (sécurité)
            min_len = min(len(y_true_arr), len(y_pred_arr))
            pool_forecast[model_name][lbl] = {
                "y_true": y_true_arr[:min_len],
                "y_pred": y_pred_arr[:min_len],
            }

        # ── Données de classification ───────────────────────────────────────
        train_mask = gold["split_label"].str.contains(f"train_{lbl}", na=False)
        regime_series, stress_thresh = _build_regime_series(gold, train_mask)

        logger.info(
            f"  Seuil stress YoY (percentile {REGIME_PERCENTILE}%) "
            f"= {stress_thresh:.3f}%  |  train_mask sum={train_mask.sum()}"
        )

        regime_test = regime_series.reindex(test_dates).dropna()
        n_pos       = int(regime_test.sum())
        logger.info(f"  Regime test : {n_pos}/{len(regime_test)} positifs ({n_pos/max(len(regime_test),1)*100:.0f}%)")

        for sig_name, sig_col in [("behavioral", beh_col_available), ("hybrid", hyb_col_available)]:
            if sig_col is None or sig_col not in gold.columns:
                continue
            scores_test = gold[sig_col].reindex(regime_test.index).ffill().bfill().fillna(0)
            pool_classif[sig_name][lbl] = {
                "y_regime": regime_test.values,
                "scores":   scores_test.values,
            }

    # ── Bootstrap par bloc ────────────────────────────────────────────────────
    logger.info(f"\n=== Block Bootstrap ({n_boot} iterations par scope) ===")
    rows = []

    all_blocs   = sorted({b for d in pool_forecast.values() for b in d.keys()})
    blocs_plus_global = all_blocs + ["global"]

    for scope in blocs_plus_global:
        logger.info(f"\n--- Scope : {scope} ---")

        # ── Prédictions forecast ──────────────────────────────────────────────
        for model_name in ["naif", "sarima", "sarimax_behavioral", "sarimax_hybrid"]:
            model_data = pool_forecast.get(model_name, {})

            if scope == "global":
                # Concaténer tous les blocs disponibles
                yt_list, yp_list = [], []
                for b in all_blocs:
                    if b in model_data:
                        yt_list.append(model_data[b]["y_true"])
                        yp_list.append(model_data[b]["y_pred"])
                if not yt_list:
                    continue
                y_true_full = np.concatenate(yt_list)
                y_pred_full = np.concatenate(yp_list)
            else:
                if scope not in model_data:
                    continue
                y_true_full = model_data[scope]["y_true"]
                y_pred_full = model_data[scope]["y_pred"]

            n = len(y_true_full)
            if n < block_size * 2:
                logger.warning(
                    f"  {model_name} | {scope} : n={n} < 2 blocs — IC non fiable"
                )

            # Point estimate
            rmse_pt = _rmse(y_true_full, y_pred_full)
            mae_pt  = _mae(y_true_full,  y_pred_full)

            # Block bootstrap pour RMSE et MAE
            boot_rmse = np.empty(n_boot)
            boot_mae  = np.empty(n_boot)
            for b in range(n_boot):
                idx = block_bootstrap_indices(n, block_size, rng)
                boot_rmse[b] = _rmse(y_true_full[idx], y_pred_full[idx])
                boot_mae[b]  = _mae(y_true_full[idx],  y_pred_full[idx])

            _, rmse_lo, rmse_hi = _ci_from_boot(boot_rmse)
            _, mae_lo,  mae_hi  = _ci_from_boot(boot_mae)

            # Bootstrap bayésien pour RMSE (bonus)
            bayes_rmse = _bayesian_bootstrap_rmse(y_true_full, y_pred_full, n_boot, rng)
            _, rmse_bay_lo, rmse_bay_hi = _ci_from_boot(bayes_rmse)

            # Classification (BESI sur forecast) — non applicable pour forecast pur
            # → AUC/F1 vient des scores BESI, pas des prédictions SARIMA
            row = {
                "model":      model_name,
                "scope":      scope,
                "n_obs":      n,
                "n_blocs_bs": int(n / block_size),
                # RMSE
                "RMSE":       round(rmse_pt,    4),
                "RMSE_lo95":  round(rmse_lo,    4),
                "RMSE_hi95":  round(rmse_hi,    4),
                "RMSE_width": round(rmse_hi - rmse_lo, 4),
                # MAE
                "MAE":        round(mae_pt,     4),
                "MAE_lo95":   round(mae_lo,     4),
                "MAE_hi95":   round(mae_hi,     4),
                "MAE_width":  round(mae_hi - mae_lo, 4),
                # Bayesian CI (bonus)
                "RMSE_bayes_lo95": round(rmse_bay_lo, 4),
                "RMSE_bayes_hi95": round(rmse_bay_hi, 4),
                # Colonnes classification remplies ci-dessous
                "AUC":        np.nan, "AUC_lo95":  np.nan, "AUC_hi95":  np.nan,
                "F1":         np.nan, "F1_lo95":   np.nan, "F1_hi95":   np.nan,
                "Recall":     np.nan, "Recall_lo95":    np.nan, "Recall_hi95":    np.nan,
                "Precision":  np.nan, "Precision_lo95": np.nan, "Precision_hi95": np.nan,
            }
            rows.append(row)
            logger.info(
                f"  {model_name:24s} | scope={scope} | "
                f"RMSE={rmse_pt:.4f} [{rmse_lo:.4f}, {rmse_hi:.4f}] | "
                f"MAE={mae_pt:.4f} [{mae_lo:.4f}, {mae_hi:.4f}]"
            )

        # ── Classification bootstrap (signaux BESI) ──────────────────────────
        for sig_name in ["behavioral", "hybrid"]:
            sig_data = pool_classif.get(sig_name, {})

            if scope == "global":
                yr_list, sc_list = [], []
                for b in all_blocs:
                    if b in sig_data:
                        yr_list.append(sig_data[b]["y_regime"])
                        sc_list.append(sig_data[b]["scores"])
                if not yr_list:
                    continue
                y_reg_full  = np.concatenate(yr_list)
                scores_full = np.concatenate(sc_list)
            else:
                if scope not in sig_data:
                    continue
                y_reg_full  = sig_data[scope]["y_regime"]
                scores_full = sig_data[scope]["scores"]

            n = len(y_reg_full)

            # Point estimate
            auc_pt      = _roc_auc(y_reg_full, scores_full)
            f1_pt, r_pt, p_pt = _best_f1_recall_precision(y_reg_full, scores_full)

            # Block bootstrap pour métriques classification
            boot_auc  = np.empty(n_boot)
            boot_f1   = np.empty(n_boot)
            boot_rec  = np.empty(n_boot)
            boot_pre  = np.empty(n_boot)
            for b in range(n_boot):
                idx             = block_bootstrap_indices(n, block_size, rng)
                boot_auc[b]     = _roc_auc(y_reg_full[idx], scores_full[idx])
                f1b, rb, pb     = _best_f1_recall_precision(y_reg_full[idx], scores_full[idx])
                boot_f1[b]      = f1b
                boot_rec[b]     = rb
                boot_pre[b]     = pb

            _, auc_lo,  auc_hi  = _ci_from_boot(boot_auc)
            _, f1_lo,   f1_hi   = _ci_from_boot(boot_f1)
            _, rec_lo,  rec_hi  = _ci_from_boot(boot_rec)
            _, pre_lo,  pre_hi  = _ci_from_boot(boot_pre)

            # Modèle associé au signal
            model_key = f"sarimax_{sig_name}"
            # Mettre à jour la ligne du modèle correspondant
            for row in rows:
                if row["model"] == model_key and row["scope"] == scope:
                    row.update({
                        "AUC":           round(auc_pt,  4),
                        "AUC_lo95":      round(auc_lo,  4),
                        "AUC_hi95":      round(auc_hi,  4),
                        "F1":            round(f1_pt,   4),
                        "F1_lo95":       round(f1_lo,   4),
                        "F1_hi95":       round(f1_hi,   4),
                        "Recall":        round(r_pt,    4),
                        "Recall_lo95":   round(rec_lo,  4),
                        "Recall_hi95":   round(rec_hi,  4),
                        "Precision":     round(p_pt,    4),
                        "Precision_lo95": round(pre_lo, 4),
                        "Precision_hi95": round(pre_hi, 4),
                    })
                    break

            logger.info(
                f"  {sig_name:12s} classif | scope={scope} | "
                f"AUC={auc_pt:.3f} [{auc_lo:.3f}, {auc_hi:.3f}] | "
                f"F1={f1_pt:.3f} [{f1_lo:.3f}, {f1_hi:.3f}]"
            )

    ci_df = pd.DataFrame(rows)

    # ── Test de chevauchement des IC ──────────────────────────────────────────
    overlap_rows = _compute_overlap_tests(ci_df)
    overlap_df   = pd.DataFrame(overlap_rows)

    # ── Sauvegarde CSV ────────────────────────────────────────────────────────
    out_ci      = REPORTS / "bootstrap_ci.csv"
    out_overlap = REPORTS / "bootstrap_ci_overlap.csv"
    ci_df.to_csv(out_ci, index=False)
    overlap_df.to_csv(out_overlap, index=False)
    logger.info(f"\nIC sauvegardees : {out_ci}")
    logger.info(f"Tests chevauchement : {out_overlap}")

    # ── Forest plot ───────────────────────────────────────────────────────────
    if plt is not None:
        _plot_forest_plot(ci_df)

    return ci_df


# ─── Test de chevauchement des IC ────────────────────────────────────────────

def _compute_overlap_tests(ci_df: pd.DataFrame) -> list:
    """
    Pour chaque paire de modèles (à scope et métrique fixés),
    teste si les IC 95% se chevauchent.

    Non-chevauchement → différence statistiquement significative à ~5%.
    (Heuristique conservative ; pour une vraie signifiance, utiliser le
     test de Diebold-Mariano dans diebold_mariano.py)
    """
    rows = []
    metrics_ci = {
        "RMSE":      ("RMSE_lo95",      "RMSE_hi95"),
        "MAE":       ("MAE_lo95",       "MAE_hi95"),
        "AUC":       ("AUC_lo95",       "AUC_hi95"),
        "F1":        ("F1_lo95",        "F1_hi95"),
        "Recall":    ("Recall_lo95",    "Recall_hi95"),
        "Precision": ("Precision_lo95", "Precision_hi95"),
    }

    scopes  = ci_df["scope"].unique()
    models  = ci_df["model"].unique()

    for scope in scopes:
        sub = ci_df[ci_df["scope"] == scope]
        for metric, (lo_col, hi_col) in metrics_ci.items():
            if lo_col not in sub.columns:
                continue
            model_ci = {}
            for _, row in sub.iterrows():
                lo = row.get(lo_col, np.nan)
                hi = row.get(hi_col, np.nan)
                if not (np.isnan(lo) or np.isnan(hi)):
                    model_ci[row["model"]] = (lo, hi)

            model_list = list(model_ci.keys())
            for i in range(len(model_list)):
                for j in range(i + 1, len(model_list)):
                    m1, m2 = model_list[i], model_list[j]
                    lo1, hi1 = model_ci[m1]
                    lo2, hi2 = model_ci[m2]
                    # Chevauchement si max(lo1, lo2) < min(hi1, hi2)
                    overlap   = max(lo1, lo2) < min(hi1, hi2)
                    signif    = not overlap
                    rows.append({
                        "scope":   scope,
                        "metric":  metric,
                        "model_A": m1,
                        "model_B": m2,
                        "A_lo95":  round(lo1, 4),
                        "A_hi95":  round(hi1, 4),
                        "B_lo95":  round(lo2, 4),
                        "B_hi95":  round(hi2, 4),
                        "overlap": overlap,
                        "significant_diff": signif,
                    })
    return rows


# ─── Forest plot ─────────────────────────────────────────────────────────────

def _plot_forest_plot(ci_df: pd.DataFrame) -> None:
    """
    Forest plot 3x2 (RMSE, MAE, AUC, F1, Recall, Precision).

    Pour chaque métrique :
        - Y-axis : modèles
        - X-axis : valeur de la métrique
        - Barres d'erreur : IC 95% via block bootstrap
        - Couleur : scope (Bloc A, Bloc B, global)
    """
    metrics_config = [
        ("RMSE",      "RMSE_lo95",      "RMSE_hi95",      "RMSE (points IPC)",       False),
        ("MAE",       "MAE_lo95",       "MAE_hi95",        "MAE (points IPC)",        False),
        ("AUC",       "AUC_lo95",       "AUC_hi95",        "AUC ROC",                 True),
        ("F1",        "F1_lo95",        "F1_hi95",         "F1-score (meilleur seuil)", True),
        ("Recall",    "Recall_lo95",    "Recall_hi95",     "Recall",                  True),
        ("Precision", "Precision_lo95", "Precision_hi95",  "Precision",               True),
    ]

    model_display = {
        "naif":               "Naif",
        "sarima":             "SARIMA",
        "sarimax_behavioral": "SARIMAX+BESI",
        "sarimax_hybrid":     "SARIMAX+Hybride",
    }
    model_order = list(model_display.keys())

    scopes   = ["A", "B", "global"]
    n_scopes = len(scopes)

    n_metrics = len(metrics_config)
    n_cols    = 3
    n_rows    = int(np.ceil(n_metrics / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(18, n_rows * 5),
        squeeze=False,
    )
    fig.suptitle(
        "Bootstrap IC 95% par modele et scope (block bootstrap, b=6, n=1000)",
        fontsize=14, fontweight="bold", y=1.01,
    )

    for ax_idx, (metric, lo_col, hi_col, metric_label, higher_is_better) in enumerate(metrics_config):
        ax_row = ax_idx // n_cols
        ax_col = ax_idx  % n_cols
        ax     = axes[ax_row][ax_col]

        if lo_col not in ci_df.columns or hi_col not in ci_df.columns:
            ax.set_visible(False)
            continue

        y_pos_base = np.arange(len(model_order))
        scope_offsets = np.linspace(-0.2, 0.2, n_scopes)

        for si, scope in enumerate(scopes):
            sub = ci_df[ci_df["scope"] == scope]
            color = BLOC_COLORS.get(scope, "#7f8c8d")

            for mi, model_name in enumerate(model_order):
                row = sub[sub["model"] == model_name]
                if row.empty:
                    continue
                row = row.iloc[0]

                val = row.get(metric, np.nan)
                lo  = row.get(lo_col,  np.nan)
                hi  = row.get(hi_col,  np.nan)

                if pd.isna(val):
                    continue

                y = float(y_pos_base[mi]) + scope_offsets[si]

                xerr_lo = max(0.0, float(val - lo)) if not pd.isna(lo) else 0.0
                xerr_hi = max(0.0, float(hi - val)) if not pd.isna(hi) else 0.0

                ax.errorbar(
                    x=val, y=y,
                    xerr=[[xerr_lo], [xerr_hi]],
                    fmt="o",
                    color=color,
                    capsize=4,
                    linewidth=1.5,
                    markersize=6,
                    label=f"Bloc {scope}" if mi == 0 else "_nolegend_",
                    alpha=0.85,
                )

        ax.set_yticks(y_pos_base)
        ax.set_yticklabels([model_display.get(m, m) for m in model_order], fontsize=9)
        ax.set_xlabel(metric_label, fontsize=10)
        ax.set_title(metric_label, fontsize=11, fontweight="bold")
        ax.axvline(x=0.5 if higher_is_better else 0, color="gray",
                   linestyle="--", linewidth=0.8, alpha=0.5)
        ax.grid(axis="x", linestyle=":", alpha=0.4)

        if ax_idx == 0:
            ax.legend(loc="lower right", fontsize=8, title="Scope")

    # Masquer les axes vides
    for ax_idx in range(n_metrics, n_rows * n_cols):
        axes[ax_idx // n_cols][ax_idx % n_cols].set_visible(False)

    plt.tight_layout()
    out_path = FIGURES / "forest_plot.png"
    plt.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Forest plot sauvegarde : {out_path}")


# ─── Affichage final ──────────────────────────────────────────────────────────

def _print_summary(ci_df: pd.DataFrame) -> None:
    """Affiche le tableau de synthèse dans la console (ASCII safe)."""
    sep = "=" * 110

    print()
    print(sep)
    print("  BOOTSTRAP IC 95% -- Block Bootstrap (b=6, n=1000) + Bayesian Bootstrap (Rubin 1981)")
    print(sep)

    col_order = ["model", "scope", "n_obs", "RMSE", "RMSE_lo95", "RMSE_hi95",
                 "MAE", "MAE_lo95", "MAE_hi95",
                 "AUC", "AUC_lo95", "AUC_hi95",
                 "F1", "F1_lo95", "F1_hi95"]

    # Header
    print(f"\n  {'Modele':<24} {'Scope':<8} {'N':>4}  "
          f"{'RMSE':>7} {'[lo':>7} {'hi]':>7}  "
          f"{'MAE':>6} {'[lo':>6} {'hi]':>6}  "
          f"{'AUC':>5} {'[lo':>5} {'hi]':>5}  "
          f"{'F1':>5} {'[lo':>5} {'hi]':>5}")
    print("  " + "-" * 106)

    for scope in ["A", "B", "global"]:
        sub = ci_df[ci_df["scope"] == scope]
        if sub.empty:
            continue
        print(f"\n  -- Scope : Bloc {scope} --")
        for _, row in sub.iterrows():
            def fmt(col, w=7):
                v = row.get(col, np.nan)
                return f"{v:{w}.4f}" if not pd.isna(v) else f"{'--':>{w}}"

            print(
                f"  {row['model']:<24} {scope:<8} {int(row.get('n_obs', 0)):>4}  "
                f"{fmt('RMSE')} {fmt('RMSE_lo95')} {fmt('RMSE_hi95')}  "
                f"{fmt('MAE', 6)} {fmt('MAE_lo95', 6)} {fmt('MAE_hi95', 6)}  "
                f"{fmt('AUC', 5)} {fmt('AUC_lo95', 5)} {fmt('AUC_hi95', 5)}  "
                f"{fmt('F1', 5)} {fmt('F1_lo95', 5)} {fmt('F1_hi95', 5)}"
            )

    print()
    print(sep)
    print()

    # Chevauchement IC
    print("  INTERPRETATION -- IC 95% non-chevauchants -> difference significative")
    print("  Note : RMSE et MAE : IC bas = meilleur  |  AUC/F1 : IC haut = meilleur")
    print()

    # SARIMAX vs SARIMA sur RMSE global
    sarima_row = ci_df[(ci_df["model"] == "sarima") & (ci_df["scope"] == "global")]
    beh_row    = ci_df[(ci_df["model"] == "sarimax_behavioral") & (ci_df["scope"] == "global")]

    if not sarima_row.empty and not beh_row.empty:
        s   = sarima_row.iloc[0]
        b   = beh_row.iloc[0]
        overlap = max(s["RMSE_lo95"], b["RMSE_lo95"]) < min(s["RMSE_hi95"], b["RMSE_hi95"])
        print(f"  SARIMA RMSE IC : [{s['RMSE_lo95']:.4f}, {s['RMSE_hi95']:.4f}]")
        print(f"  BESI   RMSE IC : [{b['RMSE_lo95']:.4f}, {b['RMSE_hi95']:.4f}]")
        print(f"  Chevauchement  : {'OUI (pas de difference sig.)' if overlap else 'NON (difference sig.)'}")

    print()
    print(sep)


# ─── Point d'entrée ───────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    ci_df = run_bootstrap_ci(n_boot=N_BOOT, block_size=BLOCK_SIZE, seed=SEED)

    _print_summary(ci_df)

    print("\nFichiers generes :")
    for fname in ["bootstrap_ci.csv", "bootstrap_ci_overlap.csv"]:
        p = REPORTS / fname
        size_kb = int(p.stat().st_size / 1024) if p.exists() else 0
        print(f"  outputs/reports/{fname:<40} {size_kb} KB")
    fig_path = FIGURES / "forest_plot.png"
    size_kb = int(fig_path.stat().st_size / 1024) if fig_path.exists() else 0
    print(f"  outputs/figures/forest_plot.png                          {size_kb} KB")


if __name__ == "__main__":
    main()
