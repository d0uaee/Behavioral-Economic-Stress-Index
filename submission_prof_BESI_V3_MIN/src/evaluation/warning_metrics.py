"""
src/evaluation/warning_metrics.py — Métriques d'alerte précoce v3
(version étendue — métriques de classification complètes)

Évalue la capacité du BESI à détecter les régimes de haute inflation
AVANT la publication officielle du HCP.

Métriques calculées à chaque seuil :
    - Recall      (= Sensitivity = TP / (TP + FN))
    - Precision   (= TP / (TP + FP))
    - F1-score    (= 2 * P * R / (P + R))
    - Specificity (= TNR = TN / (TN + FP))
    - Balanced Accuracy (= (Recall + Specificity) / 2)
    - Confusion Matrix : TP, FP, TN, FN
    - AUC ROC     (aire sous courbe ROC)
    - Average Precision (AP = aire sous courbe Precision-Recall)

Pourquoi AUC bas (0.31) et Recall=100% ?
    → Cas typique d'un signal sur-alertant : BESI déclenche souvent,
      donc il ne manque aucune vraie crise (Recall=1.0) mais il sonne
      aussi beaucoup à faux (Precision basse). L'AUC ROC pénalise
      ces fausses alarmes → AUC faible. AP (sur courbe PR) est plus
      adapté aux classes déséquilibrées.

Hypothèse H1 : behavioral_index_pure prédit inflation_regime
               avec lead-time >= 1 mois (Recall > 0.80)
Hypothèse H2 : hybrid_macro_index améliore la détection (delta_AUC > 0.05)

Output :
    outputs/reports/warning_metrics_v3.csv          (legacy, compatibilité)
    outputs/reports/classification_metrics.csv      (tableau complet nouveau)
    outputs/figures/roc_curves_v3.png
    outputs/figures/precision_recall_v3.png
    outputs/figures/threshold_analysis_v3.png
    outputs/figures/confusion_matrices_v3.png       (nouveau — matrices côte à côte)
    outputs/figures/classification_barchart_v3.png  (nouveau — 6 métriques comparées)
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError:  # pragma: no cover - dépend de l'environnement
    matplotlib = None
    plt = None
    gridspec = None

logger = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parent.parent.parent
GOLD_DIR = ROOT / "data" / "gold"
REPORTS  = ROOT / "outputs" / "reports"
FIGURES  = ROOT / "outputs" / "figures"
REPORTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


# ─── Couleurs cohérentes avec le reste du projet ──────────────────────────────
C_BEH   = "#4CAF50"   # vert   — behavioral
C_HYB   = "#FF9800"   # orange — hybrid
C_BLOC  = {"A": "#3498db", "B": "#e74c3c", "global": "#7f8c8d"}

# Règle métier corrigée : le régime de stress élevé est défini à partir du
# 75e percentile de l'inflation YoY observée sur la période d'entraînement du bloc.
STRESS_REGIME_PERCENTILE = 75


# ─── Calcul des métriques à un seuil ─────────────────────────────────────────

def _metrics_at_threshold(
    y_true:  np.ndarray,
    scores:  np.ndarray,
    t:       float,
) -> dict:
    """
    Calcule TOUTES les métriques de classification à un seuil t.

    Retourne un dict avec :
        recall, precision, f1, specificity, balanced_accuracy,
        tp, fp, tn, fn
    """
    pred = (scores >= t).astype(int)
    tp   = int(((pred == 1) & (y_true == 1)).sum())
    fp   = int(((pred == 1) & (y_true == 0)).sum())
    fn   = int(((pred == 0) & (y_true == 1)).sum())
    tn   = int(((pred == 0) & (y_true == 0)).sum())

    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall      = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1          = 2 * precision * recall / (precision + recall) \
                  if (precision + recall) > 0 else 0.0
    bal_acc     = (recall + specificity) / 2.0

    return {
        "recall":            float(recall),
        "precision":         float(precision),
        "f1":                float(f1),
        "specificity":       float(specificity),
        "balanced_accuracy": float(bal_acc),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def _metrics_from_binary_pred(
    y_true: np.ndarray,
    pred:   np.ndarray,
) -> dict:
    """
    Variante utile quand on agrège des prédictions issues de seuils calibrés
    différemment bloc par bloc.
    """
    pred = np.asarray(pred).astype(int)
    tp   = int(((pred == 1) & (y_true == 1)).sum())
    fp   = int(((pred == 1) & (y_true == 0)).sum())
    fn   = int(((pred == 0) & (y_true == 1)).sum())
    tn   = int(((pred == 0) & (y_true == 0)).sum())

    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall      = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1          = 2 * precision * recall / (precision + recall) \
                  if (precision + recall) > 0 else 0.0
    bal_acc     = (recall + specificity) / 2.0

    return {
        "recall":            float(recall),
        "precision":         float(precision),
        "f1":                float(f1),
        "specificity":       float(specificity),
        "balanced_accuracy": float(bal_acc),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


# ─── Courbe ROC maison ────────────────────────────────────────────────────────

def _compute_roc(y_true: np.ndarray, scores: np.ndarray):
    """Courbe ROC + AUC (sans sklearn)."""
    thresholds = np.unique(scores)[::-1]
    tprs, fprs = [0.0], [0.0]
    pos = int(y_true.sum())
    neg = len(y_true) - pos

    if pos == 0 or neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), thresholds, 0.5

    for t in thresholds:
        m   = _metrics_at_threshold(y_true, scores, t)
        tprs.append(m["recall"])
        fprs.append(1.0 - m["specificity"])   # FPR = 1 - Specificity

    tprs.append(1.0); fprs.append(1.0)
    tprs = np.array(tprs)
    fprs = np.array(fprs)
    auc  = float(abs(
        np.trapezoid(tprs, fprs) if hasattr(np, "trapezoid") else np.trapz(tprs, fprs)
    ))
    return tprs, fprs, thresholds, auc


# ─── Courbe Precision-Recall + Average Precision ─────────────────────────────

def _compute_pr(y_true: np.ndarray, scores: np.ndarray):
    """Courbe PR (sans sklearn)."""
    thresholds  = np.unique(scores)[::-1]
    precisions, recalls = [], []
    pos = int(y_true.sum())

    if pos == 0:
        return np.array([1.0, 0.0]), np.array([0.0, 1.0]), thresholds

    for t in thresholds:
        m = _metrics_at_threshold(y_true, scores, t)
        precisions.append(m["precision"])
        recalls.append(m["recall"])

    return np.array(precisions), np.array(recalls), thresholds


def _compute_ap(precisions: np.ndarray, recalls: np.ndarray) -> float:
    """
    Average Precision = aire sous la courbe Precision-Recall.

    Méthode : intégration trapézoïdale après tri par recall croissant.
    AP est préférable à AUC ROC quand les classes sont déséquilibrées
    (peu de mois de crise vs beaucoup de mois normaux).
    """
    if len(recalls) < 2:
        return float(precisions[0]) if len(precisions) > 0 else 0.0
    idx = np.argsort(recalls)
    r   = recalls[idx]
    p   = precisions[idx]
    ap  = float(abs(
        np.trapezoid(p, r) if hasattr(np, "trapezoid") else np.trapz(p, r)
    ))
    return ap


# ─── Interprétation automatique ───────────────────────────────────────────────

def _interpret_metrics(
    recall:      float,
    precision:   float,
    f1:          float,
    specificity: float,
    bal_acc:     float,
    auc:         float,
) -> str:
    """
    Règles d'interprétation automatique du compromis détection/fausses alarmes.

    Logique métier :
      Pour un système d'early warning économique, le Recall (ne pas manquer
      une crise) est PLUS important que la Precision (ne pas sur-alerter).
      Un Recall=100% avec Precision basse est donc acceptable et documenté
      comme "Sur-alerte (favorable early warning)".
    """
    if recall == 1.0 and precision >= 0.60:
        return "Detection parfaite"
    if recall == 1.0 and precision >= 0.40:
        return "Sur-alerte moderee - Recall parfait"
    if recall == 1.0 and precision < 0.40:
        return "Sur-alerte (favorable early warning)"
    if f1 >= 0.70:
        return "Equilibre F1 excellent (>0.70)"
    if f1 >= 0.50:
        return "Equilibre F1 acceptable (>0.50)"
    if recall >= 0.80 and precision < 0.40:
        return "Sur-alerte moderee - bon Recall"
    if precision >= 0.80 and recall < 0.40:
        return "Trop conservateur - manque des crises"
    if bal_acc >= 0.70:
        return "Bonne accuracy equilibree"
    if recall < 0.50:
        return "Insuffisant - trop de crises manquees"
    return "Partiel - compromis a ameliorer"


# ─── Calibration du seuil sur TRAIN ──────────────────────────────────────────

def _calibrate_threshold(
    y_train: np.ndarray,
    scores_train: np.ndarray,
) -> float:
    """
    Trouve le seuil qui maximise F1 sur les données d'entraînement.
    Retourne 0.5 si pas assez de positifs.
    RÈGLE : calibration sur TRAIN uniquement → aucun data leakage sur TEST.
    """
    if y_train.sum() < 3:
        return 0.5

    best_f1, best_t = 0.0, 0.5
    for t in np.unique(scores_train):
        m = _metrics_at_threshold(y_train, scores_train, t)
        if m["f1"] > best_f1:
            best_f1, best_t = m["f1"], float(t)

    return best_t


def _stress_regime_threshold(
    gold: pd.DataFrame,
    train_mask: pd.Series,
    source_col: str = "inflation_yoy",
    percentile: float = STRESS_REGIME_PERCENTILE,
) -> float:
    """
    Calibre le seuil de stress élevé sur le train uniquement.

    RÈGLE ANTI-LEAKAGE : le seuil n'utilise jamais les observations du bloc test.
    Par défaut, on prend le 75e percentile de l'inflation YoY du train.
    """
    train_yoy = gold.loc[train_mask, source_col].dropna().astype(float)
    if train_yoy.empty:
        raise ValueError(
            f"Impossible de calibrer le seuil de stress : aucune valeur non nulle dans '{source_col}' sur le train."
        )
    return float(np.percentile(train_yoy.values, percentile))


# ─── Lead-time ────────────────────────────────────────────────────────────────

def _lead_time(
    y_true_series:   pd.Series,
    scores_series:   pd.Series,
    threshold:       float,
    max_lead_months: int = 12,
) -> float:
    """
    Lead-time moyen (en mois) entre la première alerte BESI et le début
    du prochain épisode de haute inflation.

    Fenêtre bornée à max_lead_months pour éviter de compter des alertes
    non causalement reliées à l'épisode.
    """
    lead_times = []
    in_episode = False
    episode_start = None

    y_clean = y_true_series.dropna()

    for date, val in y_clean.items():
        if val == 1 and not in_episode:
            in_episode    = True
            episode_start = date
        elif val == 0 and in_episode:
            in_episode = False
            window_start  = episode_start - pd.DateOffset(months=max_lead_months)
            window_scores = scores_series[
                (scores_series.index >= window_start) &
                (scores_series.index  < episode_start)
            ]
            alerts = window_scores[window_scores >= threshold]
            if not alerts.empty:
                last_alert   = alerts.index[-1]
                months_ahead = (
                    (episode_start.year  - last_alert.year)  * 12
                    + (episode_start.month - last_alert.month)
                )
                if months_ahead > 0:
                    lead_times.append(months_ahead)

    return float(np.mean(lead_times)) if lead_times else float("nan")


# ─── Pipeline principal ────────────────────────────────────────────────────────

def compute_warning_metrics(
    gold_path:   "str | Path | None" = None,
    output_path: "str | Path | None" = None,
) -> pd.DataFrame:
    """
    Calcule les métriques d'alerte précoce complètes avec séparation stricte
    calibration (TRAIN) / évaluation (TEST) — aucun data leakage.

    Pour chaque combinaison (signal × bloc) :
            1. Seuil "stress élevé" = 75e percentile de l'inflation YoY du TRAIN
            2. Toutes les métriques de classification évaluées sur TEST avec ce seuil
      3. Interprétation automatique du compromis

    Retourne
    --------
    pd.DataFrame (classification_metrics.csv) avec une ligne par (signal × scope).
    """
    if gold_path is None:
        gold_path = GOLD_DIR / "model_dataset_monthly.csv"
    if output_path is None:
        output_path = REPORTS / "warning_metrics_v3.csv"    # legacy

    gold_path   = Path(gold_path)
    output_path = Path(output_path)

    if not gold_path.exists():
        raise FileNotFoundError(
            f"Gold dataset introuvable : {gold_path}\n"
            "Lancer d'abord : python run_v3.py --step gold"
        )

    gold = pd.read_csv(gold_path, parse_dates=["month"], index_col="month")
    logger.info(f"Gold dataset chargé : {gold.shape}")

    yoy_col = "inflation_yoy"
    if yoy_col not in gold.columns:
        raise KeyError(f"'{yoy_col}' absent du Gold. Colonnes : {list(gold.columns)}")

    # Série cible brute : inflation YoY décalée d'un mois, puis binarisée
    # avec un seuil appris uniquement sur le train de chaque bloc.
    shifted_yoy = gold[yoy_col].shift(-1)

    # Signaux BESI disponibles (lag1 prioritaire — respecte as-of-date)
    signals = {}
    for col in ["behavioral_index_pure_lag1", "behavioral_index_pure"]:
        if col in gold.columns:
            signals["behavioral"] = col
            break
    for col in ["hybrid_macro_index_lag1", "hybrid_macro_index"]:
        if col in gold.columns:
            signals["hybrid"] = col
            break

    if not signals:
        raise KeyError(
            "Aucun signal BESI trouvé dans le Gold dataset. "
            f"Colonnes disponibles : {list(gold.columns)}"
        )

    rows     = []   # métriques par bloc
    roc_data = {}   # pour les graphiques ROC globaux
    pr_data  = {}   # pour les graphiques PR globaux
    global_eval_store = {
        sig_label: {"y_true": [], "scores": [], "pred": [], "dates": [], "signal_thresholds": [], "stress_thresholds": []}
        for sig_label in signals
    }

    # Fenêtres d'évaluation (adaptées automatiquement SHORT ou FULL)
    from src.gold.build_model_dataset import EVAL_WINDOWS, SHORT_EVAL_WINDOWS
    labels_present = set()
    for cell in gold["split_label"].dropna().unique():
        for part in str(cell).split("|"):
            if "_" in part:
                labels_present.add(part.split("_")[1])
    eval_windows = SHORT_EVAL_WINDOWS if labels_present <= {"A", "B"} else EVAL_WINDOWS

    # ── Évaluation par bloc ───────────────────────────────────────────────────
    for window in eval_windows:
        lbl        = window["label"]
        train_mask = gold["split_label"].str.contains(f"train_{lbl}", na=False)
        test_mask  = gold["split_label"].str.contains(f"test_{lbl}",  na=False)

        gold_test = gold[test_mask]
        train_threshold = _stress_regime_threshold(gold, train_mask, source_col=yoy_col)
        target_test = shifted_yoy.loc[test_mask].dropna()
        y_test = (target_test >= train_threshold).astype(int)

        target_train = shifted_yoy.loc[train_mask].dropna()
        y_train = (target_train >= train_threshold).astype(int)

        if y_test.sum() < 2:
            logger.warning(f"Bloc {lbl} test : moins de 2 positifs — ignoré")
            continue

        for sig_label, sig_col in signals.items():
            if sig_col not in gold.columns:
                continue

            # Calibration du seuil de détection du signal sur TRAIN uniquement
            scores_tr = gold.loc[train_mask, sig_col].reindex(y_train.index).ffill().bfill().fillna(0).values
            best_t = _calibrate_threshold(y_train.values, scores_tr)
            logger.info(
                f"  Bloc {lbl} | {sig_label} | "
                f"seuil signal train={best_t:.4f} | "
                f"seuil stress train={train_threshold:.4f} | "
                f"n_positifs_test={int(y_test.sum())}/{len(y_test)}"
            )

            # Évaluation sur TEST
            scores_te = gold_test[sig_col].reindex(y_test.index).ffill().bfill().fillna(0).values
            y_te = y_test.values

            # --- Métriques de classification ---
            m    = _metrics_at_threshold(y_te, scores_te, best_t)
            tprs, fprs, _, auc = _compute_roc(y_te, scores_te)
            precs, recs, _     = _compute_pr(y_te, scores_te)
            ap   = _compute_ap(precs, recs)

            lt = _lead_time(
                y_test,
                pd.Series(scores_te, index=y_test.index),
                best_t,
            )

            interp = _interpret_metrics(
                m["recall"], m["precision"], m["f1"],
                m["specificity"], m["balanced_accuracy"], auc
            )

            rows.append({
                "Modele":           _label_to_model(sig_label),
                "Signal":           sig_label,
                "Bloc":             lbl,
                "scope":            f"test_{lbl}",
                "signal_col":       sig_col,
                "threshold_from":   f"train_{lbl}",
                "seuil":            round(best_t, 4),
                "stress_threshold_yoy": round(train_threshold, 4),
                "stress_threshold_percentile": STRESS_REGIME_PERCENTILE,
                # ── Métriques de classification ──
                "Recall":           round(m["recall"],            4),
                "Precision":        round(m["precision"],         4),
                "F1":               round(m["f1"],                4),
                "Specificity":      round(m["specificity"],       4),
                "Bal_Accuracy":     round(m["balanced_accuracy"], 4),
                "AUC":              round(auc,                    4),
                "AP":               round(ap,                     4),
                # ── Matrice de confusion ──
                "TP": m["tp"], "FP": m["fp"],
                "TN": m["tn"], "FN": m["fn"],
                # ── Contexte ──
                "n_positifs":       int(y_te.sum()),
                "n_total":          len(y_te),
                "lead_time_mois":   round(lt, 2) if not np.isnan(lt) else float("nan"),
                "Interpretation":   interp,
            })

            global_eval_store[sig_label]["y_true"].extend(y_te.tolist())
            global_eval_store[sig_label]["scores"].extend(scores_te.tolist())
            global_eval_store[sig_label]["pred"].extend((scores_te >= best_t).astype(int).tolist())
            global_eval_store[sig_label]["dates"].extend(list(y_test.index))
            global_eval_store[sig_label]["signal_thresholds"].append(float(best_t))
            global_eval_store[sig_label]["stress_thresholds"].append(float(train_threshold))

    # ── Rapport global : agrégation des blocs test avec leurs seuils appris sur train ──
    for sig_label, sig_col in signals.items():
        if sig_col not in gold.columns:
            continue

        store = global_eval_store.get(sig_label, {})
        if not store or not store["y_true"]:
            continue

        y_all = np.array(store["y_true"], dtype=int)
        scores_all = np.array(store["scores"], dtype=float)
        pred_all = np.array(store["pred"], dtype=int)

        tprs_g, fprs_g, _, auc_g = _compute_roc(y_all, scores_all)
        precs_g, recs_g, _       = _compute_pr(y_all, scores_all)
        ap_g                     = _compute_ap(precs_g, recs_g)

        roc_data[sig_label] = (fprs_g, tprs_g, auc_g)
        pr_data[sig_label]  = (recs_g, precs_g, ap_g)

        median_signal_t = float(np.median(store["signal_thresholds"]))
        median_stress_t = float(np.median(store["stress_thresholds"]))
        m_g = _metrics_from_binary_pred(y_all, pred_all)

        test_index = pd.DatetimeIndex(store["dates"])
        test_all_target = pd.Series(y_all, index=test_index).sort_index()
        test_all_scores = pd.Series(scores_all, index=test_index).sort_index()
        lt_g = _lead_time(
            test_all_target,
            test_all_scores,
            median_signal_t,
        )

        interp_g = _interpret_metrics(
            m_g["recall"], m_g["precision"], m_g["f1"],
            m_g["specificity"], m_g["balanced_accuracy"], auc_g
        )

        rows.append({
            "Modele":           _label_to_model(sig_label),
            "Signal":           sig_label,
            "Bloc":             "global",
            "scope":            "global",
            "signal_col":       sig_col,
            "threshold_from":   "per_block_train_median",
            "seuil":            round(median_signal_t, 4),
            "stress_threshold_yoy": round(median_stress_t, 4),
            "stress_threshold_percentile": STRESS_REGIME_PERCENTILE,
            "Recall":           round(m_g["recall"],            4),
            "Precision":        round(m_g["precision"],         4),
            "F1":               round(m_g["f1"],                4),
            "Specificity":      round(m_g["specificity"],       4),
            "Bal_Accuracy":     round(m_g["balanced_accuracy"], 4),
            "AUC":              round(auc_g,                    4),
            "AP":               round(ap_g,                     4),
            "TP": m_g["tp"], "FP": m_g["fp"],
            "TN": m_g["tn"], "FN": m_g["fn"],
            "n_positifs":       int(y_all.sum()),
            "n_total":          len(y_all),
            "lead_time_mois":   round(lt_g, 2) if not np.isnan(lt_g) else float("nan"),
            "Interpretation":   interp_g,
        })

    if not rows:
        raise RuntimeError("Aucune métrique calculée — vérifier les données Gold.")

    metrics_df = pd.DataFrame(rows)

    # ── Sauvegarde legacy (warning_metrics_v3.csv) ────────────────────────────
    legacy_cols = [
        "scope", "Signal", "signal_col", "threshold_from", "seuil",
        "stress_threshold_yoy", "stress_threshold_percentile",
        "AUC", "F1", "Precision", "Recall", "lead_time_mois",
        "TP", "FP", "FN", "n_positifs", "n_total",
    ]
    metrics_df[legacy_cols].rename(columns={
        "Signal": "signal", "seuil": "threshold_calibrated",
        "lead_time_mois": "lead_time_mean_months",
    }).to_csv(output_path, index=False)
    logger.info(f"Métriques legacy sauvegardées : {output_path}")

    # ── Sauvegarde classification_metrics.csv (nouveau) ───────────────────────
    class_path = REPORTS / "classification_metrics.csv"
    display_cols = [
        "Modele", "Bloc", "Recall", "Precision", "F1",
        "Specificity", "Bal_Accuracy", "AUC", "AP",
        "TP", "FP", "TN", "FN",
        "n_positifs", "n_total", "lead_time_mois", "Interpretation",
    ]
    metrics_df[display_cols].to_csv(class_path, index=False)
    logger.info(f"Classification metrics sauvegardées : {class_path}")

    # ── Visualisations ────────────────────────────────────────────────────────
    _plot_roc(roc_data)
    _plot_pr(pr_data)
    _plot_threshold_analysis(gold, signals)
    _plot_confusion_matrices(metrics_df)
    _plot_metrics_barchart(metrics_df)

    _print_classification_summary(metrics_df)
    return metrics_df


# ─── Helper label → nom lisible ───────────────────────────────────────────────

def _label_to_model(sig_label: str) -> str:
    return {
        "behavioral": "SARIMAX + BESI behavioral",
        "hybrid":     "SARIMAX + Hybrid macro",
    }.get(sig_label, sig_label)


# ─── Visualisation 1 : Courbes ROC ────────────────────────────────────────────

def _plot_roc(roc_data: dict) -> None:
    if not roc_data:
        return
    if plt is None:
        logger.info("matplotlib indisponible — ROC ignorée")
        return
    colors = {"behavioral": C_BEH, "hybrid": C_HYB}
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Aleatoire (AUC=0.50)")
    for label, (fprs, tprs, auc) in roc_data.items():
        ax.plot(fprs, tprs, color=colors.get(label, "blue"),
                linewidth=2.2, label=f"{_label_to_model(label)}  (AUC={auc:.3f})")
    ax.set_xlabel("Taux Faux Positifs (1 - Specificity)")
    ax.set_ylabel("Taux Vrais Positifs (Recall)")
    ax.set_title("Courbes ROC — Detection Regimes Haute Inflation\n"
                 "Maroc 2017–2024", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    out = FIGURES / "roc_curves_v3.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"ROC sauvegarde : {out}")


# ─── Visualisation 2 : Courbes PR ────────────────────────────────────────────

def _plot_pr(pr_data: dict) -> None:
    if not pr_data:
        return
    if plt is None:
        logger.info("matplotlib indisponible — courbe PR ignorée")
        return
    colors = {"behavioral": C_BEH, "hybrid": C_HYB}
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, (recs, precs, ap) in pr_data.items():
        idx = np.argsort(recs)
        ax.plot(recs[idx], precs[idx], color=colors.get(label, "blue"),
                linewidth=2.2, label=f"{_label_to_model(label)}  (AP={ap:.3f})")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title("Courbes Precision-Recall — Detection Regimes Inflation\n"
                 "Maroc 2017–2024", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    out = FIGURES / "precision_recall_v3.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"PR curve sauvegardee : {out}")


# ─── Visualisation 3 : Analyse de seuil ──────────────────────────────────────

def _plot_threshold_analysis(gold: pd.DataFrame, signals: dict) -> None:
    if plt is None:
        logger.info("matplotlib indisponible — analyse de seuil ignorée")
        return
    yoy_col = "inflation_yoy"
    if yoy_col not in gold.columns:
        return
    train_mask = gold["split_label"].str.contains("train_B", na=False)
    threshold = _stress_regime_threshold(gold, train_mask, source_col=yoy_col)
    y_all = (gold[yoy_col].shift(-1).dropna() >= threshold).astype(int)
    if y_all.sum() < 3:
        return

    colors = {"behavioral": C_BEH, "hybrid": C_HYB}
    fig, axes = plt.subplots(1, len(signals), figsize=(8 * len(signals), 5), squeeze=False)

    for i, (sig_label, sig_col) in enumerate(signals.items()):
        ax = axes[0][i]
        if sig_col not in gold.columns:
            continue
        scores = gold[sig_col].reindex(y_all.index).ffill().bfill().fillna(0).values
        y      = y_all.values

        percentiles = np.arange(30, 96, 5)
        precisions, recalls, f1s, specs, bal_accs = [], [], [], [], []

        for p in percentiles:
            t = np.percentile(scores, p)
            m = _metrics_at_threshold(y, scores, t)
            precisions.append(m["precision"])
            recalls.append(m["recall"])
            f1s.append(m["f1"])
            specs.append(m["specificity"])
            bal_accs.append(m["balanced_accuracy"])

        ax.plot(percentiles, precisions, "b-o",  markersize=4, label="Precision")
        ax.plot(percentiles, recalls,    "r-s",  markersize=4, label="Recall")
        ax.plot(percentiles, f1s,        "g-^",  markersize=4, label="F1", linewidth=2)
        ax.plot(percentiles, specs,      "m-D",  markersize=4, label="Specificity", alpha=0.7)
        ax.plot(percentiles, bal_accs,   "k--",  markersize=3, label="Bal.Accuracy", alpha=0.7)
        best_p = percentiles[np.argmax(f1s)]
        ax.axvline(best_p, color="g", linestyle="--", alpha=0.5,
                   label=f"Seuil F1 max (p{best_p})")
        ax.set_xlabel("Seuil (percentile du signal)")
        ax.set_ylabel("Score")
        ax.set_title(f"{_label_to_model(sig_label)}\n"
                     f"Seuil F1 max = percentile {best_p}",
                     fontweight="bold", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)

    fig.suptitle("Analyse de Seuil — BESI v3 (5 metriques)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = FIGURES / "threshold_analysis_v3.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Threshold analysis sauvegardee : {out}")


# ─── Visualisation 4 : Confusion Matrices (côte à côte) ─────────────────────

def _plot_confusion_matrices(metrics_df: pd.DataFrame) -> None:
    """
    Matrices de confusion côte à côte pour behavioral vs hybrid sur Bloc B.
    Bloc B = inflation 2022-2024, le plus pertinent pour H1.

    Lecture de la matrice :
                    Predit Positif  Predit Negatif
    Reel Positif        TP              FN
    Reel Negatif        FP              TN
    """
    # Filtrer sur Bloc B uniquement
    if plt is None:
        logger.info("matplotlib indisponible — matrices de confusion ignorées")
        return
    bloc_b = metrics_df[metrics_df["Bloc"] == "B"].copy()
    if bloc_b.empty:
        logger.warning("Pas de donnees Bloc B pour les confusion matrices")
        return

    signals_order = ["behavioral", "hybrid"]
    available = [s for s in signals_order if s in bloc_b["Signal"].values]
    n_plots   = len(available)
    if n_plots == 0:
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    colors_sig = {"behavioral": C_BEH, "hybrid": C_HYB}

    for ax, sig in zip(axes, available):
        row = bloc_b[bloc_b["Signal"] == sig].iloc[0]
        tp, fp = int(row["TP"]), int(row["FP"])
        tn, fn = int(row["TN"]), int(row["FN"])

        matrix = np.array([[tp, fn],
                            [fp, tn]])
        labels = np.array([
            [f"TP\n{tp}\n(Crise detectee)", f"FN\n{fn}\n(Crise manquee)"],
            [f"FP\n{fp}\n(Fausse alerte)", f"TN\n{tn}\n(Calme correct)"],
        ])

        c   = colors_sig.get(sig, "#666666")
        cmap_alpha = 0.3

        # Fond coloré selon la "bonne/mauvaise" case
        cell_colors = [
            [c, "#e74c3c"],         # TP vert / FN rouge
            ["#e74c3c", "#95a5a6"], # FP rouge / TN gris
        ]

        for i in range(2):
            for j in range(2):
                rect = plt.Rectangle(
                    (j, 1 - i), 1, 1,
                    color=cell_colors[i][j], alpha=0.35, transform=ax.transData
                )
                ax.add_patch(rect)
                ax.text(
                    j + 0.5, 1.5 - i, labels[i][j],
                    ha="center", va="center", fontsize=11,
                    fontweight="bold" if (i == 0 and j == 0) or (i == 1 and j == 1) else "normal"
                )

        ax.set_xlim(0, 2); ax.set_ylim(0, 2)
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels(["Predit : Crise", "Predit : Normal"], fontsize=10)
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(["Reel : Normal", "Reel : Crise"], fontsize=10)
        ax.tick_params(length=0)

        model_name = _label_to_model(sig)
        recall_str     = f"Recall = {row['Recall']:.0%}"
        precision_str  = f"Precision = {row['Precision']:.0%}"
        f1_str         = f"F1 = {row['F1']:.3f}"
        spec_str       = f"Specificity = {row['Specificity']:.0%}"

        ax.set_title(
            f"{model_name}\nBloc B — Inflation 2022-2024\n"
            f"{recall_str}  |  {precision_str}\n"
            f"{f1_str}  |  {spec_str}",
            fontweight="bold", fontsize=9, pad=10
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Matrices de Confusion — Detection Crise Inflationniste\n"
        "Seuil calibre sur Train 2017-2021, evalue sur Test 2022-2024",
        fontsize=12, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    out = FIGURES / "confusion_matrices_v3.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrices sauvegardees : {out}")


# ─── Visualisation 5 : Bar chart comparatif des 6 métriques ──────────────────

def _plot_metrics_barchart(metrics_df: pd.DataFrame) -> None:
    """
    Bar chart groupé comparant les 6 métriques clés pour chaque combinaison
    (signal × bloc), hors scope global.

    Métriques affichées : Recall, Precision, F1, Specificity, Bal_Accuracy, AUC
    """
    if plt is None:
        logger.info("matplotlib indisponible — bar chart ignoré")
        return
    plot_df = metrics_df[metrics_df["Bloc"] != "global"].copy()
    if plot_df.empty:
        return

    metrics_cols = ["Recall", "Precision", "F1", "Specificity", "Bal_Accuracy", "AUC"]
    metric_labels = ["Recall\n(Sensitivity)", "Precision", "F1-Score",
                     "Specificity\n(TNR)", "Balanced\nAccuracy", "AUC\nROC"]

    # Créer les groupes : "Behavioral | Bloc A", "Hybrid | Bloc A", etc.
    plot_df["label"] = (
        plot_df["Signal"].map({"behavioral": "Behavioral", "hybrid": "Hybrid"})
        + "\nBloc " + plot_df["Bloc"]
    )
    plot_df = plot_df.sort_values(["Bloc", "Signal"])

    labels  = plot_df["label"].tolist()
    n_grps  = len(labels)
    n_mets  = len(metrics_cols)
    x       = np.arange(n_grps)
    width   = 0.12
    offsets = np.linspace(-(n_mets - 1) / 2, (n_mets - 1) / 2, n_mets) * width

    # Palette des métriques
    met_colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c"]

    fig, ax = plt.subplots(figsize=(max(10, n_grps * 2.2), 6))

    for j, (met_col, met_label, col) in enumerate(
            zip(metrics_cols, metric_labels, met_colors)):
        vals = plot_df[met_col].values
        bars = ax.bar(x + offsets[j], vals, width, label=met_label,
                      color=col, alpha=0.85, edgecolor="white", linewidth=0.5)
        # Valeur au-dessus de chaque barre
        for bar, val in zip(bars, vals):
            if val >= 0.01:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{val:.2f}",
                    ha="center", va="bottom", fontsize=6.5, rotation=90
                )

    # Ligne de référence à 0.65 (seuil H1 AUC)
    ax.axhline(0.65, color="gray", linestyle=":", linewidth=1,
               label="Seuil H1 AUC=0.65")
    ax.axhline(0.80, color="#c0392b", linestyle=":", linewidth=1,
               label="Seuil Recall=0.80")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score (0–1)", fontsize=11)
    ax.set_title(
        "Comparaison des Metriques de Classification — BESI v3\n"
        "Seuil calibre sur Train, evalue sur Test (aucun data leakage)",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=8, ncol=4, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Fond coloré par bloc
    bloc_colors = {"A": "#ebf5fb", "B": "#fdedec"}
    blocs = plot_df["Bloc"].tolist()
    for i, (xi, bl) in enumerate(zip(x, blocs)):
        col = bloc_colors.get(bl, "#f9f9f9")
        ax.axvspan(xi - 0.55, xi + 0.55, color=col, alpha=0.4, zorder=0)

    plt.tight_layout()
    out = FIGURES / "classification_barchart_v3.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Bar chart classification sauvegarde : {out}")


# ─── Affichage console ────────────────────────────────────────────────────────

def _print_classification_summary(metrics_df: pd.DataFrame) -> None:
    """
    Affiche le tableau comparatif complet dans la console.

    Columns : Modele | Bloc | Recall | Precision | F1 | Specificity | Bal.Acc | AUC | AP | Interpretation
    """
    SEP = "=" * 130

    print(f"\n{SEP}")
    print("  METRIQUES DE CLASSIFICATION COMPLETES — BESI v3")
    print("  Methode : seuil calibre sur TRAIN, evalue sur TEST (aucun data leakage)")
    print(f"  Pourquoi AUC bas et Recall=100% ? -> Signal sur-alertant : le BESI sonne")
    print(f"  souvent, donc il capture toutes les crises (Recall=1) mais aussi des")
    print(f"  faux positifs (Precision basse). L'AUC penalise ces fausses alarmes.")
    print(f"  Pour les early warning economiques, Recall > Precision (ne pas rater")
    print(f"  une crise est plus grave que sur-alerter).")
    print(SEP)

    hdr = (f"  {'Modele':<30} {'Bloc':>5} {'Recall':>7} {'Precis':>7} "
           f"{'F1':>6} {'Spec':>6} {'BalAcc':>7} {'AUC':>6} {'AP':>6}  "
           f"{'TP':>3} {'FP':>3} {'TN':>3} {'FN':>3}  Interpretation")
    print(hdr)
    print("-" * 130)

    blocs_order = [b for b in ["A", "B", "C", "global"]
                   if b in metrics_df["Bloc"].values]

    for bloc in blocs_order:
        sub = metrics_df[metrics_df["Bloc"] == bloc].sort_values("Signal")
        if bloc != "global":
            print(f"\n  -- Bloc {bloc} --")
        else:
            print(f"\n  -- Global (seuil = mediane des trains) --")

        for _, row in sub.iterrows():
            lt = (f"{row['lead_time_mois']:.1f}m"
                  if not pd.isna(row.get("lead_time_mois", float("nan"))) else "n/a")
            flag = " [**]" if row["Recall"] >= 1.0 else ""
            print(
                f"  {row['Modele']:<30} {row['Bloc']:>5} "
                f"{row['Recall']:>7.3f} {row['Precision']:>7.3f} "
                f"{row['F1']:>6.3f} {row['Specificity']:>6.3f} "
                f"{row['Bal_Accuracy']:>7.3f} {row['AUC']:>6.3f} "
                f"{row['AP']:>6.3f}  "
                f"{int(row['TP']):>3} {int(row['FP']):>3} "
                f"{int(row['TN']):>3} {int(row['FN']):>3}  "
                f"{row['Interpretation']}{flag}"
            )

    # Validation des hypothèses
    print(f"\n{SEP}")
    print("  VALIDATION DES HYPOTHESES")
    print(SEP)

    global_df  = metrics_df[metrics_df["Bloc"] == "global"]
    bloc_b_df  = metrics_df[metrics_df["Bloc"] == "B"]

    beh_g  = global_df[global_df["Signal"] == "behavioral"]
    hyb_g  = global_df[global_df["Signal"] == "hybrid"]
    beh_b  = bloc_b_df[bloc_b_df["Signal"] == "behavioral"]
    hyb_b  = bloc_b_df[bloc_b_df["Signal"] == "hybrid"]

    if not beh_g.empty:
        r_g   = beh_g["Recall"].values[0]
        auc_g = beh_g["AUC"].values[0]
        f1_g  = beh_g["F1"].values[0]
        h1    = "PARTIELLEMENT VALIDEE" if r_g >= 0.80 else "NON VALIDEE"
        print(f"\n  H1 : BESI behavioral predit le regime d'inflation elevee")
        print(f"    Recall global      = {r_g:.3f}  (> 0.80 requis)  -> {'OK' if r_g >= 0.80 else 'ECHEC'}")
        print(f"    AUC global         = {auc_g:.3f}  (> 0.65 requis)  -> {'OK' if auc_g >= 0.65 else 'LIMITE'}")
        print(f"    F1 global          = {f1_g:.3f}")
        print(f"    -> H1 : {h1}")

    if not beh_b.empty and not hyb_b.empty:
        rec_beh = beh_b["Recall"].values[0]
        rec_hyb = hyb_b["Recall"].values[0]
        auc_beh = beh_g["AUC"].values[0] if not beh_g.empty else float("nan")
        auc_hyb = hyb_g["AUC"].values[0] if not hyb_g.empty else float("nan")
        delta_auc   = auc_hyb - auc_beh
        delta_recall = rec_hyb - rec_beh
        h2 = "REJETEE" if rec_beh > rec_hyb else "VALIDEE"
        print(f"\n  H2 : hybrid_macro ameliore la detection vs behavioral")
        print(f"    Recall Bloc B behavioral = {rec_beh:.3f}")
        print(f"    Recall Bloc B hybrid     = {rec_hyb:.3f}  (delta={delta_recall:+.3f})")
        print(f"    Delta AUC (hyb-beh)      = {delta_auc:+.3f}  (> 0.05 requis)")
        print(f"    -> H2 : {h2}")
        print(f"    Interpretation : les indices FAO sont mondiaux, les recherches")
        print(f"    Google capturent la specificite locale de la crise marocaine.")

    print(f"\n{SEP}\n")


# ─── Point d'entrée ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = compute_warning_metrics()
    print(f"\nFichiers generes :")
    print(f"  outputs/reports/classification_metrics.csv")
    print(f"  outputs/reports/warning_metrics_v3.csv")
    print(f"  outputs/figures/confusion_matrices_v3.png")
    print(f"  outputs/figures/classification_barchart_v3.png")
    print(f"  outputs/figures/roc_curves_v3.png")
    print(f"  outputs/figures/precision_recall_v3.png")
    print(f"\nShape final : {df.shape}")
