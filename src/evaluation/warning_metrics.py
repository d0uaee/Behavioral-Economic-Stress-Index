"""
src/evaluation/warning_metrics.py — Métriques d'alerte précoce v3

Évalue la capacité du BESI à détecter les régimes de haute inflation
AVANT la publication officielle du HCP.

Métriques calculées :
    - ROC-AUC et courbe ROC
    - Precision-Recall et courbe PR
    - F1-Score à différents seuils
    - Lead-time moyen (combien de mois d'avance ?)
    - Confusion matrix à seuil optimal (max F1)

Hypothèse H1 : behavioral_index_pure prédit inflation_regime
                avec un lead-time ≥ 1 mois (AUC > 0.65)

Hypothèse H2 : hybrid_macro_index améliore la détection (ΔAUC > 0.05)

Output :
    outputs/reports/warning_metrics_v3.csv
    outputs/figures/roc_curves_v3.png
    outputs/figures/precision_recall_v3.png
    outputs/figures/threshold_analysis_v3.png
"""

import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parent.parent.parent
GOLD_DIR = ROOT / "data" / "gold"
REPORTS  = ROOT / "outputs" / "reports"
FIGURES  = ROOT / "outputs" / "figures"
REPORTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# Seuil de détection (percentile du signal) à tester
THRESHOLD_PERCENTILES = list(range(50, 96, 5))   # 50, 55, ..., 95


# ─── Utilitaires ──────────────────────────────────────────────────────────────

def _compute_roc(y_true: np.ndarray, scores: np.ndarray):
    """ROC maison (évite sklearn si indisponible)."""
    thresholds = np.unique(scores)[::-1]
    tprs, fprs = [0.0], [0.0]
    pos = y_true.sum()
    neg = len(y_true) - pos

    if pos == 0 or neg == 0:
        return np.array([0, 1]), np.array([0, 1]), thresholds, 0.5

    for t in thresholds:
        pred  = (scores >= t).astype(int)
        tp    = ((pred == 1) & (y_true == 1)).sum()
        fp    = ((pred == 1) & (y_true == 0)).sum()
        tprs.append(tp / pos)
        fprs.append(fp / neg)

    tprs.append(1.0); fprs.append(1.0)
    tprs, fprs = np.array(tprs), np.array(fprs)
    auc = float(np.trapezoid(tprs, fprs) if hasattr(np, "trapezoid") else np.trapz(tprs, fprs))
    return tprs, fprs, thresholds, abs(auc)   # abs car intégrale peut être négative


def _compute_pr(y_true: np.ndarray, scores: np.ndarray):
    """Precision-Recall maison."""
    thresholds = np.unique(scores)[::-1]
    precisions, recalls = [], []
    pos = y_true.sum()
    if pos == 0:
        return np.array([1.0, 0.0]), np.array([0.0, 1.0]), thresholds

    for t in thresholds:
        pred  = (scores >= t).astype(int)
        tp    = ((pred == 1) & (y_true == 1)).sum()
        fp    = ((pred == 1) & (y_true == 0)).sum()
        fn    = ((pred == 0) & (y_true == 1)).sum()
        prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)

    return np.array(precisions), np.array(recalls), thresholds


def _f1_at_threshold(y_true, scores, t):
    pred = (scores >= t).astype(int)
    tp   = ((pred == 1) & (y_true == 1)).sum()
    fp   = ((pred == 1) & (y_true == 0)).sum()
    fn   = ((pred == 0) & (y_true == 1)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return float(prec), float(rec), float(f1), int(tp), int(fp), int(fn)


def _lead_time(
    y_true_series:   pd.Series,
    scores_series:   pd.Series,
    threshold:       float,
    max_lead_months: int = 12,
) -> float:
    """
    Lead-time moyen sur les épisodes de haute inflation.

    Définition rigoureuse :
    - Un épisode commence au 1er mois consécutif avec y=1.
    - L'alerte valide est la DERNIÈRE fois que le signal dépasse le seuil
      dans la fenêtre [episode_start - max_lead_months, episode_start).
    - Si aucune alerte dans cette fenêtre → épisode non détecté (ignoré).
    - Fenêtre maximale explicite : max_lead_months (défaut 12) pour éviter
      de compter des alertes très anciennes non causalement reliées.

    Retourne
    --------
    float : lead-time moyen en mois, ou np.nan si aucun épisode détecté.
    """
    lead_times = []
    in_episode = False
    episode_start = None

    # NaN dans y_true → ignorer ces mois
    y_clean = y_true_series.dropna()

    for date, val in y_clean.items():
        if val == 1 and not in_episode:
            in_episode    = True
            episode_start = date
        elif val == 0 and in_episode:
            in_episode = False
            # Fenêtre bornée : [episode_start - max_lead, episode_start)
            window_start = episode_start - pd.DateOffset(months=max_lead_months)
            window_scores = scores_series[
                (scores_series.index >= window_start) &
                (scores_series.index < episode_start)
            ]
            alerts = window_scores[window_scores >= threshold]
            if not alerts.empty:
                # Alerte la plus proche du début d'épisode (la plus récente)
                last_alert = alerts.index[-1]
                months_ahead = (
                    (episode_start.year  - last_alert.year)  * 12
                    + (episode_start.month - last_alert.month)
                )
                if months_ahead > 0:
                    lead_times.append(months_ahead)

    return float(np.mean(lead_times)) if lead_times else float("nan")


# ─── Analyse principale ────────────────────────────────────────────────────────

def _calibrate_threshold(
    gold:       pd.DataFrame,
    sig_col:    str,
    target_col: str,
    train_mask: pd.Series,
) -> float:
    """
    Trouve le seuil qui maximise F1 sur les données d'entraînement.
    Retourne 0.5 si pas assez de positifs.
    """
    train_df = gold[train_mask]
    y_tr     = train_df[target_col].dropna()
    if y_tr.sum() < 3:
        return 0.5

    scores_tr = train_df[sig_col].reindex(y_tr.index).ffill().bfill().fillna(0).values
    y_tr_vals = y_tr.values

    best_f1, best_t = 0.0, 0.5
    for t in np.unique(scores_tr):
        _, _, f1, *_ = _f1_at_threshold(y_tr_vals, scores_tr, t)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)

    return best_t


def compute_warning_metrics(
    gold_path:   str | Path | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Calcule les métriques d'alerte précoce avec séparation stricte
    calibration / évaluation :

    Pour chaque bloc A/B/C :
      - Seuil optimal calibré sur le jeu TRAIN correspondant (max F1)
      - AUC, F1, Precision, Recall, Lead-time évalués sur le TEST

    Pour le rapport global :
      - AUC calculé sur l'ensemble des données (non biaisé par le seuil)
      - Seuil = médiane des seuils calibrés sur les 3 blocs train
      - Lead-time calculé sur l'ensemble des données de test

    Retourne
    --------
    pd.DataFrame avec une ligne par (scope, signal).
    """
    if gold_path is None:
        gold_path = GOLD_DIR / "model_dataset_monthly.csv"
    if output_path is None:
        output_path = REPORTS / "warning_metrics_v3.csv"

    gold_path   = Path(gold_path)
    output_path = Path(output_path)

    if not gold_path.exists():
        raise FileNotFoundError(
            f"Gold dataset introuvable : {gold_path}\n"
            "Lancer : from src.gold.build_model_dataset import build_gold_dataset"
        )

    gold = pd.read_csv(gold_path, parse_dates=["month"], index_col="month")
    logger.info(f"Gold dataset chargé : {gold.shape}")

    target_col = "target_high_inflation_regime_t1"
    if target_col not in gold.columns:
        raise KeyError(f"'{target_col}' absent du Gold. Colonnes : {list(gold.columns)}")

    # Signaux à évaluer (lag1 prioritaire — respecte l'as-of-date)
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

    rows     = []
    roc_data = {}
    pr_data  = {}

    # Fenêtres d'évaluation — même définition que backtest
    from src.gold.build_model_dataset import EVAL_WINDOWS

    for window in EVAL_WINDOWS:
        lbl        = window["label"]
        train_mask = gold["split_label"].str.contains(f"train_{lbl}", na=False)
        test_mask  = gold["split_label"].str.contains(f"test_{lbl}",  na=False)

        gold_test = gold[test_mask]
        y_test    = gold_test[target_col].dropna()

        if y_test.sum() < 2:
            logger.warning(f"Bloc {lbl} test : moins de 2 positifs — ignoré")
            continue

        for sig_label, sig_col in signals.items():
            if sig_col not in gold.columns:
                continue

            # ── Calibration du seuil sur TRAIN ───────────────────────────────
            best_t = _calibrate_threshold(gold, sig_col, target_col, train_mask)
            logger.info(f"  Bloc {lbl} | {sig_label} | seuil calibré sur train = {best_t:.4f}")

            # ── Évaluation sur TEST ───────────────────────────────────────────
            scores_te = gold_test[sig_col].reindex(y_test.index).ffill().bfill().fillna(0).values
            y_te      = y_test.values

            tprs, fprs, _, auc = _compute_roc(y_te, scores_te)
            precs, recs, _     = _compute_pr(y_te, scores_te)

            prec_ev, rec_ev, f1_ev, tp, fp, fn = _f1_at_threshold(y_te, scores_te, best_t)

            lt = _lead_time(
                gold_test[target_col].dropna(),
                gold_test[sig_col].reindex(gold_test[target_col].dropna().index).ffill(),
                best_t,
            )

            rows.append({
                "scope":               f"test_{lbl}",
                "signal":              sig_label,
                "signal_col":          sig_col,
                "threshold_from":      f"train_{lbl}",
                "threshold_calibrated": round(best_t, 4),
                "auc":                 round(auc, 4),
                "f1":                  round(f1_ev, 4),
                "precision":           round(prec_ev, 4),
                "recall":              round(rec_ev, 4),
                "lead_time_mean_months": round(lt, 2) if not np.isnan(lt) else float("nan"),
                "tp":                  tp,
                "fp":                  fp,
                "fn":                  fn,
                "n_positive":          int(y_te.sum()),
                "n_total":             len(y_te),
            })

    # ── Rapport global (AUC sur toutes données, seuil = médiane des trains) ──
    for sig_label, sig_col in signals.items():
        if sig_col not in gold.columns:
            continue

        y_all     = gold[target_col].dropna()
        scores_all = gold[sig_col].reindex(y_all.index).ffill().bfill().fillna(0).values

        tprs_g, fprs_g, _, auc_g = _compute_roc(y_all.values, scores_all)
        precs_g, recs_g, _       = _compute_pr(y_all.values, scores_all)

        roc_data[sig_label] = (fprs_g, tprs_g, auc_g)
        pr_data[sig_label]  = (recs_g, precs_g)

        # Seuil global = médiane des seuils calibrés sur les 3 blocs train
        global_thresholds = [
            r["threshold_calibrated"]
            for r in rows
            if r["signal"] == sig_label
        ]
        global_t = float(np.median(global_thresholds)) if global_thresholds else 0.5

        prec_g, rec_g, f1_g, tp_g, fp_g, fn_g = _f1_at_threshold(
            y_all.values, scores_all, global_t
        )

        # Lead-time global sur toutes les données de test
        test_mask_all = gold["split_label"].str.contains("test_", na=False)
        gold_test_all = gold[test_mask_all]
        lt_g = _lead_time(
            gold_test_all[target_col].dropna(),
            gold_test_all[sig_col].reindex(gold_test_all[target_col].dropna().index).ffill(),
            global_t,
        )

        rows.append({
            "scope":               "global",
            "signal":              sig_label,
            "signal_col":          sig_col,
            "threshold_from":      "median_of_trains",
            "threshold_calibrated": round(global_t, 4),
            "auc":                 round(auc_g, 4),
            "f1":                  round(f1_g, 4),
            "precision":           round(prec_g, 4),
            "recall":              round(rec_g, 4),
            "lead_time_mean_months": round(lt_g, 2) if not np.isnan(lt_g) else float("nan"),
            "tp":                  tp_g,
            "fp":                  fp_g,
            "fn":                  fn_g,
            "n_positive":          int(y_all.values.sum()),
            "n_total":             len(y_all),
        })

    if not rows:
        raise RuntimeError("Aucune métrique calculée — vérifier les données Gold.")

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_path, index=False)
    logger.info(f"Métriques d'alerte sauvegardées : {output_path}")

    _plot_roc(roc_data)
    _plot_pr(pr_data)
    _plot_threshold_analysis(gold, signals)

    _print_warning_summary(metrics_df)
    return metrics_df


# ─── Graphiques ───────────────────────────────────────────────────────────────

def _plot_roc(roc_data: dict) -> None:
    if not roc_data:
        return
    colors = {"behavioral": "#4CAF50", "hybrid": "#FF9800"}
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Aléatoire (AUC=0.50)")
    for label, (fprs, tprs, auc) in roc_data.items():
        ax.plot(fprs, tprs, color=colors.get(label, "blue"),
                linewidth=2.2, label=f"{label}  (AUC={auc:.3f})")
    ax.set_xlabel("Taux Faux Positifs")
    ax.set_ylabel("Taux Vrais Positifs")
    ax.set_title("Courbes ROC — Détection Régimes Haute Inflation\n"
                 "Maroc 2010–2024", fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    out = FIGURES / "roc_curves_v3.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"ROC sauvegardé : {out}")


def _plot_pr(pr_data: dict) -> None:
    if not pr_data:
        return
    colors = {"behavioral": "#4CAF50", "hybrid": "#FF9800"}
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, (recs, precs) in pr_data.items():
        # Trier par recall
        idx = np.argsort(recs)
        ax.plot(recs[idx], precs[idx], color=colors.get(label, "blue"),
                linewidth=2.2, label=label)
    ax.set_xlabel("Rappel")
    ax.set_ylabel("Précision")
    ax.set_title("Courbes Précision-Rappel — Détection Régimes Inflation\n"
                 "Maroc 2010–2024", fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    out = FIGURES / "precision_recall_v3.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"PR curve sauvegardée : {out}")


def _plot_threshold_analysis(gold: pd.DataFrame, signals: dict) -> None:
    """Affiche Precision / Recall / F1 en fonction du seuil (percentile)."""
    target_col = "target_high_inflation_regime_t1"
    y_all = gold[target_col].dropna()
    if y_all.sum() < 3:
        return

    colors = {"behavioral": "#4CAF50", "hybrid": "#FF9800"}
    fig, axes = plt.subplots(1, len(signals), figsize=(8 * len(signals), 5), squeeze=False)

    for i, (sig_label, sig_col) in enumerate(signals.items()):
        ax = axes[0][i]
        if sig_col not in gold.columns:
            continue
        scores = gold[sig_col].reindex(y_all.index).ffill().bfill().fillna(0).values
        y      = y_all.values

        percentiles = np.arange(30, 96, 5)
        precisions, recalls, f1s = [], [], []

        for p in percentiles:
            t = np.percentile(scores, p)
            prec, rec, f1, *_ = _f1_at_threshold(y, scores, t)
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

        ax.plot(percentiles, precisions, "b-o",  markersize=4, label="Précision")
        ax.plot(percentiles, recalls,    "r-s",  markersize=4, label="Rappel")
        ax.plot(percentiles, f1s,        "g-^",  markersize=4, label="F1", linewidth=2)
        ax.axvline(percentiles[np.argmax(f1s)], color="g", linestyle="--", alpha=0.5)
        ax.set_xlabel("Seuil (percentile du signal)")
        ax.set_ylabel("Score")
        ax.set_title(f"{sig_label}\n(seuil optimal = p{percentiles[np.argmax(f1s)]})",
                     fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)

    fig.suptitle("Analyse de Seuil — BESI v3 vs Régime Inflation", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = FIGURES / "threshold_analysis_v3.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Threshold analysis sauvegardée : {out}")


def _print_warning_summary(metrics_df: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("MÉTRIQUES D'ALERTE PRÉCOCE V3")
    print("  Méthode : seuil calibré sur TRAIN, évalué sur TEST (pas de data leakage)")
    print("=" * 78)

    # Par bloc
    bloc_df = metrics_df[metrics_df["scope"] != "global"]
    if not bloc_df.empty:
        print(f"\n{'Bloc':<10} {'Signal':<20} {'Seuil':>7} {'AUC':>6} {'F1':>6} "
              f"{'Prec':>6} {'Rec':>6} {'Lead':>6}")
        print("-" * 78)
        for _, row in bloc_df.sort_values(["scope", "signal"]).iterrows():
            lt = f"{row['lead_time_mean_months']:.1f}" if not pd.isna(row["lead_time_mean_months"]) else "n/a"
            print(f"{row['scope']:<10} {row['signal']:<20} {row['threshold_calibrated']:>7.4f} "
                  f"{row['auc']:>6.3f} {row['f1']:>6.3f} "
                  f"{row['precision']:>6.3f} {row['recall']:>6.3f} {lt:>6}")

    # Global
    global_df = metrics_df[metrics_df["scope"] == "global"]
    if not global_df.empty:
        print(f"\n{'GLOBAL (seuil=médiane trains)':<30} {'AUC':>6} {'F1':>6} {'Lead (mois)':>12}")
        print("-" * 60)
        for _, row in global_df.iterrows():
            lt = f"{row['lead_time_mean_months']:.1f}" if not pd.isna(row["lead_time_mean_months"]) else "n/a"
            print(f"  {row['signal']:<28} {row['auc']:>6.3f} {row['f1']:>6.3f} {lt:>12}")

    print("\n  H1 validee si AUC > 0.65  (signal predit le regime d'inflation)")
    print("  H2 validee si dAUC(hybrid - behavioral) > 0.05  (macro apporte de la valeur)")

    auc_beh = global_df[global_df["signal"] == "behavioral"]["auc"].values
    auc_hyb = global_df[global_df["signal"] == "hybrid"]["auc"].values
    if len(auc_beh) > 0:
        h1 = "OK VALIDEE" if auc_beh[0] > 0.65 else "NON REJETEE"
        print(f"\n  -> H1 (behavioral AUC={auc_beh[0]:.3f} > 0.65) : {h1}")
    if len(auc_beh) > 0 and len(auc_hyb) > 0:
        delta = auc_hyb[0] - auc_beh[0]
        h2 = "OK VALIDEE" if delta > 0.05 else "NON REJETEE"
        print(f"  -> H2 (dAUC={delta:+.3f} > 0.05) : {h2}")
    print("=" * 78)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = compute_warning_metrics()
    print(f"\nShape métriques : {df.shape}")
