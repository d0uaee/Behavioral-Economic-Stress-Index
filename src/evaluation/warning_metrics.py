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
    auc = float(np.trapz(tprs, fprs))
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


def _lead_time(y_true_series: pd.Series, scores_series: pd.Series, threshold: float) -> float:
    """
    Lead-time moyen : nombre de mois AVANT la 1ère observation du régime
    à laquelle le signal dépasse le seuil pour la 1ère fois.
    Retourne np.nan si aucun épisode détectable.
    """
    lead_times = []

    # Identifier les épisodes de haute inflation (blocs consécutifs de 1)
    in_episode  = False
    episode_start = None

    for i, (date, val) in enumerate(y_true_series.items()):
        if val == 1 and not in_episode:
            in_episode    = True
            episode_start = date
        elif val == 0 and in_episode:
            in_episode = False
            # Chercher quand le signal a dépassé le seuil avant episode_start
            pre_episode = scores_series[scores_series.index < episode_start]
            alerts = pre_episode[pre_episode >= threshold]
            if not alerts.empty:
                first_alert = alerts.index[0]
                months_ahead = (episode_start.year - first_alert.year) * 12 + \
                               (episode_start.month - first_alert.month)
                if months_ahead > 0:
                    lead_times.append(months_ahead)

    return float(np.mean(lead_times)) if lead_times else float("nan")


# ─── Analyse principale ────────────────────────────────────────────────────────

def compute_warning_metrics(
    gold_path:   str | Path | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Calcule les métriques d'alerte précoce pour behavioral_index_pure
    et hybrid_macro_index sur l'ensemble des données (et par bloc).

    Retourne
    --------
    pd.DataFrame avec AUC, F1_optimal, lead_time_mean, etc.
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

    # Signaux à évaluer : lag1 (respecte la règle as-of-date)
    signals = {}
    for col in ["behavioral_index_pure_lag1", "hybrid_macro_index_lag1",
                "behavioral_index_pure", "hybrid_macro_index"]:
        if col in gold.columns:
            signals[col] = col
            break   # utiliser lag1 en priorité

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

    rows        = []
    roc_data    = {}   # pour le graphique
    pr_data     = {}

    # Évaluation globale + par bloc
    for scope, subset in _scopes(gold):
        y_all = subset[target_col].dropna()
        if y_all.sum() < 3:
            logger.warning(f"Scope '{scope}' : moins de 3 positifs — ignoré")
            continue

        for sig_label, sig_col in signals.items():
            if sig_col not in subset.columns:
                continue

            scores = subset[sig_col].reindex(y_all.index).ffill().bfill().fillna(0).values
            y      = y_all.values

            # ROC
            tprs, fprs, thresholds_roc, auc = _compute_roc(y, scores)
            if scope == "global":
                roc_data[sig_label] = (fprs, tprs, auc)

            # PR
            precs, recs, thresholds_pr = _compute_pr(y, scores)
            if scope == "global":
                pr_data[sig_label] = (recs, precs)

            # Seuil optimal (max F1)
            best_f1, best_t = 0.0, 0.5
            if len(thresholds_pr) > 0:
                unique_scores = np.unique(scores)
                for t in unique_scores:
                    _, _, f1, *_ = _f1_at_threshold(y, scores, t)
                    if f1 > best_f1:
                        best_f1, best_t = f1, float(t)

            prec_opt, rec_opt, f1_opt, tp, fp, fn = _f1_at_threshold(y, scores, best_t)

            # Lead-time
            if scope == "global":
                lt = _lead_time(
                    subset[target_col].dropna(),
                    subset[sig_col].reindex(subset[target_col].dropna().index).ffill(),
                    best_t,
                )
            else:
                lt = float("nan")

            rows.append({
                "scope":         scope,
                "signal":        sig_label,
                "signal_col":    sig_col,
                "auc":           round(auc, 4),
                "f1_optimal":    round(f1_opt, 4),
                "precision_opt": round(prec_opt, 4),
                "recall_opt":    round(rec_opt, 4),
                "threshold_opt": round(best_t, 4),
                "lead_time_mean_months": round(lt, 2) if not np.isnan(lt) else float("nan"),
                "tp":            tp,
                "fp":            fp,
                "fn":            fn,
                "n_positive":    int(y.sum()),
                "n_total":       len(y),
            })

    if not rows:
        raise RuntimeError("Aucune métrique calculée — vérifier les données Gold.")

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_path, index=False)
    logger.info(f"Métriques d'alerte sauvegardées : {output_path}")

    # ── Graphiques ────────────────────────────────────────────────────────────
    _plot_roc(roc_data)
    _plot_pr(pr_data)
    _plot_threshold_analysis(gold, signals)

    _print_warning_summary(metrics_df)
    return metrics_df


def _scopes(gold: pd.DataFrame):
    """Génère des sous-ensembles : global + blocs A/B/C."""
    yield "global", gold
    for lbl in ["A", "B", "C"]:
        mask = gold["split_label"].str.contains(f"test_{lbl}", na=False)
        if mask.sum() > 0:
            yield f"test_{lbl}", gold[mask]


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
    print("\n" + "=" * 72)
    print("MÉTRIQUES D'ALERTE PRÉCOCE V3")
    print("=" * 72)
    global_df = metrics_df[metrics_df["scope"] == "global"]
    print(f"\n{'Signal':<30} {'AUC':>6} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Lead (mois)':>12}")
    print("-" * 72)
    for _, row in global_df.iterrows():
        lt = f"{row['lead_time_mean_months']:.1f}" if not pd.isna(row["lead_time_mean_months"]) else "n/a"
        print(f"{row['signal']:<30} {row['auc']:>6.3f} {row['f1_optimal']:>6.3f} "
              f"{row['precision_opt']:>6.3f} {row['recall_opt']:>6.3f} {lt:>12}")

    print("\n  H1 validée si AUC > 0.65  (signal comportemental prédit le régime d'inflation)")
    print("  H2 validée si ΔAUC(hybrid - behavioral) > 0.05")

    auc_beh = global_df[global_df["signal"] == "behavioral"]["auc"].values
    auc_hyb = global_df[global_df["signal"] == "hybrid"]["auc"].values
    if len(auc_beh) > 0:
        h1 = "✓ VALIDÉE" if auc_beh[0] > 0.65 else "✗ REJETÉE"
        print(f"\n  → H1 (behavioral AUC={auc_beh[0]:.3f} > 0.65) : {h1}")
    if len(auc_beh) > 0 and len(auc_hyb) > 0:
        delta = auc_hyb[0] - auc_beh[0]
        h2 = "✓ VALIDÉE" if delta > 0.05 else "✗ REJETÉE"
        print(f"  → H2 (ΔAUC={delta:+.3f} > 0.05) : {h2}")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = compute_warning_metrics()
    print(f"\nShape métriques : {df.shape}")
