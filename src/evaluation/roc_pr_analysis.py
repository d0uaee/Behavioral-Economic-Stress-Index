"""
src/evaluation/roc_pr_analysis.py — Courbes ROC + Precision-Recall (BESI v3)

PROBLÈME MOTIVANT :
    Sur le Bloc B (2022-2024), 68.6% des mois sont en régime de haute inflation.
    → L'AUC ROC est trompeuse quand les classes sont déséquilibrées :
      avec peu de vrais négatifs, le dénominateur FPR = TN/(TN+FP) est petit,
      donc la courbe ROC semble bonne même pour un signal médiocre.
    → La courbe Precision-Recall est honnête : elle mesure directement la qualité
      des alertes positives émises, sans dépendre du nombre de TN.

BASELINE DE RÉFÉRENCE :
    ROC : ligne diagonale (AUC = 0.5 pour un classifieur aléatoire)
    PR  : ligne horizontale à y = ratio_positifs (AP = ratio_positifs pour aléatoire)
          → Bloc A : baseline PR ≈ 0.33
          → Bloc B : baseline PR ≈ 0.69  ← ROC baseline reste à 0.50

FIGURES GÉNÉRÉES (300 dpi) :
    outputs/figures/roc_pr_curves_global.png
    outputs/figures/roc_pr_curves_blocA.png
    outputs/figures/roc_pr_curves_blocB.png

RAPPORT :
    outputs/reports/roc_pr_comparison.csv
    | Modele | Bloc | AUC_ROC | AP | F1_optimal | seuil_optimal |

Usage :
    python src/evaluation/roc_pr_analysis.py
    from src.evaluation.roc_pr_analysis import run_roc_pr_analysis
"""

import logging
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parent.parent.parent
GOLD_DIR = ROOT / "data" / "gold"
REPORTS  = ROOT / "outputs" / "reports"
FIGURES  = ROOT / "outputs" / "figures"
REPORTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
C_BEH    = "#27ae60"   # vert foncé  — behavioral
C_HYB    = "#e67e22"   # orange      — hybrid
C_OPT    = "#c0392b"   # rouge       — point optimal F1
C_ISO    = "#bdc3c7"   # gris clair  — iso-courbes F1
C_BASE   = "#95a5a6"   # gris        — baseline aléatoire

MODEL_CONFIG = {
    "behavioral": {
        "label":  "SARIMAX + BESI behavioral",
        "color":  C_BEH,
        "ls":     "-",
        "lw":     2.5,
        "zorder": 4,
    },
    "hybrid": {
        "label":  "SARIMAX + Hybrid macro",
        "color":  C_HYB,
        "ls":     "--",
        "lw":     2.0,
        "zorder": 3,
    },
}


# ─── Calcul des courbes ───────────────────────────────────────────────────────

def _build_curves(
    y_true:  np.ndarray,
    scores:  np.ndarray,
) -> dict:
    """
    Calcule toutes les courbes + métriques agrégées pour un (signal, scope).

    Retourne
    --------
    dict avec :
        roc_fpr, roc_tpr, roc_thresholds  — courbe ROC
        pr_prec, pr_rec,  pr_thresholds   — courbe PR
        auc_roc  — aire sous courbe ROC
        ap       — Average Precision (aire sous courbe PR)
        f1_opt   — F1 au point optimal
        thresh_opt — seuil correspondant
        fpr_opt  — FPR au point optimal (pour ROC)
        tpr_opt  — TPR au point optimal (pour ROC)
        rec_opt  — Recall au point optimal (pour PR)
        prec_opt — Precision au point optimal (pour PR)
        baseline_pr — ratio de positifs (baseline aléatoire pour PR)
        n_pos, n_neg, n_total
    """
    n_total = len(y_true)
    n_pos   = int(y_true.sum())
    n_neg   = n_total - n_pos

    thresholds = np.unique(scores)[::-1]   # ordre décroissant

    roc_fpr, roc_tpr = [0.0], [0.0]
    pr_prec, pr_rec  = [], []
    f1s              = []
    all_thresholds   = []

    for t in thresholds:
        pred = (scores >= t).astype(int)
        tp   = int(((pred == 1) & (y_true == 1)).sum())
        fp   = int(((pred == 1) & (y_true == 0)).sum())
        fn   = int(((pred == 0) & (y_true == 1)).sum())
        tn   = int(((pred == 0) & (y_true == 0)).sum())

        tpr  = tp / n_pos if n_pos > 0 else 0.0
        fpr  = fp / n_neg if n_neg > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tpr
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        roc_fpr.append(fpr)
        roc_tpr.append(tpr)
        pr_prec.append(prec)
        pr_rec.append(rec)
        f1s.append(f1)
        all_thresholds.append(float(t))

    roc_fpr.append(1.0); roc_tpr.append(1.0)

    roc_fpr = np.array(roc_fpr)
    roc_tpr = np.array(roc_tpr)
    pr_prec = np.array(pr_prec)
    pr_rec  = np.array(pr_rec)
    f1s_arr = np.array(f1s)

    # AUC ROC (intégration trapézoïdale — abs car orientation peut varier)
    _trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    auc_roc = float(abs(_trap(roc_tpr, roc_fpr)))

    # Average Precision : intégration sur courbe PR triée par recall
    idx_sorted = np.argsort(pr_rec)
    ap = float(abs(_trap(pr_prec[idx_sorted], pr_rec[idx_sorted])))

    # Point optimal = seuil maximisant F1
    if len(f1s) > 0:
        best_idx  = int(np.argmax(f1s_arr))
        f1_opt    = float(f1s_arr[best_idx])
        thresh_opt = all_thresholds[best_idx]
        fpr_opt   = roc_fpr[best_idx + 1]   # +1 car roc_fpr commence par 0
        tpr_opt   = roc_tpr[best_idx + 1]
        rec_opt   = float(pr_rec[best_idx])
        prec_opt  = float(pr_prec[best_idx])
    else:
        f1_opt = thresh_opt = fpr_opt = tpr_opt = rec_opt = prec_opt = 0.0

    baseline_pr = n_pos / n_total if n_total > 0 else 0.0

    return {
        "roc_fpr":    roc_fpr,
        "roc_tpr":    roc_tpr,
        "pr_prec":    pr_prec,
        "pr_rec":     pr_rec,
        "auc_roc":    auc_roc,
        "ap":         ap,
        "f1_opt":     f1_opt,
        "thresh_opt": thresh_opt,
        "fpr_opt":    fpr_opt,
        "tpr_opt":    tpr_opt,
        "rec_opt":    rec_opt,
        "prec_opt":   prec_opt,
        "baseline_pr": baseline_pr,
        "n_pos":      n_pos,
        "n_neg":      n_neg,
        "n_total":    n_total,
    }


# ─── Figure ROC + PR côte à côte ─────────────────────────────────────────────

def _plot_roc_pr_panel(
    curves:     dict,     # {sig_label: curve_dict}
    scope:      str,      # "global" | "A" | "B"
    scope_info: dict,     # {n_pos, n_neg, baseline_pr, date_range}
    out_path:   Path,
) -> None:
    """
    Génère une figure 2 sous-graphes :
      [Gauche] Courbe ROC   : FPR (x) vs TPR (y)
      [Droite] Courbe PR    : Recall (x) vs Precision (y)

    Fonctionnalités :
      - Iso-courbes F1 en arrière-plan (F1=0.2, 0.4, 0.6, 0.8)
      - Point optimal F1 marqué par une étoile
      - Baseline aléatoire annotée
      - Diagnostic déséquilibre classes
    """
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.patch.set_facecolor("#fafafa")

    baseline_pr = scope_info["baseline_pr"]
    n_pos       = scope_info["n_pos"]
    n_neg       = scope_info["n_neg"]
    n_total     = scope_info["n_total"]
    date_range  = scope_info.get("date_range", "")
    imbalance   = scope_info.get("imbalance_text", "")

    # ── Titre principal ───────────────────────────────────────────────────────
    scope_label = {
        "global": "Global (tous blocs)",
        "A":      "Bloc A — COVID 2020-2021",
        "B":      "Bloc B — Inflation 2022-2024",
    }.get(scope, scope)

    fig.suptitle(
        f"Courbes ROC & Precision-Recall — Detection Stress Economique\n"
        f"{scope_label}  |  {date_range}  |  "
        f"{n_pos} mois positifs / {n_total} ({100*n_pos/n_total:.0f}%)\n"
        f"Seuil calibre sur TRAIN uniquement — aucun data leakage",
        fontsize=11, fontweight="bold", y=1.01
    )

    # ══════════════════════════════════════════════════════════════════════════
    # SOUS-GRAPHE GAUCHE : Courbe ROC
    # ══════════════════════════════════════════════════════════════════════════
    ax_roc.set_facecolor("#fdfdfd")

    # Grille légère
    ax_roc.grid(True, alpha=0.25, linewidth=0.7, color="gray")

    # Baseline aléatoire
    ax_roc.plot([0, 1], [0, 1], color=C_BASE, lw=1.2, ls=":",
                label=f"Aleatoire (AUC = 0.500)", zorder=1)

    # Zone grisée sous la diagonale
    ax_roc.fill_between([0, 1], [0, 1], alpha=0.04, color="gray")

    # Courbes ROC par modèle
    opt_points_roc = {}
    for sig, c in curves.items():
        cfg = MODEL_CONFIG.get(sig, {"label": sig, "color": "blue",
                                     "ls": "-", "lw": 2.0, "zorder": 3})
        ax_roc.plot(
            c["roc_fpr"], c["roc_tpr"],
            color=cfg["color"], lw=cfg["lw"], ls=cfg["ls"],
            label=f"{cfg['label']}  (AUC = {c['auc_roc']:.3f})",
            zorder=cfg["zorder"]
        )
        # Zone sous la courbe
        ax_roc.fill_between(c["roc_fpr"], c["roc_tpr"],
                            alpha=0.06, color=cfg["color"])
        opt_points_roc[sig] = c

    # Points optimaux F1 sur ROC
    for sig, c in opt_points_roc.items():
        cfg = MODEL_CONFIG.get(sig, {"color": "blue"})
        ax_roc.scatter(
            c["fpr_opt"], c["tpr_opt"],
            color=C_OPT, s=120, zorder=10, marker="*",
            edgecolors=cfg["color"], linewidths=1.5
        )
        ax_roc.annotate(
            f" F1={c['f1_opt']:.3f}\n seuil={c['thresh_opt']:.3f}",
            xy=(c["fpr_opt"], c["tpr_opt"]),
            xytext=(c["fpr_opt"] + 0.04, c["tpr_opt"] - 0.08),
            fontsize=7.5, color=cfg["color"],
            arrowprops=dict(arrowstyle="->", color=cfg["color"],
                            lw=0.8, connectionstyle="arc3,rad=0.2"),
        )

    ax_roc.set_xlim(-0.02, 1.02)
    ax_roc.set_ylim(-0.02, 1.05)
    ax_roc.set_xlabel("Taux de Faux Positifs (FPR = 1 - Specificity)",
                      fontsize=10)
    ax_roc.set_ylabel("Taux de Vrais Positifs (TPR = Recall = Sensitivity)",
                      fontsize=10)
    ax_roc.set_title("Courbe ROC", fontsize=11, fontweight="bold", pad=8)
    ax_roc.legend(fontsize=8.5, loc="lower right",
                  framealpha=0.9, edgecolor="lightgray")

    # Annotation diagnostic déséquilibre
    auc_vals = [c["auc_roc"] for c in curves.values()]
    ap_vals  = [c["ap"]      for c in curves.values()]
    if auc_vals and ap_vals:
        best_auc = max(auc_vals)
        best_ap  = max(ap_vals)
        if best_auc > best_ap + 0.05:
            diag = "! AUC > AP : ROC trop\noptimiste (desequilibre)"
            diag_color = "#e74c3c"
        elif best_ap > best_auc + 0.05:
            diag = "AP > AUC : PR plus\npessimiste (classes rares)"
            diag_color = "#2980b9"
        else:
            diag = "AUC ~= AP : classes\nequilibrees"
            diag_color = "#27ae60"
        ax_roc.text(0.02, 0.97, diag, transform=ax_roc.transAxes,
                    fontsize=7.5, va="top", color=diag_color,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor=diag_color, alpha=0.85))

    # ══════════════════════════════════════════════════════════════════════════
    # SOUS-GRAPHE DROITE : Courbe Precision-Recall
    # ══════════════════════════════════════════════════════════════════════════
    ax_pr.set_facecolor("#fdfdfd")
    ax_pr.grid(True, alpha=0.25, linewidth=0.7, color="gray")

    # Iso-courbes F1 en arrière-plan
    recall_grid    = np.linspace(0.01, 1.0, 300)
    precision_grid = np.linspace(0.01, 1.0, 300)
    R, P = np.meshgrid(recall_grid, precision_grid)
    F1   = 2 * R * P / (R + P)
    iso_levels = [0.2, 0.4, 0.6, 0.8]
    cs = ax_pr.contour(R, P, F1, levels=iso_levels,
                       colors=C_ISO, linewidths=0.8, linestyles="--",
                       zorder=1, alpha=0.7)
    ax_pr.clabel(cs, fmt={v: f"F1={v:.1f}" for v in iso_levels},
                 fontsize=7, colors=C_ISO)

    # Baseline aléatoire (no-skill = prédire toujours positif)
    ax_pr.axhline(baseline_pr, color=C_BASE, lw=1.2, ls=":",
                  label=f"Aleatoire (AP = {baseline_pr:.3f}  = {100*baseline_pr:.0f}% positifs)",
                  zorder=2)
    ax_pr.fill_between([0, 1], baseline_pr, alpha=0.06, color="gray")

    # Courbes PR par modèle
    for sig, c in curves.items():
        cfg = MODEL_CONFIG.get(sig, {"label": sig, "color": "blue",
                                     "ls": "-", "lw": 2.0, "zorder": 3})
        # Trier par recall croissant pour tracer proprement
        idx = np.argsort(c["pr_rec"])
        ax_pr.plot(
            c["pr_rec"][idx], c["pr_prec"][idx],
            color=cfg["color"], lw=cfg["lw"], ls=cfg["ls"],
            label=f"{cfg['label']}  (AP = {c['ap']:.3f})",
            zorder=cfg["zorder"]
        )
        ax_pr.fill_between(c["pr_rec"][idx], c["pr_prec"][idx],
                           baseline_pr,
                           where=c["pr_prec"][idx] >= baseline_pr,
                           alpha=0.08, color=cfg["color"])

    # Points optimaux F1 sur PR
    for sig, c in curves.items():
        cfg = MODEL_CONFIG.get(sig, {"color": "blue"})
        ax_pr.scatter(
            c["rec_opt"], c["prec_opt"],
            color=C_OPT, s=120, zorder=10, marker="*",
            edgecolors=cfg["color"], linewidths=1.5
        )
        offset_x = 0.04 if c["rec_opt"] < 0.7 else -0.20
        ax_pr.annotate(
            f" F1={c['f1_opt']:.3f}\n seuil={c['thresh_opt']:.3f}",
            xy=(c["rec_opt"], c["prec_opt"]),
            xytext=(c["rec_opt"] + offset_x, c["prec_opt"] + 0.05),
            fontsize=7.5, color=cfg["color"],
            arrowprops=dict(arrowstyle="->", color=cfg["color"],
                            lw=0.8, connectionstyle="arc3,rad=0.2"),
        )

    ax_pr.set_xlim(-0.02, 1.02)
    ax_pr.set_ylim(max(0, baseline_pr - 0.15), 1.05)
    ax_pr.set_xlabel("Recall (Sensitivity = TP / (TP + FN))", fontsize=10)
    ax_pr.set_ylabel("Precision (= TP / (TP + FP))", fontsize=10)
    ax_pr.set_title(
        f"Courbe Precision-Recall\n"
        f"(Baseline aleatoire = {baseline_pr:.2f} vs ROC baseline = 0.50)",
        fontsize=11, fontweight="bold", pad=8
    )
    ax_pr.legend(fontsize=8.5, loc="upper right",
                 framealpha=0.9, edgecolor="lightgray")

    # Annotation : explication déséquilibre
    n_ratio = f"{n_pos} positifs / {n_total} = {100*n_pos/n_total:.0f}%"
    ax_pr.text(
        0.02, 0.03,
        f"Classe positive : {n_ratio}\n"
        f"Plus {100*n_pos/n_total:.0f}% > 50% => ROC surestimee\n"
        f"AP < AUC => ROC est trop optimiste" if n_pos/n_total > 0.50
        else f"Classe positive : {n_ratio}\n"
             f"Classes desequilibrees (< 50%)\n"
             f"AP < AUC => PR plus conservative",
        transform=ax_pr.transAxes, fontsize=7.5, va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#eaf4fb",
                  edgecolor="#2980b9", alpha=0.85)
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Figure sauvegardee (300 dpi) : {out_path}")


# ─── Tableau comparatif ───────────────────────────────────────────────────────

def _build_comparison_table(all_results: list) -> pd.DataFrame:
    """
    Construit le tableau comparatif :
    Modele | Bloc | AUC_ROC | AP | F1_optimal | seuil_optimal
    + interprétation automatique de la relation AUC vs AP.
    """
    rows = []
    for r in all_results:
        sig      = r["signal"]
        scope    = r["scope"]
        c        = r["curves"]
        n_pos    = c["n_pos"]
        n_total  = c["n_total"]
        ratio    = n_pos / n_total if n_total > 0 else 0.0

        # Interprétation déséquilibre AUC vs AP
        delta = c["auc_roc"] - c["ap"]
        if delta > 0.15:
            roc_bias = "ROC tres optimiste (desequilibre fort)"
        elif delta > 0.05:
            roc_bias = "ROC moderement optimiste"
        elif delta < -0.05:
            roc_bias = "AP > AUC (classe dominante positive)"
        else:
            roc_bias = "AUC ~= AP (classes equilibrees)"

        # Performance AP
        if c["ap"] > 0.70:
            ap_verdict = "Excellent malgre desequilibre"
        elif c["ap"] > 0.50:
            ap_verdict = "Acceptable"
        elif c["ap"] > 0.30:
            ap_verdict = "Faible"
        else:
            ap_verdict = "Insuffisant"

        rows.append({
            "Modele":         _label(sig),
            "Signal":         sig,
            "Bloc":           scope,
            "n_positifs":     n_pos,
            "n_total":        n_total,
            "ratio_positifs": round(ratio, 3),
            "AUC_ROC":        round(c["auc_roc"],    4),
            "AP":             round(c["ap"],          4),
            "Delta_AUC_AP":   round(delta,            4),
            "F1_optimal":     round(c["f1_opt"],      4),
            "seuil_optimal":  round(c["thresh_opt"],  4),
            "Recall_opt":     round(c["tpr_opt"],     4),
            "Precision_opt":  round(c["prec_opt"],    4),
            "ROC_biais":      roc_bias,
            "AP_verdict":     ap_verdict,
        })

    df = pd.DataFrame(rows)
    # Ordre d'affichage
    order = {"global": 0, "A": 1, "B": 2}
    df["_ord"] = df["Bloc"].map(order).fillna(9)
    df = df.sort_values(["_ord", "Signal"]).drop(columns="_ord")
    return df.reset_index(drop=True)


def _label(sig: str) -> str:
    return {
        "behavioral": "SARIMAX + BESI behavioral",
        "hybrid":     "SARIMAX + Hybrid macro",
    }.get(sig, sig)


# ─── Affichage console du tableau ─────────────────────────────────────────────

def _print_table(df: pd.DataFrame) -> None:
    SEP = "=" * 110
    print(f"\n{SEP}")
    print("  TABLEAU COMPARATIF ROC vs PRECISION-RECALL — BESI v3")
    print("  Pourquoi AUC ROC > AP ? : la ROC reste haute meme avec des faux positifs")
    print("  car le denominateur FPR = TN/(TN+FP) est petit quand il y a peu de negatifs.")
    print("  La courbe PR est conservative : elle penalise directement les fausses alertes.")
    print(SEP)

    scopes = df["Bloc"].unique()
    for scope in ["global", "A", "B"]:
        if scope not in scopes:
            continue
        sub = df[df["Bloc"] == scope]
        r0  = sub.iloc[0]
        scope_label = {
            "global": "GLOBAL",
            "A":      "BLOC A — COVID 2020-2021",
            "B":      "BLOC B — Inflation 2022-2024",
        }.get(scope, scope)

        print(f"\n  [{scope_label}]")
        print(f"  Classes : {r0['n_positifs']} positifs / {r0['n_total']} total "
              f"({100*r0['ratio_positifs']:.0f}%)")

        # Déséquilibre diagnostique
        if r0["ratio_positifs"] > 0.55:
            print(f"  ** Desequilibre majorite positive ({100*r0['ratio_positifs']:.0f}%) :")
            print(f"     La ROC surestime la performance — se fier a l'AP pour comparer.")
        elif r0["ratio_positifs"] < 0.35:
            print(f"  ** Desequilibre minorite positive ({100*r0['ratio_positifs']:.0f}%) :")
            print(f"     La ROC est trop optimiste — l'AP mesure la vraie capacite de detection.")

        print()
        hdr = (f"  {'Modele':<30} {'AUC ROC':>8} {'AP':>8} "
               f"{'Delta':>7} {'F1 opt':>7} {'Seuil':>7} "
               f"{'Rec opt':>8} {'Prec opt':>9}  Verdict AP")
        print(hdr)
        print(f"  {'-'*105}")

        for _, row in sub.iterrows():
            delta_sign = "+" if row["Delta_AUC_AP"] > 0 else ""
            flag = " [**]" if row["AUC_ROC"] > row["AP"] + 0.05 else ""
            print(
                f"  {row['Modele']:<30} "
                f"{row['AUC_ROC']:>8.3f} {row['AP']:>8.3f} "
                f"({delta_sign}{row['Delta_AUC_AP']:.3f}){' ':>1}"
                f"{row['F1_optimal']:>7.3f} {row['seuil_optimal']:>7.3f} "
                f"{row['Recall_opt']:>8.3f} {row['Precision_opt']:>9.3f}  "
                f"{row['AP_verdict']}{flag}"
            )

    # Comparaison finale behavioral vs hybrid
    print(f"\n{SEP}")
    print("  COMPARAISON BEHAVIORAL vs HYBRID (par bloc)")
    print(SEP)
    print(f"\n  {'Bloc':<8} {'Metrique':<15} {'Behavioral':>12} {'Hybrid':>10} "
          f"{'Avantage':>15}")
    print(f"  {'-'*65}")

    for scope in ["A", "B", "global"]:
        sub = df[df["Bloc"] == scope]
        if sub.empty:
            continue
        beh = sub[sub["Signal"] == "behavioral"]
        hyb = sub[sub["Signal"] == "hybrid"]
        if beh.empty or hyb.empty:
            continue
        beh, hyb = beh.iloc[0], hyb.iloc[0]

        for met in ["AUC_ROC", "AP", "F1_optimal"]:
            b_val = beh[met]
            h_val = hyb[met]
            winner = "Behavioral" if b_val > h_val else "Hybrid" if h_val > b_val else "Egal"
            print(f"  {scope:<8} {met:<15} {b_val:>12.3f} {h_val:>10.3f} "
                  f"{winner:>15}")

    print(f"\n{SEP}")
    print("  CONCLUSION :")
    beh_b = df[(df["Bloc"] == "B") & (df["Signal"] == "behavioral")]
    hyb_b = df[(df["Bloc"] == "B") & (df["Signal"] == "hybrid")]
    if not beh_b.empty and not hyb_b.empty:
        b_ap = beh_b.iloc[0]["AP"]
        h_ap = hyb_b.iloc[0]["AP"]
        b_auc = beh_b.iloc[0]["AUC_ROC"]
        h_auc = hyb_b.iloc[0]["AUC_ROC"]
        print(f"  Bloc B behavioral : AUC={b_auc:.3f}  AP={b_ap:.3f}")
        print(f"  Bloc B hybrid     : AUC={h_auc:.3f}  AP={h_ap:.3f}")
        if b_auc > h_auc and b_ap < h_ap:
            print(f"  -> Paradoxe : BESI a un AUC plus faible mais un AP plus eleve.")
            print(f"     Interpretation : BESI est meilleur pour detecter les crises reelles")
            print(f"     (AP plus eleve) mais la ROC le penalise pour ses fausses alarmes")
            print(f"     sur les mois calmes (peu nombreux en Bloc B).")
        elif b_ap > h_ap:
            print(f"  -> BESI behavioral superieur sur AP (+{b_ap-h_ap:.3f})")
            print(f"     -> H2 REJETEE : le macro n'ameliore pas la detection")
        else:
            print(f"  -> Hybrid superieur sur AP (+{h_ap-b_ap:.3f})")
    print(f"{SEP}\n")


# ─── Pipeline principal ────────────────────────────────────────────────────────

def run_roc_pr_analysis(
    gold_path: "str | Path | None" = None,
) -> pd.DataFrame:
    """
    Pipeline complet :
    1. Charge le Gold dataset
    2. Pour chaque scope (global, A, B) et chaque signal (behavioral, hybrid) :
       - Calcule les courbes ROC et PR avec tous les seuils
       - Identifie le point optimal F1
    3. Génère les 3 figures côte à côte (300 dpi)
    4. Génère et sauvegarde le tableau comparatif
    5. Affiche l'analyse console

    Retourne
    --------
    pd.DataFrame : tableau comparatif (roc_pr_comparison.csv)
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

    target_col = "target_high_inflation_regime_t1"
    if target_col not in gold.columns:
        raise KeyError(f"'{target_col}' absent du Gold.")

    # Signaux disponibles (lag1 prioritaire)
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
        raise KeyError("Aucun signal BESI dans le Gold dataset.")

    # Définir les scopes
    labels_present = set()
    for cell in gold["split_label"].dropna().unique():
        for part in str(cell).split("|"):
            if "_" in part:
                labels_present.add(part.split("_")[1])

    scope_defs = []
    date_ranges = {
        "A": "Test 2020-2021",
        "B": "Test 2022-2024",
        "global": "Toutes donnees",
    }
    for lbl in sorted(labels_present):
        mask = gold["split_label"].str.contains(f"test_{lbl}", na=False)
        if mask.sum() > 0:
            scope_defs.append({
                "scope":      lbl,
                "mask":       mask,
                "date_range": date_ranges.get(lbl, lbl),
                "out_path":   FIGURES / f"roc_pr_curves_bloc{lbl}.png",
            })
    # Global
    test_mask = gold["split_label"].str.contains("test_", na=False)
    scope_defs.append({
        "scope":      "global",
        "mask":       test_mask,
        "date_range": date_ranges["global"],
        "out_path":   FIGURES / "roc_pr_curves_global.png",
    })

    all_results = []

    for sdef in scope_defs:
        scope    = sdef["scope"]
        mask     = sdef["mask"]
        gold_sc  = gold[mask]
        y_sc     = gold_sc[target_col].dropna()

        if y_sc.sum() < 2:
            logger.warning(f"Scope {scope} : moins de 2 positifs — ignore")
            continue

        logger.info(f"\nScope {scope} : {len(y_sc)} mois, {int(y_sc.sum())} positifs "
                    f"({100*y_sc.mean():.1f}%)")

        scope_curves = {}
        for sig_label, sig_col in signals.items():
            if sig_col not in gold.columns:
                continue
            scores = (
                gold_sc[sig_col]
                .reindex(y_sc.index)
                .ffill().bfill().fillna(0)
                .values
            )
            c = _build_curves(y_sc.values, scores)
            scope_curves[sig_label] = c

            logger.info(
                f"  {sig_label:<15} : AUC={c['auc_roc']:.3f}  "
                f"AP={c['ap']:.3f}  F1_opt={c['f1_opt']:.3f}  "
                f"seuil={c['thresh_opt']:.3f}"
            )
            all_results.append({
                "signal": sig_label,
                "scope":  scope,
                "curves": c,
            })

        if not scope_curves:
            continue

        # Infos pour le panneau de la figure
        first_c   = list(scope_curves.values())[0]
        scope_info = {
            "n_pos":        first_c["n_pos"],
            "n_neg":        first_c["n_neg"],
            "n_total":      first_c["n_total"],
            "baseline_pr":  first_c["baseline_pr"],
            "date_range":   sdef["date_range"],
        }

        _plot_roc_pr_panel(
            curves     = scope_curves,
            scope      = scope,
            scope_info = scope_info,
            out_path   = sdef["out_path"],
        )

    # Tableau comparatif
    comparison_df = _build_comparison_table(all_results)
    out_table = REPORTS / "roc_pr_comparison.csv"
    comparison_df.to_csv(out_table, index=False)
    logger.info(f"Tableau comparatif sauvegarde : {out_table}")

    _print_table(comparison_df)

    return comparison_df


# ─── Point d'entrée ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt= "%H:%M:%S",
        stream = open(sys.stdout.fileno(), mode="w",
                      encoding="utf-8", errors="replace",
                      closefd=False, buffering=1),
    )

    df = run_roc_pr_analysis()

    print("\nFichiers generes :")
    for f in [
        "outputs/figures/roc_pr_curves_global.png",
        "outputs/figures/roc_pr_curves_blocA.png",
        "outputs/figures/roc_pr_curves_blocB.png",
        "outputs/reports/roc_pr_comparison.csv",
    ]:
        path = ROOT / f
        size = f"{path.stat().st_size // 1024} KB" if path.exists() else "MANQUANT"
        print(f"  {f:<50} {size}")

    print(f"\nTableau final ({df.shape[0]} lignes x {df.shape[1]} colonnes) :")
    display_cols = ["Modele", "Bloc", "AUC_ROC", "AP", "Delta_AUC_AP",
                    "F1_optimal", "AP_verdict"]
    print(df[display_cols].to_string(index=False))
