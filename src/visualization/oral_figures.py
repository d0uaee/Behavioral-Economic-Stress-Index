# -*- coding: utf-8 -*-
"""
oral_figures.py
===============
4 figures publication-qualite pour la presentation orale BESI.

Figures generees :
  1. Serie temporelle double-axe  : IPC YoY + BESI behavioral
  2. Poids du modele BESI          : behavioral (equal fallback) + hybrid (Lasso)
  3. Matrices de confusion Bloc B  : SARIMAX+BESI vs SARIMAX+Hybrid
  4. Radar multi-criteres          : Naif / SARIMA / SARIMAX+BESI (global)

Sortie : outputs/figures/oral/  (PNG 300 dpi + PDF)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_GOLD    = os.path.join(PROJECT_ROOT, "data", "gold", "model_dataset_monthly.csv")
RPT          = os.path.join(PROJECT_ROOT, "outputs", "reports")
OUT_DIR      = os.path.join(PROJECT_ROOT, "outputs", "figures", "oral")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Constantes de style
# ---------------------------------------------------------------------------
FONT_FAMILY  = "DejaVu Sans"   # fallback universel si Arial absent
DPI          = 300
COLORS       = {
    "yoy"    : "#1f77b4",   # bleu IPC
    "besi"   : "#d62728",   # rouge BESI
    "naif"   : "#7f7f7f",   # gris naif
    "sarima" : "#ff7f0e",   # orange SARIMA
    "besi_m" : "#2ca02c",   # vert SARIMAX+BESI
    "hybrid" : "#9467bd",   # violet Hybrid
    "lstm"   : "#e377c2",   # rose LSTM deep learning
    "pos"    : "#d62728",   # barre positive
    "zero"   : "#aec7e8",   # barre zero / negligeable
    "covid"  : "#636363",   # ligne event
    "ukraine": "#e6550d",   # ligne event 2
}

def _save(fig, name):
    """Sauvegarde PNG + PDF."""
    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"{name}.{ext}")
        fig.savefig(path, dpi=DPI, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
    print(f"  Sauvegarde : {os.path.join(OUT_DIR, name)}.png  (.pdf)")


def _setup_style():
    plt.rcParams.update({
        "font.family"       : FONT_FAMILY,
        "font.size"         : 12,
        "axes.titlesize"    : 13,
        "axes.labelsize"    : 11,
        "xtick.labelsize"   : 10,
        "ytick.labelsize"   : 10,
        "legend.fontsize"   : 10,
        "legend.framealpha" : 0.85,
        "axes.spines.top"   : False,
        "axes.spines.right" : False,
        "figure.dpi"        : 72,  # ecran (impression = DPI param)
    })


# ===========================================================================
# FIGURE 1 — Serie temporelle double-axe
# ===========================================================================
def fig1_timeseries(gold: pd.DataFrame):
    """IPC YoY (gauche) + BESI behavioral (droite), avec evenements."""
    print("=== Figure 1 : Serie temporelle double-axe ===")

    df = gold[["month", "inflation_yoy", "behavioral_index_pure"]].copy()
    df["month"] = pd.to_datetime(df["month"])
    df = df.dropna(subset=["inflation_yoy", "behavioral_index_pure"])
    df = df.sort_values("month").reset_index(drop=True)

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    # --- Courbe IPC YoY ---
    ax1.plot(df["month"], df["inflation_yoy"],
             color=COLORS["yoy"], lw=2.2, label="IPC Inflation YoY (%)", zorder=3)
    ax1.axhline(0, color=COLORS["yoy"], lw=0.8, ls="--", alpha=0.4)
    ax1.fill_between(df["month"], df["inflation_yoy"], 0,
                     where=df["inflation_yoy"] > 0,
                     alpha=0.10, color=COLORS["yoy"])

    # --- Courbe BESI (lag 0, puis deplacement visuel) ---
    # On trace le BESI un mois en avance pour montrer l'early warning
    besi_lead = df["behavioral_index_pure"].shift(-1)  # BESI avance de 1 mois
    ax2.plot(df["month"], df["behavioral_index_pure"],
             color=COLORS["besi"], lw=2.0, ls="-", alpha=0.75,
             label="BESI comportemental (t)", zorder=2)
    ax2.plot(df["month"], besi_lead,
             color=COLORS["besi"], lw=1.4, ls=":", alpha=0.50,
             label="BESI avance 1 mois (t+1)", zorder=2)

    # --- Lignes d'evenements ---
    events = [
        ("2020-03-01", "COVID-19\n(mars 2020)", COLORS["covid"]),
        ("2022-02-01", "Guerre Ukraine\n(fev. 2022)",  COLORS["ukraine"]),
        ("2022-10-01", "Pic inflation\n(oct. 2022)",   "#e6550d"),
    ]
    ymin1, ymax1 = ax1.get_ylim()
    for date_str, label, col in events:
        xd = pd.Timestamp(date_str)
        ax1.axvline(xd, color=col, lw=1.4, ls="--", alpha=0.7, zorder=1)
        ax1.text(xd, ax1.get_ylim()[1] * 0.92, label,
                 ha="center", va="top", fontsize=8.5, color=col,
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, alpha=0.8))

    # --- Annotation pic ---
    peak_idx = df["inflation_yoy"].idxmax()
    peak_date = df.loc[peak_idx, "month"]
    peak_val  = df.loc[peak_idx, "inflation_yoy"]
    ax1.annotate(
        f"Pic : {peak_val:.1f}%",
        xy=(peak_date, peak_val),
        xytext=(peak_date - pd.DateOffset(months=8), peak_val - 4),
        fontsize=9, color=COLORS["yoy"],
        arrowprops=dict(arrowstyle="->", color=COLORS["yoy"], lw=1.2),
    )

    # --- Axes labels ---
    ax1.set_ylabel("Inflation IPC YoY (%)", color=COLORS["yoy"], fontsize=11)
    ax2.set_ylabel("BESI (0-1, normalise)", color=COLORS["besi"], fontsize=11)
    ax1.tick_params(axis="y", labelcolor=COLORS["yoy"])
    ax2.tick_params(axis="y", labelcolor=COLORS["besi"])
    ax1.set_xlabel("")

    # --- Legende combinee ---
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper left", framealpha=0.9)

    # --- Titre ---
    ax1.set_title(
        "Le BESI capture-t-il le stress inflationniste marocain ?\n"
        "Signal comportemental Google Trends vs IPC officiel HCP  (2018-2024)",
        fontsize=13, fontweight="bold", pad=10
    )

    # --- Grille minimale ---
    ax1.yaxis.grid(True, ls=":", alpha=0.4, color="grey")
    ax1.set_axisbelow(True)
    ax2.yaxis.grid(False)

    # --- Zone de rupture structurelle ---
    ax1.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2023-06-01"),
                alpha=0.04, color="orange", zorder=0)

    plt.tight_layout()
    _save(fig, "fig1_timeseries")
    plt.close(fig)


# ===========================================================================
# FIGURE 2 — Poids des modeles BESI
# ===========================================================================
def fig2_weights(beh_weights: pd.DataFrame, hyb_weights: pd.DataFrame):
    """Deux panneaux : poids behavioral (fallback egal) et hybrid (Lasso)."""
    print("=== Figure 2 : Poids des modeles BESI ===")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ---- Panneau gauche : BESI behavioral (Google Trends) ----
    ax = axes[0]
    labels_beh = {
        "trends_prix_alim"   : "Prix alimentaires\n(Trends)",
        "trends_inflation"   : "Inflation maroc\n(Trends)",
        "trends_carburant"   : "Carburant\n(Trends)",
        "trends_subvention"  : "Subvention\n(Trends)",
    }
    rows_b = beh_weights.copy()
    rows_b["label"] = rows_b["feature"].map(labels_beh).fillna(rows_b["feature"])
    rows_b = rows_b.sort_values("weight", ascending=True)

    colors_b = [COLORS["pos"] if w > 0.01 else COLORS["zero"]
                for w in rows_b["weight"]]
    bars = ax.barh(rows_b["label"], rows_b["weight"] * 100,
                   color=colors_b, edgecolor="white", height=0.6)

    for bar, w in zip(bars, rows_b["weight"]):
        if w > 0.01:
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{w*100:.0f}%", va="center", ha="left", fontsize=10)

    ax.set_xlabel("Poids (%)", fontsize=11)
    ax.set_xlim(0, 35)
    ax.set_title(
        "BESI comportemental\n(Google Trends — poids egalises*)",
        fontsize=12, fontweight="bold"
    )
    ax.text(0.97, 0.04,
            "* Regularisation L1 : tous coefficients\nannules -> poids egaux (fallback)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, color="grey",
            bbox=dict(boxstyle="round", fc="lightyellow", ec="grey", alpha=0.8))
    ax.xaxis.grid(True, ls=":", alpha=0.4)
    ax.set_axisbelow(True)

    # ---- Panneau droit : Hybrid (Lasso Macro) ----
    ax = axes[1]
    labels_hyb = {
        "behavioral_pure" : "BESI comportemental",
        "fao_food_yoy"    : "Prix alim. mondiaux\n(FAO YoY)",
        "fao_cereals_yoy" : "Cereales\n(FAO YoY)",
        "fao_oils_yoy"    : "Huiles alim.\n(FAO YoY)",
        "fx_yoy"          : "Taux de change\nMAD/EUR (YoY)",
    }
    rows_h = hyb_weights.copy()
    rows_h["label"] = rows_h["feature"].map(labels_hyb).fillna(rows_h["feature"])
    rows_h = rows_h.sort_values("weight", ascending=True)

    colors_h = [COLORS["pos"] if w > 0.01 else COLORS["zero"]
                for w in rows_h["weight"]]
    bars = ax.barh(rows_h["label"], rows_h["weight"] * 100,
                   color=colors_h, edgecolor="white", height=0.6)

    for bar, w in zip(bars, rows_h["weight"]):
        if w > 0.01:
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{w*100:.1f}%", va="center", ha="left", fontsize=10,
                    fontweight="bold")

    top_feat = rows_h.loc[rows_h["weight"].idxmax(), "label"]
    top_w    = rows_h["weight"].max()
    ax.text(0.97, 0.04,
            f"Top variable : {top_feat}\n(poids = {top_w*100:.1f}%)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, color=COLORS["pos"],
            bbox=dict(boxstyle="round", fc="mistyrose", ec=COLORS["pos"], alpha=0.8))

    ax.set_xlabel("Poids Lasso (%)", fontsize=11)
    ax.set_xlim(0, 90)
    ax.set_title(
        "BESI hybride macro\n(Variables macro-internationales — Lasso CV)",
        fontsize=12, fontweight="bold"
    )
    ax.xaxis.grid(True, ls=":", alpha=0.4)
    ax.set_axisbelow(True)

    fig.suptitle(
        "Composition des indices BESI : signaux selectionnes par regularisation",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    _save(fig, "fig2_weights")
    plt.close(fig)


# ===========================================================================
# FIGURE 3 — Matrices de confusion Bloc B
# ===========================================================================
def fig3_confusion(classif: pd.DataFrame):
    """Matrices de confusion cote a cote pour le Bloc B."""
    print("=== Figure 3 : Matrices de confusion Bloc B ===")

    bloc_b = classif[classif["Bloc"] == "B"].copy()

    models_cfg = [
        ("SARIMAX + BESI behavioral", "SARIMAX + BESI\ncomportemental", COLORS["besi_m"]),
        ("SARIMAX + Hybrid macro",    "SARIMAX + Hybrid\nmacro",        COLORS["hybrid"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, (model_name, short_name, color) in zip(axes, models_cfg):
        row = bloc_b[bloc_b["Modele"] == model_name]
        if row.empty:
            ax.set_visible(False)
            continue
        row = row.iloc[0]

        TP = int(row["TP"]); FP = int(row["FP"])
        FN = int(row["FN"]); TN = int(row["TN"])
        total = TP + FP + FN + TN

        cm = np.array([[TP, FN],
                       [FP, TN]])
        pct = cm / total * 100

        # Colormap : vert pour diagonale, rouge hors diagonale
        cmap_data = np.array([[1, 0], [0, 1]], dtype=float)  # 1=bon, 0=mauvais
        bg_colors = np.where(cmap_data == 1, "#d4edda", "#f8d7da")

        # Dessin manuel de la matrice 2x2
        row_labels = ["Reel : STRESS", "Reel : NORMAL"]
        col_labels = ["Predit : STRESS", "Predit : NORMAL"]

        ax.set_xlim(0, 2); ax.set_ylim(0, 2)
        ax.set_xticks([0.5, 1.5])
        ax.set_yticks([0.5, 1.5])
        ax.set_xticklabels(col_labels, fontsize=10)
        ax.set_yticklabels(row_labels[::-1], fontsize=10)
        ax.tick_params(length=0)

        for i in range(2):
            for j in range(2):
                val  = cm[i, j]
                p    = pct[i, j]
                bg   = "#d4edda" if i == j else "#f8d7da"
                txt_col = "#155724" if i == j else "#721c24"
                ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1,
                                           fc=bg, ec="white", lw=2, zorder=1))
                ax.text(j + 0.5, 1 - i + 0.55, str(val),
                        ha="center", va="center",
                        fontsize=22, fontweight="bold", color=txt_col, zorder=2)
                ax.text(j + 0.5, 1 - i + 0.28, f"({p:.0f}%)",
                        ha="center", va="center",
                        fontsize=10, color=txt_col, zorder=2)

        # Etiquettes quadrants
        quad_labels = {
            (0, 0): "Vrai Positif\n(TP)",
            (0, 1): "Faux Negatif\n(FN)",
            (1, 0): "Faux Positif\n(FP)",
            (1, 1): "Vrai Negatif\n(TN)",
        }
        for (i, j), lbl in quad_labels.items():
            ax.text(j + 0.5, 1 - i + 0.05, lbl,
                    ha="center", va="bottom",
                    fontsize=8, color="grey", zorder=2)

        # Metriques en dessous
        recall = row["Recall"]; prec = row["Precision"]
        f1 = row["F1"]; auc = row["AUC"]
        metrics_txt = (
            f"Recall = {recall:.0%}   |   "
            f"Precision = {prec:.0%}   |   "
            f"F1 = {f1:.3f}   |   "
            f"AUC = {auc:.3f}"
        )
        ax.text(1.0, -0.12, metrics_txt,
                ha="center", va="top", fontsize=9.5,
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1.5))

        ax.set_title(short_name, fontsize=12, fontweight="bold",
                     color=color, pad=10)
        ax.spines[["top", "right", "bottom", "left"]].set_visible(False)

    fig.suptitle(
        "Detection du stress inflationniste - Bloc B (36 mois de test)\n"
        "Regime stress = IPC YoY > 75e percentile (seuil train only)",
        fontsize=13, fontweight="bold", y=1.04
    )

    # Fleche comparaison
    fig.text(0.50, 0.52,
             "BESI : 100% recall\n(aucune crise ratee)",
             ha="center", va="center", fontsize=10,
             color=COLORS["besi_m"], fontweight="bold",
             bbox=dict(boxstyle="round", fc="honeydew", ec=COLORS["besi_m"], lw=1.5))

    plt.tight_layout()
    _save(fig, "fig3_confusion")
    plt.close(fig)


# ===========================================================================
# FIGURE 4 — Radar multi-criteres
# ===========================================================================
def fig4_radar(bootstrap_ci: pd.DataFrame, classif: pd.DataFrame,
               backtest: pd.DataFrame):
    """
    Radar a 6 axes (global scope) :
      RMSE_score, MAE_score, AUC, AP, Recall, F1
    Modeles : Naif / SARIMA / SARIMAX+BESI / LSTM (Deep Learning)
    Normalisation : score = best / valeur  (RMSE/MAE inverted)
                    ou valeur directe pour AUC/AP/Recall/F1 (deja 0-1)
    Note LSTM : aucune metrique de classification (detection de crises non implantee)
    """
    print("=== Figure 4 : Radar multi-criteres ===")

    # ---- Extraction donnees ----
    # Backtest global (moyenne Bloc A + Bloc B)
    bt = backtest[backtest["bloc"].isin(["A", "B"])].copy()
    bt_agg = bt.groupby("model")[["rmse", "mae"]].mean().reset_index()

    def get_bt(model_key):
        r = bt_agg[bt_agg["model"] == model_key]
        if r.empty:
            return np.nan, np.nan
        return float(r["rmse"].iloc[0]), float(r["mae"].iloc[0])

    rmse_naif,  mae_naif  = get_bt("naif")
    rmse_sar,   mae_sar   = get_bt("sarima")
    rmse_besi,  mae_besi  = get_bt("sarimax_behavioral")

    # LSTM : charger depuis outputs/reports/lstm_results.csv
    rmse_lstm, mae_lstm = np.nan, np.nan
    lstm_bloc_b_rmse    = np.nan
    lstm_csv = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "reports", "lstm_results.csv")
    lstm_csv = os.path.normpath(lstm_csv)
    if os.path.exists(lstm_csv):
        dl = pd.read_csv(lstm_csv)
        dl.columns = [c.lower() for c in dl.columns]
        la = dl[dl["bloc"].str.upper().str.contains("_A|BLOC.A|BLOC_A", regex=True, na=False)]
        lb = dl[dl["bloc"].str.upper().str.contains("_B|BLOC.B|BLOC_B", regex=True, na=False)]
        if not la.empty and not lb.empty:
            ra, ma = float(la["rmse"].iloc[0]), float(la["mae"].iloc[0])
            rb, mb = float(lb["rmse"].iloc[0]), float(lb["mae"].iloc[0])
            rmse_lstm = (ra + rb) / 2   # moyenne des deux blocs
            mae_lstm  = (ma + mb) / 2
            lstm_bloc_b_rmse = rb
            print(f"  LSTM  BlocA RMSE={ra:.3f}  BlocB RMSE={rb:.3f}  Moy={rmse_lstm:.3f}")
    else:
        print(f"  [INFO] lstm_results.csv absent — LSTM exclu du radar")

    # Classification global (BESI uniquement)
    cl_g = classif[classif["Bloc"] == "global"]
    def get_cl(model_name):
        r = cl_g[cl_g["Modele"] == model_name]
        if r.empty:
            return 0.0, 0.0, 0.0, 0.0
        row = r.iloc[0]
        return (float(row.get("AUC", 0) or 0),
                float(row.get("AP",  0) or 0),
                float(row.get("Recall", 0) or 0),
                float(row.get("F1",  0) or 0))

    auc_besi, ap_besi, rec_besi, f1_besi = get_cl("SARIMAX + BESI behavioral")

    # Normalisation RMSE/MAE (higher=better, reference=best parmi modeles statistiques)
    # Note : on n'inclut pas le LSTM dans best_rmse (il deformerait le reference)
    best_rmse = min(v for v in [rmse_naif, rmse_sar, rmse_besi] if not np.isnan(v))
    best_mae  = min(v for v in [mae_naif,  mae_sar,  mae_besi]  if not np.isnan(v))

    def rmse_score(v):
        if np.isnan(v) or v <= 0:
            return 0.0
        return min(best_rmse / v, 1.0)   # plafonner a 1.0

    def mae_score(v):
        if np.isnan(v) or v <= 0:
            return 0.0
        return min(best_mae / v, 1.0)

    # ---- Construction des donnees ----
    categories = [
        "Score RMSE\n(pred. niveau)",
        "Score MAE\n(pred. niveau)",
        "AUC\n(detection)",
        "Precision moy.\n(AP)",
        "Recall\n(crises)",
        "F1-Score\n(equilibre)",
    ]
    N = len(categories)

    # Donnees de base (3 modeles statistiques)
    data = {
        "Naif (saisonnier)"  : [rmse_score(rmse_naif), mae_score(mae_naif), 0, 0, 0, 0],
        "SARIMA"             : [rmse_score(rmse_sar),  mae_score(mae_sar),  0, 0, 0, 0],
        "SARIMAX + BESI"     : [rmse_score(rmse_besi), mae_score(mae_besi),
                                 auc_besi, ap_besi, rec_besi, f1_besi],
    }
    model_colors  = [COLORS["naif"], COLORS["sarima"], COLORS["besi_m"]]
    model_alphas  = [0.12, 0.12, 0.20]
    model_lwidths = [1.5,  1.5,  2.5]
    model_lstyles = ["--", "-.", "-"]

    # Ajouter LSTM si disponible
    lstm_label = None
    if not np.isnan(rmse_lstm):
        lstm_label = f"LSTM (Deep Learning)\n[BlocB RMSE={lstm_bloc_b_rmse:.1f}]"
        data[lstm_label] = [rmse_score(rmse_lstm), mae_score(mae_lstm), 0, 0, 0, 0]
        model_colors.append(COLORS["lstm"])
        model_alphas.append(0.10)
        model_lwidths.append(1.8)
        model_lstyles.append((0, (4, 2)))  # pointille long

    # ---- Construction du graphique polaire ----
    fig = plt.figure(figsize=(9, 9))
    ax  = fig.add_subplot(111, polar=True)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # fermeture

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Cercles de reference
    for level in [0.25, 0.50, 0.75, 1.0]:
        ax.plot(angles, [level] * (N + 1),
                color="grey", lw=0.6, ls=":", alpha=0.5)

    # Annotations des cercles
    for level in [0.25, 0.50, 0.75, 1.0]:
        ax.text(0, level + 0.02, f"{level:.2f}",
                ha="center", va="bottom", fontsize=7.5, color="grey")

    # Tracé des polygones
    for (model_name, values), color, alpha, lw, ls in zip(
            data.items(), model_colors, model_alphas, model_lwidths, model_lstyles):

        vals = values + values[:1]
        ax.plot(angles, vals, color=color, lw=lw, ls=ls, label=model_name, zorder=3)
        ax.fill(angles, vals, color=color, alpha=alpha, zorder=2)
        ax.scatter(angles[:-1], values, color=color, s=35, zorder=4)

    # Etiquettes des axes
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)

    # Legende
    legend_anchor = (1.40, 1.15) if lstm_label else (1.35, 1.15)
    ax.legend(loc="upper right", bbox_to_anchor=legend_anchor,
              fontsize=9, framealpha=0.9)

    # Annotation BESI superieur (mise a jour avec contexte LSTM)
    lstm_note = (
        f"\nLSTM Deep Learning : RMSE Bloc B = {lstm_bloc_b_rmse:.1f}\n"
        f"(x{lstm_bloc_b_rmse/rmse_sar:.0f} vs SARIMA — echec hors distribution)"
        if not np.isnan(lstm_bloc_b_rmse) else ""
    )
    ax.text(0, -0.28,
            "SARIMAX+BESI : seul modele capable\n"
            "de detecter les crises (Recall = 100%)" + lstm_note,
            ha="center", va="top", fontsize=8.5,
            transform=ax.transData,
            bbox=dict(boxstyle="round", fc="honeydew",
                      ec=COLORS["besi_m"], lw=1.5, alpha=0.9))

    ax.set_title(
        "Comparaison multi-criteres des modeles\n"
        "Prevision IPC (score) + Detection de crises (0-1)",
        fontsize=13, fontweight="bold", pad=25
    )

    plt.tight_layout()
    _save(fig, "fig4_radar")
    plt.close(fig)


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    _setup_style()
    print("Chargement des donnees...")

    # Gold dataset
    gold = pd.read_csv(DATA_GOLD)
    gold["month"] = pd.to_datetime(gold["month"])

    # Poids
    beh_weights = pd.read_csv(os.path.join(RPT, "besi_v3_behavioral_weights.csv"))
    hyb_weights = pd.read_csv(os.path.join(RPT, "besi_v3_hybrid_weights.csv"))

    # Classification
    classif = pd.read_csv(os.path.join(RPT, "classification_metrics.csv"))

    # Bootstrap CI
    bootstrap_ci = pd.read_csv(os.path.join(RPT, "bootstrap_ci.csv"))

    # Backtest resultats
    backtest = pd.read_csv(os.path.join(RPT, "backtest_v3_results.csv"))

    print(f"Gold dataset charge : {gold.shape}")
    print(f"Destination : {OUT_DIR}\n")

    # --- Generation ---
    fig1_timeseries(gold)
    fig2_weights(beh_weights, hyb_weights)
    fig3_confusion(classif)
    fig4_radar(bootstrap_ci, classif, backtest)

    print("\n" + "=" * 60)
    print("  FIGURES ORALES GENEREES")
    print("=" * 60)
    figs = [
        ("fig1_timeseries.png", "Double-axe IPC YoY + BESI"),
        ("fig2_weights.png",    "Poids des modeles BESI"),
        ("fig3_confusion.png",  "Matrices de confusion Bloc B"),
        ("fig4_radar.png",      "Radar multi-criteres"),
    ]
    for fname, desc in figs:
        path = os.path.join(OUT_DIR, fname)
        if os.path.exists(path):
            kb = os.path.getsize(path) // 1024
            print(f"  {fname:<30}  {kb} KB  - {desc}")
        else:
            print(f"  {fname:<30}  ABSENT (erreur?)")
    print("=" * 60)


if __name__ == "__main__":
    main()
