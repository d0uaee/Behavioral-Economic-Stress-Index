"""
Dashboard de visualisation — Projet BESI Maroc
Session 8 : src/visualization.py

Produit 6 figures de qualite publication et un dashboard combine.
Toutes les figures sont sauvegardees dans outputs/figures/ a 300 DPI.

Figures generees
----------------
fig1_ipc_inflation.png        IPC mensuel + taux YoY (double axe)
fig2_besi_stress_zones.png    BESI et zones de stress colorees
fig3_behavioral_signals.png   Signaux Trends / Reddit / YouTube
fig4_model_predictions.png    SARIMA vs SARIMAX — previsions vs reel
fig5_besi_lag_analysis.png    Correlation BESI→IPC par lag (CCF)
fig6_period_performance.png   Performance par periode (avant/pendant/apres 2022)
dashboard_combined.png        Vue d'ensemble 3x2 (optionnel)

Fonction principale
-------------------
generate_dashboard(...)  -> list[Path]
"""

import warnings
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from pathlib import Path

np.random.seed(42)

# ── Chemins ───────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "outputs" / "figures"
REP_DIR = ROOT / "outputs" / "reports"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Palette coherente avec le reste du projet ─────────────────────────────────
_C_IPC     = "#2C5F8A"   # bleu — IPC
_C_YOY     = "#E07B39"   # orange — variation YoY
_C_BESI    = "#2C2C2C"   # gris fonce — BESI
_C_NORMAL  = "#2CA02C"   # vert — etat Normal
_C_WARN    = "#FF7F0E"   # orange — etat Warning
_C_STRESS  = "#D62728"   # rouge — etat High Stress
_C_TRENDS  = "#1F77B4"   # bleu — Google Trends
_C_REDDIT  = "#FF4500"   # rouge-orange — Reddit
_C_YTUBE   = "#CC0000"   # rouge — YouTube
_C_SARIMA  = "#2C5F8A"   # bleu — SARIMA
_C_SARIMAX = "#E07B39"   # orange — SARIMAX
_C_REAL    = "#111111"   # noir — valeurs reelles
_C_BREAK   = "#9467BD"   # violet — rupture 2022

_STATE_COLORS      = {"Normal": _C_NORMAL, "Warning": _C_WARN, "High Stress": _C_STRESS}
_STRESS_THRESHOLDS = (0.35, 0.65)

# ── Style academique ──────────────────────────────────────────────────────────
_STYLE = {
    "font.family":       "DejaVu Serif",
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#333333",
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "axes.titlepad":     9,
    "axes.labelsize":    9,
    "axes.labelcolor":   "#333333",
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "legend.framealpha": 0.93,
    "legend.edgecolor":  "#cccccc",
    "grid.color":        "#e5e5e5",
    "grid.linewidth":    0.65,
    "lines.linewidth":   1.7,
    "savefig.facecolor": "white",
}


def _apply_style() -> None:
    plt.rcParams.update(_STYLE)


def _despine(ax) -> None:
    """Supprime les axes haut et droit (style minimaliste)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _fmt_xdates(ax) -> None:
    """Formate l'axe des dates en annees (pas de 2 ans)."""
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")


def _add_break_vline(
    ax,
    date: str = "2022-01-01",
    label: str = "Choc inflationniste (jan. 2022)",
) -> None:
    ax.axvline(
        pd.Timestamp(date), color=_C_BREAK,
        lw=1.5, ls="--", alpha=0.75, label=label,
    )


def _textbox(ax, text: str, loc: str = "upper left") -> None:
    """Encadre de statistiques dans un coin de l'axe."""
    coords = {
        "upper left":  (0.02, 0.97, "left",  "top"),
        "upper right": (0.97, 0.97, "right", "top"),
        "lower left":  (0.02, 0.04, "left",  "bottom"),
        "lower right": (0.97, 0.04, "right", "bottom"),
    }
    x, y, ha, va = coords.get(loc, (0.02, 0.97, "left", "top"))
    ax.text(
        x, y, text, transform=ax.transAxes,
        fontsize=7.5, va=va, ha=ha, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cccccc", alpha=0.92),
    )


# ── Chargement des donnees ────────────────────────────────────────────────────

def _load_data() -> tuple:
    """Charge ipc_processed.csv et master_dataset.csv depuis le cache."""
    ipc_path    = ROOT / "data" / "processed" / "ipc_processed.csv"
    master_path = ROOT / "data" / "processed" / "master_dataset.csv"
    for p in (ipc_path, master_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Fichier manquant : {p}\n"
                "Lancer d'abord : python src/data_pipeline.py"
            )
    ipc_df = pd.read_csv(
        ipc_path, parse_dates=["date"], index_col="date"
    )
    master_df = pd.read_csv(
        master_path, parse_dates=["date"], index_col="date"
    )
    ipc_df.index.freq    = "MS"
    master_df.index.freq = "MS"
    return ipc_df, master_df


def _quick_sarima_forecast(
    ipc: pd.Series,
    besi: pd.Series,
    train_end: str = "2021-12-01",
    pdq: tuple = (1, 1, 1),
    PDQ: tuple = (0, 1, 1),
) -> tuple:
    """
    Ajuste SARIMA et SARIMAX sur le train, prevoit sur toute la periode test.
    Utilise les valeurs realisees de BESI comme exog futur (hypothese backtesting).
    Retourne (sarima_fc, sarimax_fc, y_test, test_dates).
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    te    = pd.Timestamp(train_end)
    train = ipc[ipc.index <= te]
    test  = ipc[ipc.index >  te]
    h     = len(test)
    if h == 0:
        return None, None, test, test.index

    train_e = besi.reindex(train.index).ffill().bfill()
    test_e  = besi.reindex(test.index).ffill().bfill()

    sarima_fc, sarimax_fc = None, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            m_s = SARIMAX(
                train, order=pdq, seasonal_order=(*PDQ, 12),
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False)
            sarima_fc = m_s.forecast(steps=h)
        except Exception:
            pass
        try:
            m_sx = SARIMAX(
                train, exog=train_e, order=pdq, seasonal_order=(*PDQ, 12),
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False)
            sarimax_fc = m_sx.forecast(steps=h, exog=test_e.values)
        except Exception:
            pass

    return sarima_fc, sarimax_fc, test, test.index


# =============================================================================
# FIGURE 1 — IPC mensuel + taux d'inflation YoY (double axe)
# =============================================================================

def _fig1_ipc_inflation(ipc_df: pd.DataFrame, dpi: int) -> Path:
    """
    Axe gauche  : IPC absolu (base 100)
    Axe droit   : taux d'inflation YoY en %
    Zone ombree : periode post-2022 (choc inflationniste)
    """
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    ipc = ipc_df["ipc"]
    if "ipc_yoy" in ipc_df.columns:
        yoy_raw = ipc_df["ipc_yoy"]
        yoy = yoy_raw * 100 if yoy_raw.abs().max() < 2.0 else yoy_raw
    else:
        yoy = ipc.pct_change(12) * 100

    bp = pd.Timestamp("2022-01-01")

    # Zone post-choc
    ax1.axvspan(bp, ipc.index[-1], color=_C_STRESS, alpha=0.055,
                label="Periode post-choc (2022+)")

    # Aire sous la courbe IPC
    ax1.fill_between(ipc.index, ipc.values, ipc.min() * 0.997,
                     color=_C_IPC, alpha=0.10)
    ax1.plot(ipc.index, ipc.values, color=_C_IPC, lw=2.0,
             label="IPC mensuel (base 100)")

    # Taux YoY
    ax2.plot(yoy.index, yoy.values, color=_C_YOY, lw=1.5, alpha=0.90,
             label="Inflation YoY (%)")
    ax2.axhline(2.0, color=_C_YOY,    lw=0.9, ls=":", alpha=0.60,
                label="Seuil modere (+2%)")
    ax2.axhline(4.0, color=_C_STRESS, lw=0.9, ls=":", alpha=0.60,
                label="Seuil eleve (+4%)")
    ax2.axhline(0.0, color="black", lw=0.5, alpha=0.22)

    _add_break_vline(ax1)

    ax1.set_ylabel("Indice des Prix a la Consommation (IPC)",
                   color=_C_IPC, fontsize=9)
    ax2.set_ylabel("Variation annuelle (%)", color=_C_YOY, fontsize=9)
    ax1.set_xlabel("Annee", fontsize=9)
    ax1.set_title(
        "IPC mensuel et taux d'inflation annuel — Maroc (2010–2024)\n"
        "Source : HCP / Banque Mondiale",
        fontsize=11, fontweight="bold",
    )
    ax1.tick_params(axis="y", labelcolor=_C_IPC)
    ax2.tick_params(axis="y", labelcolor=_C_YOY)

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2,
               loc="upper left", fontsize=7.5, ncol=2)

    _despine(ax1)
    ax2.spines["top"].set_visible(False)
    ax1.grid(True, alpha=0.35)
    _fmt_xdates(ax1)

    # Annotation pic d'inflation
    yoy_valid = yoy.dropna()
    if len(yoy_valid) > 0:
        yoy_max_date = yoy_valid.idxmax()
        yoy_max_val  = yoy_valid.max()
        ax2.annotate(
            f"Pic : {yoy_max_val:.1f}%\n({yoy_max_date.strftime('%b %Y')})",
            xy=(yoy_max_date, yoy_max_val),
            xytext=(15, -30), textcoords="offset points",
            fontsize=7.5, color=_C_STRESS,
            arrowprops=dict(arrowstyle="->", color=_C_STRESS, lw=1.0),
        )

    plt.tight_layout()
    path = FIG_DIR / "fig1_ipc_inflation.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


# =============================================================================
# FIGURE 2 — BESI avec zones de stress colorees
# =============================================================================

def _fig2_besi_zones(master_df: pd.DataFrame, dpi: int) -> Path:
    """
    Zones colorees : vert (Normal), orange (Warning), rouge (High Stress).
    Coloration temporelle de la courbe BESI selon l'etat courant.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    besi       = master_df["besi"]
    stress_raw = master_df.get("stress_level", pd.Series(dtype=str))
    bp         = pd.Timestamp("2022-01-01")

    # Bandes horizontales de fond
    ax.axhspan(0,                     _STRESS_THRESHOLDS[0],
               color=_C_NORMAL, alpha=0.12, zorder=0,
               label=f"Zone Normal  (BESI < {_STRESS_THRESHOLDS[0]})")
    ax.axhspan(_STRESS_THRESHOLDS[0], _STRESS_THRESHOLDS[1],
               color=_C_WARN,   alpha=0.12, zorder=0,
               label=f"Zone Warning ({_STRESS_THRESHOLDS[0]}–{_STRESS_THRESHOLDS[1]})")
    ax.axhspan(_STRESS_THRESHOLDS[1], 1.05,
               color=_C_STRESS, alpha=0.12, zorder=0,
               label=f"Zone High Stress (BESI > {_STRESS_THRESHOLDS[1]})")

    # Aire coloree sous la courbe par etat
    if len(stress_raw) > 0:
        for label, c in _STATE_COLORS.items():
            mask = stress_raw.reindex(besi.index, fill_value="").eq(label)
            if mask.any():
                ax.fill_between(
                    besi.index, 0, besi.values,
                    where=mask.values, color=c, alpha=0.30,
                    step="post", zorder=1,
                )

    # Courbe BESI principale
    ax.plot(besi.index, besi.values, color=_C_BESI, lw=2.0,
            zorder=4, label="BESI")

    # Lignes de seuil
    ax.axhline(_STRESS_THRESHOLDS[0], color=_C_WARN, lw=1.0, ls="--",
               alpha=0.70, label=f"Seuil Warning  = {_STRESS_THRESHOLDS[0]}")
    ax.axhline(_STRESS_THRESHOLDS[1], color=_C_STRESS, lw=1.0, ls="--",
               alpha=0.70, label=f"Seuil H. Stress = {_STRESS_THRESHOLDS[1]}")

    _add_break_vline(ax)

    # Encadre statistiques
    if len(stress_raw.dropna()) > 0:
        freq_n = (stress_raw == "Normal").mean() * 100
        freq_w = (stress_raw == "Warning").mean() * 100
        freq_s = (stress_raw == "High Stress").mean() * 100
        _textbox(
            ax,
            f"Frequence des etats\n"
            f"Normal      : {freq_n:.0f}%\n"
            f"Warning     : {freq_w:.0f}%\n"
            f"High Stress : {freq_s:.0f}%",
            loc="upper right",
        )

    ax.set_ylim(-0.02, 1.08)
    ax.set_ylabel("BESI (indice normalise 0–1)", fontsize=9)
    ax.set_xlabel("Annee", fontsize=9)
    ax.set_title(
        "Indice BESI et etats de stress economique des menages marocains\n"
        "BESI = 0.40 x Trends + 0.30 x Reddit + 0.20 x YouTube + 0.10 x IPC_change",
        fontsize=11, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=7.5, ncol=2)
    _despine(ax)
    ax.grid(True, alpha=0.35)
    _fmt_xdates(ax)

    plt.tight_layout()
    path = FIG_DIR / "fig2_besi_stress_zones.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


# =============================================================================
# FIGURE 3 — Signaux comportementaux individuels
# =============================================================================

def _fig3_behavioral_signals(master_df: pd.DataFrame, dpi: int) -> Path:
    """
    4 panneaux empiles (axe x partage) :
      1. Google Trends (trends_composite)
      2. Reddit r/Morocco (reddit_composite)
      3. YouTube (youtube_composite)
      4. BESI composite + zones de stress
    """
    signal_info = [
        ("Google Trends",    "trends_composite",  _C_TRENDS),
        ("Reddit r/Morocco", "reddit_composite",  _C_REDDIT),
        ("YouTube",          "youtube_composite", _C_YTUBE),
    ]

    fig, axes = plt.subplots(
        4, 1, figsize=(12, 11), sharex=True,
        gridspec_kw={"hspace": 0.12},
    )
    fig.suptitle(
        "Signaux comportementaux normalises — Sources numeriques (2010–2024)\n"
        "Normalisation Min-Max 0–1 par source | Geo = MA (Maroc)",
        fontsize=11, fontweight="bold",
    )

    bp = pd.Timestamp("2022-01-01")

    # Panneaux 1-3 : signaux individuels ─────────────────────────────────────
    for ax, (label, col, color) in zip(axes[:3], signal_info):
        if col in master_df.columns:
            s    = master_df[col].dropna()
            s_ma = s.rolling(3, center=True).mean()

            ax.fill_between(s.index, s.values, alpha=0.15, color=color)
            ax.plot(s.index, s.values,
                    color=color, lw=1.2, alpha=0.60)
            ax.plot(s_ma.index, s_ma.values,
                    color=color, lw=2.0, label=f"{label} (moy. mob. 3m)")

            # Seuil Warning BESI en reference
            ax.axhline(_STRESS_THRESHOLDS[0], color="gray",
                       lw=0.7, ls=":", alpha=0.50)
        else:
            ax.text(
                0.5, 0.5,
                f"Signal {label} non disponible\n(relancer data_pipeline.py)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=8, color="gray",
            )

        ax.axvline(bp, color=_C_BREAK, lw=1.1, ls="--", alpha=0.55)
        ax.set_ylabel(label, fontsize=8, labelpad=3)
        ax.set_ylim(-0.05, 1.20)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
        _despine(ax)
        ax.grid(True, alpha=0.28)
        ax.legend(fontsize=7.5, loc="upper left")

    # Panneau 4 : BESI composite ──────────────────────────────────────────────
    ax4   = axes[3]
    besi  = master_df["besi"]
    s_raw = master_df.get("stress_level", pd.Series(dtype=str))

    ax4.axhspan(0, _STRESS_THRESHOLDS[0], color=_C_NORMAL, alpha=0.10)
    ax4.axhspan(_STRESS_THRESHOLDS[0], _STRESS_THRESHOLDS[1],
                color=_C_WARN, alpha=0.10)
    ax4.axhspan(_STRESS_THRESHOLDS[1], 1.1, color=_C_STRESS, alpha=0.10)

    ax4.fill_between(besi.index, besi.values, alpha=0.20, color=_C_BESI)
    ax4.plot(besi.index, besi.values, color=_C_BESI, lw=2.0,
             label="BESI composite")
    ax4.axhline(_STRESS_THRESHOLDS[0], color=_C_WARN,   lw=0.8, ls=":", alpha=0.65)
    ax4.axhline(_STRESS_THRESHOLDS[1], color=_C_STRESS, lw=0.8, ls=":", alpha=0.65)
    ax4.axvline(bp, color=_C_BREAK, lw=1.1, ls="--", alpha=0.55,
                label="Choc 2022")

    ax4.set_ylabel("BESI", fontsize=8, labelpad=3)
    ax4.set_ylim(-0.05, 1.20)
    ax4.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
    _despine(ax4)
    ax4.grid(True, alpha=0.28)
    ax4.legend(fontsize=7.5, loc="upper left", ncol=2)
    ax4.set_xlabel("Annee", fontsize=9)

    _fmt_xdates(ax4)
    plt.tight_layout()
    path = FIG_DIR / "fig3_behavioral_signals.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


# =============================================================================
# FIGURE 4 — SARIMA vs SARIMAX : previsions vs valeurs reelles
# =============================================================================

def _fig4_model_predictions(
    ipc: pd.Series,
    besi: pd.Series,
    sarima_results: "dict | None",
    sarimax_results: "dict | None",
    train_end: str,
    dpi: int,
) -> Path:
    """
    Panneau haut : IPC reel + previsions SARIMA et SARIMAX sur la periode test.
    Panneau bas  : erreurs absolues par modele.
    """
    te = pd.Timestamp(train_end)

    # ── Recuperation ou calcul des previsions ─────────────────────────────────
    if (sarima_results is not None and "y_pred" in sarima_results
            and sarima_results["y_pred"] is not None):
        test_dates  = sarima_results.get("test_dates", ipc[ipc.index > te].index)
        fc_sarima   = pd.Series(sarima_results["y_pred"], index=test_dates)
        y_test      = (pd.Series(sarima_results["y_true"], index=test_dates)
                       if "y_true" in sarima_results
                       else ipc.reindex(test_dates))
        fc_sarimax  = (pd.Series(sarimax_results["y_pred"],
                                  index=sarimax_results.get("test_dates", test_dates))
                       if sarimax_results and "y_pred" in sarimax_results else None)
        method_lbl  = "walk-forward validation"
    else:
        print("    Ajustement SARIMA/SARIMAX (prevision directe)...")
        fc_sarima, fc_sarimax, y_test, test_dates = _quick_sarima_forecast(
            ipc, besi, train_end=train_end,
        )
        method_lbl = "prevision directe multi-pas"

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 9),
        gridspec_kw={"height_ratios": [2.5, 1], "hspace": 0.28},
    )
    fig.suptitle(
        f"SARIMA vs SARIMAX — Previsions vs valeurs reelles ({method_lbl})\n"
        f"Train : jusqu'au {te.strftime('%B %Y')}  |  "
        f"Test  : {y_test.index[0].strftime('%b. %Y')} – {y_test.index[-1].strftime('%b. %Y')}",
        fontsize=11, fontweight="bold",
    )

    # Panneau 1 : valeurs absolues ─────────────────────────────────────────────
    train_ipc = ipc[ipc.index <= te]
    ax1.plot(train_ipc.index, train_ipc.values,
             color="#bbbbbb", lw=1.3, zorder=1, label="IPC (entrainement)")
    ax1.plot(y_test.index, y_test.values,
             color=_C_REAL, lw=2.3, zorder=5, label="IPC reel (test)")
    ax1.axvline(te, color=_C_BREAK, lw=1.3, ls="--", alpha=0.60,
                label=f"Coupure {te.strftime('%b. %Y')}")

    rmse_s, rmse_sx = np.nan, np.nan
    if fc_sarima is not None:
        yt = y_test.values[:len(fc_sarima)]
        rmse_s = float(np.sqrt(np.mean((yt - fc_sarima.values) ** 2)))
        ax1.plot(fc_sarima.index, fc_sarima.values,
                 color=_C_SARIMA, lw=1.8, ls="-.",
                 label=f"SARIMA(1,1,1)(0,1,1)[12]   RMSE = {rmse_s:.4f}",
                 zorder=4)

    if fc_sarimax is not None:
        yt = y_test.values[:len(fc_sarimax)]
        rmse_sx = float(np.sqrt(np.mean((yt - fc_sarimax.values) ** 2)))
        ax1.plot(fc_sarimax.index, fc_sarimax.values,
                 color=_C_SARIMAX, lw=1.8, ls="--",
                 label=f"SARIMAX + BESI                  RMSE = {rmse_sx:.4f}",
                 zorder=4)

    # Encadre comparatif
    if not (np.isnan(rmse_s) or np.isnan(rmse_sx)):
        gain = (rmse_s - rmse_sx) / rmse_s * 100
        sign = "amelioration" if gain > 0 else "degradation"
        _textbox(
            ax1,
            f"RMSE SARIMA  : {rmse_s:.5f}\n"
            f"RMSE SARIMAX : {rmse_sx:.5f}\n"
            f"Gain BESI    : {gain:+.1f}% ({sign})",
            loc="lower right",
        )

    ax1.set_ylabel("IPC (base 100)", fontsize=9)
    ax1.legend(fontsize=7.5, ncol=2, loc="upper left")
    _despine(ax1)
    ax1.grid(True, alpha=0.35)
    _fmt_xdates(ax1)

    # Panneau 2 : erreurs absolues ─────────────────────────────────────────────
    if fc_sarima is not None:
        err_s = np.abs(y_test.values[:len(fc_sarima)] - fc_sarima.values)
        ax2.plot(fc_sarima.index, err_s, color=_C_SARIMA, lw=1.4, ls="-.",
                 label=f"Erreur SARIMA   MAE = {err_s.mean():.4f}")
        ax2.fill_between(fc_sarima.index, err_s, alpha=0.12, color=_C_SARIMA)

    if fc_sarimax is not None:
        err_sx = np.abs(y_test.values[:len(fc_sarimax)] - fc_sarimax.values)
        ax2.plot(fc_sarimax.index, err_sx, color=_C_SARIMAX, lw=1.4, ls="--",
                 label=f"Erreur SARIMAX  MAE = {err_sx.mean():.4f}")
        ax2.fill_between(fc_sarimax.index, err_sx, alpha=0.10, color=_C_SARIMAX)

    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_ylabel("Erreur absolue", fontsize=9)
    ax2.set_xlabel("Annee", fontsize=9)
    ax2.legend(fontsize=7.5, loc="upper left")
    _despine(ax2)
    ax2.grid(True, alpha=0.35)
    _fmt_xdates(ax2)

    plt.tight_layout()
    path = FIG_DIR / "fig4_model_predictions.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


# =============================================================================
# FIGURE 5 — Analyse lag CCF : correlation BESI → IPC par lag
# =============================================================================

def _fig5_besi_lag_analysis(
    ipc: pd.Series,
    besi: pd.Series,
    max_lag: int,
    dpi: int,
) -> Path:
    """
    Panneau gauche : barplot horizontal CCF par lag avec couleurs.
    Panneau droit  : courbe CCF avec annotation lead time.
    """
    # Calcul de la CCF : corr(BESI[t], IPC_YoY[t+lag])
    ipc_yoy = ipc.pct_change(12).dropna() * 100
    common  = besi.index.intersection(ipc_yoy.index)
    b_al    = besi.reindex(common).ffill().bfill()
    y_al    = ipc_yoy.reindex(common)

    lags, corrs = [], []
    for lag in range(0, max_lag + 1):
        if lag == 0:
            r = float(b_al.corr(y_al))
        else:
            r = float(b_al.iloc[:-lag].corr(y_al.iloc[lag:]))
        lags.append(lag)
        corrs.append(r if not np.isnan(r) else 0.0)

    opt_lag  = int(lags[int(np.argmax(corrs))])
    opt_corr = corrs[opt_lag]

    # Test de Granger (optionnel)
    granger_pval = np.nan
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
        g_lag = max(1, min(opt_lag, max_lag // 2))
        g_df  = pd.DataFrame({
            "ipc":  y_al.diff().dropna(),
            "besi": b_al.diff().dropna(),
        }).dropna()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gc_res = grangercausalitytests(
                g_df[["ipc", "besi"]], maxlag=g_lag, verbose=False
            )
        granger_pval = float(gc_res[g_lag][0]["ssr_ftest"][1])
    except Exception:
        pass

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5),
                                    gridspec_kw={"width_ratios": [1.8, 1]})
    fig.suptitle(
        "Analyse de causalite : BESI anticipe-t-il le stress IPC officiel ?\n"
        "Cross-Correlation Function (CCF) — corr( BESI[t] , IPC_YoY[t+lag] )",
        fontsize=11, fontweight="bold",
    )

    # Panneau gauche : barplot horizontal ─────────────────────────────────────
    bar_colors = [
        (_C_SARIMAX if i == opt_lag else (_C_NORMAL if r >= 0 else _C_STRESS))
        for i, r in enumerate(corrs)
    ]
    bars = ax1.barh(
        lags, corrs, color=bar_colors, alpha=0.82,
        edgecolor="white", linewidth=0.5, height=0.70,
    )
    for lag, r, bar in zip(lags, corrs, bars):
        ax1.text(
            r + (0.006 if r >= 0 else -0.006),
            bar.get_y() + bar.get_height() / 2,
            f"{r:+.3f}",
            va="center", ha="left" if r >= 0 else "right",
            fontsize=7.5,
            fontweight="bold" if lag == opt_lag else "normal",
            color=_C_SARIMAX if lag == opt_lag else "#333333",
        )

    ax1.axvline(0, color="black", lw=0.7)
    ax1.axhline(opt_lag, color=_C_SARIMAX, lw=1.2, ls="--", alpha=0.55,
                label=f"Lag optimal = {opt_lag} mois")
    ax1.set_xlabel("Correlation r", fontsize=9)
    ax1.set_ylabel("Lag (mois) — BESI precede IPC de lag mois", fontsize=9)
    ax1.set_yticks(lags)
    ax1.set_yticklabels([f"lag = {l}" for l in lags], fontsize=8)
    ax1.invert_yaxis()
    ax1.legend(fontsize=8, loc="lower right")
    _despine(ax1)
    ax1.grid(True, alpha=0.30, axis="x")

    # Encadre statistiques
    gc_str = (
        f"Granger p = {granger_pval:.3f} "
        f"({'sig.' if granger_pval < 0.05 else 'n.s.'})"
        if not np.isnan(granger_pval) else "Granger : n/a"
    )
    _textbox(
        ax1,
        f"Lag optimal   : {opt_lag} mois\n"
        f"Correlation r : {opt_corr:+.3f}\n"
        f"{gc_str}",
        loc="upper right",
    )

    # Panneau droit : courbe CCF ───────────────────────────────────────────────
    ax2.plot(lags, corrs, color=_C_BESI, lw=2.0, marker="o",
             ms=5, label="CCF")
    ax2.fill_between(lags, 0, corrs,
                     where=[r > 0 for r in corrs],
                     alpha=0.13, color=_C_NORMAL)
    ax2.fill_between(lags, 0, corrs,
                     where=[r <= 0 for r in corrs],
                     alpha=0.13, color=_C_STRESS)
    ax2.axvline(opt_lag, color=_C_SARIMAX, lw=1.4, ls="--", alpha=0.65)
    ax2.axhline(0, color="black", lw=0.6)

    # Annotation lead time
    y_ann = opt_corr - 0.06 if opt_corr > 0.5 else opt_corr + 0.06
    ax2.annotate(
        f"Lead time\nBESI -> IPC\n{opt_lag} mois",
        xy=(opt_lag, opt_corr),
        xytext=(max(0, opt_lag - 3), y_ann),
        fontsize=7.5, color=_C_SARIMAX, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=_C_SARIMAX, lw=1.0),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=_C_SARIMAX, alpha=0.88),
    )

    ax2.set_xlabel("Lag (mois)", fontsize=9)
    ax2.set_ylabel("Correlation r", fontsize=9)
    ax2.set_title("Evolution de la CCF selon le lag", fontsize=9)
    ax2.set_xticks(lags)
    _despine(ax2)
    ax2.grid(True, alpha=0.30)

    plt.tight_layout()
    path = FIG_DIR / "fig5_besi_lag_analysis.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


# =============================================================================
# FIGURE 6 — Performance par periode (grouped bars SARIMA vs SARIMAX)
# =============================================================================

def _fig6_period_performance(dpi: int) -> Path:
    """
    Charge period_performance.csv.
    Affiche un grouped bar chart RMSE / MAE / MAPE par periode.
    Si le CSV n'est pas disponible, affiche un placeholder informatif.
    """
    csv_path = REP_DIR / "period_performance.csv"

    df_perf = None
    if csv_path.exists():
        try:
            df_perf = pd.read_csv(csv_path, index_col=0)
            # Filtrer les lignes avec RMSE aberrant (divergence numerique)
            if "SARIMA_RMSE" in df_perf.columns:
                df_perf = df_perf[
                    df_perf["SARIMA_RMSE"].notna() &
                    (df_perf["SARIMA_RMSE"] < 1.0)
                ]
        except Exception:
            df_perf = None

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
    fig.suptitle(
        "Performances SARIMA vs SARIMAX par sous-periode — Maroc (2010–2024)\n"
        "Walk-forward h=1 | annotation : gain(+) ou perte(-) du BESI vs SARIMA seul",
        fontsize=11, fontweight="bold",
    )

    if df_perf is not None and len(df_perf) > 0:
        periods_lbl = [
            p.strip().split("(")[0].strip() for p in df_perf.index
        ]
        x     = np.arange(len(df_perf))
        w     = 0.35
        has_sx = "SARIMAX_RMSE" in df_perf.columns

        metric_defs = [
            ("RMSE",   "SARIMA_RMSE",  "SARIMAX_RMSE"),
            ("MAE",    "SARIMA_MAE",   "SARIMAX_MAE"),
            ("MAPE (%)", "SARIMA_MAPE", "SARIMAX_MAPE"),
        ]

        for ax, (title, col_s, col_sx) in zip(axes, metric_defs):
            vals_s = (df_perf[col_s].values
                      if col_s in df_perf.columns
                      else np.full(len(df_perf), np.nan))

            bars_s = ax.bar(x - w / 2, vals_s, width=w, color=_C_SARIMA,
                            alpha=0.85, label="SARIMA", edgecolor="white")

            if has_sx and col_sx in df_perf.columns:
                vals_sx = df_perf[col_sx].values
                ax.bar(x + w / 2, vals_sx, width=w, color=_C_SARIMAX,
                       alpha=0.85, label="SARIMAX + BESI", edgecolor="white")

                for xi, (vs, vsx) in enumerate(zip(vals_s, vals_sx)):
                    if not (np.isnan(vs) or np.isnan(vsx) or vs == 0):
                        gain = (vs - vsx) / vs * 100
                        c_ann = _C_NORMAL if gain > 0 else _C_STRESS
                        ymax  = max(vs, vsx)
                        ax.text(
                            xi, ymax * 1.04,
                            f"{gain:+.0f}%",
                            ha="center", fontsize=7.5,
                            color=c_ann, fontweight="bold",
                        )

            ax.set_xticks(x)
            ax.set_xticklabels(periods_lbl, rotation=15,
                                ha="right", fontsize=8)
            ax.set_title(title, fontsize=10)
            ax.legend(fontsize=7.5)
            _despine(ax)
            ax.grid(True, alpha=0.30, axis="y")

        # Legende Gain_RMSE
        if "Gain_RMSE_%" in df_perf.columns and df_perf["Gain_RMSE_%"].notna().any():
            best     = df_perf["Gain_RMSE_%"].idxmax()
            best_val = df_perf.loc[best, "Gain_RMSE_%"]
            fig.text(
                0.5, 0.01,
                f"BESI aide le plus durant : "
                f"{best.strip().split('(')[0].strip()} "
                f"(gain RMSE = {best_val:+.1f}%)",
                ha="center", fontsize=8, style="italic", color="#555555",
            )
    else:
        # Placeholder
        for ax in axes:
            ax.text(
                0.5, 0.5,
                "Donnees non disponibles.\nLancer d'abord :\n\n"
                "from src.analysis import period_performance\n"
                "period_performance(ipc, exog=besi_df, save_fig=False)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=8, color="gray",
                bbox=dict(boxstyle="round", facecolor="#f8f8f8",
                          edgecolor="#cccccc"),
            )
            ax.set_xticks([])
            ax.set_yticks([])
            _despine(ax)

    plt.tight_layout()
    path = FIG_DIR / "fig6_period_performance.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


# =============================================================================
# DASHBOARD COMBINE (grille 3 x 2)
# =============================================================================

def _combined_dashboard(saved_paths: list, dpi: int) -> Path:
    """Assemble les 6 figures en une vue d'ensemble 3x2."""
    import matplotlib.image as mpimg

    ncols, nrows = 3, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(21, 14))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Dashboard BESI — Detection precoce du stress economique des menages marocains\n"
        "Douae & Adama  |  Cours Series Temporelles  |  ENSAM Meknes",
        fontsize=13, fontweight="bold", y=1.003,
    )

    for i, ax in enumerate(axes.flat):
        if i < len(saved_paths) and Path(saved_paths[i]).exists():
            img = mpimg.imread(str(saved_paths[i]))
            ax.imshow(img)
        ax.axis("off")

    plt.tight_layout(pad=0.4)
    path = FIG_DIR / "dashboard_combined.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def generate_dashboard(
    sarima_results:  "dict | None" = None,
    sarimax_results: "dict | None" = None,
    train_end:     str  = "2021-12-01",
    max_lag:       int  = 12,
    dpi:           int  = 300,
    save_combined: bool = True,
    verbose:       bool = True,
) -> list:
    """
    Genere les 6 figures de qualite publication du projet BESI Maroc.

    Figures produites (300 DPI par defaut)
    ---------------------------------------
    1. fig1_ipc_inflation.png        IPC + taux d'inflation YoY (double axe)
    2. fig2_besi_stress_zones.png    BESI + zones Normal / Warning / High Stress
    3. fig3_behavioral_signals.png   Signaux Trends / Reddit / YouTube normalises
    4. fig4_model_predictions.png    SARIMA vs SARIMAX vs valeurs reelles
    5. fig5_besi_lag_analysis.png    CCF BESI->IPC par lag (0..max_lag)
    6. fig6_period_performance.png   Performance avant / pendant / apres 2022
    dashboard_combined.png           Vue d'ensemble 3x2 (si save_combined=True)

    Parametres
    ----------
    sarima_results   : dict {y_pred, y_true, test_dates, rmse}
                       depuis walk_forward_validation() ou compare_models().
                       Si None, ajuste SARIMA rapidement en interne.
    sarimax_results  : idem pour SARIMAX + BESI
    train_end        : coupure train/test identique aux autres modules
    max_lag          : lag maximal pour la CCF (defaut 12 mois)
    dpi              : resolution PNG (300 pour publication, 150 pour apercu)
    save_combined    : generer aussi dashboard_combined.png
    verbose          : afficher la progression dans la console

    Retourne
    --------
    list[Path] — chemins des figures sauvegardees dans outputs/figures/
    """
    _apply_style()
    t0 = time.time()

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    _log(f"\n{'='*64}")
    _log("  GENERATION DU DASHBOARD BESI MAROC")
    _log(f"  DPI : {dpi}  |  Destination : {FIG_DIR}")
    _log(f"{'='*64}")

    ipc_df, master_df = _load_data()
    ipc  = ipc_df["ipc"]
    besi = master_df["besi"]
    _log(
        f"\n  Donnees chargees : IPC {ipc.index[0].date()} -> "
        f"{ipc.index[-1].date()} ({len(ipc)} mois)"
    )

    saved: list = []

    _log("\n  [1/6] IPC + inflation YoY...")
    saved.append(_fig1_ipc_inflation(ipc_df, dpi))

    _log("  [2/6] BESI + zones de stress...")
    saved.append(_fig2_besi_zones(master_df, dpi))

    _log("  [3/6] Signaux comportementaux...")
    saved.append(_fig3_behavioral_signals(master_df, dpi))

    _log("  [4/6] Predictions SARIMA vs SARIMAX...")
    saved.append(
        _fig4_model_predictions(
            ipc, besi, sarima_results, sarimax_results, train_end, dpi
        )
    )

    _log("  [5/6] Analyse CCF lag BESI -> IPC...")
    saved.append(_fig5_besi_lag_analysis(ipc, besi, max_lag, dpi))

    _log("  [6/6] Performances par periode...")
    saved.append(_fig6_period_performance(dpi))

    if save_combined:
        _log("\n  [+] Dashboard combine 3x2...")
        saved.append(_combined_dashboard(saved, dpi=150))

    elapsed = time.time() - t0
    _log(f"\n{'='*64}")
    _log(f"  Termine en {elapsed:.1f}s — {len(saved)} fichier(s) genere(s)")
    for p in saved:
        _log(f"    {p.name}")
    _log(f"{'='*64}\n")

    return saved


# ── Point d'entree ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    paths = generate_dashboard(
        train_end="2021-12-01",
        max_lag=12,
        dpi=300,
        save_combined=True,
        verbose=True,
    )
