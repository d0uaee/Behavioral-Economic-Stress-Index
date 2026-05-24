"""
src/analysis/besi_diagnostics.py — Diagnostics complets du signal BESI v1

Objectif : vérifier que le BESI behavioral porte un signal AUTONOME,
non redondant avec l'inertie de l'IPC, et qu'il PRÉCÈDE effectivement l'IPC.

Tests réalisés :
    1. Stationnarité du BESI : ADF + KPSS
       -> Si non stationnaire : BESI non utilisable en niveau pour SARIMAX
       -> En pratique : BESI normalisé 0-1 → attendu stationnaire

    2. Structure d'autocorrélation du BESI : ACF + PACF
       -> ACF décroît rapidement → BESI est un processus MA (bruit structuré)
       -> ACF décroît lentement → BESI suit une tendance → différencier

    3. Décomposition STL du BESI + ACF des résidus
       -> Vérifie que les résidus STL sont un bruit blanc
       -> Résidus non blancs → structure restante non capturée

    4. Cross-corrélogramme BESI vs inflation YoY (lags -12 à +12)
       -> Lag k positif : corr(BESI(t), IPC(t+k)) → BESI précède IPC de k mois
       -> Pic à k=1 ou k=2 → confirme le lead de 1-2 mois du BESI (early warning)
       -> Pic à k<0 → IPC précède BESI (signal retardé, moins utile)

    5. Tableau récapitulatif des corrélations à lags clés

Output :
    outputs/figures/besi_diagnostics.png      (figure principale 300 dpi)
    outputs/reports/besi_diagnostics.csv      (corrélations croisées tabulaires)
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
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Patch
except ImportError:
    matplotlib = None
    plt = None

logger = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parent.parent.parent
GOLD_DIR = ROOT / "data" / "gold"
REPORTS  = ROOT / "outputs" / "reports"
FIGURES  = ROOT / "outputs" / "figures"
REPORTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# Colonnes utilisées
BESI_COL   = "behavioral_index_pure"
IPC_COL    = "ipc_level"
YOY_COL    = "inflation_yoy"
MOM_COL    = "inflation_mom"
LAG_COLS   = ["behavioral_index_pure_lag1", "behavioral_index_pure_lag2"]

# Paramètres STL
STL_PERIOD = 12   # saisonnalité mensuelle

# Paramètres graphiques
C_BESI = "#2ecc71"
C_IPC  = "#e74c3c"
C_ACF  = "#3498db"
C_NEG  = "#e74c3c"
C_ZERO = "#7f8c8d"
C_CRISIS = "#e74c3c"


# ─── Tests de stationnarité ───────────────────────────────────────────────────

def _adf_test(series: pd.Series) -> dict:
    """ADF test (H0: racine unitaire = non stationnaire)."""
    try:
        from statsmodels.tsa.stattools import adfuller
        s = series.dropna()
        res = adfuller(s, autolag="AIC")
        return {
            "stat":   round(float(res[0]), 4),
            "pvalue": round(float(res[1]), 4),
            "lags":   int(res[2]),
            "nobs":   int(res[3]),
            "cv1":    round(float(res[4]["1%"]),  4),
            "cv5":    round(float(res[4]["5%"]),  4),
            "cv10":   round(float(res[4]["10%"]), 4),
            "stationary": bool(res[1] < 0.05),
        }
    except Exception as exc:
        logger.warning(f"ADF echec : {exc}")
        return {}


def _kpss_test(series: pd.Series) -> dict:
    """KPSS test (H0: stationnaire)."""
    try:
        from statsmodels.tsa.stattools import kpss
        s = series.dropna()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = kpss(s, regression="c", nlags="auto")
        return {
            "stat":   round(float(res[0]), 4),
            "pvalue": round(float(res[1]), 4),
            "lags":   int(res[2]),
            "cv1":    round(float(res[3]["1%"]),   4),
            "cv5":    round(float(res[3]["5%"]),   4),
            "cv10":   round(float(res[3]["10%"]),  4),
            "stationary": bool(res[1] > 0.05),   # H0 = statio, donc p>0.05 = ne rejette pas statio
        }
    except Exception as exc:
        logger.warning(f"KPSS echec : {exc}")
        return {}


def _ljung_box(series: pd.Series, lags: int = 12) -> dict:
    """Ljung-Box test (H0: pas d'autocorrélation = bruit blanc)."""
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        s = series.dropna()
        res = acorr_ljungbox(s, lags=[lags], return_df=True)
        pval = float(res["lb_pvalue"].iloc[-1])
        return {
            "stat":   round(float(res["lb_stat"].iloc[-1]), 4),
            "pvalue": round(pval, 4),
            "lag":    lags,
            "white_noise": bool(pval > 0.05),
        }
    except Exception as exc:
        logger.warning(f"Ljung-Box echec : {exc}")
        return {}


# ─── Calcul ACF / PACF ────────────────────────────────────────────────────────

def _manual_acf(series: pd.Series, max_lag: int = 24) -> np.ndarray:
    """
    ACF manuelle : corr(x_t, x_{t-k}) pour k in 1..max_lag.
    Retourne un array de longueur max_lag (sans le lag 0).
    """
    s  = series.dropna().values
    n  = len(s)
    mu = s.mean()
    var = np.var(s, ddof=0)
    if var == 0:
        return np.zeros(max_lag)
    acfs = []
    for k in range(1, max_lag + 1):
        cov = np.mean((s[k:] - mu) * (s[:n-k] - mu))
        acfs.append(cov / var)
    return np.array(acfs)


def _confidence_band(n: int, alpha: float = 0.05) -> float:
    """Bande de confiance à (1-alpha)% pour ACF : +/- z_{alpha/2} / sqrt(n)."""
    from scipy import stats as sp_stats
    z = float(sp_stats.norm.ppf(1 - alpha / 2))
    return z / np.sqrt(n)


# ─── Décomposition STL ────────────────────────────────────────────────────────

def _stl_decompose(series: pd.Series, period: int = STL_PERIOD):
    """
    Décomposition STL (Seasonal and Trend decomposition using Loess).
    Retourne (trend, seasonal, residual) ou None si indisponible.
    """
    try:
        from statsmodels.tsa.seasonal import STL
        s = series.dropna()
        if len(s) < period * 2:
            return None, None, None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = STL(s, period=period, robust=True).fit()
        return res.trend, res.seasonal, res.resid
    except Exception as exc:
        logger.warning(f"STL echec : {exc}")
        return None, None, None


# ─── Cross-corrélogramme ──────────────────────────────────────────────────────

def _cross_correlations(
    x: pd.Series,
    y: pd.Series,
    max_lag: int = 12,
) -> pd.DataFrame:
    """
    Calcule corr(x[t], y[t+k]) pour k in -max_lag..+max_lag.

    Convention :
        k > 0 : x PRÉCÈDE y de k mois (x est early warning pour y)
        k < 0 : y PRÉCÈDE x de k mois (y est early warning pour x)
        k = 0 : corrélation contemporaine

    Retourne pd.DataFrame avec colonnes [lag, correlation, abs_corr].
    """
    common = x.dropna().index.intersection(y.dropna().index)
    xs = x.reindex(common)
    ys = y.reindex(common)

    rows = []
    for k in range(-max_lag, max_lag + 1):
        if k > 0:
            r = xs.iloc[:-k].corr(ys.iloc[k:])
        elif k < 0:
            r = xs.iloc[-k:].corr(ys.iloc[:k])
        else:
            r = xs.corr(ys)
        rows.append({"lag": k, "correlation": float(r) if not np.isnan(r) else 0.0})

    df = pd.DataFrame(rows)
    df["abs_corr"] = df["correlation"].abs()
    return df


# ─── Visualisation principale ─────────────────────────────────────────────────

def _plot_besi_diagnostics(
    besi:          pd.Series,
    ipc:           pd.Series,
    yoy:           pd.Series,
    gold:          pd.DataFrame,
    stl_res:       "pd.Series | None",
    xcorr_df:      pd.DataFrame,
    xcorr_diff_df: pd.DataFrame,
    adf_res:       dict,
    kpss_res:      dict,
    lb_besi:       dict,
    lb_resid:      "dict | None",
    out_path:      Path,
) -> None:
    """
    Figure diagnostique complète en 2 × 3 subplots :

    [0,0] BESI time series + IPC (axe droit) + zones de crise
    [0,1] ACF du BESI (lags 1-24)
    [0,2] PACF du BESI (lags 1-24)
    [1,0] Cross-corrélogramme BESI -> inflation YoY (-12 à +12)
    [1,1] ACF des résidus STL du BESI
    [1,2] Tableau récapitulatif (tests + corrélations clés)
    """
    if plt is None:
        logger.error("matplotlib non disponible — figure non générée")
        return

    n_besi = len(besi.dropna())
    ci_besi = _confidence_band(n_besi)

    fig = plt.figure(figsize=(20, 12))
    gs  = gridspec.GridSpec(
        2, 3,
        figure=fig,
        hspace=0.38, wspace=0.32,
        top=0.92, bottom=0.07, left=0.06, right=0.97,
    )

    # ── [0,0] Série temporelle BESI + IPC ─────────────────────────────────────
    ax00 = fig.add_subplot(gs[0, 0])
    ax00r = ax00.twinx()

    # Zone de crise (inflation élevée 2022+)
    crisis_mask = gold["split_label"].str.contains("test_B", na=False)
    if crisis_mask.any():
        crisis_start = gold.index[crisis_mask].min()
        crisis_end   = gold.index[crisis_mask].max()
        ax00.axvspan(crisis_start, crisis_end, alpha=0.08, color=C_CRISIS, label="Bloc B (2022-2024)")

    ax00.plot(besi.index, besi.values, color=C_BESI, linewidth=2.0, label="BESI behavioral")
    ax00.fill_between(besi.index, besi.values, alpha=0.15, color=C_BESI)
    ax00r.plot(ipc.index, ipc.values, color=C_IPC, linewidth=1.5,
               linestyle="--", alpha=0.75, label="IPC niveau")

    ax00.set_ylabel("BESI (0-1)", color=C_BESI, fontsize=9)
    ax00r.set_ylabel("IPC niveau", color=C_IPC, fontsize=9)
    ax00.tick_params(axis="y", labelcolor=C_BESI)
    ax00r.tick_params(axis="y", labelcolor=C_IPC)
    ax00.set_title("BESI behavioral + IPC niveau", fontsize=11, fontweight="bold")
    ax00.tick_params(axis="x", rotation=30, labelsize=8)

    lines1, labels1 = ax00.get_legend_handles_labels()
    lines2, labels2 = ax00r.get_legend_handles_labels()
    ax00.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")
    ax00.grid(axis="y", linestyle=":", alpha=0.4)

    # ── [0,1] ACF du BESI ─────────────────────────────────────────────────────
    ax01 = fig.add_subplot(gs[0, 1])

    max_lag_acf = 24
    acf_vals = _manual_acf(besi, max_lag=max_lag_acf)
    lags_acf = np.arange(1, max_lag_acf + 1)

    colors_acf = [C_BESI if v >= 0 else C_NEG for v in acf_vals]
    ax01.bar(lags_acf, acf_vals, color=colors_acf, alpha=0.85, width=0.7)
    ax01.axhline(y=0,       color="black",  linewidth=0.8)
    ax01.axhline(y=+ci_besi, color=C_ZERO, linewidth=1.5, linestyle="--",
                 alpha=0.7, label=f"+/-95% CI")
    ax01.axhline(y=-ci_besi, color=C_ZERO, linewidth=1.5, linestyle="--", alpha=0.7)

    ax01.set_xlabel("Lag (mois)", fontsize=9)
    ax01.set_ylabel("ACF", fontsize=9)
    ax01.set_title("ACF du BESI (lags 1-24)", fontsize=11, fontweight="bold")
    ax01.set_xlim(0, max_lag_acf + 1)
    ax01.set_ylim(-0.6, 1.05)
    ax01.legend(fontsize=8)
    ax01.grid(axis="y", linestyle=":", alpha=0.4)

    # Annotation
    n_sig_acf = int((np.abs(acf_vals) > ci_besi).sum())
    ax01.text(0.97, 0.97, f"{n_sig_acf}/{max_lag_acf} sig.", ha="right", va="top",
              transform=ax01.transAxes, fontsize=8,
              bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

    # ── [0,2] PACF du BESI ────────────────────────────────────────────────────
    ax02 = fig.add_subplot(gs[0, 2])

    try:
        from statsmodels.tsa.stattools import pacf as sm_pacf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pacf_vals = sm_pacf(besi.dropna().values, nlags=max_lag_acf, method="ywm")
        pacf_vals = pacf_vals[1:]   # retirer lag 0 (=1 par définition)
    except Exception:
        pacf_vals = np.zeros(max_lag_acf)

    colors_pacf = [C_BESI if v >= 0 else C_NEG for v in pacf_vals]
    ax02.bar(lags_acf, pacf_vals, color=colors_pacf, alpha=0.85, width=0.7)
    ax02.axhline(y=0,        color="black",  linewidth=0.8)
    ax02.axhline(y=+ci_besi, color=C_ZERO,  linewidth=1.5, linestyle="--", alpha=0.7)
    ax02.axhline(y=-ci_besi, color=C_ZERO,  linewidth=1.5, linestyle="--", alpha=0.7)

    ax02.set_xlabel("Lag (mois)", fontsize=9)
    ax02.set_ylabel("PACF", fontsize=9)
    ax02.set_title("PACF du BESI (lags 1-24)", fontsize=11, fontweight="bold")
    ax02.set_xlim(0, max_lag_acf + 1)
    ax02.set_ylim(-0.6, 1.05)
    ax02.grid(axis="y", linestyle=":", alpha=0.4)

    # ── [1,0] Cross-corrélogramme sur PREMIÈRES DIFFÉRENCES ──────────────────
    # (version correcte : corrige la corrélation spurieuse des séries non-stat.)
    ax10 = fig.add_subplot(gs[1, 0])

    # Sous-graphique : inset avec CCF en niveaux (pour montrer le problème)
    ax10_inset = ax10.inset_axes([0.62, 0.55, 0.36, 0.42])

    # ── CCF sur differences (graphique principal) ─────────────────────────────
    lags_d    = xcorr_diff_df["lag"].values
    corrs_d   = xcorr_diff_df["correlation"].values
    best_lag_d  = int(xcorr_diff_df.loc[xcorr_diff_df["abs_corr"].idxmax(), "lag"])
    best_corr_d = float(xcorr_diff_df.loc[xcorr_diff_df["abs_corr"].idxmax(), "correlation"])

    n_diff   = len(besi.dropna()) - 1
    ci_diff  = _confidence_band(n_diff)

    bar_colors_d = []
    for lg, cr in zip(lags_d, corrs_d):
        if lg == best_lag_d:
            bar_colors_d.append("#f39c12")
        elif lg > 0:
            bar_colors_d.append(C_BESI)
        elif lg < 0:
            bar_colors_d.append("#95a5a6")
        else:
            bar_colors_d.append(C_ZERO)

    ax10.bar(lags_d, corrs_d, color=bar_colors_d, alpha=0.85, width=0.7)
    ax10.axhline(y=0, color="black", linewidth=0.8)
    ax10.axvline(x=0, color=C_ZERO,  linewidth=1.0, linestyle="--", alpha=0.5)
    ax10.axhline(y=+ci_diff, color=C_ZERO, linewidth=1.2, linestyle=":", alpha=0.7,
                 label=f"+/-95% CI")
    ax10.axhline(y=-ci_diff, color=C_ZERO, linewidth=1.2, linestyle=":", alpha=0.7)

    # Annoter le pic
    ann_x = best_lag_d + (2 if best_lag_d <= 8 else -4)
    ann_y = best_corr_d + 0.06 * np.sign(best_corr_d)
    ax10.annotate(
        f"Max: lag={best_lag_d:+d}\nr={best_corr_d:.3f}",
        xy=(best_lag_d, best_corr_d),
        xytext=(ann_x, ann_y),
        fontsize=8,
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0),
        bbox=dict(boxstyle="round,pad=0.2", fc="#f39c12", alpha=0.9),
    )

    ax10.set_xlabel("Lag k  (BESI leads YoY si k > 0)", fontsize=9)
    ax10.set_ylabel("Corrélation (differences)", fontsize=9)
    ax10.set_title(
        "Cross-corrélogramme : delta-BESI vs delta-YoY\n"
        "(series différenciées — évite corrélation spurieuse)",
        fontsize=10, fontweight="bold",
    )
    ax10.set_xticks(lags_d[::2])
    ax10.grid(axis="y", linestyle=":", alpha=0.4)
    ax10.legend(fontsize=7, loc="upper left")

    # ── Inset : CCF en niveaux (spurieuse) ────────────────────────────────────
    lags_l  = xcorr_df["lag"].values
    corrs_l = xcorr_df["correlation"].values
    inset_colors = [C_ZERO if lg == 0 else ("#bdc3c7" if lg < 0 else "#a8d5b5")
                    for lg in lags_l]
    ax10_inset.bar(lags_l, corrs_l, color=inset_colors, alpha=0.7, width=0.7)
    ax10_inset.axhline(y=0, color="black", linewidth=0.5)
    ax10_inset.set_title("En niveaux\n(spurieux)", fontsize=6.5, color="#c0392b",
                          fontweight="bold")
    ax10_inset.set_xticks([-12, -6, 0, 6, 12])
    ax10_inset.tick_params(labelsize=5.5)
    ax10_inset.set_ylim(-0.2, 1.0)

    legend_patches = [
        Patch(color=C_BESI,    label="delta-BESI precede delta-YoY (k>0)"),
        Patch(color="#95a5a6", label="delta-YoY precede delta-BESI (k<0)"),
        Patch(color="#f39c12", label="Correlation max"),
    ]
    ax10.legend(handles=legend_patches, fontsize=7, loc="lower right")

    # ── [1,1] ACF des résidus STL ──────────────────────────────────────────────
    ax11 = fig.add_subplot(gs[1, 1])

    if stl_res is not None and len(stl_res.dropna()) > 10:
        n_resid = len(stl_res.dropna())
        ci_res  = _confidence_band(n_resid)
        acf_res = _manual_acf(stl_res, max_lag=max_lag_acf)

        bar_col_res = [C_ACF if v >= 0 else C_NEG for v in acf_res]
        ax11.bar(lags_acf, acf_res, color=bar_col_res, alpha=0.85, width=0.7)
        ax11.axhline(y=0,       color="black", linewidth=0.8)
        ax11.axhline(y=+ci_res, color=C_ZERO,  linewidth=1.5, linestyle="--", alpha=0.7)
        ax11.axhline(y=-ci_res, color=C_ZERO,  linewidth=1.5, linestyle="--", alpha=0.7)

        n_sig_res = int((np.abs(acf_res) > ci_res).sum())
        wb_str = "Bruit blanc" if (lb_resid or {}).get("white_noise", False) else "Non-blanc"
        ax11.text(0.97, 0.97,
                  f"{n_sig_res} sig. | {wb_str}\nLB p={lb_resid.get('pvalue', float('nan')):.3f}" if lb_resid else f"{n_sig_res} sig.",
                  ha="right", va="top", transform=ax11.transAxes, fontsize=8,
                  bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

        ax11.set_title("ACF résidus STL du BESI\n(apres decomp. tendance+saison)", fontsize=11, fontweight="bold")
    else:
        ax11.text(0.5, 0.5, "STL non disponible\n(< 24 obs.)",
                  ha="center", va="center", transform=ax11.transAxes, fontsize=11)
        ax11.set_title("ACF résidus STL du BESI", fontsize=11, fontweight="bold")

    ax11.set_xlabel("Lag (mois)", fontsize=9)
    ax11.set_ylabel("ACF résidus STL", fontsize=9)
    ax11.set_xlim(0, max_lag_acf + 1)
    ax11.grid(axis="y", linestyle=":", alpha=0.4)

    # ── [1,2] Tableau récapitulatif ────────────────────────────────────────────
    ax12 = fig.add_subplot(gs[1, 2])
    ax12.axis("off")

    # Préparer les données du tableau
    def bool_str(v, true_txt, false_txt):
        if isinstance(v, bool):
            return true_txt if v else false_txt
        return "--"

    def pval_marker(p):
        if isinstance(p, float):
            if p < 0.01: return "***"
            if p < 0.05: return "**"
            if p < 0.10: return "*"
            return ""
        return ""

    adf_stat = f"{adf_res.get('stat', float('nan')):.3f}" if adf_res else "--"
    adf_p    = adf_res.get("pvalue", float("nan"))
    adf_str  = f"p={adf_p:.4f} {pval_marker(adf_p)}" if adf_res else "--"
    adf_conc = bool_str(adf_res.get("stationary"), "Stationnaire", "Non stationnaire") if adf_res else "--"

    kpss_stat = f"{kpss_res.get('stat', float('nan')):.3f}" if kpss_res else "--"
    kpss_p    = kpss_res.get("pvalue", float("nan"))
    kpss_str  = f"p={kpss_p:.4f} {pval_marker(kpss_p)}" if kpss_res else "--"
    kpss_conc = bool_str(kpss_res.get("stationary"), "Stationnaire", "Non stationnaire") if kpss_res else "--"

    lb_str  = f"p={lb_besi.get('pvalue', float('nan')):.4f}" if lb_besi else "--"
    lb_conc = bool_str(lb_besi.get("white_noise"), "Bruit blanc", "Autocorrelé") if lb_besi else "--"

    # Corrélations clés — utiliser le CCF sur différences (correct statistiquement)
    def xcorr_at_diff(k):
        row = xcorr_diff_df[xcorr_diff_df["lag"] == k]
        if row.empty:
            return float("nan")
        return float(row["correlation"].iloc[0])

    xcorr_m1  = xcorr_at_diff(-1)
    xcorr_0   = xcorr_at_diff(0)
    xcorr_p1  = xcorr_at_diff(1)
    xcorr_p2  = xcorr_at_diff(2)
    xcorr_p3  = xcorr_at_diff(3)

    # best_lag_d et best_corr_d calculés plus haut dans la fonction
    best_lag_sign = "lead" if best_lag_d > 0 else ("lag" if best_lag_d < 0 else "contemp.")

    cell_data = [
        # Section stationnarité
        ["-- Tests de stationnarité --", "", ""],
        ["ADF  (H0=non stat.)", adf_str, adf_conc],
        ["KPSS (H0=stat.)",     kpss_str, kpss_conc],
        ["Ljung-Box BESI",      lb_str,   lb_conc],
        ["", "", ""],
        # Section corrélations sur DIFFERENCES
        ["-- CCF delta-BESI vs delta-YoY --", "", ""],
        ["d(BESI)[t-1] vs d(YoY)[t]", f"{xcorr_m1:.3f}", "YoY leads BESI"],
        ["d(BESI)[t]   vs d(YoY)[t]", f"{xcorr_0:.3f}",  "Contemp."],
        ["d(BESI)[t+1] vs d(YoY)[t]", f"{xcorr_p1:.3f}", "BESI lead 1m"],
        ["d(BESI)[t+2] vs d(YoY)[t]", f"{xcorr_p2:.3f}", "BESI lead 2m"],
        ["d(BESI)[t+3] vs d(YoY)[t]", f"{xcorr_p3:.3f}", "BESI lead 3m"],
        ["", "", ""],
        ["-- Lead optimal (diff.) --", "", ""],
        ["Max corr. abs.", f"lag={best_lag_d:+d}  r={best_corr_d:.3f}", best_lag_sign],
    ]

    col_widths = [0.45, 0.30, 0.25]
    row_height = 0.063
    y0 = 0.98

    for ri, row in enumerate(cell_data):
        y = y0 - ri * row_height
        x = 0.0
        is_header = row[0].startswith("──")
        is_empty  = all(c == "" for c in row)
        if is_empty:
            continue
        for ci, (cell, cw) in enumerate(zip(row, col_widths)):
            weight = "bold" if (is_header or ci == 0) else "normal"
            color  = "#2c3e50" if is_header else "black"
            fs     = 8.5 if is_header else 8
            ax12.text(x, y, cell, ha="left", va="top",
                      transform=ax12.transAxes,
                      fontsize=fs, fontweight=weight, color=color)
            x += cw

    ax12.set_title("Synthèse diagnostique BESI", fontsize=11, fontweight="bold")

    # ── Titre global ──────────────────────────────────────────────────────────
    stat_verdict = ""
    if adf_res and kpss_res:
        adf_ok  = adf_res.get("stationary", False)
        kpss_ok = kpss_res.get("stationary", False)
        if adf_ok and kpss_ok:
            stat_verdict = " | Stationnarité : CONFIRMEE (ADF + KPSS)"
        elif adf_ok or kpss_ok:
            stat_verdict = " | Stationnarité : PARTIELLE (1 test sur 2)"
        else:
            stat_verdict = " | Stationnarité : REJETEE (différencier avant SARIMAX)"

    # Utiliser le lead sur différences (plus fiable)
    best_lag_d  = int(xcorr_diff_df.loc[xcorr_diff_df["abs_corr"].idxmax(), "lag"])
    best_corr_d = float(xcorr_diff_df.loc[xcorr_diff_df["abs_corr"].idxmax(), "correlation"])
    lead_verdict = ""
    if best_lag_d > 0:
        lead_verdict = f" | delta-BESI precede delta-YoY de {best_lag_d} mois (early warning)"
    elif best_lag_d == 0:
        lead_verdict = " | Correlation max contemporaine sur differences"
    else:
        lead_verdict = f" | delta-YoY precede delta-BESI de {-best_lag_d} mois"

    fig.suptitle(
        f"Diagnostics du signal BESI behavioral (n={n_besi} mois, 2017-2024)"
        f"{stat_verdict}{lead_verdict}",
        fontsize=12, fontweight="bold", y=0.97,
    )

    plt.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Figure sauvegardee : {out_path}  ({int(out_path.stat().st_size/1024)} KB)")


# ─── Pipeline principal ────────────────────────────────────────────────────────

def run_besi_diagnostics(
    gold_path: "str | Path | None" = None,
) -> dict:
    """
    Lance les diagnostics complets du BESI behavioral.

    Retourne un dict avec tous les résultats (stationnarité, ACF, lead optimal).
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

    # ── Vérifications colonnes ────────────────────────────────────────────────
    for col in (BESI_COL, IPC_COL, YOY_COL):
        if col not in gold.columns:
            raise KeyError(
                f"Colonne '{col}' absente du Gold dataset. "
                f"Colonnes disponibles : {list(gold.columns)}"
            )

    besi = gold[BESI_COL].dropna()
    ipc  = gold[IPC_COL].dropna()
    yoy  = gold[YOY_COL].dropna()

    logger.info(f"BESI : n={len(besi)}, mean={besi.mean():.3f}, std={besi.std():.3f}")
    logger.info(f"  min={besi.min():.3f}, max={besi.max():.3f}")
    logger.info(f"  Periode : {besi.index.min().date()} -> {besi.index.max().date()}")

    # ── 1. Tests de stationnarité ─────────────────────────────────────────────
    logger.info("\n=== Test ADF (H0 : racine unitaire = non stationnaire) ===")
    adf_res = _adf_test(besi)
    if adf_res:
        logger.info(
            f"  stat={adf_res['stat']:.4f}  p={adf_res['pvalue']:.4f}  "
            f"CV5%={adf_res['cv5']}  "
            f"-> {'STATIONNAIRE' if adf_res['stationary'] else 'NON STATIONNAIRE'}"
        )

    logger.info("\n=== Test KPSS (H0 : stationnaire) ===")
    kpss_res = _kpss_test(besi)
    if kpss_res:
        logger.info(
            f"  stat={kpss_res['stat']:.4f}  p={kpss_res['pvalue']:.4f}  "
            f"CV5%={kpss_res['cv5']}  "
            f"-> {'STATIONNAIRE' if kpss_res['stationary'] else 'NON STATIONNAIRE'}"
        )

    # ── 2. Ljung-Box sur BESI ─────────────────────────────────────────────────
    logger.info("\n=== Test Ljung-Box BESI (H0 : bruit blanc) ===")
    lb_besi = _ljung_box(besi, lags=12)
    if lb_besi:
        logger.info(
            f"  stat={lb_besi['stat']:.4f}  p={lb_besi['pvalue']:.4f}  "
            f"-> {'BRUIT BLANC' if lb_besi['white_noise'] else 'AUTOCORRELÉ'}"
        )

    # ── 3. Décomposition STL ──────────────────────────────────────────────────
    logger.info("\n=== Décomposition STL du BESI ===")
    trend, seasonal, residual = _stl_decompose(besi)
    stl_resid = None
    lb_resid  = None

    if residual is not None:
        stl_resid = pd.Series(residual, index=besi.dropna().index)
        logger.info(
            f"  STL residuals : n={len(stl_resid)}, "
            f"std={stl_resid.std():.4f}"
        )
        lb_resid = _ljung_box(stl_resid, lags=12)
        if lb_resid:
            logger.info(
                f"  Ljung-Box residus : p={lb_resid['pvalue']:.4f}  "
                f"-> {'BRUIT BLANC' if lb_resid['white_noise'] else 'AUTOCORRELÉ'}"
            )
    else:
        logger.warning("  STL non disponible")

    # ── 4a. Cross-corrélogramme en niveaux (informatif mais spurieux si non stat.) ─
    logger.info("\n=== Cross-corrélogramme BESI vs Inflation YoY (en niveaux) ===")
    xcorr_df = _cross_correlations(besi, yoy, max_lag=12)
    best_lag_lvl  = int(xcorr_df.loc[xcorr_df["abs_corr"].idxmax(), "lag"])
    best_corr_lvl = float(xcorr_df.loc[xcorr_df["abs_corr"].idxmax(), "correlation"])

    logger.info(f"  Lag optimal (niveaux) : {best_lag_lvl:+d}  r={best_corr_lvl:.4f}")
    logger.info("  ATTENTION : corrélation en niveaux potentiellement spurieuse")
    logger.info("  (symetrie CCF = signe de tendance commune non statio)")

    # ── 4b. Cross-corrélogramme sur PREMIÈRES DIFFÉRENCES (méthode correcte) ──
    logger.info("\n=== Cross-corrélogramme delta-BESI vs delta-YoY (1eres differences) ===")
    besi_diff = besi.diff().dropna()
    yoy_diff  = yoy.diff().dropna()
    xcorr_diff_df = _cross_correlations(besi_diff, yoy_diff, max_lag=12)
    best_lag  = int(xcorr_diff_df.loc[xcorr_diff_df["abs_corr"].idxmax(), "lag"])
    best_corr = float(xcorr_diff_df.loc[xcorr_diff_df["abs_corr"].idxmax(), "correlation"])

    logger.info(f"  Lag optimal (differences) : {best_lag:+d} mois  (r={best_corr:.4f})")
    logger.info(f"  => {'delta-BESI PRECEDE delta-YoY' if best_lag > 0 else 'delta-YoY PRECEDE delta-BESI' if best_lag < 0 else 'CONTEMPORAIN'}")

    logger.info("\n  Correlations delta-BESI vs delta-YoY par lag :")
    for k in [-3, -2, -1, 0, 1, 2, 3]:
        row = xcorr_diff_df[xcorr_diff_df["lag"] == k]
        if not row.empty:
            r = float(row["correlation"].iloc[0])
            tag = "(BESI lead)" if k > 0 else ("(YoY lead)" if k < 0 else "(contemp.)")
            logger.info(f"    lag={k:+2d}  r={r:+.4f}  {tag}")

    # Sauvegarder les deux CCF dans le CSV
    xcorr_df["type"]      = "niveaux"
    xcorr_diff_df["type"] = "differences"
    xcorr_combined        = pd.concat([xcorr_df, xcorr_diff_df], ignore_index=True)
    xcorr_out = REPORTS / "besi_diagnostics.csv"
    xcorr_combined.to_csv(xcorr_out, index=False)
    logger.info(f"\n  CSV sauvegarde : {xcorr_out}")

    # ── 5. Figure ─────────────────────────────────────────────────────────────
    logger.info("\n=== Génération de la figure ===")
    out_fig = FIGURES / "besi_diagnostics.png"
    _plot_besi_diagnostics(
        besi          = besi,
        ipc           = ipc,
        yoy           = yoy,
        gold          = gold,
        stl_res       = stl_resid,
        xcorr_df      = xcorr_df,
        xcorr_diff_df = xcorr_diff_df,
        adf_res       = adf_res,
        kpss_res      = kpss_res,
        lb_besi       = lb_besi,
        lb_resid      = lb_resid,
        out_path      = out_fig,
    )

    return {
        "adf":           adf_res,
        "kpss":          kpss_res,
        "lb_besi":       lb_besi,
        "lb_resid":      lb_resid,
        "best_lag":      best_lag,
        "best_corr":     best_corr,
        "xcorr_df":      xcorr_df,
        "xcorr_diff_df": xcorr_diff_df,
        "n_besi":        len(besi),
    }


# ─── Affichage console ────────────────────────────────────────────────────────

def _print_report(res: dict) -> None:
    sep = "=" * 80

    print()
    print(sep)
    print("  DIAGNOSTICS DU SIGNAL BESI behavioral")
    print(sep)

    adf  = res.get("adf", {})
    kpss = res.get("kpss", {})
    lb   = res.get("lb_besi", {})

    print("\n  1. STATIONNARITÉ")
    print(f"     ADF  : stat={adf.get('stat','--'):.4f}  p={adf.get('pvalue','--'):.4f}  "
          f"-> {'STATIONNAIRE ***' if adf.get('stationary') else 'Non stationnaire'}")
    print(f"     KPSS : stat={kpss.get('stat','--'):.4f}  p={kpss.get('pvalue','--'):.4f}  "
          f"-> {'Stationnaire' if kpss.get('stationary') else 'NON STATIONNAIRE ***'}")

    adf_ok  = adf.get("stationary", False)
    kpss_ok = kpss.get("stationary", False)
    if adf_ok and kpss_ok:
        print("     => Stationnarité CONFIRMÉE (ADF + KPSS). Utilisation en niveau valide.")
    elif adf_ok or kpss_ok:
        print("     => Stationnarité PARTIELLE. Résultats mitigés — interpréter avec prudence.")
    else:
        print("     => Stationnarité REJETÉE. Différencier le BESI avant SARIMAX.")

    print("\n  2. AUTOCORRÉLATION DU BESI")
    print(f"     Ljung-Box (12 lags) : p={lb.get('pvalue','--'):.4f}  "
          f"-> {'Bruit blanc — pas d autocorrelation significative' if lb.get('white_noise') else 'AUTOCORRELÉ — structure temporelle présente'}")

    print("\n  3. LEAD DU BESI (EARLY WARNING)")
    best_lag  = res.get("best_lag", 0)
    best_corr = res.get("best_corr", 0.0)

    # Préférer le CCF sur différences (correct statistiquement)
    xcorr = res.get("xcorr_diff_df") if res.get("xcorr_diff_df") is not None else res.get("xcorr_df")
    label = "delta-BESI vs delta-YoY (1eres differences)" if res.get("xcorr_diff_df") is not None else "niveaux"
    print(f"     [CCF sur {label}]")
    if xcorr is not None:
        sub = xcorr[xcorr["type"] == "differences"] if "type" in xcorr.columns else xcorr
        for k in [-1, 0, 1, 2]:
            row = sub[sub["lag"] == k]
            if not row.empty:
                r = float(row["correlation"].iloc[0])
                tag = (
                    "corr(dBESI[t-1], dYoY[t]) -> YoY leads BESI" if k == -1 else
                    "corr(dBESI[t],   dYoY[t]) -> Contemporain"    if k ==  0 else
                    "corr(dBESI[t+1], dYoY[t]) -> BESI lead 1 mois" if k == 1 else
                    "corr(dBESI[t+2], dYoY[t]) -> BESI lead 2 mois"
                )
                print(f"     {tag} : r = {r:+.4f}")

    if best_lag > 0:
        print(f"\n     => BESI PRECEDE l'inflation YoY de {best_lag} mois")
        print(f"        (corrélation maximale : r={best_corr:.4f} à lag={best_lag:+d})")
        print("     => H1 VALIDÉE : BESI est un indicateur avancé de l'IPC")
    elif best_lag == 0:
        print("\n     => Corrélation max contemporaine — BESI suit l'IPC en temps réel")
        print("        (Utile pour nowcasting, moins pour early warning)")
    else:
        print(f"\n     => IPC précède BESI de {-best_lag} mois — signal retardé")
        print("        (BESI reflète l'IPC passé — utilité pour early warning limitée)")

    print()
    print(sep)

    print("\nFichiers générés :")
    for p in [FIGURES/"besi_diagnostics.png", REPORTS/"besi_diagnostics.csv"]:
        sz = int(p.stat().st_size / 1024) if p.exists() else 0
        print(f"  {str(p.relative_to(ROOT)):<55} {sz} KB")
    print()


# ─── Point d'entrée ───────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt = "%H:%M:%S",
    )
    results = run_besi_diagnostics()
    _print_report(results)


if __name__ == "__main__":
    main()
