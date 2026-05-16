"""
Modélisation SARIMA/SARIMAX — Projet BESI Maroc
Session 5 : stationnarité, préparation des séries, identification des ordres ARIMA
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

np.random.seed(42)

# ─── Chemins ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
FIG_DIR     = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Palette commune (cohérence visuelle avec les autres modules)
_COL_ORIG   = "#2C5F8A"   # bleu HCP
_COL_DIFF   = "#E07B39"   # orange
_COL_TREND  = "#2CA02C"   # vert
_COL_SEAS   = "#9467BD"   # violet
_COL_RESID  = "#8C8C8C"   # gris


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS INTERNES
# ═══════════════════════════════════════════════════════════════════════════════

def _run_adf(series: pd.Series) -> dict:
    """Augmented Dickey-Fuller — H₀ : série non-stationnaire (racine unitaire)."""
    result = adfuller(series.dropna(), autolag="AIC")
    return {
        "stat":      result[0],
        "p_value":   result[1],
        "lags_used": result[2],
        "n_obs":     result[3],
        "critical":  result[4],   # dict {"1%": ..., "5%": ..., "10%": ...}
    }


def _run_kpss(series: pd.Series, regression: str = "c") -> dict:
    """KPSS — H₀ : série stationnaire autour d'une constante (ou tendance)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # p-value interpolée aux bornes : normal
        result = kpss(series.dropna(), regression=regression, nlags="auto")
    return {
        "stat":     result[0],
        "p_value":  result[1],
        "lags":     result[2],
        "critical": result[3],
    }


def _is_stationary_combined(adf: dict, kpss_res: dict) -> tuple[bool, str]:
    """
    Règle de décision combinée ADF + KPSS.

    ADF  H₀ = non-stationnaire  → p < 0.05 signifie STATIONNAIRE
    KPSS H₀ = stationnaire      → p < 0.05 signifie NON-STATIONNAIRE

    Retourne (is_stationary, interprétation textuelle)
    """
    adf_stat  = adf["p_value"]  < 0.05   # True  → rejette H₀ → stationnaire
    kpss_stat = kpss_res["p_value"] > 0.05  # True  → ne rejette pas H₀ → stationnaire

    if adf_stat and kpss_stat:
        return True,  "Stationnaire (ADF et KPSS concordent)"
    if not adf_stat and not kpss_stat:
        return False, "Non-stationnaire — racine unitaire probable (ADF et KPSS concordent)"
    if adf_stat and not kpss_stat:
        return False, "Trend-stationnaire — tendance deterministe presente (ADF : stat, KPSS : non-stat)"
    # adf non-stat + kpss stat
    return False,     "Inconclusive — possible processus a memoire longue (ADF : non-stat, KPSS : stat)"


def _count_diffs_needed(series: pd.Series, max_d: int = 2) -> int:
    """
    Nombre de différenciations pour rendre la série stationnaire.
    Teste successivement d = 0, 1, 2 avec la règle ADF+KPSS combinée.
    """
    s = series.dropna().copy()
    for d in range(max_d + 1):
        adf_r  = _run_adf(s)
        kpss_r = _run_kpss(s)
        is_stat, _ = _is_stationary_combined(adf_r, kpss_r)
        if is_stat:
            return d
        if d < max_d:
            s = s.diff().dropna()
    return max_d


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ANALYSE DE STATIONNARITÉ
# ═══════════════════════════════════════════════════════════════════════════════

def stationarity_analysis(
    series: pd.Series,
    name: str = "IPC",
    period: int = 12,
    save_fig: bool = True,
) -> tuple[bool, int]:
    """
    Analyse complète de stationnarité : ADF + KPSS + décomposition STL.

    Paramètres
    ----------
    series   : série temporelle mensuelle (index DatetimeIndex, freq='MS')
    name     : nom affiché dans les titres et sauvegardes
    period   : période saisonnière (12 pour données mensuelles)
    save_fig : sauvegarder la figure dans outputs/figures/

    Affiche
    -------
    - Résultats ADF et KPSS avec valeurs critiques et interprétation
    - Décomposition STL (tendance / saisonnalité / résidu)

    Retourne
    --------
    is_stationary  : True si la série est stationnaire selon ADF + KPSS
    n_diffs_needed : nombre de différenciations recommandées (0, 1 ou 2)
    """
    s = series.dropna().copy()

    # ── Tests statistiques ────────────────────────────────────────────────────
    adf_r  = _run_adf(s)
    kpss_r = _run_kpss(s)
    is_stat, interpretation = _is_stationary_combined(adf_r, kpss_r)
    n_diffs = 0 if is_stat else _count_diffs_needed(s)

    # ── Affichage console ─────────────────────────────────────────────────────
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  ANALYSE DE STATIONNARITE -- {name}")
    print(sep)

    print(f"\n{'-'*40}")
    print("  TEST ADF (Augmented Dickey-Fuller)")
    print("  H0 : la serie possede une racine unitaire (non-stationnaire)")
    print(f"{'-'*40}")
    print(f"  Statistique ADF  : {adf_r['stat']:>10.4f}")
    print(f"  p-value          : {adf_r['p_value']:>10.4f}  {'*** Rejette H0 (stationnaire)' if adf_r['p_value'] < 0.05 else 'Ne rejette pas H0 (non-stationnaire)'}")
    print(f"  Lags utilises    : {adf_r['lags_used']}")
    print(f"  Valeurs critiques:")
    for level, val in adf_r["critical"].items():
        marker = " <--" if adf_r["stat"] < val else ""
        print(f"    {level:>4s} : {val:>8.4f}{marker}")

    print(f"\n{'-'*40}")
    print("  TEST KPSS")
    print("  H0 : la serie est stationnaire autour d'une constante")
    print(f"{'-'*40}")
    print(f"  Statistique KPSS : {kpss_r['stat']:>10.4f}")
    print(f"  p-value          : {kpss_r['p_value']:>10.4f}  {'*** Rejette H0 (non-stationnaire)' if kpss_r['p_value'] < 0.05 else 'Ne rejette pas H0 (stationnaire)'}")
    print(f"  Valeurs critiques:")
    for level, val in kpss_r["critical"].items():
        marker = " <--" if kpss_r["stat"] > val else ""
        print(f"    {level:>4s} : {val:>8.4f}{marker}")

    print(f"\n{'-'*40}")
    print("  CONCLUSION")
    print(f"{'-'*40}")
    print(f"  {interpretation}")
    if not is_stat:
        print(f"  --> Differentiation(s) recommandee(s) : d = {n_diffs}")
    else:
        print("  --> Aucune transformation necessaire")
    print(sep)

    # ── Décomposition STL ─────────────────────────────────────────────────────
    stl = STL(s, period=period, robust=True)
    stl_fit = stl.fit()

    fig = plt.figure(figsize=(13, 9))
    fig.suptitle(f"Décomposition STL — {name}", fontsize=13, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(4, 1, hspace=0.55)

    ax_orig   = fig.add_subplot(gs[0])
    ax_trend  = fig.add_subplot(gs[1])
    ax_seas   = fig.add_subplot(gs[2])
    ax_resid  = fig.add_subplot(gs[3])

    ax_orig.plot(s.index,              s.values,             color=_COL_ORIG,  lw=1.5)
    ax_trend.plot(s.index,             stl_fit.trend,        color=_COL_TREND, lw=1.5)
    ax_seas.plot(s.index,              stl_fit.seasonal,     color=_COL_SEAS,  lw=1.0)
    ax_resid.bar(s.index,              stl_fit.resid,        color=_COL_RESID, width=20, alpha=0.7)

    ax_orig.set_ylabel("Série originale",   fontsize=9)
    ax_trend.set_ylabel("Tendance",          fontsize=9)
    ax_seas.set_ylabel("Saisonnalité",       fontsize=9)
    ax_resid.set_ylabel("Résidu",            fontsize=9)

    # Annotation rupture 2022
    for ax in [ax_orig, ax_trend, ax_seas, ax_resid]:
        ax.axvline(pd.Timestamp("2022-01-01"), color="red", lw=0.8, ls="--", alpha=0.6)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    ax_resid.set_xlabel("Date", fontsize=9)

    if save_fig:
        path = FIG_DIR / f"stl_{name.lower().replace(' ', '_')}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"\n  Figure sauvegardee : {path}")

    plt.show()

    return is_stat, n_diffs


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PRÉPARATION DE LA SÉRIE + ACF / PACF
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_series(
    series: pd.Series,
    name: str = "IPC",
    max_lags: int = 36,
    save_fig: bool = True,
) -> pd.DataFrame:
    """
    Applique les différenciations nécessaires et produit les diagnostics visuels
    pour identifier les ordres p, q, P, Q du modèle SARIMA.

    Paramètres
    ----------
    series   : série mensuelle brute (index DatetimeIndex)
    name     : nom pour les titres et sauvegardes
    max_lags : nombre de lags affichés dans ACF/PACF (36 = 3 ans)
    save_fig : sauvegarder dans outputs/figures/

    Étapes
    ------
    1. Appelle stationarity_analysis pour déterminer d
    2. Applique d différenciations successives
    3. Plot : série originale | série transformée
    4. Plot : ACF | PACF de la série transformée

    Retourne
    --------
    DataFrame avec colonnes 'original' et 'transformed'

    Lecture des graphiques ACF/PACF
    --------------------------------
    ACF coupe brusquement après lag q  → ordre MA = q
    PACF coupe brusquement après lag p → ordre AR = p
    Pics aux lags 12, 24 ...           → composante saisonnière S(P,Q)
    """
    s = series.dropna().copy()

    # ── Analyse de stationnarité ──────────────────────────────────────────────
    is_stat, n_diffs = stationarity_analysis(s, name=name, save_fig=save_fig)

    # ── Application des différenciations ─────────────────────────────────────
    s_transf = s.copy()
    diff_label = "Série originale"

    if n_diffs == 0:
        print(f"\n[{name}] Série déjà stationnaire — aucune transformation appliquée.")
    else:
        for i in range(1, n_diffs + 1):
            s_transf = s_transf.diff().dropna()
        label_map = {1: "première", 2: "deuxième"}
        diff_label = f"Différence d'ordre {n_diffs} ({label_map.get(n_diffs, str(n_diffs))})"
        print(f"\n[{name}] {n_diffs} différenciation(s) appliquée(s).")

        # Vérification post-différenciation
        adf_check  = _run_adf(s_transf)
        kpss_check = _run_kpss(s_transf)
        stat_check, interp_check = _is_stationary_combined(adf_check, kpss_check)
        print(f"  Vérification post-diff : {interp_check}")
        print(f"  ADF p={adf_check['p_value']:.4f}  KPSS p={kpss_check['p_value']:.4f}")

    # ── Figure 1 : Série originale vs transformée ─────────────────────────────
    fig1, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=False)
    fig1.suptitle(f"Transformation pour stationnarité — {name}", fontsize=12, fontweight="bold")

    axes[0].plot(s.index,       s.values,       color=_COL_ORIG,  lw=1.4)
    axes[0].set_title("Série originale",          fontsize=10)
    axes[0].axvline(pd.Timestamp("2022-01-01"),  color="red", lw=0.8, ls="--", alpha=0.6, label="Rupture 2022")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylabel(name, fontsize=9)

    axes[1].plot(s_transf.index, s_transf.values, color=_COL_DIFF, lw=1.4)
    axes[1].set_title(diff_label,                  fontsize=10)
    axes[1].axhline(0, color="black", lw=0.6, ls="-")
    axes[1].axvline(pd.Timestamp("2022-01-01"),   color="red", lw=0.8, ls="--", alpha=0.6)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylabel(f"diff({name})" if n_diffs > 0 else name, fontsize=9)
    axes[1].set_xlabel("Date", fontsize=9)

    plt.tight_layout()
    if save_fig:
        path = FIG_DIR / f"diff_{name.lower().replace(' ', '_')}.png"
        fig1.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Figure sauvegardée : {path}")
    plt.show()

    # ── Figure 2 : ACF et PACF ────────────────────────────────────────────────
    fig2, (ax_acf, ax_pacf) = plt.subplots(2, 1, figsize=(13, 7))
    fig2.suptitle(
        f"ACF / PACF — {name} ({diff_label})\n"
        "Lecture : ACF → ordre q (MA) | PACF → ordre p (AR) | Pics lag 12,24 → saisonnalité",
        fontsize=10, fontweight="bold",
    )

    plot_acf(
        s_transf,
        lags=max_lags,
        ax=ax_acf,
        alpha=0.05,
        color=_COL_ORIG,
        vlines_kwargs={"colors": _COL_ORIG},
    )
    ax_acf.set_title("ACF — Autocorrélation (→ ordre q du MA)", fontsize=10)
    ax_acf.set_xlabel("Lag (mois)", fontsize=9)
    ax_acf.grid(True, alpha=0.3)

    # Repères saisonniers (lags 12 et 24)
    for lag in [12, 24]:
        ax_acf.axvline(lag, color=_COL_SEAS, lw=0.8, ls=":", alpha=0.8)
        ax_acf.text(lag + 0.3, ax_acf.get_ylim()[1] * 0.92, f"lag {lag}",
                    color=_COL_SEAS, fontsize=7)

    plot_pacf(
        s_transf,
        lags=max_lags,
        ax=ax_pacf,
        alpha=0.05,
        method="ywm",
        color=_COL_DIFF,
        vlines_kwargs={"colors": _COL_DIFF},
    )
    ax_pacf.set_title("PACF — Autocorrélation partielle (→ ordre p du AR)", fontsize=10)
    ax_pacf.set_xlabel("Lag (mois)", fontsize=9)
    ax_pacf.grid(True, alpha=0.3)

    for lag in [12, 24]:
        ax_pacf.axvline(lag, color=_COL_SEAS, lw=0.8, ls=":", alpha=0.8)
        ax_pacf.text(lag + 0.3, ax_pacf.get_ylim()[1] * 0.92, f"lag {lag}",
                     color=_COL_SEAS, fontsize=7)

    plt.tight_layout()
    if save_fig:
        path = FIG_DIR / f"acf_pacf_{name.lower().replace(' ', '_')}.png"
        fig2.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Figure sauvegardée : {path}")
    plt.show()

    # ── Guide de lecture dans la console ─────────────────────────────────────
    print(f"\n{'-'*60}")
    print("  GUIDE LECTURE ACF/PACF POUR SARIMA(p,d,q)(P,D,Q)[12]")
    print(f"{'-'*60}")
    print(f"  d (differenciation ordinaire)   = {n_diffs}")
    print("  p  --> lag ou la PACF coupe (avant lag 12)")
    print("  q  --> lag ou l'ACF coupe   (avant lag 12)")
    print("  P  --> pic significatif PACF au lag 12")
    print("  Q  --> pic significatif ACF  au lag 12")
    print("  D  --> differenciation saisonniere (0 ou 1 selon pics lag 12)")
    print(f"{'-'*60}")

    # ── DataFrame résultat ────────────────────────────────────────────────────
    result = pd.DataFrame({
        "original":    s,
        "transformed": s_transf,
    })
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MÉTRIQUES
# ═══════════════════════════════════════════════════════════════════════════════

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FIT SARIMA BASELINE
# ═══════════════════════════════════════════════════════════════════════════════

def fit_sarima_baseline(
    series: pd.Series,
    train_end: str = "2021-12-01",
    seasonal_period: int = 12,
    save_fig: bool = True,
) -> tuple:
    """
    Identifie et ajuste le meilleur modèle SARIMA sur la période d'entraînement.

    Utilise pmdarima.auto_arima (recherche pas-à-pas par AIC) si disponible,
    sinon bascule sur une grille réduite avec statsmodels.SARIMAX.

    Paramètres
    ----------
    series          : série mensuelle complète (train + test)
    train_end       : dernière date d'entraînement (défaut = avant choc 2022)
    seasonal_period : période saisonnière (12 pour mensuel)
    save_fig        : sauvegarder les diagnostics résidus

    Affiche
    -------
    - Résumé du modèle (ordres, AIC, BIC)
    - Diagnostics résidus : Ljung-Box, Jarque-Bera, graphiques

    Retourne
    --------
    (model, (p,d,q), (P,D,Q))
    """
    train = series.loc[:train_end].dropna()

    print(f"\n{'='*60}")
    print("  FIT SARIMA BASELINE")
    print(f"  Periode entrainement : {train.index[0].date()} -> {train.index[-1].date()}")
    print(f"  N observations       : {len(train)}")
    print(f"{'='*60}")

    # ── Tentative pmdarima ────────────────────────────────────────────────────
    model = None
    try:
        import pmdarima as pm
        print("\n  [auto_arima] Recherche des ordres optimaux (AIC, stepwise) ...")
        model = pm.auto_arima(
            train,
            seasonal=True, m=seasonal_period,
            information_criterion="aic",
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            max_p=3, max_q=3, max_P=2, max_Q=2,
            d=None, D=None,
            trend="c",
            random_state=42,
        )
        order         = model.order           # (p, d, q)
        seasonal_order = model.seasonal_order  # (P, D, Q, m)
        print(f"\n  Ordres selectionnes : SARIMA{order}x{seasonal_order}")
        print(f"  AIC = {model.aic():.2f}  |  BIC = {model.bic():.2f}")

    except ImportError:
        # ── Fallback : grille réduite statsmodels ─────────────────────────────
        print("\n  [pmdarima non installe] Grille manuelle statsmodels ...")
        print("  Pour une recherche automatique : pip install pmdarima\n")

        from statsmodels.tsa.statespace.sarimax import SARIMAX

        candidate_orders = [
            ((1,1,1), (1,1,1,12)),
            ((1,1,1), (0,1,1,12)),
            ((2,1,1), (1,1,1,12)),
            ((1,1,2), (1,1,1,12)),
            ((0,1,1), (0,1,1,12)),
            ((2,1,2), (1,1,1,12)),
        ]
        best_aic, best_order, best_seasonal = np.inf, None, None
        for ord_, sord_ in candidate_orders:
            try:
                res_ = SARIMAX(
                    train, order=ord_, seasonal_order=sord_,
                    enforce_stationarity=False, enforce_invertibility=False,
                ).fit(disp=False)
                if res_.aic < best_aic:
                    best_aic, best_order, best_seasonal = res_.aic, ord_, sord_
                print(f"    SARIMA{ord_}x{sord_}  AIC={res_.aic:.2f}")
            except Exception:
                pass

        order          = best_order
        seasonal_order = best_seasonal
        print(f"\n  Meilleur modele : SARIMA{order}x{seasonal_order}  AIC={best_aic:.2f}")

        model = SARIMAX(
            train, order=order, seasonal_order=seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)

        # Adapter l'interface pour la compatibilité avec walk_forward_validation
        model._order          = order
        model._seasonal_order = seasonal_order
        model._is_statsmodels  = True

    # ── Résumé complet ────────────────────────────────────────────────────────
    try:
        print(model.summary())
    except Exception:
        pass

    # ── Diagnostics des résidus ───────────────────────────────────────────────
    # pmdarima : resid() est une méthode ; statsmodels : resid est une propriété
    raw_resid = model.resid() if callable(getattr(model, "resid", None)) else model.resid
    resid = pd.Series(np.asarray(raw_resid).ravel(), index=train.index[-len(np.asarray(raw_resid).ravel()):])

    resid = resid.dropna()

    # Ljung-Box (H0 : résidus non autocorrélés)
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from scipy.stats import jarque_bera, shapiro

    lb_res = acorr_ljungbox(resid, lags=[6, 12, 24], return_df=True)

    # Jarque-Bera (H0 : normalité)
    jb_stat, jb_p = jarque_bera(resid)

    # Shapiro-Wilk sur un sous-échantillon (max 5000 obs)
    sw_stat, sw_p = shapiro(resid[:5000])

    print(f"\n{'-'*60}")
    print("  DIAGNOSTICS RESIDUS")
    print(f"{'-'*60}")
    print("  Ljung-Box (H0 : pas d'autocorrelation residuelle) :")
    for _, row in lb_res.iterrows():
        verdict = "OK" if row["lb_pvalue"] > 0.05 else "*** PROBLEME"
        print(f"    Lag {int(row.name):>2d} : stat={row['lb_stat']:.3f}  p={row['lb_pvalue']:.4f}  {verdict}")
    print(f"  Jarque-Bera (normalite)  : stat={jb_stat:.3f}  p={jb_p:.4f}  "
          f"{'OK' if jb_p > 0.05 else 'Non normal (attention aux IC)'}")
    print(f"  Shapiro-Wilk (normalite) : stat={sw_stat:.4f}  p={sw_p:.4f}  "
          f"{'OK' if sw_p > 0.05 else 'Non normal'}")
    print(f"{'-'*60}")

    # ── Figure diagnostics résidus ────────────────────────────────────────────
    from scipy.stats import probplot

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(
        f"Diagnostics residus — SARIMA{order}x{seasonal_order[:3]}[{seasonal_period}]",
        fontsize=11, fontweight="bold",
    )

    # Résidus dans le temps
    axes[0, 0].plot(resid.index, resid.values, color=_COL_ORIG, lw=0.9)
    axes[0, 0].axhline(0, color="black", lw=0.6)
    axes[0, 0].set_title("Residus dans le temps", fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    # Histogramme + KDE
    resid.plot.hist(ax=axes[0, 1], bins=25, color=_COL_ORIG, alpha=0.7, density=True)
    resid.plot.kde(ax=axes[0, 1], color=_COL_DIFF, lw=1.5)
    axes[0, 1].set_title("Distribution des residus", fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    # Q-Q plot
    probplot(resid, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q plot (normalite)", fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    # ACF des résidus
    plot_acf(resid, lags=24, ax=axes[1, 1], alpha=0.05,
             color=_COL_ORIG, vlines_kwargs={"colors": _COL_ORIG})
    axes[1, 1].set_title("ACF des residus (doit etre dans les IC)", fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_fig:
        path = FIG_DIR / f"residus_sarima{''.join(map(str, order))}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Figure sauvegardee : {path}")
    plt.show()

    pdq = order
    PDQ = seasonal_order[:3] if len(seasonal_order) == 4 else seasonal_order
    return model, pdq, PDQ


# ═══════════════════════════════════════════════════════════════════════════════
# 4. WALK-FORWARD VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def _predict_steps(model, steps: int) -> np.ndarray:
    """Interface unifiée pmdarima / statsmodels pour la prévision."""
    try:
        # pmdarima
        return np.array(model.predict(n_periods=steps))
    except AttributeError:
        pass
    try:
        # statsmodels FittedResult
        return np.array(model.forecast(steps=steps))
    except Exception:
        return np.array(model.get_forecast(steps=steps).predicted_mean)


def walk_forward_validation(
    series: pd.Series,
    model_func,
    n_test: int = 36,
    horizons: list[int] | None = None,
    save_fig: bool = True,
    name: str = "IPC",
) -> dict:
    """
    Validation par fenêtre glissante (walk-forward / expanding window).

    Principe
    --------
    Pour chaque mois t dans la période de test :
      1. Entraîner model_func sur toutes les données jusqu'à t-1
      2. Prévoir t+h pour h in horizons
      3. Calculer l'erreur par rapport à la valeur réelle

    Paramètres
    ----------
    series      : série temporelle mensuelle complète
    model_func  : callable(train: pd.Series) -> modèle avec predict/forecast
                  Exemple : lambda s: fit_sarima_baseline(s)[0]
    n_test      : nombre de mois de test (36 = 3 ans, dernier quart 2022-2024)
    horizons    : horizons de prévision (défaut : [1, 2, 3])
    save_fig    : sauvegarder la figure
    name        : nom affiché dans les graphiques

    Retourne
    --------
    dict avec pour chaque horizon h :
      {h: {"rmse": float, "mae": float, "mape": float,
           "y_true": array, "y_pred": array, "dates": DatetimeIndex}}
    """
    if horizons is None:
        horizons = [1, 2, 3]

    s = series.dropna().copy()
    n_total = len(s)
    n_train_init = n_total - n_test

    if n_train_init < 24:
        raise ValueError(f"Trop peu de données d'entraînement ({n_train_init} mois). Réduire n_test.")

    test_dates = s.index[n_train_init:]

    print(f"\n{'='*60}")
    print("  WALK-FORWARD VALIDATION")
    print(f"  Entrainement initial : {s.index[0].date()} -> {s.index[n_train_init-1].date()}")
    print(f"  Test (rolling)       : {test_dates[0].date()} -> {test_dates[-1].date()}")
    print(f"  Horizons             : {horizons} mois")
    print(f"  Etapes               : {n_test}")
    print(f"{'='*60}\n")

    # Stockage des prédictions par horizon
    preds: dict[int, list[float]] = {h: [] for h in horizons}
    actuals: dict[int, list[float]] = {h: [] for h in horizons}
    pred_dates: dict[int, list] = {h: [] for h in horizons}

    max_h = max(horizons)

    for step in range(n_test):
        train_end_idx = n_train_init + step
        train_s = s.iloc[:train_end_idx]

        if (step % 6 == 0) or step == n_test - 1:
            print(f"  Etape {step+1:>3}/{n_test}  train jusqu'a {train_s.index[-1].date()} ...")

        try:
            fitted = model_func(train_s)
            fc = _predict_steps(fitted, steps=max_h)
        except Exception as exc:
            print(f"    [WARN] Etape {step+1} : erreur modele ({exc}) — NaN insere.")
            fc = np.full(max_h, np.nan)

        for h in horizons:
            target_idx = train_end_idx + h - 1
            if target_idx < n_total:
                preds[h].append(fc[h - 1])
                actuals[h].append(s.iloc[target_idx])
                pred_dates[h].append(s.index[target_idx])

    # ── Calcul des métriques ──────────────────────────────────────────────────
    results: dict = {}
    print(f"\n{'-'*60}")
    print("  METRIQUES PAR HORIZON")
    print(f"{'Horizon':>8} | {'RMSE':>8} | {'MAE':>8} | {'MAPE (%)':>10}")
    print(f"{'-'*60}")

    for h in horizons:
        yt = np.array(actuals[h])
        yp = np.array(preds[h])
        mask = ~np.isnan(yp)
        yt, yp = yt[mask], yp[mask]

        rmse = _rmse(yt, yp)
        mae  = _mae(yt, yp)
        mape = _mape(yt, yp)

        results[h] = {
            "rmse":   rmse,
            "mae":    mae,
            "mape":   mape,
            "y_true": yt,
            "y_pred": yp,
            "dates":  pd.DatetimeIndex(pred_dates[h])[mask],
        }
        print(f"  h = {h:>2d}   | {rmse:>8.4f} | {mae:>8.4f} | {mape:>9.2f}%")

    print(f"{'-'*60}")

    # ── Figure : prédictions vs réel ─────────────────────────────────────────
    horizon_colors = {1: _COL_DIFF, 2: _COL_TREND, 3: _COL_SEAS}
    horizon_labels = {1: "h=1 mois", 2: "h=2 mois", 3: "h=3 mois"}

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(
        f"Walk-Forward Validation — {name}  ({test_dates[0].date()} -> {test_dates[-1].date()})",
        fontsize=11, fontweight="bold",
    )

    # Série complète en arrière-plan
    ax.plot(s.index, s.values, color="lightgray", lw=1.5, label="Serie complete", zorder=1)

    # Valeurs réelles de test (référence principale)
    ax.plot(
        s.index[n_train_init:], s.values[n_train_init:],
        color=_COL_ORIG, lw=2.0, label="Valeurs reelles (test)", zorder=2,
    )

    # Prédictions par horizon
    for h in horizons:
        r = results[h]
        ax.plot(
            r["dates"], r["y_pred"],
            color=horizon_colors.get(h, "black"),
            lw=1.2, ls="--",
            label=f"Prevision {horizon_labels.get(h, f'h={h}')}  "
                  f"RMSE={r['rmse']:.4f}  MAPE={r['mape']:.1f}%",
            zorder=3,
        )

    # Délimiteur train / test
    ax.axvline(test_dates[0], color="red", lw=1.0, ls="--", alpha=0.7, label="Debut test")
    ax.axvline(pd.Timestamp("2022-01-01"), color="purple", lw=0.8, ls=":", alpha=0.6,
               label="Rupture 2022")

    ax.set_xlabel("Date", fontsize=9)
    ax.set_ylabel(name, fontsize=9)
    ax.legend(fontsize=7.5, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_fig:
        path = FIG_DIR / f"walk_forward_{name.lower().replace(' ', '_')}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"\n  Figure sauvegardee : {path}")
    plt.show()

    # ── Figure : erreurs absolues par horizon ─────────────────────────────────
    fig2, axes2 = plt.subplots(len(horizons), 1, figsize=(13, 3 * len(horizons)), sharex=False)
    if len(horizons) == 1:
        axes2 = [axes2]
    fig2.suptitle(f"Erreurs absolues par horizon — {name}", fontsize=10, fontweight="bold")

    for ax2, h in zip(axes2, horizons):
        r = results[h]
        errors = np.abs(r["y_true"] - r["y_pred"])
        ax2.bar(r["dates"], errors, width=20,
                color=horizon_colors.get(h, _COL_ORIG), alpha=0.7)
        ax2.axhline(r["mae"], color="red", lw=1.0, ls="--",
                    label=f"MAE = {r['mae']:.4f}")
        ax2.set_title(f"h = {h} mois", fontsize=9)
        ax2.set_ylabel("|erreur|", fontsize=8)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

    axes2[-1].set_xlabel("Date", fontsize=9)
    plt.tight_layout()
    if save_fig:
        path = FIG_DIR / f"walk_forward_errors_{name.lower().replace(' ', '_')}.png"
        fig2.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Figure sauvegardee : {path}")
    plt.show()

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 5. FIT SARIMAX
# ═══════════════════════════════════════════════════════════════════════════════

def _residual_diagnostics(resid: pd.Series, title: str, save_fig: bool) -> None:
    """Diagnostics résidus réutilisables (Ljung-Box, JB, Shapiro + 4 graphiques)."""
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from scipy.stats import jarque_bera, shapiro, probplot

    lb_res = acorr_ljungbox(resid.dropna(), lags=[6, 12, 24], return_df=True)
    jb_stat, jb_p = jarque_bera(resid.dropna())
    sw_stat, sw_p  = shapiro(resid.dropna()[:5000])

    print(f"\n{'-'*60}")
    print(f"  DIAGNOSTICS RESIDUS -- {title}")
    print(f"{'-'*60}")
    print("  Ljung-Box (H0 : pas d'autocorrelation residuelle) :")
    for _, row in lb_res.iterrows():
        v = "OK" if row["lb_pvalue"] > 0.05 else "*** PROBLEME"
        print(f"    Lag {int(row.name):>2d} : stat={row['lb_stat']:.3f}  p={row['lb_pvalue']:.4f}  {v}")
    print(f"  Jarque-Bera  : stat={jb_stat:.3f}  p={jb_p:.4f}  "
          f"{'OK' if jb_p > 0.05 else 'Non normal (attention aux IC)'}")
    print(f"  Shapiro-Wilk : stat={sw_stat:.4f}  p={sw_p:.4f}  "
          f"{'OK' if sw_p > 0.05 else 'Non normal'}")
    print(f"{'-'*60}")

    if save_fig:
        from scipy.stats import probplot as _probplot
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        fig.suptitle(f"Diagnostics residus — {title}", fontsize=11, fontweight="bold")

        axes[0, 0].plot(resid.index, resid.values, color=_COL_ORIG, lw=0.9)
        axes[0, 0].axhline(0, color="black", lw=0.6)
        axes[0, 0].set_title("Residus dans le temps", fontsize=9)
        axes[0, 0].grid(True, alpha=0.3)

        resid.plot.hist(ax=axes[0, 1], bins=25, color=_COL_ORIG, alpha=0.7, density=True)
        resid.plot.kde(ax=axes[0, 1], color=_COL_DIFF, lw=1.5)
        axes[0, 1].set_title("Distribution des residus", fontsize=9)
        axes[0, 1].grid(True, alpha=0.3)

        _probplot(resid, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title("Q-Q plot (normalite)", fontsize=9)
        axes[1, 0].grid(True, alpha=0.3)

        plot_acf(resid.dropna(), lags=24, ax=axes[1, 1], alpha=0.05,
                 color=_COL_ORIG, vlines_kwargs={"colors": _COL_ORIG})
        axes[1, 1].set_title("ACF des residus (doit rester dans les IC)", fontsize=9)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        slug = title.lower().replace(" ", "_").replace("(", "").replace(")", "")
        path = FIG_DIR / f"residus_{slug}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Figure sauvegardee : {path}")
        plt.show()


def fit_sarimax(
    series: pd.Series,
    exog: pd.DataFrame,
    orders: tuple,
    train_end: str = "2021-12-01",
    name: str = "SARIMAX",
    save_fig: bool = True,
) -> object:
    """
    Ajuste un modèle SARIMAX avec variables exogènes sur la période d'entraînement.

    Paramètres
    ----------
    series    : série cible mensuelle (IPC)
    exog      : DataFrame des variables exogènes, même index DatetimeIndex
    orders    : tuple ((p,d,q), (P,D,Q)) — sortie de fit_sarima_baseline()
    train_end : dernière date d'entraînement
    name      : label du modèle (ex. "SARIMAX_BESI")
    save_fig  : sauvegarder les diagnostics résidus

    Remarque sur les variables exogènes
    ------------------------------------
    Les variables exogènes doivent etre stationnaires ou préalablement
    différenciées (cf. stationarity_analysis). Les signaux BESI normalisés
    0-1 satisfont généralement cette condition.

    Retourne
    --------
    Objet statsmodels SARIMAXResults (accès à .aic, .bic, .forecast, .summary)
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    pdq, PDQ = orders

    # Aligner sur l'intersection train + exog (gère les NaN et gaps)
    train_raw = series.loc[:train_end].dropna()
    exog_clean = exog.loc[:train_end].dropna()
    common_idx = train_raw.index.intersection(exog_clean.index)
    if len(common_idx) < 24:
        raise ValueError(
            f"Trop peu de mois communs series/exog ({len(common_idx)}). "
            "Vérifier les plages de dates."
        )
    train_s = train_raw.loc[common_idx]
    train_e = exog_clean.loc[common_idx]

    print(f"\n{'='*60}")
    print(f"  FIT {name}")
    print(f"  Exog colonnes     : {list(exog.columns)}")
    print(f"  Periode           : {common_idx[0].date()} -> {common_idx[-1].date()}")
    print(f"  N observations    : {len(common_idx)}")
    print(f"  Ordres            : SARIMAX{pdq}x{PDQ}[12]")
    print(f"{'='*60}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            train_s,
            exog=train_e,
            order=pdq,
            seasonal_order=(*PDQ, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)

    print(model.summary())

    # Diagnostics résidus
    raw_r = np.asarray(model.resid).ravel()
    resid = pd.Series(raw_r, index=train_s.index[-len(raw_r):])
    _residual_diagnostics(resid, title=name, save_fig=save_fig)

    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 6. COMPARE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

def _wf_one_model(
    series: pd.Series,
    exog: pd.DataFrame | None,
    pdq: tuple,
    PDQ: tuple,
    n_test: int,
    horizons: list[int],
    label: str = "",
) -> dict:
    """
    Walk-forward interne pour un modèle SARIMA(X).

    Hypothèse exog : les valeurs futures des variables exogènes sont connues
    au moment de la prévision (utilisation des valeurs réalisées).
    Justification : dans un cadre de backtesting post-hoc, les signaux Google Trends
    et Reddit sont rétrospectivement disponibles — cette hypothèse est standard en
    évaluation de modèles de prévision macroéconomique.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    s = series.dropna().copy()
    n_total     = len(s)
    n_train_init = n_total - n_test
    max_h       = max(horizons)

    preds      = {h: [] for h in horizons}
    actuals    = {h: [] for h in horizons}
    pred_dates = {h: [] for h in horizons}

    for step in range(n_test):
        t        = n_train_init + step
        train_s  = s.iloc[:t]
        train_e  = exog.iloc[:t]          if exog is not None else None
        future_e = exog.iloc[t : t+max_h] if exog is not None else None

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = SARIMAX(
                    train_s, exog=train_e,
                    order=pdq, seasonal_order=(*PDQ, 12),
                    enforce_stationarity=False, enforce_invertibility=False,
                ).fit(disp=False)

            fc = np.array(
                m.forecast(steps=max_h, exog=future_e.values)
                if exog is not None
                else m.forecast(steps=max_h)
            )
        except Exception as exc:
            print(f"    [WARN] {label} etape {step+1}: {exc}")
            fc = np.full(max_h, np.nan)

        for h in horizons:
            tidx = t + h - 1
            if tidx < n_total:
                preds[h].append(fc[h - 1])
                actuals[h].append(s.iloc[tidx])
                pred_dates[h].append(s.index[tidx])

    results = {}
    for h in horizons:
        yt = np.array(actuals[h])
        yp = np.array(preds[h])
        mask = ~np.isnan(yp)
        yt, yp = yt[mask], yp[mask]
        results[h] = {
            "rmse":   _rmse(yt, yp),
            "mae":    _mae(yt, yp),
            "mape":   _mape(yt, yp),
            "y_true": yt,
            "y_pred": yp,
            "dates":  pd.DatetimeIndex(pred_dates[h])[mask],
        }
    return results


def compare_models(
    series: pd.Series,
    exog_variants: dict,
    orders: tuple,
    train_end: str = "2021-12-01",
    n_test: int = 36,
    horizons: list[int] | None = None,
    save_fig: bool = True,
) -> pd.DataFrame:
    """
    Compare SARIMA baseline vs plusieurs variantes SARIMAX sur walk-forward.

    Paramètres
    ----------
    series        : série cible mensuelle (IPC)
    exog_variants : dict de variantes exogènes, par exemple :
                      {
                        "SARIMA":         None,
                        "SARIMAX_Google": df[["trends_composite"]],
                        "SARIMAX_BESI":   df[["besi"]],
                        "SARIMAX_All":    df[["trends_composite",
                                              "reddit_composite",
                                              "youtube_composite"]],
                      }
                    Le modèle avec exog=None sert de baseline.
    orders        : ((p,d,q), (P,D,Q)) — sortie de fit_sarima_baseline()
    train_end     : limite de la période d'entraînement pour AIC/BIC
    n_test        : nombre de mois de test en walk-forward
    horizons      : horizons de prévision (défaut : [1, 2, 3])
    save_fig      : sauvegarder les graphiques

    Affiche
    -------
    - Progression walk-forward par modèle
    - Tableau comparatif : AIC, BIC, RMSE, MAE, MAPE par horizon
    - % d'amélioration vs SARIMA baseline
    - Figure : toutes les prédictions superposées
    - Heatmap des métriques normalisées

    Retourne
    --------
    DataFrame de comparaison (sauvegardé dans outputs/reports/model_comparison.csv)
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    import seaborn as sns

    if horizons is None:
        horizons = [1, 2, 3]

    pdq, PDQ = orders

    # Identifier le baseline (premier modèle avec exog=None)
    baseline_key = next(
        (k for k, v in exog_variants.items() if v is None),
        list(exog_variants.keys())[0],
    )

    all_wf:   dict[str, dict] = {}
    aic_bic:  dict[str, dict] = {}

    # Palettes de couleurs : baseline en bleu, variants en dégradé
    _palette = [_COL_ORIG, _COL_DIFF, _COL_TREND, _COL_SEAS,
                "#E377C2", "#17BECF", "#BCBD22"]
    model_colors = {k: _palette[i % len(_palette)]
                    for i, k in enumerate(exog_variants.keys())}

    for model_name, exog in exog_variants.items():
        print(f"\n{'='*60}")
        print(f"  MODELE : {model_name}")
        print(f"{'='*60}")

        # ── AIC / BIC (fit complet sur train) ────────────────────────────────
        train_raw = series.loc[:train_end].dropna()
        if exog is not None:
            exog_cl = exog.loc[:train_end].dropna()
            cidx    = train_raw.index.intersection(exog_cl.index)
            ts_fit  = train_raw.loc[cidx]
            te_fit  = exog_cl.loc[cidx]
        else:
            ts_fit  = train_raw
            te_fit  = None

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m_fit = SARIMAX(
                    ts_fit, exog=te_fit,
                    order=pdq, seasonal_order=(*PDQ, 12),
                    enforce_stationarity=False, enforce_invertibility=False,
                ).fit(disp=False)
            aic_bic[model_name] = {"aic": m_fit.aic, "bic": m_fit.bic}
            print(f"  AIC = {m_fit.aic:.2f}  BIC = {m_fit.bic:.2f}")
        except Exception as exc:
            print(f"  [WARN] AIC/BIC indisponible : {exc}")
            aic_bic[model_name] = {"aic": np.nan, "bic": np.nan}

        # ── Walk-forward ─────────────────────────────────────────────────────
        if exog is not None:
            # Aligner series et exog sur l'intersection complète
            exog_full = exog.reindex(series.dropna().index).ffill()
            s_al = series.dropna()
            e_al = exog_full.dropna()
            common = s_al.index.intersection(e_al.index)
            s_al, e_al = s_al.loc[common], e_al.loc[common]
        else:
            s_al, e_al = series.dropna(), None

        print(f"  Walk-forward : {n_test} etapes, horizons={horizons} ...")
        wf = _wf_one_model(s_al, e_al, pdq, PDQ, n_test, horizons, label=model_name)
        all_wf[model_name] = wf

        for h in horizons:
            r = wf[h]
            print(f"    h={h}: RMSE={r['rmse']:.4f}  MAE={r['mae']:.4f}  MAPE={r['mape']:.2f}%")

    # ── Tableau comparatif ────────────────────────────────────────────────────
    base_rmse_h = {h: all_wf[baseline_key][h]["rmse"] for h in horizons}
    rows = []
    for mname, wf in all_wf.items():
        row = {"Modele": mname,
               "AIC":    round(aic_bic[mname]["aic"], 2),
               "BIC":    round(aic_bic[mname]["bic"], 2)}
        for h in horizons:
            r = wf[h]
            gain = (base_rmse_h[h] - r["rmse"]) / base_rmse_h[h] * 100 \
                   if base_rmse_h[h] != 0 else 0.0
            row[f"RMSE_h{h}"] = round(r["rmse"], 5)
            row[f"MAE_h{h}"]  = round(r["mae"],  5)
            row[f"MAPE_h{h}"] = round(r["mape"], 2)
            row[f"Gain%_h{h}"] = round(gain, 1)
        rows.append(row)

    df_cmp = pd.DataFrame(rows).set_index("Modele")

    print(f"\n{'='*70}")
    print("  TABLEAU COMPARATIF COMPLET")
    print(f"{'='*70}")
    print(df_cmp.to_string())

    # Meilleur modèle sur RMSE h=1
    best_col = f"RMSE_h{horizons[0]}"
    best_model = df_cmp[best_col].idxmin()
    best_rmse  = df_cmp.loc[best_model, best_col]
    base_rmse  = df_cmp.loc[baseline_key, best_col]
    pct        = (base_rmse - best_rmse) / base_rmse * 100 if base_rmse > 0 else 0.0

    print(f"\n  Meilleur (RMSE h={horizons[0]}) : {best_model}  RMSE={best_rmse:.5f}")
    sign = "amelioration" if pct >= 0 else "degradation"
    print(f"  {sign.capitalize()} vs {baseline_key} : {pct:+.1f}%")

    # Sauvegarde CSV
    rep_dir = ROOT / "outputs" / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    csv_path = rep_dir / "model_comparison.csv"
    df_cmp.to_csv(csv_path)
    print(f"  Tableau sauvegarde : {csv_path}")

    # ── Figures ───────────────────────────────────────────────────────────────
    if save_fig:
        h1 = horizons[0]

        # Figure 1 : prédictions superposées (horizon h=1)
        s_ref     = series.dropna()
        n_total   = len(s_ref)
        n_tr_init = n_total - n_test

        fig, ax = plt.subplots(figsize=(14, 6))
        fig.suptitle(
            f"Comparaison SARIMA vs SARIMAX — h={h1} mois | "
            f"Meilleur : {best_model} ({pct:+.1f}% vs {baseline_key})",
            fontsize=10, fontweight="bold",
        )

        # Série complète + test
        ax.plot(s_ref.index, s_ref.values,
                color="lightgray", lw=1.5, label="Serie complete", zorder=1)
        ax.plot(s_ref.index[n_tr_init:], s_ref.values[n_tr_init:],
                color="black", lw=2.0, label="Valeurs reelles (test)", zorder=2)

        for mname, wf in all_wf.items():
            r = wf[h1]
            ls = "-." if mname == baseline_key else "--"
            ax.plot(r["dates"], r["y_pred"],
                    color=model_colors[mname], lw=1.3, ls=ls, alpha=0.9,
                    label=f"{mname}  RMSE={r['rmse']:.4f}  MAPE={r['mape']:.1f}%",
                    zorder=3)

        ax.axvline(s_ref.index[n_tr_init], color="red", lw=1.0, ls="--",
                   alpha=0.7, label="Debut test")
        ax.axvline(pd.Timestamp("2022-01-01"), color="purple", lw=0.8,
                   ls=":", alpha=0.6, label="Rupture 2022")
        ax.set_xlabel("Date", fontsize=9)
        ax.set_ylabel("IPC", fontsize=9)
        ax.legend(fontsize=7.5, loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        p1 = FIG_DIR / "model_comparison.png"
        fig.savefig(p1, dpi=150, bbox_inches="tight")
        print(f"  Figure sauvegardee : {p1}")
        plt.show()

        # Figure 2 : RMSE par horizon par modèle (barres groupées)
        rmse_cols = [f"RMSE_h{h}" for h in horizons]
        df_rmse   = df_cmp[rmse_cols].copy()
        df_rmse.columns = [f"h={h}" for h in horizons]

        fig2, ax2 = plt.subplots(figsize=(max(8, len(exog_variants) * 2), 5))
        fig2.suptitle("RMSE par modele et par horizon", fontsize=10, fontweight="bold")
        x = np.arange(len(df_rmse))
        w = 0.8 / len(horizons)
        for i, hcol in enumerate(df_rmse.columns):
            bars = ax2.bar(x + i * w, df_rmse[hcol].values, width=w,
                           color=_palette[i % len(_palette)], alpha=0.8,
                           label=hcol)
            for bar in bars:
                ax2.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.0001,
                         f"{bar.get_height():.4f}",
                         ha="center", va="bottom", fontsize=7)
        ax2.set_xticks(x + w * (len(horizons) - 1) / 2)
        ax2.set_xticklabels(df_rmse.index, fontsize=9)
        ax2.set_ylabel("RMSE", fontsize=9)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        p2 = FIG_DIR / "model_comparison_rmse.png"
        fig2.savefig(p2, dpi=150, bbox_inches="tight")
        print(f"  Figure sauvegardee : {p2}")
        plt.show()

        # Figure 3 : heatmap normalisée des métriques
        heat_cols = [c for c in df_cmp.columns
                     if any(c.startswith(x) for x in ("AIC", "BIC", "RMSE", "MAE", "MAPE"))]
        df_heat = df_cmp[heat_cols].copy()
        df_norm = df_heat.copy()
        for col in df_norm.columns:
            rng = df_norm[col].max() - df_norm[col].min()
            if rng > 0:
                df_norm[col] = (df_norm[col] - df_norm[col].min()) / rng

        fig3, ax3 = plt.subplots(
            figsize=(max(8, len(heat_cols) * 1.3), max(3, len(df_cmp) * 0.9))
        )
        fig3.suptitle(
            "Heatmap comparaison (valeurs brutes, vert = meilleur pour AIC/BIC/RMSE/MAE/MAPE)",
            fontsize=9, fontweight="bold",
        )
        sns.heatmap(
            df_norm, ax=ax3, cmap="RdYlGn_r",
            annot=df_heat.round(3), fmt="g",
            linewidths=0.5, cbar=True,
        )
        plt.tight_layout()
        p3 = FIG_DIR / "model_comparison_heatmap.png"
        fig3.savefig(p3, dpi=150, bbox_inches="tight")
        print(f"  Figure sauvegardee : {p3}")
        plt.show()

    return df_cmp


# ═══════════════════════════════════════════════════════════════════════════════
# 7. COMPARE MODELS V2 — Sans Reddit/YouTube simulés
# ═══════════════════════════════════════════════════════════════════════════════

def _download_ipc_worldbank(save_raw: bool = True) -> pd.Series | None:
    """
    Télécharge l'IPC Maroc depuis la Banque Mondiale (FP.CPI.TOTL, country='MA').
    Retourne une Series mensuelle (freq='MS') normalisée base 2010=1.0, ou None si échec.

    Sauvegarde :
    - data/raw/ipc_real.csv      (données brutes annuelles WB)
    - data/processed/ipc_clean.csv (série mensuelle interpolée, normalisée)
    """
    raw_dir  = ROOT / "data" / "raw"
    proc_dir = ROOT / "data" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    src_log  = ROOT / "outputs" / "reports"
    src_log.mkdir(parents=True, exist_ok=True)

    try:
        import pandas_datareader.wb as wb
        print("  [WB] Téléchargement FP.CPI.TOTL, MA, 2010-2024 ...")
        df_wb = wb.download(indicator="FP.CPI.TOTL", country="MA",
                            start=2010, end=2024)
        df_wb = df_wb.reset_index()
        # Colonnes attendues : country, year, FP.CPI.TOTL
        df_wb = df_wb.rename(columns={"FP.CPI.TOTL": "ipc_wb", "year": "year"})
        df_wb["year"] = df_wb["year"].astype(int)
        df_wb = df_wb[["year", "ipc_wb"]].dropna().sort_values("year")

        if save_raw:
            df_wb.to_csv(raw_dir / "ipc_real.csv", index=False)
            print(f"  [WB] Données brutes sauvegardées : {raw_dir / 'ipc_real.csv'}")

        # Interpolation mensuelle (12 mois par an)
        dates_annual = pd.date_range(f"{df_wb['year'].min()}-01-01",
                                     f"{df_wb['year'].max()}-01-01", freq="YS")
        s_annual = pd.Series(df_wb["ipc_wb"].values,
                             index=dates_annual[:len(df_wb)])
        s_monthly = s_annual.resample("MS").interpolate("cubic")

        # Normalisation base 2010-01 = 1.0
        base = s_monthly.loc["2010-01-01"] if "2010-01-01" in s_monthly.index \
               else s_monthly.iloc[0]
        s_norm = s_monthly / base

        # Sauvegarder ipc_clean.csv
        df_clean = pd.DataFrame({
            "ipc":        s_norm,
            "ipc_source": "WorldBank_FP.CPI.TOTL",
        })
        df_clean.index.name = "date"
        df_clean.to_csv(proc_dir / "ipc_clean.csv")
        print(f"  [WB] ipc_clean.csv sauvegardé ({len(df_clean)} mois)")

        # Journal des sources
        with open(src_log / "data_sources.txt", "w", encoding="utf-8") as f:
            f.write("=== Sources de données — BESI Maroc ===\n\n")
            f.write("IPC (variable cible) :\n")
            f.write("  Source   : World Bank Open Data — FP.CPI.TOTL\n")
            f.write("  Pays     : MA (Maroc)\n")
            f.write("  Période  : 2010-2024\n")
            f.write("  Méthode  : téléchargé via pandas_datareader.wb, ")
            f.write("interpolation cubique mensuelle, normalisé base 2010-01=1.0\n")
            f.write(f"  Date DL  : {pd.Timestamp.now().strftime('%Y-%m-%d')}\n")
            f.write("  Fichiers : data/raw/ipc_real.csv, data/processed/ipc_clean.csv\n\n")
            f.write("Signaux comportementaux :\n")
            f.write("  Google Trends : réel (pytrends, geo='MA', 2010-2024)\n")
            f.write("  Reddit        : simulé (données indisponibles pour la période historique)\n")
            f.write("  YouTube       : simulé (quota API insuffisant pour historique long)\n")
            f.write("  → Pour v2 : seul Google Trends utilisé comme exogène SARIMAX\n")

        return s_norm

    except Exception as exc:
        print(f"  [WB] Échec téléchargement WB : {exc}")

        # Documenter l'échec
        with open(src_log / "data_sources.txt", "w", encoding="utf-8") as f:
            f.write("=== Sources de données — BESI Maroc ===\n\n")
            f.write("IPC (variable cible) :\n")
            f.write("  Source attendue : World Bank FP.CPI.TOTL (MA)\n")
            f.write(f"  Erreur API      : {exc}\n")
            f.write("  Fallback        : ipc_processed.csv (données HCP traitées en session précédente)\n\n")
            f.write("Signaux comportementaux :\n")
            f.write("  Google Trends : réel (pytrends, geo='MA', 2010-2024)\n")
            f.write("  Reddit        : simulé\n")
            f.write("  YouTube       : simulé\n")
            f.write("  → Pour v2 : seul Google Trends utilisé comme exogène SARIMAX\n")

        return None


def _compute_besi_trends(
    trends: pd.Series,
    ipc_change: pd.Series,
    alpha_t: float = 0.70,
    alpha_c: float = 0.30,
) -> pd.Series:
    """
    Calcule BESI_trends = alpha_t * Trends + alpha_c * |IPC_change| (normalisé 0-1).

    Utilise uniquement Google Trends (signal réel) et la variation IPC (observable).
    Évite la contamination par Reddit/YouTube simulés.
    """
    # Normalisation 0-1 de ipc_change sur la plage disponible
    ic = ipc_change.copy().abs()
    ic_min, ic_max = ic.min(), ic.max()
    if ic_max > ic_min:
        ic_norm = (ic - ic_min) / (ic_max - ic_min)
    else:
        ic_norm = ic * 0.0

    # Normalisation trends (déjà 0-1 mais on réapplique pour robustesse)
    t = trends.copy()
    t_min, t_max = t.min(), t.max()
    if t_max > t_min:
        t_norm = (t - t_min) / (t_max - t_min)
    else:
        t_norm = t * 0.0

    besi_t = alpha_t * t_norm + alpha_c * ic_norm
    besi_t.name = "besi_trends"
    return besi_t


def _naive_walk_forward(
    series: pd.Series,
    n_test: int,
    h: int = 1,
) -> dict:
    """Modèle naïf (Random Walk) : prévision = dernière valeur observée."""
    s = series.dropna().copy()
    n_total = len(s)
    n_tr    = n_total - n_test

    preds, actuals, dates = [], [], []
    for step in range(n_test):
        t = n_tr + step
        for horizon in [h]:
            tidx = t + horizon - 1
            if tidx < n_total:
                preds.append(s.iloc[t - 1])   # y_hat = y[t-1] (RW)
                actuals.append(s.iloc[tidx])
                dates.append(s.index[tidx])

    yt = np.array(actuals)
    yp = np.array(preds)
    return {
        "rmse":   _rmse(yt, yp),
        "mae":    _mae(yt, yp),
        "mape":   _mape(yt, yp),
        "y_true": yt,
        "y_pred": yp,
        "dates":  pd.DatetimeIndex(dates),
    }


def compare_models_v2(
    series: "pd.Series | None" = None,
    master_df: "pd.DataFrame | None" = None,
    train_start: str = "2015-01-01",
    train_end:   str = "2021-12-01",
    test_end:    str = "2024-12-01",
    horizons: "list[int] | None" = None,
    try_worldbank: bool = True,
    save_fig: bool = True,
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """
    Comparaison v2 — Correction du biais Reddit/YouTube simulés.

    Modèles comparés
    ----------------
    - Naif        : Random Walk (y_hat = y[t-1])
    - SARIMA      : SARIMA(p,d,q)(P,D,Q)[12] sans exogènes
    - SARIMAX_T   : SARIMA + trends_composite (Google Trends uniquement)
    - SARIMAX_BT  : SARIMA + besi_trends (0.70*Trends + 0.30*|IPC_change|)

    Période d'entraînement
    ----------------------
    train_start à train_end (défaut : 2015-01 → 2021-12, 84 mois)

    Période de test
    ---------------
    train_end+1 à test_end (défaut : 2022-01 → 2024-12, 36 mois)

    Sous-périodes de test
    ---------------------
    - choc_2022  : 2022-01 → 2022-12 (12 mois choc inflationniste)
    - post_2022  : 2023-01 → fin (mois de normalisation)

    Paramètres
    ----------
    series        : IPC mensuel ; si None, chargé depuis data/processed/
    master_df     : DataFrame maître ; si None, chargé depuis data/processed/
    train_start   : début de la fenêtre d'entraînement (2015-01 recommandé)
    train_end     : fin de la fenêtre d'entraînement (avant choc 2022)
    test_end      : fin de la période de test
    horizons      : horizons de prévision (défaut : [1])
    try_worldbank : tenter le téléchargement WB avant fallback CSV
    save_fig      : sauvegarder les graphiques

    Retourne
    --------
    (df_comparison_v2, df_period_v2)
    Sauvegarde :
      outputs/reports/model_comparison_v2.csv
      outputs/reports/period_performance_v2.csv
      outputs/reports/data_sources.txt
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    import seaborn as sns

    if horizons is None:
        horizons = [1]

    rep_dir = ROOT / "outputs" / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print("  COMPARE MODELS V2 — Google Trends only, sans Reddit/YouTube")
    print(f"  Train : {train_start} -> {train_end}")
    print(f"  Test  : apres {train_end} jusqu'a {test_end}")
    print(f"  Horizons : {horizons}")
    print(f"{'='*65}")

    # ── 1. Charger / récupérer les données ────────────────────────────────────
    proc_dir = ROOT / "data" / "processed"

    if series is None or master_df is None:
        # Essayer WB d'abord
        ipc_wb = None
        if try_worldbank:
            ipc_wb = _download_ipc_worldbank(save_raw=True)

        # Charger le master dataset existant
        master_csv = proc_dir / "master_dataset.csv"
        if not master_csv.exists():
            raise FileNotFoundError(
                f"{master_csv} introuvable — lancer data_pipeline.py d'abord."
            )
        df_m = pd.read_csv(master_csv, index_col=0, parse_dates=True)
        try:
            df_m.index = pd.DatetimeIndex(df_m.index, freq="MS")
        except Exception:
            df_m.index = pd.DatetimeIndex(df_m.index)
            df_m = df_m.asfreq("MS")

        master_df = df_m

        # IPC : préférer WB si disponible
        if ipc_wb is not None:
            ipc_wb.index = pd.DatetimeIndex(ipc_wb.index).to_period("M").to_timestamp("MS")
            series = ipc_wb.reindex(df_m.index)
            print("  [INFO] IPC source : World Bank (FP.CPI.TOTL)")
        else:
            ipc_csv = proc_dir / "ipc_processed.csv"
            df_ipc  = pd.read_csv(ipc_csv, index_col=0, parse_dates=True)
            try:
                df_ipc.index = pd.DatetimeIndex(df_ipc.index, freq="MS")
            except Exception:
                df_ipc.index = pd.DatetimeIndex(df_ipc.index)
                df_ipc = df_ipc.asfreq("MS")
            series = df_ipc["ipc"]
            print("  [INFO] IPC source : ipc_processed.csv (HCP/fallback)")

    # ── 2. Calculer BESI_trends ───────────────────────────────────────────────
    trends_col = master_df["trends_composite"] if "trends_composite" in master_df.columns \
                 else pd.Series(np.nan, index=master_df.index)
    ipc_chg    = master_df["ipc_change"]       if "ipc_change"       in master_df.columns \
                 else series.pct_change()

    besi_trends = _compute_besi_trends(trends_col, ipc_chg)
    print(f"  [INFO] BESI_trends calculé — mean={besi_trends.mean():.3f}  "
          f"std={besi_trends.std():.3f}")

    # Ajouter besi_trends au master_dataset et sauvegarder
    if "besi_trends" not in master_df.columns:
        master_df["besi_trends"] = besi_trends
        master_df.to_csv(proc_dir / "master_dataset.csv")
        print("  [INFO] besi_trends ajouté à master_dataset.csv")

    # ── 3. Filtrer sur la plage d'analyse ────────────────────────────────────
    s_full = series.loc[train_start:test_end].dropna().copy()
    trends_full = trends_col.reindex(s_full.index).ffill().fillna(0.0)
    besi_t_full = besi_trends.reindex(s_full.index).ffill().fillna(0.0)

    # ── 3b. Sous-indices thématiques Google Trends ────────────────────────────
    # Critique prof : regrouper tous les mots-clés dans un seul indice peut
    # être biaisé. On crée 3 sous-indices thématiques depuis trends_monthly.csv.
    trends_csv = proc_dir / "trends_monthly.csv"
    _prix_full    = pd.Series(np.nan, index=s_full.index, name="trends_prix")
    _inf_full     = pd.Series(np.nan, index=s_full.index, name="trends_inflation")
    _stress_full  = pd.Series(np.nan, index=s_full.index, name="trends_stress")

    if trends_csv.exists():
        df_tr = pd.read_csv(trends_csv, index_col=0, parse_dates=True)
        try:
            df_tr.index = pd.DatetimeIndex(df_tr.index, freq="MS")
        except Exception:
            df_tr.index = pd.DatetimeIndex(df_tr.index)
            df_tr = df_tr.asfreq("MS")

        # trends_prix     = moyenne("prix huile", "hausse prix") — signal prix
        prix_cols = [c for c in ["prix huile", "hausse prix"] if c in df_tr.columns]
        if prix_cols:
            _prix_full = df_tr[prix_cols].mean(axis=1).reindex(s_full.index).ffill().fillna(0.0)
            _prix_full.name = "trends_prix"

        # trends_inflation = "inflation maroc" seul — signal ancrage
        if "inflation maroc" in df_tr.columns:
            _inf_full = df_tr["inflation maroc"].reindex(s_full.index).ffill().fillna(0.0)
            _inf_full.name = "trends_inflation"

        # trends_stress    = moyenne("credit consommation", "chomage maroc") — stress ménages
        stress_cols = [c for c in ["credit consommation", "chomage maroc"] if c in df_tr.columns]
        if stress_cols:
            _stress_full = df_tr[stress_cols].mean(axis=1).reindex(s_full.index).ffill().fillna(0.0)
            _stress_full.name = "trends_stress"

        print(f"  [INFO] Sous-indices thématiques chargés depuis trends_monthly.csv")
        print(f"         trends_prix     : {_prix_full.notna().sum()} obs  mean={_prix_full.mean():.3f}")
        print(f"         trends_inflation: {_inf_full.notna().sum()} obs  mean={_inf_full.mean():.3f}")
        print(f"         trends_stress   : {_stress_full.notna().sum()} obs  mean={_stress_full.mean():.3f}")
    else:
        print("  [WARN] trends_monthly.csv introuvable — sous-indices thématiques à NaN")

    # besi_enrichi depuis master_dataset (si disponible)
    _be_full = pd.Series(np.nan, index=s_full.index, name="besi_enrichi")
    if "besi_enrichi" in master_df.columns:
        _be_full = master_df["besi_enrichi"].reindex(s_full.index).ffill().fillna(0.0)
        _be_full.name = "besi_enrichi"
        print(f"  [INFO] besi_enrichi chargé depuis master_dataset.csv  mean={_be_full.mean():.3f}")

    # Index de coupure train/test
    test_start = (pd.Timestamp(train_end) + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d")
    n_test = len(s_full.loc[test_start:test_end])
    n_total = len(s_full)
    n_tr_init = n_total - n_test

    print(f"\n  N total  : {n_total} mois ({s_full.index[0].date()} -> {s_full.index[-1].date()})")
    print(f"  N train  : {n_tr_init} mois -> {s_full.index[n_tr_init-1].date()}")
    print(f"  N test   : {n_test}  mois -> {s_full.index[n_tr_init].date()} to {s_full.index[-1].date()}")

    # ── 4. Sélection des ordres SARIMA (sur train) ────────────────────────────
    train_s_sel = s_full.iloc[:n_tr_init]
    print(f"\n  Sélection ordres SARIMA sur {len(train_s_sel)} mois ...")

    candidate_orders = [
        ((2, 1, 1), (0, 1, 1)),
        ((1, 1, 1), (0, 1, 1)),
        ((1, 1, 2), (0, 1, 1)),
        ((0, 1, 1), (0, 1, 1)),
    ]
    best_aic, best_pdq, best_PDQ = np.inf, (1, 1, 1), (0, 1, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for pdq_, PDQ_ in candidate_orders:
            try:
                m_ = SARIMAX(train_s_sel, order=pdq_,
                             seasonal_order=(*PDQ_, 12),
                             enforce_stationarity=False,
                             enforce_invertibility=False).fit(disp=False)
                if m_.aic < best_aic:
                    best_aic, best_pdq, best_PDQ = m_.aic, pdq_, PDQ_
            except Exception:
                pass
    print(f"  Meilleurs ordres : SARIMA{best_pdq}x{best_PDQ}[12]  AIC={best_aic:.2f}")
    pdq, PDQ = best_pdq, best_PDQ

    # ── 5. Walk-forward pour chaque modèle ────────────────────────────────────
    models_def = {
        "Naif":           None,           # traité séparément
        "SARIMA":         None,
        "SARIMAX_T":      "trends",
        "SARIMAX_BT":     "besi_trends",
        # ── Sous-indices thématiques (critique prof) ──────────────────────────
        "SARIMAX_prix":   "trends_prix",       # signal prix directs
        "SARIMAX_inf":    "trends_inflation",   # signal ancrage inflation
        "SARIMAX_stress": "trends_stress",      # signal stress ménages
        # ── BESI enrichi (validation formelle) ───────────────────────────────
        "SARIMAX_BE":     "besi_enrichi",
    }

    exog_map = {
        "trends":           trends_full.to_frame("trends_composite"),
        "besi_trends":      besi_t_full.to_frame("besi_trends"),
        "trends_prix":      _prix_full.to_frame("trends_prix"),
        "trends_inflation": _inf_full.to_frame("trends_inflation"),
        "trends_stress":    _stress_full.to_frame("trends_stress"),
        "besi_enrichi":     _be_full.to_frame("besi_enrichi"),
    }

    all_wf: dict[str, dict] = {}
    aic_bic_v2: dict[str, dict] = {}

    for model_name, exog_key in models_def.items():
        print(f"\n  --- {model_name} ---")

        if model_name == "Naif":
            r_naive = _naive_walk_forward(s_full, n_test, h=1)
            all_wf["Naif"] = {1: r_naive}
            aic_bic_v2["Naif"] = {"aic": np.nan, "bic": np.nan}
            print(f"    h=1: RMSE={r_naive['rmse']:.5f}  MAE={r_naive['mae']:.5f}  "
                  f"MAPE={r_naive['mape']:.2f}%")
            continue

        # AIC/BIC sur l'ensemble train (une seule fois)
        exog_tr = exog_map[exog_key].iloc[:n_tr_init] if exog_key else None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m_fit = SARIMAX(
                    s_full.iloc[:n_tr_init], exog=exog_tr,
                    order=pdq, seasonal_order=(*PDQ, 12),
                    enforce_stationarity=False, enforce_invertibility=False,
                ).fit(disp=False)
            aic_bic_v2[model_name] = {"aic": m_fit.aic, "bic": m_fit.bic}
            print(f"    AIC={m_fit.aic:.2f}  BIC={m_fit.bic:.2f}")
        except Exception as exc:
            aic_bic_v2[model_name] = {"aic": np.nan, "bic": np.nan}
            print(f"    [WARN] AIC/BIC : {exc}")

        # Walk-forward
        exog_full = exog_map[exog_key] if exog_key else None
        wf = _wf_one_model(s_full, exog_full, pdq, PDQ, n_test, horizons,
                           label=model_name)
        all_wf[model_name] = wf
        for h in horizons:
            r = wf[h]
            print(f"    h={h}: RMSE={r['rmse']:.5f}  MAE={r['mae']:.5f}  "
                  f"MAPE={r['mape']:.2f}%")

    # ── 6. Tableau comparatif global ──────────────────────────────────────────
    baseline_key = "SARIMA"
    base_rmse_h = {h: all_wf[baseline_key][h]["rmse"] for h in horizons}

    rows = []
    for mname, wf in all_wf.items():
        row = {"Modele": mname,
               "AIC":    round(aic_bic_v2[mname]["aic"], 2),
               "BIC":    round(aic_bic_v2[mname]["bic"], 2)}
        for h in horizons:
            r = wf[h]
            gain = (base_rmse_h[h] - r["rmse"]) / base_rmse_h[h] * 100 \
                   if base_rmse_h[h] != 0 else 0.0
            row[f"RMSE_h{h}"]  = round(r["rmse"], 5)
            row[f"MAE_h{h}"]   = round(r["mae"],  5)
            row[f"MAPE_h{h}"]  = round(r["mape"], 2)
            row[f"Gain%_h{h}"] = round(gain, 1)
        rows.append(row)

    df_cmp_v2 = pd.DataFrame(rows).set_index("Modele")

    print(f"\n{'='*65}")
    print("  TABLEAU COMPARATIF V2 (h=1)")
    print(f"{'='*65}")
    print(df_cmp_v2.to_string())

    df_cmp_v2.to_csv(rep_dir / "model_comparison_v2.csv")
    print(f"\n  Sauvegardé : {rep_dir / 'model_comparison_v2.csv'}")

    # ── 7. Métriques par sous-période ─────────────────────────────────────────
    sub_periods = {
        "full_test":  (test_start,       test_end),
        "choc_2022":  (test_start,       "2022-12-01"),
        "post_2022":  ("2023-01-01",     test_end),
    }

    period_rows = []
    h1 = horizons[0]

    for mname, wf in all_wf.items():
        r = wf[h1]
        for period_name, (p_start, p_end) in sub_periods.items():
            mask = (r["dates"] >= p_start) & (r["dates"] <= p_end)
            yt_p = r["y_true"][mask]
            yp_p = r["y_pred"][mask]
            if len(yt_p) == 0:
                continue
            rmse_p = _rmse(yt_p, yp_p)
            mae_p  = _mae(yt_p, yp_p)
            mape_p = _mape(yt_p, yp_p)
            base_rmse_p = all_wf[baseline_key][h1]["rmse"]
            gain_p = (base_rmse_p - rmse_p) / base_rmse_p * 100 \
                     if mname != baseline_key and base_rmse_p > 0 else 0.0

            period_rows.append({
                "Modele":  mname,
                "Periode": period_name,
                "N":       int(mask.sum()),
                "RMSE":    round(rmse_p, 5),
                "MAE":     round(mae_p,  5),
                "MAPE":    round(mape_p, 2),
                "Gain%_vs_SARIMA": round(gain_p, 1) if mname != baseline_key else 0.0,
            })

    df_period_v2 = pd.DataFrame(period_rows)
    df_period_v2.to_csv(rep_dir / "period_performance_v2.csv", index=False)
    print(f"  Sauvegardé : {rep_dir / 'period_performance_v2.csv'}")

    print(f"\n  METRIQUES PAR SOUS-PERIODE (h=1)")
    print(df_period_v2.to_string(index=False))

    # ── 8. Figures ────────────────────────────────────────────────────────────
    if save_fig:
        import matplotlib.pyplot as _plt
        import matplotlib.gridspec as _gs

        _palette_v2 = {
            "Naif":           "#AAAAAA",
            "SARIMA":         _COL_ORIG,
            "SARIMAX_T":      _COL_DIFF,
            "SARIMAX_BT":     _COL_TREND,
            # sous-indices thématiques
            "SARIMAX_prix":   "#E07B39",
            "SARIMAX_inf":    "#9467BD",
            "SARIMAX_stress": "#17BECF",
            # BESI enrichi
            "SARIMAX_BE":     "#D62728",
        }

        # Figure 1 : Prédictions superposées (h=1)
        fig, ax = _plt.subplots(figsize=(14, 6))
        fig.suptitle(
            f"Comparaison modèles v2 (Google Trends only) — h=1 mois\n"
            f"Train : {train_start} -> {train_end}  |  Test : {test_start} -> {test_end}",
            fontsize=10, fontweight="bold",
        )

        ax.plot(s_full.index, s_full.values,
                color="lightgray", lw=2.0, label="IPC réel", zorder=1)
        ax.plot(s_full.index[n_tr_init:], s_full.values[n_tr_init:],
                color="black", lw=2.5, label="Test (réel)", zorder=2)

        for mname, wf in all_wf.items():
            r = wf[h1]
            ax.plot(r["dates"], r["y_pred"],
                    color=_palette_v2.get(mname, "blue"),
                    lw=1.4, ls="--" if mname == "Naif" else "-.",
                    label=f"{mname}  RMSE={r['rmse']:.4f}",
                    alpha=0.85, zorder=3)

        ax.axvline(s_full.index[n_tr_init],
                   color="red", lw=1.2, ls="--", alpha=0.8, label="Début test")
        ax.axvline(pd.Timestamp("2022-01-01"),
                   color="purple", lw=0.9, ls=":", alpha=0.6, label="Rupture 2022")
        ax.set_xlabel("Date", fontsize=9)
        ax.set_ylabel("IPC (base 2010=1)", fontsize=9)
        ax.legend(fontsize=8, loc="upper left", ncol=2)
        ax.grid(True, alpha=0.3)
        _plt.tight_layout()
        p1 = FIG_DIR / "compare_all_predictions_v2.png"
        fig.savefig(p1, dpi=150, bbox_inches="tight")
        print(f"\n  Figure sauvegardée : {p1}")
        _plt.close(fig)

        # Figure 2 : Barres RMSE par modèle et sous-période
        pivot_rmse = df_period_v2.pivot(
            index="Modele", columns="Periode", values="RMSE"
        )
        fig2, ax2 = _plt.subplots(figsize=(10, 5))
        fig2.suptitle(
            "RMSE par modèle et sous-période (v2 — sans Reddit/YouTube)",
            fontsize=10, fontweight="bold",
        )
        x = np.arange(len(pivot_rmse))
        cols = [c for c in ["full_test", "choc_2022", "post_2022"]
                if c in pivot_rmse.columns]
        w = 0.8 / len(cols)
        bar_colors = [_COL_ORIG, _COL_DIFF, _COL_TREND]
        for i, col in enumerate(cols):
            vals = pivot_rmse[col].values
            bars = ax2.bar(x + i * w, vals, width=w,
                           color=bar_colors[i % len(bar_colors)], alpha=0.8,
                           label=col)
            for bar in bars:
                ax2.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.00005,
                         f"{bar.get_height():.4f}",
                         ha="center", va="bottom", fontsize=7.5)
        ax2.set_xticks(x + w * (len(cols) - 1) / 2)
        ax2.set_xticklabels(pivot_rmse.index, fontsize=9)
        ax2.set_ylabel("RMSE", fontsize=9)
        ax2.legend(fontsize=8, title="Sous-période")
        ax2.grid(True, alpha=0.3, axis="y")
        _plt.tight_layout()
        p2 = FIG_DIR / "period_performance_v2.png"
        fig2.savefig(p2, dpi=150, bbox_inches="tight")
        print(f"  Figure sauvegardée : {p2}")
        _plt.close(fig2)

        # Figure 3 : Gain % vs SARIMA par sous-période (SARIMAX_T et SARIMAX_BT)
        fig3, ax3 = _plt.subplots(figsize=(9, 4))
        fig3.suptitle(
            "Gain RMSE (%) vs SARIMA par sous-période",
            fontsize=10, fontweight="bold",
        )
        gain_models = [m for m in df_period_v2["Modele"].unique() if m != "SARIMA"]
        for mname in gain_models:
            sub = df_period_v2[df_period_v2["Modele"] == mname]
            sub = sub[sub["Periode"].isin(["full_test", "choc_2022", "post_2022"])]
            ax3.plot(sub["Periode"].values, sub["Gain%_vs_SARIMA"].values,
                     marker="o", lw=1.8,
                     color=_palette_v2.get(mname, "blue"),
                     label=mname)
        ax3.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)
        ax3.set_ylabel("Gain RMSE (%) vs SARIMA", fontsize=9)
        ax3.set_xlabel("Sous-période", fontsize=9)
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        _plt.tight_layout()
        p3 = FIG_DIR / "gain_vs_sarima_v2.png"
        fig3.savefig(p3, dpi=150, bbox_inches="tight")
        print(f"  Figure sauvegardée : {p3}")
        _plt.close(fig3)

    return df_cmp_v2, df_period_v2


# ─── Point d'entrée (test rapide sur l'IPC simulé) ────────────────────────────
if __name__ == "__main__":
    # Charge le master dataset pour tester sur l'IPC réel/simulé
    from pathlib import Path as _P
    _root = _P(__file__).resolve().parent.parent
    _csv  = _root / "data" / "processed" / "ipc_processed.csv"

    _master_csv = _root / "data" / "processed" / "master_dataset.csv"

    if _csv.exists() and _master_csv.exists():
        df_ipc    = pd.read_csv(_csv,        parse_dates=["date"], index_col="date")
        df_master = pd.read_csv(_master_csv, parse_dates=["date"], index_col="date")
        df_ipc.index.freq    = "MS"
        df_master.index.freq = "MS"
        ipc = df_ipc["ipc"]

        print("\n>>> Fit SARIMA baseline")
        model, pdq, PDQ = fit_sarima_baseline(ipc, train_end="2021-12-01", save_fig=True)
        orders = (pdq, PDQ)

        print("\n>>> Fit SARIMAX avec BESI composite")
        exog_besi = df_master[["besi"]].reindex(ipc.index).ffill()
        fit_sarimax(ipc, exog_besi, orders, train_end="2021-12-01",
                    name="SARIMAX_BESI", save_fig=True)

        print("\n>>> Comparaison SARIMA vs SARIMAX (n_test=12 pour rapidite)")
        exog_variants = {
            "SARIMA":         None,
            "SARIMAX_Google": df_master[["trends_composite"]],
            "SARIMAX_BESI":   df_master[["besi"]],
            "SARIMAX_All":    df_master[["trends_composite",
                                         "reddit_composite",
                                         "youtube_composite"]],
        }
        df_cmp = compare_models(
            ipc, exog_variants, orders,
            train_end="2021-12-01", n_test=12,
            horizons=[1, 2, 3], save_fig=True,
        )
        print("\n\nTableau final :")
        print(df_cmp.to_string())
    else:
        print("Fichiers manquants — lancer d'abord : python src/data_pipeline.py")
