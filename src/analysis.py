"""
Analyse de rupture structurelle et performances par période — Projet BESI Maroc
Session 7 : Test de Chow, CUSUM, comparaison SARIMA vs SARIMAX par sous-période

Fonctions principales
---------------------
chow_test(series, exog, breakpoint)  -> dict
    Test de Chow F + CUSUM pour détecter la rupture 2022
period_performance(series, exog, periods, orders) -> pd.DataFrame
    RMSE/MAE/MAPE de SARIMA vs SARIMAX dans chaque sous-période
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import f as f_dist

np.random.seed(42)

# ─── Chemins ──────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "outputs" / "figures"
REP_DIR = ROOT / "outputs" / "reports"
FIG_DIR.mkdir(parents=True, exist_ok=True)
REP_DIR.mkdir(parents=True, exist_ok=True)

# Palette (cohérente avec models.py)
_COL_PRE    = "#2C5F8A"   # bleu  — période pré-choc
_COL_POST   = "#D62728"   # rouge — période post-choc
_COL_BREAK  = "#9467BD"   # violet — ligne de rupture
_COL_SARIMA  = "#2C5F8A"
_COL_SARIMAX = "#E07B39"
_PERIOD_COLORS = ["#2C5F8A", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD"]


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS INTERNES
# ═══════════════════════════════════════════════════════════════════════════════

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _build_ols_features(series: pd.Series, exog: pd.DataFrame | None) -> pd.DataFrame:
    """
    Matrice de régresseurs OLS pour le test de Chow.
    X = [constante, tendance, sin/cos saisonniers, colonnes exog]

    On reste sur la série en niveau (non-différenciée) pour que les
    coefficients restent interprétables lors de la comparaison pré/post.
    La tendance linéaire absorbe la composante I(1).
    """
    months = series.index.month
    X = pd.DataFrame({
        "const": 1.0,
        "trend": np.arange(len(series), dtype=float),
        "sin12": np.sin(2 * np.pi * months / 12),
        "cos12": np.cos(2 * np.pi * months / 12),
    }, index=series.index)

    if exog is not None:
        for col in exog.columns:
            X[col] = exog.reindex(series.index).ffill().bfill()[col].values

    return X


def _ols(y: np.ndarray, X: np.ndarray) -> tuple:
    """OLS via numpy lstsq. Retourne (beta, y_hat, rss)."""
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    rss   = float(np.sum((y - y_hat) ** 2))
    return beta, y_hat, rss


def _cusum_recursive(residuals: np.ndarray) -> tuple:
    """
    CUSUM récursif des résidus OLS (Brown-Durbin-Evans, 1975).
    S_t = cumsum(ε) / (σ * sqrt(n))
    Bornes à 5% : ±0.948 * (1 + 2 * t/n)  — approximation linéaire standard.
    """
    n     = len(residuals)
    sigma = np.std(residuals, ddof=1) or 1.0
    cusum = np.cumsum(residuals) / (sigma * np.sqrt(n))
    t_norm = np.arange(1, n + 1) / n
    bounds = 0.948 * (1 + 2 * t_norm)
    return cusum, bounds, -bounds


def _wf_period(
    series: pd.Series,
    exog: pd.DataFrame | None,
    pdq: tuple,
    PDQ: tuple,
    train_idx_end: int,
    test_slice: slice,
) -> dict:
    """
    Walk-forward sur une sous-période. Entraîne sur [0, train_idx_end+step],
    prédit 1 pas en avant, accumule RMSE/MAE/MAPE.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    s      = series.values
    n_test = len(series.iloc[test_slice])
    preds, actuals, dates = [], [], []

    for step in range(n_test):
        t       = train_idx_end + step
        train_s = series.iloc[:t]
        train_e = exog.iloc[:t] if exog is not None else None
        fut_e   = exog.iloc[t:t+1] if exog is not None else None

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = SARIMAX(
                    train_s, exog=train_e,
                    order=pdq, seasonal_order=(*PDQ, 12),
                    enforce_stationarity=False, enforce_invertibility=False,
                ).fit(disp=False)
            fc = float(m.forecast(steps=1, exog=fut_e.values if fut_e is not None else None))
        except Exception:
            fc = np.nan

        preds.append(fc)
        actuals.append(series.iloc[t])
        dates.append(series.index[t])

    yt = np.array(actuals)
    yp = np.array(preds)
    mask = ~np.isnan(yp)
    yt, yp = yt[mask], yp[mask]

    return {
        "rmse":   _rmse(yt, yp) if len(yp) else np.nan,
        "mae":    _mae(yt, yp)  if len(yp) else np.nan,
        "mape":   _mape(yt, yp) if len(yp) else np.nan,
        "y_true": yt, "y_pred": yp,
        "dates":  pd.DatetimeIndex(dates)[mask],
        "n":      int(mask.sum()),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TEST DE CHOW
# ═══════════════════════════════════════════════════════════════════════════════

def chow_test(
    series: pd.Series,
    exog: pd.DataFrame | None = None,
    breakpoint: str = "2022-01-01",
    save_fig: bool = True,
) -> dict:
    """
    Test de Chow pour détecter une rupture structurelle dans la dynamique de l'IPC.

    Méthode
    -------
    Régression OLS : IPC_t = α + β*t + γ*sin/cos + δ*X_t + ε_t
    H0 : les paramètres (α, β, δ) sont stables avant et après le breakpoint
    H1 : au moins un paramètre change (rupture structurelle)

    Statistique F :
        F = [(RSS_R - RSS_U) / k] / [RSS_U / (n - 2k)]
        RSS_R = résidus modèle complet (contraint)
        RSS_U = RSS_pre + RSS_post (modèles non contraints)
        k = nombre de paramètres

    Complété par un test CUSUM des résidus récursifs.

    Paramètres
    ----------
    series     : série IPC mensuelle
    exog       : variables exogènes (signaux comportementaux) — optionnel
    breakpoint : date supposée de la rupture (défaut = "2022-01-01")
    save_fig   : sauvegarder les 4 graphiques

    Retourne
    --------
    dict : f_stat, p_value, is_break, beta_pre, beta_post, cusum, cusum_break
    """
    bp = pd.Timestamp(breakpoint)
    s  = series.dropna().copy()

    # Aligner exog sur l'index de la série
    if exog is not None:
        exog_al = exog.reindex(s.index).ffill().bfill().dropna()
        common  = s.index.intersection(exog_al.index)
        s, exog_al = s.loc[common], exog_al.loc[common]
    else:
        exog_al = None

    # Trouver le breakpoint dans l'index
    if bp not in s.index:
        bp = s.index[s.index.get_indexer([bp], method="nearest")[0]]
    bp_idx = s.index.get_loc(bp)

    n1, n2 = bp_idx, len(s) - bp_idx
    if n1 < 12 or n2 < 12:
        raise ValueError(
            f"Le breakpoint {bp.date()} laisse trop peu d'observations "
            f"(pre={n1}, post={n2}). Minimum = 12 mois chaque côté."
        )

    # ── Matrices OLS ─────────────────────────────────────────────────────────
    X_df   = _build_ols_features(s, exog_al)
    feat_names = list(X_df.columns)
    X_arr, y_arr = X_df.values, s.values
    k = X_arr.shape[1]
    n = len(y_arr)

    beta_full, y_hat_full, rss_full = _ols(y_arr,          X_arr)
    beta_pre,  _,          rss_pre  = _ols(y_arr[:bp_idx], X_arr[:bp_idx])
    beta_post, _,          rss_post = _ols(y_arr[bp_idx:], X_arr[bp_idx:])

    rss_unres = rss_pre + rss_post
    dof_num   = k
    dof_den   = n - 2 * k

    chow_f  = ((rss_full - rss_unres) / dof_num) / (rss_unres / dof_den)
    p_val   = float(1 - f_dist.cdf(chow_f, dof_num, dof_den))
    is_break = p_val < 0.05

    # ── CUSUM des résidus OLS ─────────────────────────────────────────────────
    resid_ols = y_arr - y_hat_full
    cusum_vals, cusum_ub, cusum_lb = _cusum_recursive(resid_ols)
    cusum_break = bool(np.any(np.abs(cusum_vals) > np.abs(cusum_ub)))

    # ── Affichage console ─────────────────────────────────────────────────────
    sep = "=" * 62
    print(f"\n{sep}")
    print("  TEST DE CHOW -- RUPTURE STRUCTURELLE IPC MAROC")
    print(f"  Breakpoint         : {bp.date()}")
    print(f"  Periode pre-choc   : {s.index[0].date()} -> {s.index[bp_idx-1].date()} ({n1} mois)")
    print(f"  Periode post-choc  : {s.index[bp_idx].date()} -> {s.index[-1].date()} ({n2} mois)")
    print(f"  Regresseurs OLS    : {feat_names}")
    print(sep)
    print(f"\n  RSS contraint   (plein echantillon) : {rss_full:.8f}")
    print(f"  RSS non-contraint (pre + post)      : {rss_unres:.8f}")
    print(f"  Reduction RSS                        : {rss_full - rss_unres:.8f}")
    print(f"\n  Statistique F   : {chow_f:.4f}")
    print(f"  Degres liberte  : F({dof_num}, {dof_den})")
    print(f"  p-value         : {p_val:.6f}")
    verdict = "*** RUPTURE STRUCTURELLE CONFIRMEE (p < 0.05)" if is_break \
              else "Pas de rupture significative (p >= 0.05)"
    print(f"\n  Conclusion      : {verdict}")

    print(f"\n{'-'*62}")
    print("  COMPARAISON COEFFICIENTS  (pre vs post rupture)")
    print(f"  {'Parametre':<22}  {'Pre':>10}  {'Post':>10}  {'Variation':>10}")
    print(f"  {'-'*58}")
    for i, fn in enumerate(feat_names):
        d   = beta_post[i] - beta_pre[i]
        pct = (d / abs(beta_pre[i]) * 100) if beta_pre[i] != 0 else float("inf")
        print(f"  {fn:<22}  {beta_pre[i]:>10.5f}  {beta_post[i]:>10.5f}  {d:>+9.5f}  ({pct:+.1f}%)")

    print(f"\n{'-'*62}")
    cusum_verdict = "*** Rupture detectable (CUSUM sort des bornes)" \
                    if cusum_break else "Stable dans les bornes (aucune rupture CUSUM)"
    print(f"  CUSUM  : {cusum_verdict}")
    print(sep)

    # ── Figures ───────────────────────────────────────────────────────────────
    if save_fig:
        fig = plt.figure(figsize=(14, 11))
        fig.suptitle(
            f"Test de Chow — Rupture structurelle IPC Maroc (breakpoint = {bp.date()})\n"
            f"F({dof_num},{dof_den}) = {chow_f:.3f}   p = {p_val:.4f}   {verdict}",
            fontsize=10, fontweight="bold",
        )
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.6, wspace=0.38)
        ax1 = fig.add_subplot(gs[0, :])   # série + lignes de tendance
        ax2 = fig.add_subplot(gs[1, 0])   # résidus par régime
        ax3 = fig.add_subplot(gs[1, 1])   # barres coefficients pré vs post
        ax4 = fig.add_subplot(gs[2, :])   # CUSUM

        # --- Série + tendances OLS par régime ---
        ax1.plot(s.index, s.values, color="lightgray", lw=1.2, zorder=1)
        ax1.plot(s.index[:bp_idx], s.values[:bp_idx],
                 color=_COL_PRE,  lw=1.8, label=f"IPC pre-{bp.year}", zorder=2)
        ax1.plot(s.index[bp_idx:], s.values[bp_idx:],
                 color=_COL_POST, lw=1.8, label=f"IPC post-{bp.year}", zorder=2)

        # Droites de tendance (composante const+trend uniquement)
        trend_pre  = X_arr[:bp_idx, :2] @ beta_pre[:2]
        trend_post = X_arr[bp_idx:, :2] @ beta_post[:2]
        ax1.plot(s.index[:bp_idx], trend_pre,
                 color=_COL_PRE,  lw=1.2, ls="--", alpha=0.75, label="Tendance pre")
        ax1.plot(s.index[bp_idx:], trend_post,
                 color=_COL_POST, lw=1.2, ls="--", alpha=0.75, label="Tendance post")

        ax1.axvline(bp, color=_COL_BREAK, lw=1.8, ls="--",
                    label=f"Rupture {bp.date()}")
        ax1.set_title("IPC Maroc — Deux regimes detectes par le test de Chow", fontsize=9)
        ax1.legend(fontsize=7.5, ncol=3)
        ax1.grid(True, alpha=0.3)

        # --- Résidus pré vs post ---
        res_pre  = y_arr[:bp_idx] - X_arr[:bp_idx] @ beta_pre
        res_post = y_arr[bp_idx:] - X_arr[bp_idx:] @ beta_post
        ax2.plot(s.index[:bp_idx], res_pre,
                 color=_COL_PRE,  lw=0.9,
                 label=f"Residus pre  sigma={np.std(res_pre):.5f}")
        ax2.plot(s.index[bp_idx:], res_post,
                 color=_COL_POST, lw=0.9,
                 label=f"Residus post sigma={np.std(res_post):.5f}")
        ax2.axhline(0, color="black", lw=0.5)
        ax2.axvline(bp, color=_COL_BREAK, lw=1.0, ls="--", alpha=0.7)
        ax2.set_title("Residus OLS par regime", fontsize=9)
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)

        # --- Barres coefficients ---
        xpos = np.arange(k)
        w    = 0.38
        bars_pre  = ax3.bar(xpos - w/2, beta_pre,  width=w,
                            color=_COL_PRE,  alpha=0.85, label="Pre")
        bars_post = ax3.bar(xpos + w/2, beta_post, width=w,
                            color=_COL_POST, alpha=0.85, label="Post")
        ax3.set_xticks(xpos)
        ax3.set_xticklabels(feat_names, rotation=28, fontsize=7)
        ax3.axhline(0, color="black", lw=0.5)
        ax3.set_title("Coefficients OLS pre vs post rupture", fontsize=9)
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3, axis="y")

        # --- CUSUM ---
        ax4.plot(s.index, cusum_vals, color=_COL_BREAK, lw=1.6, label="CUSUM")
        ax4.fill_between(s.index, cusum_lb, cusum_ub,
                         alpha=0.12, color=_COL_BREAK)
        ax4.plot(s.index, cusum_ub,  color=_COL_BREAK, lw=0.9, ls="--",
                 alpha=0.55, label="Bornes 5%")
        ax4.plot(s.index, cusum_lb,  color=_COL_BREAK, lw=0.9, ls="--", alpha=0.55)
        ax4.axhline(0, color="black", lw=0.5)
        ax4.axvline(bp, color="red", lw=1.4, ls="--", alpha=0.8,
                    label=f"Breakpoint {bp.date()}")
        cusum_title = ("*** CUSUM sort des bornes — rupture detectable"
                       if cusum_break
                       else "CUSUM dans les bornes — pas de rupture CUSUM")
        ax4.set_title(cusum_title, fontsize=9)
        ax4.legend(fontsize=7.5)
        ax4.set_xlabel("Date", fontsize=9)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        path = FIG_DIR / "chow_test.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"\n  Figure sauvegardee : {path}")
        plt.show()

    return {
        "f_stat":      chow_f,
        "p_value":     p_val,
        "is_break":    is_break,
        "breakpoint":  bp,
        "beta_pre":    dict(zip(feat_names, beta_pre)),
        "beta_post":   dict(zip(feat_names, beta_post)),
        "rss_full":    rss_full,
        "rss_pre":     rss_pre,
        "rss_post":    rss_post,
        "feat_names":  feat_names,
        "cusum":       cusum_vals,
        "cusum_break": cusum_break,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PERFORMANCES PAR PÉRIODE
# ═══════════════════════════════════════════════════════════════════════════════

_DEFAULT_PERIODS = {
    "Pre-COVID  (2010-2019)": ("2010", "2019"),
    "Choc       (2020-2022)": ("2020", "2022"),
    "Post-choc  (2022-2026)": ("2022", "2026"),
}

_MIN_TRAIN = 24   # mois minimum d'entraînement avant walk-forward


def period_performance(
    series: pd.Series,
    exog: pd.DataFrame | None = None,
    periods: dict | None = None,
    orders: tuple | None = None,
    save_fig: bool = True,
) -> pd.DataFrame:
    """
    Compare SARIMA vs SARIMAX dans chaque sous-période.

    Pour chaque période :
      - Entraînement : toutes les données AVANT le début de la période
        (ou les 24 premiers mois de la période si données insuffisantes)
      - Test : walk-forward horizon h=1 sur la durée de la période
      - Métriques : RMSE, MAE, MAPE pour SARIMA et SARIMAX

    Paramètres
    ----------
    series  : série IPC mensuelle
    exog    : variables exogènes BESI (si None, seul SARIMA est évalué)
    periods : dict {label: (annee_debut, annee_fin)}  — défaut : 3 périodes
    orders  : ((p,d,q), (P,D,Q)) depuis fit_sarima_baseline()
              Si None, utilise SARIMA(1,1,1)x(0,1,1)[12] par défaut
    save_fig: sauvegarder les graphiques

    Retourne
    --------
    DataFrame avec pour chaque période :
      SARIMA_RMSE, SARIMA_MAE, SARIMA_MAPE,
      SARIMAX_RMSE, SARIMAX_MAE, SARIMAX_MAPE, Gain_RMSE_%
    Sauvegardé dans outputs/reports/period_performance.csv
    """
    if periods is None:
        periods = _DEFAULT_PERIODS

    if orders is None:
        # Ordres par défaut — robustes pour l'IPC mensuel
        pdq, PDQ = (1, 1, 1), (0, 1, 1)
        print("  [INFO] orders=None -> utilisation SARIMA(1,1,1)x(0,1,1)[12] par defaut")
    else:
        pdq, PDQ = orders

    s = series.dropna().copy()

    # Aligner exog sur l'index complet de la série
    if exog is not None:
        exog_full = exog.reindex(s.index).ffill().bfill()
    else:
        exog_full = None

    print(f"\n{'='*64}")
    print("  PERFORMANCES PAR PERIODE -- SARIMA vs SARIMAX")
    print(f"  Ordres : SARIMA{pdq} x {PDQ}[12]")
    print(f"  Exog   : {list(exog.columns) if exog is not None else 'aucune (SARIMA seul)'}")
    print(f"{'='*64}")

    rows   = []
    all_wf = {}   # pour les graphiques : {period_label: {"sarima": wf, "sarimax": wf}}

    for period_label, (yr_start, yr_end) in periods.items():
        dt_start = pd.Timestamp(f"{yr_start}-01-01")
        dt_end   = pd.Timestamp(f"{yr_end}-12-01")

        # Clamper sur les bornes disponibles
        dt_start = max(dt_start, s.index[0])
        dt_end   = min(dt_end,   s.index[-1])

        # Index dans la série
        idx_start = s.index.get_indexer([dt_start], method="nearest")[0]
        idx_end   = s.index.get_indexer([dt_end],   method="nearest")[0]

        # Taille de la fenêtre de test
        n_test_period = idx_end - idx_start + 1
        if n_test_period < 3:
            print(f"  [SKIP] {period_label} : trop court ({n_test_period} mois)")
            continue

        # Fenêtre d'entraînement initiale
        #   - Préférence : toutes les données avant dt_start
        #   - Fallback : premiers _MIN_TRAIN mois disponibles dans la période
        if idx_start >= _MIN_TRAIN:
            train_init_end = idx_start   # entraîne sur [0, idx_start[
        else:
            train_init_end = min(idx_start + _MIN_TRAIN, idx_end - 1)

        print(f"\n  Periode : {period_label}")
        print(f"    Entrainement initial  : {s.index[0].date()} -> {s.index[train_init_end-1].date()} ({train_init_end} mois)")
        print(f"    Test (walk-forward)   : {s.index[train_init_end].date()} -> {dt_end.date()} ({idx_end - train_init_end + 1} mois)")

        test_slice = slice(train_init_end, idx_end + 1)

        # ── SARIMA ──────────────────────────────────────────────────────────
        print(f"    SARIMA  ...")
        wf_sarima = _wf_period(s, None, pdq, PDQ, train_init_end, test_slice)
        print(f"      RMSE={wf_sarima['rmse']:.5f}  MAE={wf_sarima['mae']:.5f}  MAPE={wf_sarima['mape']:.2f}%")

        # ── SARIMAX ─────────────────────────────────────────────────────────
        if exog_full is not None:
            print(f"    SARIMAX ...")
            wf_sarimax = _wf_period(s, exog_full, pdq, PDQ, train_init_end, test_slice)
            print(f"      RMSE={wf_sarimax['rmse']:.5f}  MAE={wf_sarimax['mae']:.5f}  MAPE={wf_sarimax['mape']:.2f}%")

            gain_rmse = (wf_sarima["rmse"] - wf_sarimax["rmse"]) / wf_sarima["rmse"] * 100 \
                        if wf_sarima["rmse"] > 0 else 0.0
            sign = "amelioration" if gain_rmse > 0 else "degradation"
            print(f"    --> Gain RMSE SARIMAX vs SARIMA : {gain_rmse:+.1f}%  ({sign})")
        else:
            wf_sarimax = None
            gain_rmse  = np.nan

        all_wf[period_label] = {"sarima": wf_sarima, "sarimax": wf_sarimax}

        # ── Ligne du tableau ─────────────────────────────────────────────────
        row = {
            "Periode":        period_label.strip(),
            "N_test":         wf_sarima["n"],
            "SARIMA_RMSE":    round(wf_sarima["rmse"],  5),
            "SARIMA_MAE":     round(wf_sarima["mae"],   5),
            "SARIMA_MAPE":    round(wf_sarima["mape"],  2),
        }
        if wf_sarimax is not None:
            row["SARIMAX_RMSE"]   = round(wf_sarimax["rmse"],  5)
            row["SARIMAX_MAE"]    = round(wf_sarimax["mae"],   5)
            row["SARIMAX_MAPE"]   = round(wf_sarimax["mape"],  2)
            row["Gain_RMSE_%"]    = round(gain_rmse, 1)
        rows.append(row)

    # ── Tableau récapitulatif ─────────────────────────────────────────────────
    df_perf = pd.DataFrame(rows).set_index("Periode")

    print(f"\n{'='*70}")
    print("  TABLEAU RECAPITULATIF PAR PERIODE")
    print(f"{'='*70}")
    print(df_perf.to_string())

    # La période où BESI aide le plus
    if "Gain_RMSE_%" in df_perf.columns and df_perf["Gain_RMSE_%"].notna().any():
        best_period = df_perf["Gain_RMSE_%"].idxmax()
        best_gain   = df_perf.loc[best_period, "Gain_RMSE_%"]
        print(f"\n  BESI aide le plus durant : {best_period}  (gain RMSE = {best_gain:+.1f}%)")
    print(f"{'='*70}")

    csv_path = REP_DIR / "period_performance.csv"
    df_perf.to_csv(csv_path)
    print(f"  Tableau sauvegarde : {csv_path}")

    # ── Figures ───────────────────────────────────────────────────────────────
    if save_fig and rows:
        # Figure 1 : RMSE comparé par période (barres)
        fig1, axes1 = plt.subplots(1, 3, figsize=(14, 4.5))
        fig1.suptitle("SARIMA vs SARIMAX — Performances par sous-periode",
                      fontsize=10, fontweight="bold")

        metrics_pairs = [("RMSE", "SARIMA_RMSE", "SARIMAX_RMSE"),
                         ("MAE",  "SARIMA_MAE",  "SARIMAX_MAE"),
                         ("MAPE", "SARIMA_MAPE", "SARIMAX_MAPE")]

        for ax, (metric, col_s, col_sx) in zip(axes1, metrics_pairs):
            x    = np.arange(len(df_perf))
            w    = 0.35
            vals_s  = df_perf[col_s].values
            ax.bar(x - w/2, vals_s, width=w, color=_COL_SARIMA,  alpha=0.82,
                   label="SARIMA")
            if col_sx in df_perf.columns:
                vals_sx = df_perf[col_sx].values
                ax.bar(x + w/2, vals_sx, width=w, color=_COL_SARIMAX, alpha=0.82,
                       label="SARIMAX")
                # Annotations gain/perte
                for xi, (vs, vsx) in enumerate(zip(vals_s, vals_sx)):
                    gain = (vs - vsx) / vs * 100 if vs > 0 else 0
                    color_ann = "green" if gain > 0 else "red"
                    ax.text(xi, max(vs, vsx) * 1.02,
                            f"{gain:+.0f}%", ha="center", fontsize=7,
                            color=color_ann, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(
                [r.strip().split("(")[0].strip() for r in df_perf.index],
                rotation=20, fontsize=7.5,
            )
            ax.set_title(metric, fontsize=10)
            ax.legend(fontsize=7.5)
            ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        p1 = FIG_DIR / "period_performance_metrics.png"
        fig1.savefig(p1, dpi=150, bbox_inches="tight")
        print(f"  Figure sauvegardee : {p1}")
        plt.show()

        # Figure 2 : Prédictions vs réel par période
        n_periods = len(all_wf)
        fig2, axes2 = plt.subplots(n_periods, 1,
                                    figsize=(13, 4 * n_periods),
                                    sharex=False)
        if n_periods == 1:
            axes2 = [axes2]
        fig2.suptitle("Predictions walk-forward par periode — SARIMA vs SARIMAX",
                       fontsize=10, fontweight="bold")

        for ax, ((plabel, wf_pair), pcol) in zip(
            axes2, zip(all_wf.items(), _PERIOD_COLORS)
        ):
            wf_s  = wf_pair["sarima"]
            wf_sx = wf_pair["sarimax"]

            ax.plot(wf_s["dates"], wf_s["y_true"],
                    color="black", lw=1.8, label="Valeurs reelles", zorder=3)
            ax.plot(wf_s["dates"], wf_s["y_pred"],
                    color=_COL_SARIMA, lw=1.1, ls="-.", alpha=0.85,
                    label=f"SARIMA  RMSE={wf_s['rmse']:.4f}", zorder=2)
            if wf_sx is not None and len(wf_sx["y_pred"]) > 0:
                ax.plot(wf_sx["dates"], wf_sx["y_pred"],
                        color=_COL_SARIMAX, lw=1.1, ls="--", alpha=0.85,
                        label=f"SARIMAX RMSE={wf_sx['rmse']:.4f}", zorder=2)

            ax.set_title(plabel.strip(), fontsize=9)
            ax.legend(fontsize=7.5)
            ax.grid(True, alpha=0.3)
            ax.set_ylabel("IPC", fontsize=8)

        axes2[-1].set_xlabel("Date", fontsize=9)
        plt.tight_layout()
        p2 = FIG_DIR / "period_performance_predictions.png"
        fig2.savefig(p2, dpi=150, bbox_inches="tight")
        print(f"  Figure sauvegardee : {p2}")
        plt.show()

        # Figure 3 : Gain RMSE par période (barres + axe zéro)
        if "Gain_RMSE_%" in df_perf.columns:
            fig3, ax3 = plt.subplots(figsize=(9, 4))
            fig3.suptitle(
                "Gain RMSE de SARIMAX vs SARIMA par sous-periode\n"
                "(positif = SARIMAX meilleur, negatif = SARIMA meilleur)",
                fontsize=9, fontweight="bold",
            )
            gains  = df_perf["Gain_RMSE_%"].values
            x      = np.arange(len(gains))
            colors = ["green" if g > 0 else "red" for g in gains]
            bars   = ax3.bar(x, gains, color=colors, alpha=0.8, width=0.5)
            for bar, g in zip(bars, gains):
                ax3.text(bar.get_x() + bar.get_width() / 2,
                         g + (0.5 if g >= 0 else -1.2),
                         f"{g:+.1f}%", ha="center", fontsize=9, fontweight="bold")
            ax3.set_xticks(x)
            ax3.set_xticklabels(
                [r.strip().split("(")[0].strip() for r in df_perf.index],
                fontsize=9,
            )
            ax3.axhline(0, color="black", lw=0.8)
            ax3.set_ylabel("Gain RMSE (%)", fontsize=9)
            ax3.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
            p3 = FIG_DIR / "period_performance_gain.png"
            fig3.savefig(p3, dpi=150, bbox_inches="tight")
            print(f"  Figure sauvegardee : {p3}")
            plt.show()

    return df_perf


# ═══════════════════════════════════════════════════════════════════════════════
# 3. EARLY WARNING ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

_BESI_WARN_THR  = 0.35   # seuil alerte BESI (Warning + High Stress)
_IPC_STRESS_THR = 0.02   # seuil stress IPC : variation YoY >= 2%
_MAX_LEAD       = 12     # fenetre maximale de lead time analysee (mois)
_MATCH_WINDOW   = 6      # fenetre d'appariement alerte -> evenement (mois)


def early_warning_analysis(
    besi_series: pd.Series,
    ipc_series:  pd.Series,
    besi_warn_thr:  float = _BESI_WARN_THR,
    ipc_stress_thr: float = _IPC_STRESS_THR,
    max_lead:       int   = _MAX_LEAD,
    match_window:   int   = _MATCH_WINDOW,
    save_fig: bool  = True,
) -> dict:
    """
    Analyse d'alerte precoce : BESI anticipe-t-il le stress IPC officiel ?

    Methode
    -------
    1. Signal de stress IPC   : variation YoY (12 mois) >= ipc_stress_thr
    2. Signal d'alerte BESI   : BESI >= besi_warn_thr
    3. Cross-correlation (CCF): corr(BESI[t], IPC_YoY[t+lag]) pour lag=0..max_lead
       -- lag optimal = argmax CCF (BESI precede IPC de lag_optimal mois)
    4. Test de causalite de Granger : BESI -> delta_IPC
    5. Lead time par evenement : pour chaque onset de stress IPC,
       trouver la premiere alerte BESI dans les max_lead mois precedents
    6. Precision / Recall a differents seuils de lead time
       TP : alerte BESI suivie d'un stress IPC dans match_window mois
       FP : alerte BESI sans stress IPC dans les match_window mois suivants
       FN : stress IPC non precede d'alerte BESI dans les max_lead mois

    Parametres
    ----------
    besi_series    : indice BESI normalise 0-1
    ipc_series     : serie IPC mensuelle (valeurs absolues)
    besi_warn_thr  : seuil d'alerte BESI (defaut 0.35)
    ipc_stress_thr : seuil de stress IPC en variation YoY (defaut 0.02 = 2%)
    max_lead       : fenetre maximale de lead time analysee (mois)
    match_window   : fenetre pour apparier alerte BESI -> stress IPC
    save_fig       : sauvegarder les 4 graphiques

    Retourne
    --------
    dict : lag_optimal, lead_time_mean, lead_time_median, precision, recall,
           f1, granger_pval, ccf_values, onset_dates, lead_times, tp, fp, fn
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    # ── Alignement sur l'intersection ────────────────────────────────────────
    common = besi_series.dropna().index.intersection(ipc_series.dropna().index)
    besi   = besi_series.reindex(common).ffill().bfill().rename("besi")
    ipc    = ipc_series.reindex(common).ffill().bfill().rename("ipc")

    # ── Signal de stress IPC : variation YoY sur 12 mois ─────────────────────
    ipc_yoy = ipc.pct_change(12).dropna()
    besi_al = besi.reindex(ipc_yoy.index).ffill().bfill()

    if len(ipc_yoy) < max_lead + 2:
        raise ValueError(
            f"Serie trop courte ({len(ipc_yoy)} mois) pour analyser "
            f"un lead time de {max_lead} mois."
        )

    # ── Cross-correlation : BESI[t] vs IPC_YoY[t+lag] ────────────────────────
    # Un lag positif signifie que BESI precede l'IPC de `lag` mois
    ccf_lags   = list(range(0, max_lead + 1))
    ccf_values = []
    for lag in ccf_lags:
        if lag == 0:
            r = float(besi_al.corr(ipc_yoy))
        else:
            r = float(besi_al.iloc[:-lag].corr(ipc_yoy.iloc[lag:]))
        ccf_values.append(r if not np.isnan(r) else 0.0)

    lag_optimal = int(ccf_lags[int(np.argmax(ccf_values))])

    # ── Test de causalite de Granger (BESI -> delta_IPC) ─────────────────────
    # Utilise les premieres differences pour la stationnarite
    df_granger = pd.DataFrame({
        "ipc_diff":  ipc_yoy.diff(),
        "besi_diff": besi_al.diff(),
    }).dropna()

    granger_pval = np.nan
    granger_lag  = max(1, min(lag_optimal, max_lead // 2))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gc_res = grangercausalitytests(
                df_granger[["ipc_diff", "besi_diff"]],
                maxlag=granger_lag,
                verbose=False,
            )
        granger_pval = float(gc_res[granger_lag][0]["ssr_ftest"][1])
    except Exception:
        granger_pval = np.nan

    # ── Detection des onsets de stress IPC (transitions 0->1) ────────────────
    stress_ipc  = (ipc_yoy >= ipc_stress_thr).astype(int)
    onset_mask  = (stress_ipc.diff() > 0)
    onset_dates = ipc_yoy.index[onset_mask].tolist()

    # ── Lead time par evenement ───────────────────────────────────────────────
    besi_alert = (besi_al >= besi_warn_thr).astype(int)
    lead_times, tp_dates, fn_dates = [], [], []

    for onset in onset_dates:
        t_start = onset - pd.DateOffset(months=max_lead)
        # Fenetre BESI strictement avant le debut du stress
        window  = besi_alert.loc[
            (besi_alert.index >= t_start) & (besi_alert.index < onset)
        ]
        if len(window) > 0 and window.any():
            first_alert = window[window == 1].index[0]
            lead = int(round((onset - first_alert).days / 30.44))
            lead_times.append(lead)
            tp_dates.append(onset)
        else:
            fn_dates.append(onset)

    tp_events = len(tp_dates)
    fn_events = len(fn_dates)

    # Fausses alertes BESI : alerte sans stress IPC dans match_window mois
    fp_events = 0
    alert_starts = besi_alert.index[
        (besi_alert.diff() > 0) | ((besi_alert.index == besi_alert.index[0]) & (besi_alert.iloc[0] == 1))
    ]
    for alert_t in alert_starts:
        t_end_fp   = alert_t + pd.DateOffset(months=match_window)
        fut_stress = stress_ipc.loc[
            (stress_ipc.index > alert_t) & (stress_ipc.index <= t_end_fp)
        ]
        if len(fut_stress) == 0 or not fut_stress.any():
            fp_events += 1

    # ── Precision / Recall / F1 ───────────────────────────────────────────────
    precision = tp_events / (tp_events + fp_events) if (tp_events + fp_events) > 0 else 0.0
    recall    = tp_events / (tp_events + fn_events)  if (tp_events + fn_events) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    lead_arr    = np.array(lead_times, dtype=float) if lead_times else np.array([np.nan])
    lead_mean   = float(np.nanmean(lead_arr))
    lead_median = float(np.nanmedian(lead_arr))

    # ── Courbe Precision/Recall selon le seuil de lead time ──────────────────
    pr_rows = []
    for lt_thr in range(1, max_lead + 1):
        tp_lt = sum(1 for lt in lead_times if lt <= lt_thr)
        fn_lt = len(onset_dates) - tp_lt
        fp_lt = fp_events
        p_lt  = tp_lt / (tp_lt + fp_lt) if (tp_lt + fp_lt) > 0 else 0.0
        r_lt  = tp_lt / (tp_lt + fn_lt) if (tp_lt + fn_lt) > 0 else 0.0
        pr_rows.append({"lead_threshold": lt_thr, "precision": p_lt, "recall": r_lt})
    pr_df = pd.DataFrame(pr_rows)

    # ── Affichage console ─────────────────────────────────────────────────────
    sep = "=" * 66
    print(f"\n{sep}")
    print("  ANALYSE D'ALERTE PRECOCE -- BESI vs IPC OFFICIEL")
    print(f"  Seuil alerte BESI     : BESI >= {besi_warn_thr}")
    print(f"  Seuil stress IPC      : variation YoY >= {ipc_stress_thr*100:.0f}%")
    print(f"  Fenetre lead time     : {max_lead} mois")
    print(f"  Fenetre appariement   : {match_window} mois")
    print(sep)
    print("\n  Cross-correlation CCF -- BESI[t] leads IPC[t+lag] :")
    for lag, r in zip(ccf_lags, ccf_values):
        bar    = "#" * max(0, int(abs(r) * 25))
        marker = " <-- OPTIMAL" if lag == lag_optimal else ""
        print(f"    lag={lag:2d} mois : r={r:+.3f}  {bar}{marker}")
    print(f"\n  Lag optimal (CCF max) : {lag_optimal} mois")
    print(f"\n  Causalite de Granger (BESI -> delta_IPC, lag={granger_lag}) :")
    if not np.isnan(granger_pval):
        gc_verdict = "*** CAUSALITE SIGNIFICATIVE (p < 0.05)" if granger_pval < 0.05 \
                     else "Non significatif (p >= 0.05)"
        print(f"    p-value : {granger_pval:.4f}  {gc_verdict}")
    else:
        print("    Test de Granger non disponible.")
    print(f"\n  Evenements stress IPC detectes : {len(onset_dates)}")
    print(f"    TP (BESI alerte en avance)     : {tp_events}")
    print(f"    FN (stress IPC rate par BESI)  : {fn_events}")
    print(f"    FP (fausses alertes BESI)       : {fp_events}")
    print(f"\n  Lead time moyen    : {lead_mean:.1f} mois")
    print(f"  Lead time median   : {lead_median:.1f} mois")
    print(f"\n  Precision          : {precision:.3f}  ({precision*100:.1f}%)")
    print(f"  Recall             : {recall:.3f}  ({recall*100:.1f}%)")
    print(f"  F1-score           : {f1:.3f}")
    phrase = (f"  >>> BESI detecte le stress economique {lead_mean:.1f} mois "
              f"avant l'IPC officiel <<<")
    print(f"\n{phrase}")
    print(sep)

    # ── Figures ───────────────────────────────────────────────────────────────
    if save_fig:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"Alerte precoce BESI -- Lead time moyen = {lead_mean:.1f} mois\n"
            f"Precision={precision:.2f}  Recall={recall:.2f}  F1={f1:.2f}  "
            f"[lag optimal CCF = {lag_optimal} mois]",
            fontsize=10, fontweight="bold",
        )

        # Panneau 1 : BESI + IPC YoY avec seuils et onsets ───────────────────
        ax1  = axes[0, 0]
        ax1b = ax1.twinx()
        ax1.fill_between(besi_al.index, besi_al.values,
                         alpha=0.20, color=_COL_SARIMAX)
        ax1.plot(besi_al.index, besi_al.values,
                 color=_COL_SARIMAX, lw=1.3, label="BESI")
        ax1.axhline(besi_warn_thr, color=_COL_SARIMAX, lw=1.0, ls="--",
                    alpha=0.75, label=f"Seuil BESI={besi_warn_thr}")
        ax1b.plot(ipc_yoy.index, ipc_yoy.values * 100,
                  color=_COL_PRE, lw=1.5, alpha=0.85, label="IPC YoY %")
        ax1b.axhline(ipc_stress_thr * 100, color=_COL_PRE, lw=1.0, ls="--",
                     alpha=0.7, label=f"Seuil stress IPC {ipc_stress_thr*100:.0f}%")
        for od in onset_dates:
            ax1.axvline(od, color=_COL_POST, lw=0.8, alpha=0.45)
        ax1.set_ylabel("BESI", fontsize=8, color=_COL_SARIMAX)
        ax1b.set_ylabel("IPC YoY (%)", fontsize=8, color=_COL_PRE)
        ax1.set_title(
            f"BESI vs Stress IPC (lignes rouges = {len(onset_dates)} onsets)",
            fontsize=9
        )
        lines1, labs1 = ax1.get_legend_handles_labels()
        lines2, labs2 = ax1b.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labs1 + labs2, fontsize=7, loc="upper left")
        ax1.grid(True, alpha=0.25)

        # Panneau 2 : CCF ─────────────────────────────────────────────────────
        ax2 = axes[0, 1]
        ccf_colors = ["#2CA02C" if r > 0 else "#D62728" for r in ccf_values]
        ax2.bar(ccf_lags, ccf_values, color=ccf_colors, alpha=0.78, width=0.65)
        ax2.axvline(lag_optimal, color="black", lw=1.4, ls="--",
                    label=f"Lag optimal = {lag_optimal} mois")
        ax2.axhline(0, color="black", lw=0.5)
        ax2.set_xlabel("Lag (mois) -- BESI precede IPC", fontsize=8)
        ax2.set_ylabel("Correlation r", fontsize=8)
        ax2.set_title("Cross-correlation CCF : BESI[t] vs IPC_YoY[t+lag]", fontsize=9)
        ax2.legend(fontsize=8)
        ax2.set_xticks(ccf_lags)
        ax2.grid(True, alpha=0.3, axis="y")

        # Panneau 3 : Distribution des lead times ─────────────────────────────
        ax3 = axes[1, 0]
        if len(lead_times) > 0:
            bins = np.arange(0, max_lead + 2) - 0.5
            ax3.hist(lead_times, bins=bins, color=_COL_SARIMAX, alpha=0.82,
                     edgecolor="white", rwidth=0.85)
            ax3.axvline(lead_mean, color="red", lw=2.0, ls="--",
                        label=f"Moyenne = {lead_mean:.1f} mois")
            ax3.axvline(lead_median, color="#FF7F0E", lw=1.5, ls=":",
                        label=f"Mediane = {lead_median:.1f} mois")
            ax3.set_xlabel("Lead time (mois)", fontsize=8)
            ax3.set_ylabel("Nombre d'evenements", fontsize=8)
            ax3.set_title(
                f"Distribution lead times (N={len(lead_times)} evenements detectes)",
                fontsize=9
            )
            ax3.legend(fontsize=8)
            ax3.set_xticks(range(0, max_lead + 1))
        else:
            ax3.text(0.5, 0.5, "Aucun evenement detecte",
                     transform=ax3.transAxes, ha="center", va="center", fontsize=10)
            ax3.set_title("Distribution des lead times", fontsize=9)
        ax3.grid(True, alpha=0.3, axis="y")

        # Panneau 4 : Courbe Precision & Recall vs seuil de lead time ─────────
        ax4 = axes[1, 1]
        if len(pr_df) > 0:
            ax4.plot(pr_df["lead_threshold"], pr_df["precision"] * 100,
                     color=_COL_PRE, lw=2.0, marker="o", ms=4,
                     label="Precision (%)")
            ax4.plot(pr_df["lead_threshold"], pr_df["recall"] * 100,
                     color=_COL_SARIMAX, lw=2.0, marker="s", ms=4,
                     label="Recall (%)")
            ax4.axvline(lag_optimal, color="black", lw=1.0, ls="--",
                        alpha=0.55, label=f"Lag optimal={lag_optimal}")
            ax4.fill_between(
                pr_df["lead_threshold"],
                pr_df["precision"] * 100,
                pr_df["recall"] * 100,
                alpha=0.08, color="gray",
            )
            ax4.set_xlabel("Lead threshold admis (mois)", fontsize=8)
            ax4.set_ylabel("Score (%)", fontsize=8)
            ax4.set_title(
                "Precision & Recall selon le lead time autorise", fontsize=9
            )
            ax4.legend(fontsize=8)
            ax4.set_ylim(0, 108)
            ax4.set_xticks(range(1, max_lead + 1))
        ax4.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        path = FIG_DIR / "early_warning_analysis.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"\n  Figure sauvegardee : {path}")
        plt.show()

    # Sauvegarder le rapport CSV des evenements ────────────────────────────────
    ev_rows = (
        [{"onset_date": d, "lead_time_months": lt, "detected": True}
         for d, lt in zip(tp_dates, lead_times)]
        + [{"onset_date": d, "lead_time_months": np.nan, "detected": False}
           for d in fn_dates]
    )
    if ev_rows:
        ev_df = pd.DataFrame(ev_rows).sort_values("onset_date")
        csv_path = REP_DIR / "early_warning_events.csv"
        ev_df.to_csv(csv_path, index=False)
        print(f"  CSV evenements sauvegarde : {csv_path}")

    return {
        "lag_optimal":      lag_optimal,
        "lead_time_mean":   lead_mean,
        "lead_time_median": lead_median,
        "precision":        precision,
        "recall":           recall,
        "f1":               f1,
        "granger_pval":     granger_pval,
        "ccf_values":       dict(zip(ccf_lags, ccf_values)),
        "onset_dates":      onset_dates,
        "lead_times":       lead_times,
        "tp":               tp_events,
        "fp":               fp_events,
        "fn":               fn_events,
        "pr_curve":         pr_df,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MATRICE DE TRANSITION DES ETATS DE STRESS
# ═══════════════════════════════════════════════════════════════════════════════

_STATE_LABELS  = ["Normal", "Warning", "High Stress"]
_STATE_COLORS  = {"Normal": "#2CA02C", "Warning": "#FF7F0E", "High Stress": "#D62728"}
_STATE_MAP     = {"Normal": 0, "Warning": 1, "High Stress": 2}
_STATE_REV_MAP = {0: "Normal", 1: "Warning", 2: "High Stress"}


def stress_transition_matrix(
    stress_levels: pd.Series,
    save_fig: bool = True,
) -> dict:
    """
    Matrice de transition entre etats de stress economique (chaine de Markov).

    Etats : Normal (BESI < 0.35), Warning (0.35-0.65), High Stress (> 0.65)

    Methode
    -------
    - Comptage des transitions observees t -> t+1
    - Normalisation par ligne -> probabilites de transition P(j | i)
    - Statistiques par etat : frequence, duree moyenne/max des episodes
    - Distribution stationnaire : vecteur propre de P associe a lambda=1
    - Heatmap + historique temporel + barres de duree

    Parametres
    ----------
    stress_levels : pd.Series avec valeurs 'Normal', 'Warning', 'High Stress'
                    (ou codes numeriques 0, 1, 2 — compatibles data_pipeline)
    save_fig      : sauvegarder les 3 graphiques

    Retourne
    --------
    dict : transition_matrix (DataFrame proba), count_matrix (DataFrame brut),
           state_stats (dict par etat), steady_state (dict)
    """
    import seaborn as sns
    from matplotlib.patches import Patch

    # ── Encodage des etats ────────────────────────────────────────────────────
    sl = stress_levels.dropna().copy()
    if sl.dtype == object or str(sl.dtype) == "category":
        codes = sl.map(_STATE_MAP)
        # Gerer les labels non reconnus (fallback Normal=0)
        codes = codes.fillna(0).astype(int)
    else:
        codes = sl.astype(int).clip(0, 2)

    n_states = len(_STATE_LABELS)
    vals     = codes.values

    # ── Matrice de comptage ───────────────────────────────────────────────────
    count_mat = np.zeros((n_states, n_states), dtype=int)
    for i in range(len(vals) - 1):
        s_from, s_to = int(vals[i]), int(vals[i + 1])
        if 0 <= s_from < n_states and 0 <= s_to < n_states:
            count_mat[s_from, s_to] += 1

    # ── Normalisation par ligne -> probabilites ───────────────────────────────
    row_sums = count_mat.sum(axis=1, keepdims=True).astype(float)
    row_sums[row_sums == 0] = 1.0          # eviter division par zero
    trans_mat = count_mat / row_sums

    count_df = pd.DataFrame(count_mat, index=_STATE_LABELS, columns=_STATE_LABELS)
    trans_df  = pd.DataFrame(trans_mat, index=_STATE_LABELS, columns=_STATE_LABELS)

    # ── Statistiques par etat ─────────────────────────────────────────────────
    state_stats = {}
    for label in _STATE_LABELS:
        code = _STATE_MAP[label]
        freq = float((codes == code).mean())

        # Duree des episodes : compter les runs consecutifs
        runs, cur_len = [], 0
        for v in vals:
            if v == code:
                cur_len += 1
            else:
                if cur_len > 0:
                    runs.append(cur_len)
                cur_len = 0
        if cur_len > 0:
            runs.append(cur_len)

        state_stats[label] = {
            "frequency_%":  round(freq * 100, 1),
            "n_episodes":   len(runs),
            "avg_duration": round(float(np.mean(runs))  if runs else 0.0, 1),
            "max_duration": int(np.max(runs))  if runs else 0,
            "total_months": int(sum(runs)),
        }

    # ── Distribution stationnaire (vecteur propre, lambda=1) ─────────────────
    steady_state: dict = {}
    try:
        eigvals, eigvecs = np.linalg.eig(trans_mat.T)
        idx_ss  = int(np.argmin(np.abs(eigvals - 1.0)))
        ss_vec  = np.real(eigvecs[:, idx_ss])
        ss_vec  = ss_vec / ss_vec.sum()
        steady_state = {
            lbl: round(float(ss_vec[i]), 4)
            for i, lbl in enumerate(_STATE_LABELS)
        }
    except Exception:
        steady_state = {lbl: np.nan for lbl in _STATE_LABELS}

    # ── Affichage console ─────────────────────────────────────────────────────
    sep = "=" * 60
    print(f"\n{sep}")
    print("  MATRICE DE TRANSITION DES ETATS DE STRESS ECONOMIQUE")
    print(f"  Nombre d'observations : {len(vals)} mois")
    print(sep)

    # Matrice de comptage
    print("\n  Matrice de comptage (transitions observees) :")
    hdr = f"  {'':16s}" + "".join(f"  {c:>12s}" for c in _STATE_LABELS)
    print(hdr)
    for row_l in _STATE_LABELS:
        row_str = f"  {row_l:<16s}"
        row_str += "".join(f"  {count_df.loc[row_l, col]:>12d}" for col in _STATE_LABELS)
        print(row_str)

    # Matrice de probabilites
    print("\n  Matrice de transition (probabilites P(j|i)) :")
    print(hdr)
    for row_l in _STATE_LABELS:
        row_str = f"  {row_l:<16s}"
        row_str += "".join(f"  {trans_df.loc[row_l, col]:>12.3f}" for col in _STATE_LABELS)
        print(row_str)

    # Statistiques par etat
    print("\n  Statistiques par etat :")
    print(f"  {'Etat':<14}  {'Freq%':>7}  {'Episodes':>9}  {'Dur.moy':>8}  {'Dur.max':>8}")
    print(f"  {'-'*54}")
    for label in _STATE_LABELS:
        st = state_stats[label]
        print(f"  {label:<14}  {st['frequency_%']:>6.1f}%  "
              f"{st['n_episodes']:>9d}  "
              f"{st['avg_duration']:>7.1f}m  "
              f"{st['max_duration']:>7d}m")

    # Distribution stationnaire
    print("\n  Distribution stationnaire (chaine de Markov) :")
    for label in _STATE_LABELS:
        prob = steady_state.get(label, np.nan)
        if not np.isnan(prob):
            bar = "#" * max(0, int(prob * 35))
            print(f"    {label:<14} : {prob:.3f}  {bar}")
        else:
            print(f"    {label:<14} : n/a")
    print(sep)

    # ── Figures ───────────────────────────────────────────────────────────────
    if save_fig:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(
            "Matrice de Transition des Etats de Stress Economique -- BESI Maroc",
            fontsize=10, fontweight="bold",
        )

        # Panneau 1 : Heatmap des probabilites de transition ──────────────────
        ax1 = axes[0]
        sns.heatmap(
            trans_df,
            annot=True, fmt=".3f",
            cmap="YlOrRd",
            linewidths=0.6, linecolor="white",
            cbar_kws={"label": "Probabilite", "shrink": 0.85},
            ax=ax1, vmin=0, vmax=1,
            annot_kws={"size": 10, "weight": "bold"},
        )
        ax1.set_title("Probabilites de transition\n(Chaine de Markov)", fontsize=9)
        ax1.set_xlabel("Etat suivant (t+1)", fontsize=8)
        ax1.set_ylabel("Etat actuel (t)", fontsize=8)
        ax1.tick_params(labelsize=8)

        # Panneau 2 : Historique temporel des etats ───────────────────────────
        ax2 = axes[1]
        t_idx = codes.index
        prev_v = None
        seg_start = 0
        for i in range(len(vals)):
            v = int(vals[i])
            if v != prev_v and prev_v is not None:
                lbl = _STATE_REV_MAP.get(prev_v, "Normal")
                ax2.fill_betweenx(
                    [0, 1],
                    [t_idx[seg_start], t_idx[seg_start]],
                    [t_idx[i - 1], t_idx[i - 1]],
                    color=_STATE_COLORS[lbl], alpha=0.82,
                )
                seg_start = i
            prev_v = v
        # Dernier segment
        if prev_v is not None:
            lbl = _STATE_REV_MAP.get(prev_v, "Normal")
            ax2.fill_betweenx(
                [0, 1],
                [t_idx[seg_start], t_idx[seg_start]],
                [t_idx[-1], t_idx[-1]],
                color=_STATE_COLORS[lbl], alpha=0.82,
            )
        legend_patches = [
            Patch(
                color=_STATE_COLORS[l],
                label=f"{l} ({state_stats[l]['frequency_%']:.1f}%)"
            )
            for l in _STATE_LABELS
        ]
        ax2.legend(handles=legend_patches, fontsize=7.5, loc="upper left")
        ax2.set_yticks([])
        ax2.set_title("Historique des etats de stress", fontsize=9)
        ax2.set_xlabel("Date", fontsize=8)
        ax2.tick_params(axis="x", labelsize=7, rotation=30)
        ax2.grid(True, axis="x", alpha=0.2)
        ax2.set_xlim(t_idx[0], t_idx[-1])
        ax2.set_ylim(0, 1)

        # Panneau 3 : Duree moyenne et distribution stationnaire ──────────────
        ax3  = axes[2]
        ax3b = ax3.twinx()

        labels_bar = _STATE_LABELS
        avg_durs   = [state_stats[l]["avg_duration"]  for l in labels_bar]
        ss_probs   = [steady_state.get(l, 0) * 100    for l in labels_bar]
        colors_bar = [_STATE_COLORS[l]                for l in labels_bar]
        x_pos      = np.arange(len(labels_bar))
        w          = 0.38

        bars1 = ax3.bar(x_pos - w/2, avg_durs, width=w,
                        color=colors_bar, alpha=0.85, label="Duree moy. (mois)")
        bars2 = ax3b.bar(x_pos + w/2, ss_probs, width=w,
                         color=colors_bar, alpha=0.45, hatch="//",
                         label="Etat stationnaire (%)")

        for bar, d, n in zip(
            bars1, avg_durs,
            [state_stats[l]["n_episodes"] for l in labels_bar]
        ):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                d + 0.05,
                f"{d:.1f}m\nN={n}",
                ha="center", fontsize=7.5,
            )
        for bar, p in zip(bars2, ss_probs):
            ax3b.text(
                bar.get_x() + bar.get_width() / 2,
                p + 0.5,
                f"{p:.1f}%",
                ha="center", fontsize=7.5, color="gray",
            )

        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(labels_bar, fontsize=8)
        ax3.set_ylabel("Duree moyenne (mois)", fontsize=8)
        ax3b.set_ylabel("Etat stationnaire (%)", fontsize=8, color="gray")
        ax3.set_title("Duree des episodes & distribution\nstationnaire", fontsize=9)
        lines1, labs1 = ax3.get_legend_handles_labels()
        lines2, labs2 = ax3b.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labs1 + labs2, fontsize=7.5)
        ax3.grid(True, alpha=0.3, axis="y")

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        path = FIG_DIR / "stress_transition_matrix.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"\n  Figure sauvegardee : {path}")
        plt.show()

    # CSV matrices ─────────────────────────────────────────────────────────────
    csv_trans = REP_DIR / "stress_transition_matrix.csv"
    csv_count = REP_DIR / "stress_count_matrix.csv"
    trans_df.to_csv(csv_trans)
    count_df.to_csv(csv_count)
    print(f"  CSV probabilites sauvegarde : {csv_trans}")
    print(f"  CSV comptages sauvegarde    : {csv_count}")

    return {
        "transition_matrix": trans_df,
        "count_matrix":      count_df,
        "state_stats":       state_stats,
        "steady_state":      steady_state,
    }


# ─── Point d'entrée ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    from pathlib import Path as _P
    _root   = _P(__file__).resolve().parent.parent
    _ipc    = _root / "data" / "processed" / "ipc_processed.csv"
    _master = _root / "data" / "processed" / "master_dataset.csv"

    if _ipc.exists() and _master.exists():
        df_ipc    = pd.read_csv(_ipc,    parse_dates=["date"], index_col="date")
        df_master = pd.read_csv(_master, parse_dates=["date"], index_col="date")
        df_ipc.index.freq = df_master.index.freq = "MS"

        ipc  = df_ipc["ipc"]
        exog = df_master[["besi"]]

        # Récupérer les ordres depuis un fit préalable (ou utiliser les défauts)
        try:
            from src.models import fit_sarima_baseline
            _, pdq, PDQ = fit_sarima_baseline(ipc, train_end="2021-12-01", save_fig=False)
            orders = (pdq, PDQ)
        except Exception:
            orders = ((1, 1, 1), (0, 1, 1))

        print("\n>>> Test de Chow (rupture 2022)")
        result = chow_test(ipc, exog=exog, breakpoint="2022-01-01", save_fig=True)
        print(f"\nF = {result['f_stat']:.4f}  p = {result['p_value']:.6f}  "
              f"is_break = {result['is_break']}")

        print("\n>>> Performances par periode")
        df_perf = period_performance(
            ipc, exog=exog, orders=orders, save_fig=True
        )
        print(df_perf.to_string())

        print("\n>>> Early warning analysis")
        besi_s = df_master["besi"]
        ew = early_warning_analysis(besi_s, ipc, save_fig=True)
        print(f"\nLag optimal   : {ew['lag_optimal']} mois")
        print(f"Lead time moy : {ew['lead_time_mean']:.1f} mois")
        print(f"Precision     : {ew['precision']:.3f}")
        print(f"Recall        : {ew['recall']:.3f}")
        print(f"F1            : {ew['f1']:.3f}")
        print(f"Granger p-val : {ew['granger_pval']:.4f}")

        print("\n>>> Matrice de transition des etats de stress")
        stress_s = df_master["stress_level"]
        tm = stress_transition_matrix(stress_s, save_fig=True)
        print("\nDistribution stationnaire :")
        for lbl, prob in tm["steady_state"].items():
            print(f"  {lbl:<14} : {prob:.3f}")
    else:
        print("Fichiers manquants -- lancer d'abord : python src/data_pipeline.py")
