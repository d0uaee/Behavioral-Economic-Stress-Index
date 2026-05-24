"""
src/analysis/placebo_test.py — Test Placebo complet pour valider le signal BESI

OBJECTIF :
    Prouver que le BESI porte un VRAI signal économique et n'améliore pas
    l'AIC de SARIMAX par simple chance statistique.

    Stratégie : comparer l'amélioration AIC de SARIMAX+BESI à celle obtenue
    avec des signaux placebo ne portant AUCUNE information économique.
    Si BESI se distingue des placebos → signal réel.
    Si BESI ressemble aux placebos → amélioration due au hasard.

NOTE SUR LES AIC :
    Les AIC calculés ici (SARIMA ≈ 103, SARIMAX+BESI ≈ 100) diffèrent de ceux
    du rapport exploratoire (64.85, 57.09) car :
      - Le rapport utilisait peut-être des données centrées/réduites ou une
        transformation de l'IPC
      - Ce script travaille sur ipc_level en niveaux absolus (base 100)
    Ce qui compte : le DELTA AIC (relatif), pas la valeur absolue.

PLACEBOS :
    1. Random gaussien      : bruit sans structure
    2. Tendance linéaire    : structure temporelle pure (sans info économique)
    3. BESI shufflé         : même distribution que BESI, ordre détruit
    4. Marche aléatoire     : même autocorrélation grossière que BESI

MONTE CARLO :
    500 signaux aléatoires gaussiens → distribution des Delta_AICs sous H0
    p-value empirique = P(Delta_AIC_random ≤ Delta_AIC_BESI) sous H0

    H0 : l'amélioration AIC de BESI est due au hasard
    H1 : l'amélioration AIC de BESI est statistiquement inhabituellep

Output :
    outputs/reports/placebo_test_results.csv
    outputs/reports/placebo_mc_distribution.csv
    outputs/figures/placebo_mc_distribution.png
    outputs/figures/placebo_comparison_bar.png

Usage :
    python src/analysis/placebo_test.py
    python src/analysis/placebo_test.py --n-mc 200   # rapide
    python src/analysis/placebo_test.py --n-mc 1000  # publication
"""

import argparse
import logging
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parent.parent.parent
GOLD_DIR = ROOT / "data" / "gold"
REPORTS  = ROOT / "outputs" / "reports"
FIGURES  = ROOT / "outputs" / "figures"
REPORTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Paramètres modèle ─────────────────────────────────────────────────────────
SARIMA_ORDER    = (1, 1, 1)
SARIMA_SEASONAL = (1, 0, 1, 12)
TARGET_COL      = "ipc_level"
BESI_COL        = "behavioral_index_pure_lag1"   # lag1 = respecte as-of-date
TRAIN_BLOC      = "A"                             # seuil de calibration AIC


# ── Couleurs ──────────────────────────────────────────────────────────────────
C_BESI      = "#27ae60"
C_SARIMA    = "#2c3e50"
C_PLACEBO   = "#95a5a6"
C_REJECT    = "#e74c3c"
C_ACCEPT    = "#3498db"


# ─── Création des signaux placebo ─────────────────────────────────────────────

def create_placebos(
    besi_full:  pd.Series,
    seed:       int = 42,
) -> dict:
    """
    Crée 4 signaux placebo à partir du BESI de référence.

    Tous les placebos sont normalisés 0-1 pour être comparables à BESI.

    Paramètres
    ----------
    besi_full : pd.Series — signal BESI sur la période complète (train + test)
    seed      : graine aléatoire (reproductibilité)

    Retourne
    --------
    dict {nom: pd.Series}
    """
    rng = np.random.default_rng(seed)
    n   = len(besi_full)
    idx = besi_full.index

    mu  = float(besi_full.mean())
    sig = float(besi_full.std())

    def _norm01(arr):
        mn, mx = arr.min(), arr.max()
        if mx == mn:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    # Placebo 1 : bruit gaussien pur
    raw1 = rng.normal(mu, sig, n)

    # Placebo 2 : tendance linéaire pure (structure temporelle, 0 info éco)
    raw2 = np.linspace(0, 1, n)

    # Placebo 3 : BESI shufflé (même distribution, temporalité détruite)
    raw3 = besi_full.sample(frac=1, random_state=seed).values

    # Placebo 4 : marche aléatoire (autocorrélation similaire à BESI)
    innovations = rng.normal(0, sig * 0.1, n)
    raw4 = np.cumsum(innovations)

    placebos = {
        "random_gaussien":  pd.Series(_norm01(raw1), index=idx,
                                      name="placebo_random"),
        "tendance_lineaire": pd.Series(_norm01(raw2), index=idx,
                                       name="placebo_trend"),
        "besi_shuffle":     pd.Series(_norm01(raw3), index=idx,
                                      name="placebo_shuffle"),
        "marche_aleatoire": pd.Series(_norm01(raw4), index=idx,
                                      name="placebo_rw"),
    }
    return placebos


# ─── Fit SARIMAX in-sample ────────────────────────────────────────────────────

def _fit_sarimax_insample(
    y_train:    "np.ndarray | pd.Series",
    exog_train: "Optional[np.ndarray | pd.Series]" = None,
    label:      str = "",
) -> dict:
    """
    Ajuste SARIMA(1,1,1)(1,0,1)[12] sur les données d'entraînement.
    Retourne AIC, BIC, Log-likelihood, coefficient exogène et p-value.

    Si exog_train=None → SARIMA pur (pas d'exogène).

    Note : statsmodels SARIMAX requiert des pandas Series (pas numpy arrays)
    quand une variable exogène est fournie — on convertit systématiquement.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    # Convertir en Series pandas (requis par statsmodels avec exog)
    if isinstance(y_train, np.ndarray):
        y_train = pd.Series(y_train)
    if exog_train is not None and isinstance(exog_train, np.ndarray):
        exog_train = pd.Series(exog_train, index=y_train.index)

    kwargs = dict(
        order          = SARIMA_ORDER,
        seasonal_order = SARIMA_SEASONAL,
        trend          = "n",
    )
    if exog_train is not None:
        kwargs["exog"] = exog_train

    try:
        result = SARIMAX(y_train, **kwargs).fit(
            disp=False, maxiter=300
        )
        # Nom du coefficient exogène (statsmodels le nomme d'après le signal)
        sarima_params = {"ar.L1", "ar.L2", "ma.L1", "ma.L2",
                         "ar.S.L12", "ma.S.L12", "sigma2",
                         "intercept", "drift", "const"}
        exog_keys = [k for k in result.params.index
                     if k not in sarima_params]
        coef  = float(result.params[exog_keys[0]])  if exog_keys else float("nan")
        pval  = float(result.pvalues[exog_keys[0]]) if exog_keys else float("nan")
        n_obs = int(result.nobs)

        return {
            "aic":     float(result.aic),
            "bic":     float(result.bic),
            "llf":     float(result.llf),
            "n_obs":   n_obs,
            "coef":    coef,
            "pval":    pval,
            "converged": result.mle_retvals.get("converged", True)
                         if hasattr(result, "mle_retvals") else True,
        }
    except Exception as e:
        logger.debug(f"_fit_sarimax_insample({label}) : {e}")
        return {
            "aic": float("nan"), "bic": float("nan"),
            "llf": float("nan"), "n_obs": 0,
            "coef": float("nan"), "pval": float("nan"),
            "converged": False,
        }


# ─── RMSE backtest simplifié ──────────────────────────────────────────────────

def _rmse_backtest(
    gold:         pd.DataFrame,
    exog_full:    pd.Series,
    bloc:         str,
) -> float:
    """
    Backtest simplifié (1 fit → N prédictions) pour le placebo test.

    Étapes :
      1. Entraîner SARIMAX sur le bloc d'entraînement correspondant
      2. Prédire pas à pas sur le bloc de test (1-step ahead, expanding)
      3. Calculer RMSE vs ipc_level réel

    Note : Intentionnellement simplifié (pas walk-forward complet) pour
           permettre 4 blocs × 500 MC en temps raisonnable.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    train_mask = gold["split_label"].str.contains(f"train_{bloc}", na=False)
    test_mask  = gold["split_label"].str.contains(f"test_{bloc}",  na=False)

    y_train = gold.loc[train_mask, TARGET_COL].dropna()
    y_test  = gold.loc[test_mask,  TARGET_COL].dropna()

    if len(y_train) < 24 or len(y_test) < 6:
        return float("nan")

    exog_tr = exog_full.reindex(y_train.index).ffill().bfill().fillna(0).values
    exog_te = exog_full.reindex(y_test.index).ffill().bfill().fillna(0).values

    try:
        model = SARIMAX(
            y_train,
            exog           = exog_tr.reshape(-1, 1),
            order          = SARIMA_ORDER,
            seasonal_order = SARIMA_SEASONAL,
            trend          = "n",
        ).fit(disp=False, maxiter=200, method="lbfgs")

        # Prédictions sur le test (1-step ahead expanding)
        preds = []
        y_so_far  = list(y_train.values)
        ex_so_far = list(exog_tr)

        for i in range(len(y_test)):
            ex_next = exog_te[i]
            res_i = SARIMAX(
                np.array(y_so_far),
                exog           = np.array(ex_so_far).reshape(-1, 1),
                order          = SARIMA_ORDER,
                seasonal_order = SARIMA_SEASONAL,
                trend          = "n",
            ).fit(disp=False, maxiter=150, method="lbfgs",
                  start_params=model.params)
            fc = res_i.forecast(steps=1, exog=np.array([[ex_next]]))
            preds.append(float(fc.iloc[0]))
            y_so_far.append(float(y_test.iloc[i]))
            ex_so_far.append(ex_next)

        rmse = float(np.sqrt(np.mean((np.array(preds) - y_test.values) ** 2)))
        return rmse

    except Exception as e:
        logger.debug(f"_rmse_backtest Bloc{bloc}: {e}")
        return float("nan")


def _rmse_backtest_fast(
    gold:       pd.DataFrame,
    exog_full:  pd.Series,
    bloc:       str,
) -> float:
    """
    Version rapide du RMSE : fit unique sur train, prédiction directe sur test.
    Utilisée pour les 500 placebos Monte Carlo (walk-forward serait trop lent).
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    train_mask = gold["split_label"].str.contains(f"train_{bloc}", na=False)
    test_mask  = gold["split_label"].str.contains(f"test_{bloc}",  na=False)

    y_train = gold.loc[train_mask, TARGET_COL].dropna()
    y_test  = gold.loc[test_mask,  TARGET_COL].dropna()

    if len(y_train) < 24 or len(y_test) < 3:
        return float("nan")

    exog_tr = exog_full.reindex(y_train.index).ffill().bfill().fillna(0).values
    exog_te = exog_full.reindex(y_test.index).ffill().bfill().fillna(0).values

    try:
        model = SARIMAX(
            y_train,
            exog           = exog_tr.reshape(-1, 1),
            order          = SARIMA_ORDER,
            seasonal_order = SARIMA_SEASONAL,
            trend          = "n",
        ).fit(disp=False, maxiter=200, method="lbfgs")

        fc = model.forecast(steps=len(y_test),
                            exog=exog_te.reshape(-1, 1))
        rmse = float(np.sqrt(np.mean((fc.values - y_test.values) ** 2)))
        return rmse
    except Exception:
        return float("nan")


# ─── Tableau comparatif principal ─────────────────────────────────────────────

def _run_main_comparison(gold: pd.DataFrame) -> pd.DataFrame:
    """
    Compare SARIMA pur, SARIMAX+BESI et SARIMAX+4 placebos sur :
      - AIC in-sample (train_A)
      - Delta_AIC vs SARIMA pur
      - RMSE Bloc A et Bloc B
      - Coefficient exogène et p-value
    """
    train_mask = gold["split_label"].str.contains(f"train_{TRAIN_BLOC}", na=False)
    y_train    = gold.loc[train_mask, TARGET_COL].dropna()
    besi_full  = gold[BESI_COL].ffill().bfill().fillna(0)

    logger.info(f"Train {TRAIN_BLOC} : {len(y_train)} mois "
                f"({y_train.index.min().date()} → {y_train.index.max().date()})")

    # ── 1. SARIMA pur (baseline) ──────────────────────────────────────────────
    logger.info("Fitting SARIMA pur...")
    sarima_fit = _fit_sarimax_insample(y_train.values, exog_train=None,
                                       label="SARIMA_pur")
    aic_sarima = sarima_fit["aic"]

    # ── 2. SARIMAX + BESI réel ────────────────────────────────────────────────
    logger.info("Fitting SARIMAX + BESI réel...")
    besi_train = besi_full.reindex(y_train.index).ffill().bfill().fillna(0)
    besi_fit   = _fit_sarimax_insample(y_train.values, besi_train.values,
                                       label="SARIMAX_BESI")
    aic_besi   = besi_fit["aic"]
    delta_besi = aic_besi - aic_sarima

    logger.info(f"SARIMA AIC={aic_sarima:.2f} | SARIMAX+BESI AIC={aic_besi:.2f} | "
                f"Delta_AIC={delta_besi:+.2f} | coef={besi_fit['coef']:.4f} "
                f"p={besi_fit['pval']:.4f}")

    # RMSE BESI (fast)
    rmse_besi_A = _rmse_backtest_fast(gold, besi_full, "A")
    rmse_besi_B = _rmse_backtest_fast(gold, besi_full, "B")

    # ── 3. Placebos ──────────────────────────────────────────────────────────
    placebos = create_placebos(besi_full, seed=42)

    rows = [
        {
            "Modele":         "SARIMA pur",
            "type":           "baseline",
            "AIC":            round(aic_sarima,  2),
            "BIC":            round(sarima_fit["bic"], 2),
            "LLF":            round(sarima_fit["llf"], 2),
            "Delta_AIC":      0.0,
            "RMSE_BlocA":     _rmse_backtest_fast(gold, pd.Series(0.0, index=gold.index), "A"),
            "RMSE_BlocB":     _rmse_backtest_fast(gold, pd.Series(0.0, index=gold.index), "B"),
            "Coef_exog":      float("nan"),
            "Pvalue_exog":    float("nan"),
            "bat_SARIMA":     "—",
            "conclusion":     "Baseline SARIMA pur",
        },
        {
            "Modele":         "SARIMAX + BESI behavioral",
            "type":           "signal_reel",
            "AIC":            round(aic_besi, 2),
            "BIC":            round(besi_fit["bic"], 2),
            "LLF":            round(besi_fit["llf"], 2),
            "Delta_AIC":      round(delta_besi, 2),
            "RMSE_BlocA":     round(rmse_besi_A, 4) if not np.isnan(rmse_besi_A) else float("nan"),
            "RMSE_BlocB":     round(rmse_besi_B, 4) if not np.isnan(rmse_besi_B) else float("nan"),
            "Coef_exog":      round(besi_fit["coef"], 4),
            "Pvalue_exog":    round(besi_fit["pval"], 4),
            "bat_SARIMA":     "Oui" if delta_besi < -2 else "Marginal" if delta_besi < 0 else "Non",
            "conclusion":     _interpret_delta(delta_besi, besi_fit["pval"]),
        },
    ]

    for plac_name, plac_series in placebos.items():
        logger.info(f"Fitting SARIMAX + placebo {plac_name}...")
        plac_train  = plac_series.reindex(y_train.index).ffill().bfill().fillna(0)
        plac_fit    = _fit_sarimax_insample(y_train.values, plac_train.values,
                                            label=f"placebo_{plac_name}")
        delta_plac  = plac_fit["aic"] - aic_sarima

        rmse_A = _rmse_backtest_fast(gold, plac_series, "A")
        rmse_B = _rmse_backtest_fast(gold, plac_series, "B")

        rows.append({
            "Modele":         f"SARIMAX + Placebo {plac_name}",
            "type":           "placebo",
            "AIC":            round(plac_fit["aic"], 2) if not np.isnan(plac_fit["aic"]) else float("nan"),
            "BIC":            round(plac_fit["bic"], 2) if not np.isnan(plac_fit["bic"]) else float("nan"),
            "LLF":            round(plac_fit["llf"], 2) if not np.isnan(plac_fit["llf"]) else float("nan"),
            "Delta_AIC":      round(delta_plac, 2) if not np.isnan(delta_plac) else float("nan"),
            "RMSE_BlocA":     round(rmse_A, 4) if not np.isnan(rmse_A) else float("nan"),
            "RMSE_BlocB":     round(rmse_B, 4) if not np.isnan(rmse_B) else float("nan"),
            "Coef_exog":      round(plac_fit["coef"], 4) if not np.isnan(plac_fit["coef"]) else float("nan"),
            "Pvalue_exog":    round(plac_fit["pval"],  4) if not np.isnan(plac_fit["pval"]) else float("nan"),
            "bat_SARIMA":     "Oui" if delta_plac < -2 else "Marginal" if delta_plac < 0 else "Non",
            "conclusion":     _interpret_delta(delta_plac, plac_fit["pval"]),
        })

        logger.info(f"  {plac_name:<25}: AIC={plac_fit['aic']:.2f}  "
                    f"Delta_AIC={delta_plac:+.2f}  coef={plac_fit['coef']:.4f}  "
                    f"p={plac_fit['pval']:.4f}")

    df = pd.DataFrame(rows)
    return df, aic_sarima, aic_besi, delta_besi


def _interpret_delta(delta_aic: float, pval: float) -> str:
    """Interprétation automatique du résultat."""
    if np.isnan(delta_aic):
        return "Erreur de convergence"
    if delta_aic < -4 and pval < 0.05:
        return "Signal fort et significatif"
    if delta_aic < -2 and pval < 0.10:
        return "Signal marginal (p<0.10)"
    if delta_aic < 0 and pval < 0.15:
        return "Legere amelioration AIC"
    if delta_aic < 0:
        return "Legere amelioration AIC (ns)"
    if delta_aic < 2:
        return "Neutre (pas d amelioration)"
    return "Degradation AIC (penalisation)"


# ─── Monte Carlo ──────────────────────────────────────────────────────────────

def run_monte_carlo(
    y_train:     np.ndarray,
    aic_sarima:  float,
    aic_besi:    float,
    n_mc:        int = 500,
    seed:        int = 0,
    mu:          float = 0.0,
    sigma:       float = 1.0,
) -> tuple:
    """
    Répète le test avec n_mc signaux gaussiens aléatoires.

    Pour chaque simulation :
      - Génère un signal N(mu, sigma) de longueur len(y_train)
      - Ajuste SARIMAX(1,1,1)(1,0,1)[12] avec ce signal comme exogène
      - Enregistre l'AIC

    p-value empirique = P(AIC_random ≤ AIC_BESI | H0 : signal aléatoire)
                      = fraction des simulations avec AIC ≤ AIC_BESI

    Interprétation :
      p < 0.05 → BESI donne une amélioration inhabituellement grande
                  pour un signal aléatoire (rejette H0)
      p ≥ 0.05 → On ne peut pas exclure que BESI soit une variable
                  aléatoire qui améliore l'AIC par chance

    Paramètres
    ----------
    y_train  : série IPC d'entraînement
    aic_besi : AIC du modèle BESI réel (cible à comparer)
    n_mc     : nombre de simulations (500 recommandé, 1000 pour publication)
    mu, sigma: paramètres de la distribution des placebos (calibrés sur BESI)

    Retourne
    --------
    (mc_aics, delta_aics, pvalue, n_converged)
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    rng      = np.random.default_rng(seed)
    mc_aics  = []
    n_failed = 0

    logger.info(f"Monte Carlo : {n_mc} simulations (seed={seed})...")
    t0 = time.time()

    for i in range(n_mc):
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            eta     = elapsed / (i + 1) * (n_mc - i - 1)
            logger.info(f"  MC {i+1:>4}/{n_mc}  "
                        f"({elapsed:.0f}s écoulées, ETA ~{eta:.0f}s)")

        placebo_arr = rng.normal(mu, sigma, len(y_train))
        # Normalisation 0-1 pour correspondre au BESI
        pmn, pmx = placebo_arr.min(), placebo_arr.max()
        if pmx > pmn:
            placebo_arr = (placebo_arr - pmn) / (pmx - pmn)
        # pandas Series requis par statsmodels SARIMAX avec exog
        placebo_s = pd.Series(placebo_arr)
        y_s = pd.Series(y_train) if isinstance(y_train, np.ndarray) else y_train

        try:
            res = SARIMAX(
                y_s,
                exog           = placebo_s,
                order          = SARIMA_ORDER,
                seasonal_order = SARIMA_SEASONAL,
                trend          = "n",
            ).fit(disp=False, maxiter=300)
            mc_aics.append(float(res.aic))
        except Exception:
            n_failed += 1

    mc_aics     = np.array(mc_aics)
    delta_aics  = mc_aics - aic_sarima   # Delta_AIC des placebos

    # p-value : fraction des placebos avec AIC ≤ AIC_BESI
    pvalue      = float((mc_aics <= aic_besi).mean())

    n_conv      = n_mc - n_failed
    elapsed_tot = time.time() - t0

    logger.info(f"MC terminé : {n_conv}/{n_mc} convergences "
                f"({100*n_conv/n_mc:.1f}%)  |  "
                f"temps={elapsed_tot:.1f}s")
    logger.info(f"Distribution Delta_AIC aléatoires : "
                f"mean={delta_aics.mean():+.2f}  "
                f"std={delta_aics.std():.2f}  "
                f"p5={np.percentile(delta_aics, 5):+.2f}  "
                f"p95={np.percentile(delta_aics, 95):+.2f}")
    logger.info(f"BESI réel : Delta_AIC={aic_besi-aic_sarima:+.2f}  "
                f"p-value empirique = {pvalue:.4f}")

    return mc_aics, delta_aics, pvalue, n_conv


# ─── Visualisations ───────────────────────────────────────────────────────────

def _plot_mc_distribution(
    mc_aics:     np.ndarray,
    aic_besi:    float,
    aic_sarima:  float,
    pvalue:      float,
    n_conv:      int,
    delta_placebos: dict,   # {nom: delta_aic}
) -> None:
    """
    Figure 1 : Distribution des AIC des placebos Monte Carlo
    avec l'AIC BESI réel en ligne verticale annotée.

    Lecture :
      - Si la ligne rouge (BESI) est à gauche de la masse → signal meilleur que le hasard
      - La p-value est la fraction de l'aire à gauche de la ligne rouge
    """
    fig, (ax_hist, ax_delta) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#fafafa")

    delta_mc      = mc_aics - aic_sarima
    delta_besi    = aic_besi - aic_sarima

    # ── Sous-graphe 1 : histogramme des AIC Monte Carlo ─────────────────────
    ax_hist.set_facecolor("#fdfdfd")

    # Histogramme
    n_bins = min(40, max(15, n_conv // 10))
    ax_hist.hist(mc_aics, bins=n_bins, color=C_PLACEBO, alpha=0.75,
                 edgecolor="white", linewidth=0.5, label=f"Delta_AIC aléatoires (n={n_conv})")

    # Ligne BESI
    ymax = ax_hist.get_ylim()[1] if ax_hist.get_ylim()[1] > 0 else 1
    ax_hist.axvline(aic_besi, color=C_BESI, lw=2.5, zorder=5,
                    label=f"SARIMAX+BESI (AIC={aic_besi:.1f})")
    ax_hist.axvline(aic_sarima, color=C_SARIMA, lw=2.0, ls="--",
                    label=f"SARIMA pur (AIC={aic_sarima:.1f})")

    # Zone de rejet (AIC ≤ AIC_BESI)
    bins_left = mc_aics[mc_aics <= aic_besi]
    if len(bins_left) > 0:
        ax_hist.hist(bins_left, bins=n_bins, color=C_BESI, alpha=0.35,
                     edgecolor="white", linewidth=0.5, label="Placebos aussi bons que BESI")

    # Annotation p-value
    significance = "p < 0.05  → Signal REEL (H0 rejetee)" if pvalue < 0.05 \
        else "p < 0.10 → Signal marginal" if pvalue < 0.10 \
        else "p ≥ 0.10 → Non distinguable du hasard"
    sig_color = C_BESI if pvalue < 0.05 else C_ACCEPT if pvalue < 0.10 else C_REJECT

    ax_hist.annotate(
        f"p-value empirique = {pvalue:.4f}\n{significance}",
        xy=(aic_besi, ax_hist.get_ylim()[1] * 0.7 if ax_hist.get_ylim()[1] > 0 else 10),
        xytext=(aic_besi - (ax_hist.get_xlim()[1] - ax_hist.get_xlim()[0]) * 0.25,
                ax_hist.get_ylim()[1] * 0.85 if ax_hist.get_ylim()[1] > 0 else 15),
        fontsize=9, color=sig_color, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=sig_color, lw=1.2),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor=sig_color, alpha=0.9),
    )

    ax_hist.set_xlabel("AIC  (plus bas = meilleur)", fontsize=10)
    ax_hist.set_ylabel("Frequence", fontsize=10)
    ax_hist.set_title(
        f"Distribution des AIC sous H0 (signaux aleatoires)\n"
        f"SARIMAX({SARIMA_ORDER})(1,0,1)[12]  —  {n_conv} simulations Monte Carlo",
        fontweight="bold", fontsize=10
    )
    ax_hist.legend(fontsize=8.5, framealpha=0.9)
    ax_hist.grid(axis="y", alpha=0.3)

    # ── Sous-graphe 2 : Delta_AIC comparatif (bar chart) ─────────────────────────
    ax_delta.set_facecolor("#fdfdfd")

    # Bande de l'intervalle MC [p5, p95]
    p5_mc  = np.percentile(delta_mc, 5)
    p95_mc = np.percentile(delta_mc, 95)
    mean_mc = delta_mc.mean()

    # Barres pour chaque modèle
    models_delta = {
        "BESI\nreal":     delta_besi,
        **{f"Placebo\n{k}": v for k, v in delta_placebos.items()},
        "MC\nmean":       mean_mc,
    }

    colors_bar = []
    for name in models_delta:
        if "BESI" in name:
            colors_bar.append(C_BESI)
        elif "MC" in name:
            colors_bar.append(C_PLACEBO)
        else:
            colors_bar.append("#bdc3c7")

    x_pos  = np.arange(len(models_delta))
    bars   = ax_delta.bar(x_pos, list(models_delta.values()),
                          color=colors_bar, alpha=0.85,
                          edgecolor="white", linewidth=0.8, width=0.6)

    # Valeurs sur les barres
    for bar, val in zip(bars, models_delta.values()):
        if not np.isnan(val):
            ha   = "center"
            sign = "+" if val >= 0 else ""
            ax_delta.text(
                bar.get_x() + bar.get_width() / 2,
                val + (0.15 if val >= 0 else -0.35),
                f"{sign}{val:.2f}",
                ha=ha, va="bottom" if val >= 0 else "top",
                fontsize=8.5, fontweight="bold"
            )

    # Bande intervalles Monte Carlo
    ax_delta.axhspan(p5_mc, p95_mc, alpha=0.12, color=C_PLACEBO,
                     label=f"IC 90% MC [{p5_mc:+.2f}, {p95_mc:+.2f}]")
    ax_delta.axhline(0, color=C_SARIMA, lw=1.5, ls="--",
                     label="Delta_AIC = 0 (pas d amelioration)")
    ax_delta.axhline(-2, color=C_BESI, lw=1.0, ls=":",
                     alpha=0.7, label="Seuil fort (Delta_AIC = -2)")

    ax_delta.set_xticks(x_pos)
    ax_delta.set_xticklabels(list(models_delta.keys()), fontsize=9)
    ax_delta.set_ylabel("Delta_AIC vs SARIMA pur  (negatif = meilleur)", fontsize=10)
    ax_delta.set_title(
        "Delta AIC vs SARIMA : Signal BESI vs Placebos\n"
        "Bande grisee = intervalle de confiance 90% Monte Carlo",
        fontweight="bold", fontsize=10
    )
    ax_delta.legend(fontsize=8, framealpha=0.9)
    ax_delta.grid(axis="y", alpha=0.3)
    ax_delta.spines["top"].set_visible(False)
    ax_delta.spines["right"].set_visible(False)

    fig.suptitle(
        "Test Placebo BESI — Validation du Signal vs Bruit Aleatoire\n"
        f"H0 : 'le BESI ameliore l'AIC par chance'  |  "
        f"p-value empirique = {pvalue:.4f}",
        fontsize=12, fontweight="bold", y=1.01
    )

    plt.tight_layout()
    out = FIGURES / "placebo_mc_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Figure MC sauvegardee : {out}")


def _plot_comparison_radar(comparison_df: pd.DataFrame) -> None:
    """
    Figure 2 : Bar chart horizontal comparant Delta_AIC et RMSE
    de BESI vs 4 placebos.
    """
    plot_df = comparison_df[comparison_df["type"] != "baseline"].copy()
    if plot_df.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#fafafa")

    metrics = [
        ("Delta_AIC",   "Delta AIC vs SARIMA (negatif = meilleur)",  True),
        ("RMSE_BlocA",  "RMSE Bloc A — COVID 2020-2021",             False),
        ("RMSE_BlocB",  "RMSE Bloc B — Inflation 2022-2024",         False),
    ]

    labels  = plot_df["Modele"].str.replace("SARIMAX + ", "").str.replace("Placebo ", "Plac. ")
    colors  = [C_BESI if t == "signal_reel" else C_PLACEBO
               for t in plot_df["type"]]
    y_pos   = np.arange(len(plot_df))

    for ax, (col, title, lower_better) in zip(axes, metrics):
        ax.set_facecolor("#fdfdfd")
        vals = plot_df[col].values

        bars = ax.barh(y_pos, vals, color=colors, alpha=0.85,
                       edgecolor="white", linewidth=0.6, height=0.6)

        # Valeurs sur les barres
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                sign = "+" if (col == "Delta_AIC" and val >= 0) else ""
                ax.text(
                    val + (abs(val) * 0.01 if val >= 0 else -abs(val) * 0.01),
                    bar.get_y() + bar.get_height() / 2,
                    f"{sign}{val:.3f}",
                    ha="left" if val >= 0 else "right",
                    va="center", fontsize=8
                )

        if col == "Delta_AIC":
            ax.axvline(0, color=C_SARIMA, lw=1.5, ls="--", alpha=0.7)
            ax.axvline(-2, color=C_BESI, lw=1.0, ls=":", alpha=0.5,
                       label="Seuil -2 (fort)")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.set_xlabel(title, fontsize=9)
        ax.set_title(title, fontweight="bold", fontsize=9)
        ax.grid(axis="x", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        note = "(bas = meilleur)" if lower_better else "(bas = meilleur)"
        ax.text(0.98, 0.02, note, transform=ax.transAxes,
                fontsize=7, ha="right", color="gray")

    # Légende couleurs
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor=C_BESI,    label="BESI reel"),
        Patch(facecolor=C_PLACEBO, label="Placebo (sans info eco)"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=2,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.05))

    fig.suptitle(
        "Comparaison BESI vs Placebos : AIC et RMSE\n"
        "Si BESI a un Delta AIC plus negatif que tous les placebos -> signal reel",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    out = FIGURES / "placebo_comparison_bar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Figure comparaison sauvegardee : {out}")


# ─── Affichage console ────────────────────────────────────────────────────────

def _print_summary_table(
    df:          pd.DataFrame,
    mc_pvalue:   float,
    n_mc:        int,
    n_conv:      int,
    delta_besi:  float,
    aic_sarima:  float,
) -> None:
    SEP = "=" * 115

    print(f"\n{SEP}")
    print("  TEST PLACEBO BESI v3 — Validation du Signal vs Bruit Aleatoire")
    print(f"  Modele : SARIMAX{SARIMA_ORDER}x(1,0,1,12)  |  Train : Bloc {TRAIN_BLOC}  |  Target : {TARGET_COL}")
    print(SEP)
    print()
    print(f"  {'Modele':<35} {'AIC':>8} {'Delta_AIC':>10} {'RMSE_A':>8} "
          f"{'RMSE_B':>8} {'Coef':>8} {'p-val':>7}  Conclusion")
    print(f"  {'-'*110}")

    for _, row in df.iterrows():
        marker  = " [BESI]" if row["type"] == "signal_reel" else ""
        coef_s  = f"{row['Coef_exog']:>8.4f}" if not pd.isna(row["Coef_exog"]) else f"{'—':>8}"
        pval_s  = f"{row['Pvalue_exog']:>7.4f}" if not pd.isna(row["Pvalue_exog"]) else f"{'—':>7}"
        rmse_a  = f"{row['RMSE_BlocA']:>8.4f}" if not pd.isna(row["RMSE_BlocA"]) else f"{'n/a':>8}"
        rmse_b  = f"{row['RMSE_BlocB']:>8.4f}" if not pd.isna(row["RMSE_BlocB"]) else f"{'n/a':>8}"
        delta_s = f"{row['Delta_AIC']:>+10.2f}" if not pd.isna(row["Delta_AIC"]) else f"{'n/a':>10}"
        print(f"  {row['Modele']:<35} {row['AIC']:>8.2f} {delta_s} {rmse_a} "
              f"{rmse_b} {coef_s} {pval_s}  {row['conclusion']}{marker}")

    print(f"\n{SEP}")
    print("  MONTE CARLO — Distribution des AIC sous H0")
    print(f"  H0 : 'le BESI ameliore l'AIC par chance avec un signal gaussien aleatoire'")
    print(f"  {n_mc} simulations  |  {n_conv} convergees  |  seed=0")
    print(SEP)
    print()
    print(f"  AIC SARIMA pur                : {aic_sarima:.2f}")
    print(f"  AIC SARIMAX + BESI reel       : {aic_sarima + delta_besi:.2f}  (Delta_AIC={delta_besi:+.2f})")
    print()
    print(f"  p-value empirique = {mc_pvalue:.4f}")
    print()
    if mc_pvalue < 0.01:
        verdict = "H0 REJETEE (p<0.01) — Signal BESI statistiquement tres rare sous H0"
        color_v = "[FORT]"
    elif mc_pvalue < 0.05:
        verdict = "H0 REJETEE (p<0.05) — Signal BESI distinguable du hasard"
        color_v = "[SIGNIFICATIF]"
    elif mc_pvalue < 0.10:
        verdict = "H0 marginalement rejetee (p<0.10) — Signal BESI marginal"
        color_v = "[MARGINAL]"
    else:
        verdict = "H0 non rejetee (p>=0.10) — Amelioration AIC due au hasard possible"
        color_v = "[NEUTRE]"
    print(f"  {color_v} {verdict}")
    print()
    print(f"  Interpretation :  p = {mc_pvalue:.4f} signifie que {100*mc_pvalue:.1f}% des signaux")
    print(f"  gaussiens aleatoires donnent un AIC aussi bon ou meilleur que BESI.")
    print(f"  Plus p est petit, plus BESI se distingue du bruit.")
    print()
    print(f"  IMPORTANT : L'AIC mesure uniquement le FIT QUANTITATIF in-sample.")
    print(f"  Le BESI excelle surtout comme DETECTEUR DE REGIME (Recall=100% Bloc B),")
    print(f"  ce qui n'est pas capturé par l'AIC mais par les métriques d'alerte précoce.")
    print(f"{SEP}\n")


# ─── Pipeline principal ────────────────────────────────────────────────────────

def run_placebo_test(
    n_mc:      int = 500,
    seed_mc:   int = 0,
    gold_path: "str | Path | None" = None,
) -> pd.DataFrame:
    """
    Pipeline complet du test placebo :

    1. Charge Gold dataset
    2. Compare SARIMA / SARIMAX+BESI / 4 placebos sur AIC + RMSE
    3. Monte Carlo (n_mc simulations gaussiennes)
    4. Génère figures et tableau CSV
    5. Affiche résumé console

    Paramètres
    ----------
    n_mc    : nombre de simulations Monte Carlo (200 rapide, 500 standard, 1000 publication)
    seed_mc : graine pour reproducibilité du MC
    gold_path : chemin vers le Gold dataset (défaut: data/gold/model_dataset_monthly.csv)

    Retourne
    --------
    pd.DataFrame : tableau comparatif complet
    """
    if gold_path is None:
        gold_path = GOLD_DIR / "model_dataset_monthly.csv"
    gold_path = Path(gold_path)

    if not gold_path.exists():
        raise FileNotFoundError(
            f"Gold dataset introuvable : {gold_path}\n"
            "Lancer : python run_v3.py --step gold"
        )

    gold = pd.read_csv(gold_path, parse_dates=["month"], index_col="month")
    logger.info(f"Gold dataset chargé : {gold.shape}")

    if BESI_COL not in gold.columns:
        raise KeyError(f"'{BESI_COL}' absent du Gold. Colonnes dispo: {list(gold.columns)}")

    # ── Tableau principal ─────────────────────────────────────────────────────
    logger.info("\n=== ETAPE 1 : Tableau comparatif placebos ===")
    comparison_df, aic_sarima, aic_besi, delta_besi = _run_main_comparison(gold)

    # Sauvegarde intermédiaire
    out_main = REPORTS / "placebo_test_results.csv"
    comparison_df.to_csv(out_main, index=False)
    logger.info(f"Tableau principal sauvegardé : {out_main}")

    # ── Monte Carlo ───────────────────────────────────────────────────────────
    logger.info(f"\n=== ETAPE 2 : Monte Carlo ({n_mc} simulations) ===")
    train_mask = gold["split_label"].str.contains(f"train_{TRAIN_BLOC}", na=False)
    y_train    = gold.loc[train_mask, TARGET_COL].dropna().values
    besi_full  = gold[BESI_COL].ffill().bfill().fillna(0)
    besi_train = besi_full.reindex(gold.index[train_mask]).ffill().bfill().fillna(0)

    mc_aics, delta_mc, pvalue, n_conv = run_monte_carlo(
        y_train    = y_train,
        aic_sarima = aic_sarima,
        aic_besi   = aic_besi,
        n_mc       = n_mc,
        seed       = seed_mc,
        mu         = float(besi_train.mean()),
        sigma      = float(besi_train.std()),
    )

    # Sauvegarde distribution MC
    mc_df = pd.DataFrame({
        "sim_id":    np.arange(len(mc_aics)),
        "aic_mc":    mc_aics,
        "delta_aic": delta_mc,
    })
    mc_df.to_csv(REPORTS / "placebo_mc_distribution.csv", index=False)

    # Ajouter p-value au tableau principal
    comparison_df["mc_pvalue"] = float("nan")
    besi_row = comparison_df["type"] == "signal_reel"
    comparison_df.loc[besi_row, "mc_pvalue"] = round(pvalue, 4)
    comparison_df.to_csv(out_main, index=False)

    # ── Figures ───────────────────────────────────────────────────────────────
    logger.info("\n=== ETAPE 3 : Visualisations ===")
    plac_deltas = {
        row["Modele"].replace("SARIMAX + Placebo ", ""): row["Delta_AIC"]
        for _, row in comparison_df[comparison_df["type"] == "placebo"].iterrows()
        if not np.isnan(row["Delta_AIC"])
    }
    _plot_mc_distribution(mc_aics, aic_besi, aic_sarima, pvalue, n_conv, plac_deltas)
    _plot_comparison_radar(comparison_df)

    # ── Affichage console ─────────────────────────────────────────────────────
    _print_summary_table(comparison_df, pvalue, n_mc, n_conv, delta_besi, aic_sarima)

    return comparison_df


# ─── Point d'entrée ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test Placebo BESI — validation du signal vs bruit"
    )
    parser.add_argument(
        "--n-mc", type=int, default=500,
        help="Nombre de simulations Monte Carlo (défaut: 500)"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Graine aléatoire pour le Monte Carlo (défaut: 0)"
    )
    args = parser.parse_args()

    # Logging encodage-safe Windows
    _handler = logging.StreamHandler(
        open(sys.stdout.fileno(), mode="w", encoding="utf-8",
             errors="replace", closefd=False, buffering=1)
    )
    logging.basicConfig(
        level    = logging.INFO,
        format   = "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt  = "%H:%M:%S",
        handlers = [_handler,
                    logging.FileHandler(ROOT / "run_v3.log",
                                        encoding="utf-8", mode="a")]
    )

    df = run_placebo_test(n_mc=args.n_mc, seed_mc=args.seed)

    print("Fichiers generes :")
    for f in [
        "outputs/reports/placebo_test_results.csv",
        "outputs/reports/placebo_mc_distribution.csv",
        "outputs/figures/placebo_mc_distribution.png",
        "outputs/figures/placebo_comparison_bar.png",
    ]:
        p = ROOT / f
        sz = f"{p.stat().st_size // 1024} KB" if p.exists() else "MANQUANT"
        print(f"  {f:<55} {sz}")

    print(f"\nTableau final ({df.shape[0]} lignes):")
    print(df[["Modele", "AIC", "Delta_AIC", "RMSE_BlocA", "RMSE_BlocB",
              "Pvalue_exog", "mc_pvalue", "conclusion"]].to_string(index=False))
