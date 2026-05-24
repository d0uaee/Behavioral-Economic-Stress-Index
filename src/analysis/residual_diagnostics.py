"""
Diagnostics des residus SARIMA vs SARIMAX+BESI.

Objectif
--------
Verifier si les residus ressemblent a un bruit blanc, puis comparer
si l'ajout de BESI reduit la structure residuelle.

Sorties
-------
results/residual_diagnostics.csv
results/figures/residual_diagnostics_sarima.png
results/figures/residual_diagnostics_sarimax_besi.png
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - depends on environment
    matplotlib = None
    plt = None

from scipy.stats import jarque_bera, norm, shapiro
from scipy.stats import probplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch, het_breuschpagan
from statsmodels.tsa.statespace.sarimax import SARIMAX


logger = logging.getLogger(__name__)
np.random.seed(42)

ROOT = Path(__file__).resolve().parent.parent.parent
GOLD_PATH = ROOT / "data" / "gold" / "model_dataset_monthly.csv"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = RESULTS_DIR / "residual_diagnostics.csv"

SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL = (1, 0, 1, 12)


def _load_training_data(gold_path: Path = GOLD_PATH) -> tuple[pd.Series, pd.DataFrame]:
    if not gold_path.exists():
        raise FileNotFoundError(
            f"Gold dataset introuvable : {gold_path}\n"
            "Lancer d'abord: python run_v3.py --step gold"
        )

    gold = pd.read_csv(gold_path, parse_dates=["month"], index_col="month").sort_index()

    train_mask = gold["split_label"].str.contains("train_B", na=False)
    if not train_mask.any():
        train_mask = gold.index <= pd.Timestamp("2021-12-01")

    y = gold.loc[train_mask, "ipc_level"].dropna()

    besi_col = "behavioral_index_pure_lag1" if "behavioral_index_pure_lag1" in gold.columns else "behavioral_index_pure"
    if besi_col not in gold.columns:
        raise KeyError(
            "Colonne BESI absente du Gold dataset. Attendu: 'behavioral_index_pure_lag1' ou 'behavioral_index_pure'."
        )

    exog = gold.loc[train_mask, [besi_col]].reindex(y.index).ffill().bfill()

    common_idx = y.index.intersection(exog.dropna().index)
    y = y.loc[common_idx]
    exog = exog.loc[common_idx]

    if len(y) < 36:
        raise ValueError(f"Trop peu d'observations pour diagnostic robuste: {len(y)}")

    return y, exog


def _fit_models(y: pd.Series, exog: pd.DataFrame) -> dict[str, object]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        sarima = SARIMAX(
            y,
            order=SARIMA_ORDER,
            seasonal_order=SARIMA_SEASONAL,
            trend="n",
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=300)

        sarimax_besi = SARIMAX(
            y,
            exog=exog,
            order=SARIMA_ORDER,
            seasonal_order=SARIMA_SEASONAL,
            trend="n",
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=300)

    return {
        "SARIMA": sarima,
        "SARIMAX+BESI": sarimax_besi,
    }


def _safe_shapiro(resid: pd.Series) -> tuple[float, float]:
    sample = resid.dropna().iloc[:5000]
    if len(sample) < 3:
        return float("nan"), float("nan")
    stat, pval = shapiro(sample)
    return float(stat), float(pval)


def _diagnose_one_model(model_name: str, fitted_model, resid: pd.Series) -> tuple[list[dict], str]:
    resid = resid.dropna()
    if len(resid) < 12:
        raise ValueError(f"Residus insuffisants pour {model_name}: {len(resid)}")

    lb = acorr_ljungbox(resid, lags=[6, 12, 24], return_df=True)
    jb_stat, jb_p = jarque_bera(resid)
    sw_stat, sw_p = _safe_shapiro(resid)

    arch_stat, arch_p, _, _ = het_arch(resid, nlags=12)

    exog_bp = getattr(fitted_model.model, "exog", None)
    if exog_bp is None or (isinstance(exog_bp, np.ndarray) and exog_bp.ndim == 1):
        exog_bp = np.column_stack([np.ones(len(resid)), np.arange(len(resid), dtype=float)])

    try:
        bp_stat, bp_p, _, _ = het_breuschpagan(resid.values, exog_bp)
    except Exception:
        bp_stat, bp_p = np.nan, np.nan

    rows: list[dict] = []

    for lag, row in lb.iterrows():
        rows.append(
            {
                "model": model_name,
                "test": f"Ljung-Box lag {int(lag)}",
                "statistic": float(row["lb_stat"]),
                "p_value": float(row["lb_pvalue"]),
            }
        )

    rows.extend(
        [
            {"model": model_name, "test": "Jarque-Bera", "statistic": float(jb_stat), "p_value": float(jb_p)},
            {"model": model_name, "test": "Shapiro-Wilk", "statistic": float(sw_stat), "p_value": float(sw_p)},
            {"model": model_name, "test": "ARCH-LM (lag 12)", "statistic": float(arch_stat), "p_value": float(arch_p)},
            {"model": model_name, "test": "Breusch-Pagan", "statistic": float(bp_stat), "p_value": float(bp_p)},
        ]
    )

    lb_ok = bool((lb["lb_pvalue"] > 0.05).all())
    if lb_ok:
        interpretation = "Residus = bruit blanc OK"
    else:
        interpretation = "Reste de la structure a capturer - augmenter p ou q ?"

    return rows, interpretation


def _plot_diagnostics(resid: pd.Series, model_name: str, output_path: Path) -> None:
    if plt is None:
        logger.warning("matplotlib indisponible - figure ignoree pour %s", model_name)
        return

    resid = resid.dropna()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(f"Diagnostics residus - {model_name}", fontsize=12, fontweight="bold")

    # 1) Residus dans le temps
    axes[0, 0].plot(resid.index, resid.values, color="#2C5F8A", lw=1.0)
    axes[0, 0].axhline(0, color="black", lw=0.7)
    axes[0, 0].set_title("Residus dans le temps")
    axes[0, 0].grid(alpha=0.3)

    # 2) ACF
    plot_acf(resid, lags=24, ax=axes[0, 1], alpha=0.05, color="#E07B39", vlines_kwargs={"colors": "#E07B39"})
    axes[0, 1].set_title("ACF des residus")
    axes[0, 1].grid(alpha=0.3)

    # 3) PACF
    plot_pacf(resid, lags=24, ax=axes[1, 0], alpha=0.05, method="ywm", color="#4CAF50", vlines_kwargs={"colors": "#4CAF50"})
    axes[1, 0].set_title("PACF des residus")
    axes[1, 0].grid(alpha=0.3)

    # 4) Histogramme + Q-Q plot (inset)
    ax = axes[1, 1]
    ax.hist(resid.values, bins=25, density=True, color="#7f8c8d", alpha=0.75, edgecolor="white")
    mu = float(resid.mean())
    sigma = float(resid.std(ddof=1)) if float(resid.std(ddof=1)) > 0 else 1.0
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
    ax.plot(x, norm.pdf(x, mu, sigma), color="#d62728", lw=1.4, label="Normale ajustee")
    ax.set_title("Histogramme + Q-Q plot")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    inset = ax.inset_axes([0.58, 0.08, 0.4, 0.4])
    probplot(resid.values, dist="norm", plot=inset)
    inset.set_title("Q-Q", fontsize=8)
    inset.tick_params(labelsize=7)

    plt.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def run_residual_diagnostics() -> pd.DataFrame:
    y, exog = _load_training_data()
    models = _fit_models(y, exog)

    all_rows: list[dict] = []
    interpretations: dict[str, str] = {}

    for model_name, fit in models.items():
        resid = pd.Series(np.asarray(fit.resid).ravel(), index=y.index[-len(np.asarray(fit.resid).ravel()):]).dropna()
        rows, interpretation = _diagnose_one_model(model_name, fit, resid)
        all_rows.extend(rows)
        interpretations[model_name] = interpretation

        fig_path = FIG_DIR / f"residual_diagnostics_{model_name.lower().replace('+', '_').replace(' ', '_')}.png"
        _plot_diagnostics(resid, model_name, fig_path)

    df = pd.DataFrame(all_rows)
    df["statistic"] = df["statistic"].round(4)
    df["p_value"] = df["p_value"].round(6)
    df["interpretation_model"] = df["model"].map(interpretations)

    # Comparaison SARIMA vs SARIMAX+BESI (argument informationnel BESI)
    lb_sarima = df[(df["model"] == "SARIMA") & (df["test"].str.contains("Ljung-Box"))]["p_value"].astype(float)
    lb_sarimax = df[(df["model"] == "SARIMAX+BESI") & (df["test"].str.contains("Ljung-Box"))]["p_value"].astype(float)

    if not lb_sarima.empty and not lb_sarimax.empty:
        if lb_sarimax.mean() > lb_sarima.mean():
            comparison_note = "BESI reduit la structure residuelle: p-values Ljung-Box plus elevees."
        else:
            comparison_note = "Reduction residuelle non evidente: BESI ne blanchit pas clairement les residus."
    else:
        comparison_note = "Comparaison Ljung-Box indisponible."

    df["comparison_note"] = comparison_note
    df.to_csv(CSV_PATH, index=False, encoding="utf-8")

    print("\n=== INTERPRETATION AUTOMATIQUE ===")
    for k, v in interpretations.items():
        print(f"- {k}: {v}")
    print(f"- Comparaison SARIMA vs SARIMAX+BESI: {comparison_note}")

    logger.info("CSV sauvegarde: %s", CSV_PATH)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    out = run_residual_diagnostics()
    print(out.to_string(index=False))