"""
src/evaluation/diebold_mariano.py

Test de Diebold-Mariano (DM) pour comparer la précision prédictive
de deux modèles sur les mêmes dates de prévision.

Référence originale
-------------------
Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy.
Journal of Business & Economic Statistics, 13(3), 253-263.

Formule (notation DM 1995)
--------------------------
Soit d_t = g(e1_t) - g(e2_t), avec g(.) une fonction de perte.

H0 : E[d_t] = 0  (précision égale)
H1 (unilatéral) : E[d_t] > 0  (model2 meilleur que model1)

Statistique :
    DM = d_bar / sqrt( S_d(0) / T )

où S_d(0) est l'estimateur HAC de la variance de long terme de d_t.
Ici on utilise Newey-West (poids de Bartlett) avec q = h-1.

Sorties
-------
    results/diebold_mariano_results.csv
    results/figures/diebold_mariano_pvalues.png
"""

from __future__ import annotations

import logging
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.evaluation.backtest import _derive_windows, _walk_forward_predict
except ModuleNotFoundError:
    # Support exécution directe: python src/evaluation/diebold_mariano.py
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from src.evaluation.backtest import _derive_windows, _walk_forward_predict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - dépend de l'environnement
    matplotlib = None
    plt = None


logger = logging.getLogger(__name__)
np.random.seed(42)

ROOT = Path(__file__).resolve().parent.parent.parent
GOLD_PATH = ROOT / "data" / "gold" / "model_dataset_monthly.csv"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

DM_CSV_PATH = RESULTS_DIR / "diebold_mariano_results.csv"
DM_FIG_PATH = FIG_DIR / "diebold_mariano_pvalues.png"


def _norm_cdf(x: float) -> float:
    """CDF de la loi normale standard sans dépendance externe."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _newey_west_long_run_variance(series: np.ndarray, lag: int) -> float:
    """Estimateur HAC Newey-West (poids Bartlett) de la variance long-terme."""
    x = np.asarray(series, dtype=float)
    x = x[~np.isnan(x)]
    t = len(x)
    if t < 2:
        return float("nan")

    x_centered = x - x.mean()
    gamma0 = float(np.dot(x_centered, x_centered) / t)
    if lag <= 0:
        return gamma0

    lrv = gamma0
    max_lag = min(lag, t - 1)
    for k in range(1, max_lag + 1):
        weight = 1.0 - k / (max_lag + 1)
        gamma_k = float(np.dot(x_centered[k:], x_centered[:-k]) / t)
        lrv += 2.0 * weight * gamma_k

    return max(lrv, 1e-12)


def diebold_mariano_test(
    errors_model1: np.ndarray,
    errors_model2: np.ndarray,
    h: int = 1,
    power: int = 2,
    alternative: str = "one-sided",
    small_sample_correction: bool = True,
) -> tuple[float, float]:
    """
    Test Diebold-Mariano avec correction Newey-West.

    Paramètres
    ----------
    errors_model1, errors_model2 : erreurs de prévision alignées dans le temps
    h : horizon de prévision (1 pour one-step-ahead)
    power : 2 pour MSE, 1 pour MAE
    alternative : 'one-sided' (H1: model2 meilleur) ou 'two-sided'
    small_sample_correction : applique l'ajustement de Harvey et al. (1997)

    Retour
    ------
    (dm_stat, p_value)
    """
    e1 = np.asarray(errors_model1, dtype=float)
    e2 = np.asarray(errors_model2, dtype=float)
    mask = ~(np.isnan(e1) | np.isnan(e2))
    e1, e2 = e1[mask], e2[mask]

    if len(e1) < 8:
        return float("nan"), float("nan")

    loss1 = np.abs(e1) ** power
    loss2 = np.abs(e2) ** power
    d = loss1 - loss2

    t = len(d)
    d_bar = float(d.mean())
    q = max(h - 1, 0)
    lrv = _newey_west_long_run_variance(d, lag=q)
    dm_stat = d_bar / math.sqrt(lrv / t)

    if small_sample_correction and t > 2 * h:
        # Ajustement de Harvey, Leybourne & Newbold (1997), souvent utilisé avec DM.
        adj = math.sqrt((t + 1 - 2 * h + h * (h - 1) / t) / t)
        dm_stat *= adj

    if alternative == "two-sided":
        p_value = 2.0 * (1.0 - _norm_cdf(abs(dm_stat)))
    elif alternative == "one-sided":
        # H1: model2 meilleur => d_bar > 0 => DM élevé positif
        p_value = 1.0 - _norm_cdf(dm_stat)
    else:
        raise ValueError("alternative doit être 'one-sided' ou 'two-sided'.")

    return float(dm_stat), float(max(min(p_value, 1.0), 0.0))


def _model_specs(gold: pd.DataFrame) -> dict[str, dict]:
    specs = {
        "Naif": {"model_type": "naif", "exog": None},
        "SARIMA": {"model_type": "sarima", "exog": None},
    }
    if "behavioral_index_pure_lag1" in gold.columns:
        specs["SARIMAX+BESI"] = {"model_type": "sarimax", "exog": "behavioral_index_pure_lag1"}
    if "hybrid_macro_index_lag1" in gold.columns:
        specs["SARIMAX+Hybride"] = {"model_type": "sarimax", "exog": "hybrid_macro_index_lag1"}
    return specs


def _build_forecast_errors(gold: pd.DataFrame) -> tuple[dict[str, dict[str, pd.Series]], list[str]]:
    """Reconstruit les erreurs (y_true - y_pred) par modèle et par bloc."""
    target_col = "ipc_level"
    windows = _derive_windows(gold)
    specs = _model_specs(gold)

    errors: dict[str, dict[str, pd.Series]] = {m: {} for m in specs.keys()}
    available_blocs: list[str] = []

    for window in windows:
        bloc = window["label"]
        available_blocs.append(bloc)
        train_start = window["train_start"]
        test_start = window["test_start"]
        test_end = window["test_end"]

        y_test = gold.loc[test_start:test_end, target_col].dropna()
        if y_test.empty:
            continue

        for model_name, cfg in specs.items():
            pred = _walk_forward_predict(
                gold=gold,
                target_col=target_col,
                train_start=train_start,
                test_dates=y_test.index,
                model_type=cfg["model_type"],
                exog_col=cfg["exog"],
            )
            pred_s = pd.Series(pred, index=y_test.index, name="pred")
            err_s = (y_test - pred_s).rename("error")
            errors[model_name][bloc] = err_s

    # Scope global = concat A/B/(C si présent)
    for model_name in errors:
        series_list = [errors[model_name][b] for b in available_blocs if b in errors[model_name]]
        if series_list:
            errors[model_name]["global"] = pd.concat(series_list).sort_index()

    return errors, available_blocs


def _dm_conclusion(dm_stat: float, p_value: float, alternative: str) -> str:
    if np.isnan(p_value):
        return "Echantillon insuffisant"
    if alternative == "one-sided":
        if p_value < 0.05:
            return "Amelioration statistiquement significative"
        return "Amelioration directionnelle non significative - a nuancer dans la presentation"
    if p_value < 0.05:
        return "Difference significative entre modeles"
    return "Difference non significative"


def _compute_pairwise_dm(errors: dict[str, dict[str, pd.Series]], blocs: list[str]) -> pd.DataFrame:
    model_names = list(errors.keys())
    pairs = list(combinations(model_names, 2))
    scopes = blocs + ["global"]

    rows = []
    for scope in scopes:
        for m1, m2 in pairs:
            if scope not in errors[m1] or scope not in errors[m2]:
                continue

            s1 = errors[m1][scope]
            s2 = errors[m2][scope]
            common_idx = s1.dropna().index.intersection(s2.dropna().index)
            if len(common_idx) < 8:
                continue

            e1 = s1.loc[common_idx].values
            e2 = s2.loc[common_idx].values

            for power, loss_name in [(2, "MSE"), (1, "MAE")]:
                for alternative in ["two-sided", "one-sided"]:
                    dm_stat, p_value = diebold_mariano_test(
                        e1,
                        e2,
                        h=1,
                        power=power,
                        alternative=alternative,
                    )
                    rows.append(
                        {
                            "scope": scope,
                            "model_1": m1,
                            "model_2": m2,
                            "loss": loss_name,
                            "alternative": alternative,
                            "n_obs": int(len(common_idx)),
                            "dm_stat": round(dm_stat, 4) if pd.notna(dm_stat) else np.nan,
                            "p_value": round(p_value, 6) if pd.notna(p_value) else np.nan,
                            "conclusion": _dm_conclusion(dm_stat, p_value, alternative),
                        }
                    )

    return pd.DataFrame(rows)


def _plot_forest_pvalues(results_df: pd.DataFrame, output_path: Path = DM_FIG_PATH) -> None:
    """Forest plot des p-values (version principale: one-sided + MSE)."""
    if plt is None:
        logger.warning("matplotlib indisponible — forest plot non genere")
        return

    plot_df = results_df[
        (results_df["alternative"] == "one-sided") &
        (results_df["loss"] == "MSE")
    ].copy()
    if plot_df.empty:
        logger.warning("Aucune ligne one-sided/MSE pour le forest plot")
        return

    plot_df["label"] = (
        plot_df["scope"].astype(str)
        + " | "
        + plot_df["model_1"].astype(str)
        + " vs "
        + plot_df["model_2"].astype(str)
    )
    plot_df = plot_df.sort_values("p_value", ascending=True).reset_index(drop=True)

    y = np.arange(len(plot_df))
    sig_color = np.where(plot_df["p_value"] < 0.05, "#2ca02c", "#1f77b4")

    fig, ax = plt.subplots(figsize=(11, max(6, 0.35 * len(plot_df))))
    ax.scatter(plot_df["p_value"], y, c=sig_color, s=45, alpha=0.9)
    ax.axvline(0.05, color="#d62728", linestyle="--", linewidth=1.2, label="Seuil 0.05")

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"], fontsize=8)
    ax.set_xlabel("p-value (DM one-sided, perte MSE)")
    ax.set_title("Diebold-Mariano — Forest plot des p-values", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def run_diebold_mariano(
    gold_path: str | Path = GOLD_PATH,
    output_csv: str | Path = DM_CSV_PATH,
    output_fig: str | Path = DM_FIG_PATH,
) -> pd.DataFrame:
    """Exécute tous les tests DM demandés et sauvegarde les sorties."""
    gold_path = Path(gold_path)
    output_csv = Path(output_csv)
    output_fig = Path(output_fig)

    if not gold_path.exists():
        raise FileNotFoundError(
            f"Gold dataset introuvable : {gold_path}\n"
            "Lancer d'abord : python run_v3.py --step gold"
        )

    gold = pd.read_csv(gold_path, parse_dates=["month"], index_col="month")
    errors, blocs = _build_forecast_errors(gold)
    results_df = _compute_pairwise_dm(errors, blocs)

    if results_df.empty:
        raise RuntimeError("Aucun résultat DM calculé. Vérifier les données et les modèles disponibles.")

    results_df.to_csv(output_csv, index=False, encoding="utf-8")
    _plot_forest_pvalues(results_df, output_fig)

    logger.info("Resultats DM sauvegardes : %s", output_csv)
    if output_fig.exists():
        logger.info("Forest plot sauvegarde : %s", output_fig)

    return results_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = run_diebold_mariano()
    print(df.to_string(index=False))