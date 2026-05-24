"""
Robustesse BESI sans le choc inflationniste 2022.

But
---
Prouver que SARIMAX + BESI apporte encore de l'information en dehors de la
fenêtre de crise 2021-12 -> 2022-06, puis vérifier un test complémentaire où
le Bloc B entier (2022-2024) est retiré.

Sorties
-------
    results/robustness_results.csv
    results/robustness_report.md

Métriques calculées
-------------------
    - AIC / BIC sur l'ajustement in-sample
    - RMSE / MAE / MAPE via walk-forward 1-step-ahead
    - Recall / AUC via un seuil de stress appris sur l'entraînement

Référence modèle complète
-------------------------
Les chiffres de référence issus du rapport principal sont :
    - SARIMA(1,1,1)(1,0,1)[12] : AIC 64.85, RMSE 1.923
    - SARIMAX + BESI behavioral : AIC 57.09, RMSE 1.891
    - Delta AIC (BESI - SARIMA) : -7.77

Le script ci-dessous recalcule ces métriques sur les jeux de données filtrés.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(42)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
GOLD_PATH = ROOT / "data" / "gold" / "model_dataset_monthly.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ROBUSTNESS_CSV = RESULTS_DIR / "robustness_results.csv"
ROBUSTNESS_MD = RESULTS_DIR / "robustness_report.md"

# Ordres utilisés dans le rapport principal / README
SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL = (1, 0, 1)
SEASONAL_PERIOD = 12

# Seuil de stress élevé appris uniquement sur le train
STRESS_PERCENTILE = 75


@dataclass
class ModelMetrics:
    scenario: str
    model: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train: int
    n_test: int
    threshold_yoy: float
    stress_pct_test: float
    inflation_yoy_mean_test: float
    aic: float
    bic: float
    rmse: float
    mae: float
    mape: float
    recall: float
    auc: float
    ap: float


def _get_sarimax():
    """Charge SARIMAX à la demande pour garder le module importable."""
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError as exc:  # pragma: no cover - dépend de l'environnement
        raise ImportError(
            "statsmodels est requis pour exécuter la robustesse. Installez les dépendances du projet."
        ) from exc
    return SARIMAX


def _load_gold() -> pd.DataFrame:
    if not GOLD_PATH.exists():
        raise FileNotFoundError(
            f"Gold dataset introuvable : {GOLD_PATH}\n"
            "Lancer d'abord : python run_v3.py --step gold"
        )

    gold = pd.read_csv(GOLD_PATH, parse_dates=["month"], index_col="month")
    gold = gold.sort_index()
    return gold


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _roc_ap(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """ROC-AUC et Average Precision sans sklearn."""
    if len(y_true) == 0:
        return float("nan"), float("nan")

    pos = int(y_true.sum())
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        baseline = float(pos / len(y_true)) if len(y_true) else float("nan")
        return 0.5, baseline

    thresholds = np.unique(scores)[::-1]
    tprs = [0.0]
    fprs = [0.0]
    precisions = []
    recalls = []

    for thr in thresholds:
        pred = (scores >= thr).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        tprs.append(recall)
        fprs.append(1.0 - specificity)
        precisions.append(precision)
        recalls.append(recall)

    tprs.append(1.0)
    fprs.append(1.0)

    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    auc = float(abs(trap(tprs, fprs)))
    idx = np.argsort(recalls)
    ap = float(abs(trap(np.array(precisions)[idx], np.array(recalls)[idx])))
    return auc, ap


def _threshold_from_train(train_yoy: pd.Series) -> float:
    train_yoy = train_yoy.dropna().astype(float)
    if train_yoy.empty:
        raise ValueError("Aucune observation YoY disponible pour calibrer le seuil.")
    return float(np.percentile(train_yoy.values, STRESS_PERCENTILE))


def _fit_model(
    SARIMAX,
    y_train: pd.Series,
    exog_train: pd.DataFrame | None = None,
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            y_train,
            exog=exog_train,
            order=SARIMA_ORDER,
            seasonal_order=(*SARIMA_SEASONAL, SEASONAL_PERIOD),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        return model.fit(disp=False, maxiter=250)


def _walk_forward_forecast(
    SARIMAX,
    gold: pd.DataFrame,
    target_col: str,
    exog_col: str | None,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> pd.Series:
    """Walk-forward 1-step-ahead sur la période de test retenue."""
    actual = gold.loc[test_start:test_end, target_col].dropna()
    preds: list[float] = []

    for dt in actual.index:
        cutoff = dt - pd.offsets.MonthBegin(1)
        y_train = gold.loc[train_start:cutoff, target_col].dropna()

        if len(y_train) < 24:
            preds.append(float(y_train.iloc[-1]) if len(y_train) > 0 else np.nan)
            continue

        exog_train = None
        exog_future = None
        if exog_col is not None and exog_col in gold.columns:
            exog_series = gold[exog_col].reindex(gold.index).ffill().bfill()
            exog_train_slice = exog_series.loc[y_train.index]
            if exog_train_slice.notna().mean() > 0.7:
                exog_train = exog_train_slice.to_frame(exog_col)
                exog_future = pd.DataFrame({exog_col: [float(exog_series.loc[dt])]}, index=[dt])

        try:
            fit = _fit_model(SARIMAX, y_train, exog_train)
            forecast = fit.get_forecast(steps=1, exog=exog_future)
            pred = float(forecast.predicted_mean.iloc[0])
        except Exception:
            pred = float(y_train.iloc[-1])

        preds.append(pred)

    return pd.Series(preds, index=actual.index, name="forecast")


def _classification_metrics(
    gold: pd.DataFrame,
    forecast_ipc: pd.Series,
    threshold_yoy: float,
) -> dict:
    actual_ipc = gold["ipc_level"].reindex(forecast_ipc.index)
    actual_yoy = gold["inflation_yoy"].reindex(forecast_ipc.index)
    lag12 = actual_ipc.shift(12)

    score_yoy = (forecast_ipc / lag12 - 1.0) * 100.0
    score_yoy = score_yoy.replace([np.inf, -np.inf], np.nan)

    valid = actual_yoy.dropna().index.intersection(score_yoy.dropna().index)
    if len(valid) == 0:
        return {"recall": np.nan, "auc": np.nan, "ap": np.nan, "stress_pct": np.nan, "mean_yoy": np.nan}

    y_true = (actual_yoy.loc[valid] >= threshold_yoy).astype(int).values
    y_score = score_yoy.loc[valid].values
    pred = (y_score >= threshold_yoy).astype(int)

    tp = int(((pred == 1) & (y_true == 1)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    auc, ap = _roc_ap(y_true, y_score)

    return {
        "recall": float(recall),
        "auc": float(auc),
        "ap": float(ap),
        "stress_pct": float(y_true.mean() * 100.0),
        "mean_yoy": float(actual_yoy.loc[valid].mean()),
    }


def _regression_metrics(actual: pd.Series, forecast: pd.Series) -> dict:
    valid = actual.dropna().index.intersection(forecast.dropna().index)
    if len(valid) == 0:
        return {"rmse": np.nan, "mae": np.nan, "mape": np.nan}
    y_true = actual.loc[valid].values
    y_pred = forecast.loc[valid].values
    return {
        "rmse": _rmse(y_true, y_pred),
        "mae": _mae(y_true, y_pred),
        "mape": _mape(y_true, y_pred),
    }


def _scenario_bounds() -> dict[str, dict]:
    return {
        "Sans mars 2022 ±3 mois": {
            "drop_start": pd.Timestamp("2021-12-01"),
            "drop_end": pd.Timestamp("2022-06-01"),
            "train_start": pd.Timestamp("2017-01-01"),
            "train_end": pd.Timestamp("2021-11-01"),
            "test_start": pd.Timestamp("2022-07-01"),
            "test_end": pd.Timestamp("2024-12-01"),
            "comment": "Crisis window retiré, on évalue seulement l'après-crise restante.",
        },
        "Sans Bloc B entier": {
            "drop_start": pd.Timestamp("2022-01-01"),
            "drop_end": pd.Timestamp("2024-12-01"),
            "train_start": pd.Timestamp("2017-01-01"),
            "train_end": pd.Timestamp("2019-12-01"),
            "test_start": pd.Timestamp("2020-01-01"),
            "test_end": pd.Timestamp("2021-12-01"),
            "comment": "Bloc B exclu, on vérifie si BESI reste utile sur 2017-2021.",
        },
    }


def _filter_gold(gold: pd.DataFrame, drop_start: pd.Timestamp, drop_end: pd.Timestamp) -> pd.DataFrame:
    mask = ~((gold.index >= drop_start) & (gold.index <= drop_end))
    return gold.loc[mask].copy()


def _baseline_comparison() -> dict:
    """Référence du rapport principal pour la comparaison orale."""
    return {
        "SARIMA": {"AIC": 64.85, "RMSE": 1.923},
        "SARIMAX+BESI": {"AIC": 57.09, "RMSE": 1.891},
        "Delta_AIC": -7.77,
    }


def _evaluate_scenario(
    scenario_name: str,
    bounds: dict,
    gold: pd.DataFrame,
) -> list[ModelMetrics]:
    SARIMAX = _get_sarimax()

    filtered = _filter_gold(gold, bounds["drop_start"], bounds["drop_end"])

    # Seuil appris sur l'entraînement strictement antérieur à la période test.
    train_yoy = filtered.loc[bounds["train_start"]:bounds["train_end"], "inflation_yoy"]
    threshold_yoy = _threshold_from_train(train_yoy)

    rows: list[ModelMetrics] = []
    for model_name, exog_col in [("SARIMA", None), ("SARIMAX+BESI", "behavioral_index_pure_lag1")]:
        train_actual = filtered.loc[bounds["train_start"]:bounds["train_end"], "ipc_level"].dropna()
        exog_train = None
        if exog_col is not None and exog_col in filtered.columns:
            exog_train = filtered.loc[train_actual.index, exog_col].to_frame(exog_col).ffill().bfill()

        fit = _fit_model(SARIMAX, train_actual, exog_train)
        forecast = _walk_forward_forecast(
            SARIMAX,
            filtered,
            target_col="ipc_level",
            exog_col=exog_col,
            train_start=bounds["train_start"],
            train_end=bounds["train_end"],
            test_start=bounds["test_start"],
            test_end=bounds["test_end"],
        )

        actual_test = filtered.loc[bounds["test_start"]:bounds["test_end"], "ipc_level"]
        reg = _regression_metrics(actual_test, forecast)
        cls = _classification_metrics(filtered.loc[bounds["test_start"]:bounds["test_end"]], forecast, threshold_yoy)

        rows.append(
            ModelMetrics(
                scenario=scenario_name,
                model=model_name,
                train_start=str(bounds["train_start"].date()),
                train_end=str(bounds["train_end"].date()),
                test_start=str(bounds["test_start"].date()),
                test_end=str(bounds["test_end"].date()),
                n_train=int(len(train_actual)),
                n_test=int(len(actual_test.dropna())),
                threshold_yoy=round(threshold_yoy, 4),
                stress_pct_test=round(cls["stress_pct"], 2) if pd.notna(cls["stress_pct"]) else np.nan,
                inflation_yoy_mean_test=round(cls["mean_yoy"], 4) if pd.notna(cls["mean_yoy"]) else np.nan,
                aic=round(float(fit.aic), 2),
                bic=round(float(fit.bic), 2),
                rmse=round(reg["rmse"], 4),
                mae=round(reg["mae"], 4),
                mape=round(reg["mape"], 4),
                recall=round(cls["recall"], 4),
                auc=round(cls["auc"], 4),
                ap=round(cls["ap"], 4),
            )
        )

    return rows


def _build_summary_table(results: list[ModelMetrics]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(r) for r in results])
    pivot = df.pivot(index=["scenario", "model"], values=["aic", "bic", "rmse", "mae", "mape", "recall", "auc", "ap"], columns=[])
    return df


def _format_delta(new: float, base: float) -> float:
    return round(new - base, 4)


def _generate_markdown(results: pd.DataFrame) -> str:
    baseline = _baseline_comparison()
    lines = []
    lines.append("# Robustesse BESI hors fenêtre 2022")
    lines.append("")
    lines.append("## Méthodologie")
    lines.append("- Scénario 1 : exclusion de 2021-12 à 2022-06 (7 mois).")
    lines.append("- Scénario 2 : exclusion de tout le Bloc B (2022-2024).")
    lines.append("- Modèles : SARIMA(1,1,1)(1,0,1)[12] et SARIMAX + BESI behavioral.")
    lines.append("- Seuil de stress élevé : 75e percentile de l'inflation YoY appris sur l'entraînement uniquement.")
    lines.append("")
    lines.append("## Référence modèle complète")
    lines.append("")
    lines.append("| Modèle | AIC | RMSE | Delta AIC vs SARIMA |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| SARIMA | {baseline['SARIMA']['AIC']:.2f} | {baseline['SARIMA']['RMSE']:.3f} | — |")
    lines.append(f"| SARIMAX+BESI | {baseline['SARIMAX+BESI']['AIC']:.2f} | {baseline['SARIMAX+BESI']['RMSE']:.3f} | {baseline['Delta_AIC']:.2f} |")
    lines.append("")

    lines.append("## Résultats robustesse")
    lines.append("")
    for scenario in results["scenario"].unique():
        sub = results[results["scenario"] == scenario].copy()
        lines.append(f"### {scenario}")
        lines.append("")
        lines.append("| Modèle | Train | Test | AIC | BIC | RMSE | Recall | AUC | Stress % test | YoY moyen test |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for _, row in sub.iterrows():
            lines.append(
                f"| {row['model']} | {row['train_start']} → {row['train_end']} | {row['test_start']} → {row['test_end']} | "
                f"{row['aic']:.2f} | {row['bic']:.2f} | {row['rmse']:.4f} | {row['recall']:.4f} | {row['auc']:.4f} | {row['stress_pct_test']:.2f}% | {row['inflation_yoy_mean_test']:.4f} |"
            )
        lines.append("")

        if len(sub) == 2:
            sarima = sub[sub["model"] == "SARIMA"].iloc[0]
            besi = sub[sub["model"] == "SARIMAX+BESI"].iloc[0]
            delta_aic = _format_delta(float(besi["aic"]), float(sarima["aic"]))
            delta_rmse = _format_delta(float(besi["rmse"]), float(sarima["rmse"]))
            if delta_aic < -2:
                conclusion = "BESI apporte de l'info AUSSI hors crise."
            else:
                conclusion = "BESI utilise principalement le signal de crise, à reconnaître honnêtement."
            lines.append(f"- Delta AIC (BESI - SARIMA) = {delta_aic:.2f}")
            lines.append(f"- Delta RMSE (BESI - SARIMA) = {delta_rmse:+.4f}")
            lines.append(f"- Conclusion orale : {conclusion}")
            lines.append("")

    lines.append("## Phrase défensive pour l'oral")
    lines.append("Le seuil de stress élevé est fixé à partir du train uniquement, au 75e percentile de l'inflation YoY, puis appliqué tel quel aux périodes de test sans recalibration ni fuite d'information future.")
    lines.append("")
    return "\n".join(lines)


def main() -> pd.DataFrame:
    gold = _load_gold()
    results: list[ModelMetrics] = []

    for scenario_name, bounds in _scenario_bounds().items():
        logger.info("Scenario: %s", scenario_name)
        results.extend(_evaluate_scenario(scenario_name, bounds, gold))

    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(ROBUSTNESS_CSV, index=False, encoding="utf-8")

    md = _generate_markdown(df)
    ROBUSTNESS_MD.write_text(md, encoding="utf-8")

    logger.info("Resultats sauvegardes : %s", ROBUSTNESS_CSV)
    logger.info("Rapport sauvegarde : %s", ROBUSTNESS_MD)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    out = main()
    print(out.to_string(index=False))