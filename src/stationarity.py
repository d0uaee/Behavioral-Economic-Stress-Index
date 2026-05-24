"""
Module d'analyse de stationnarité pour l'IPC marocain.

Objectif
--------
Produire un tableau propre et reproductible comparant les tests de
stationnarité sur le niveau de l'IPC et sa première différence.
"""

from __future__ import annotations

from functools import lru_cache
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


np.random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def _get_stattools():
    """Charge les fonctions de statsmodels utilisées par les tests."""
    try:
        from statsmodels.tsa.stattools import adfuller as _adfuller
        from statsmodels.tsa.stattools import kpss as _kpss
    except ImportError as exc:  # pragma: no cover - dépend de l'environnement
        raise ImportError(
            "statsmodels est requis pour l'analyse de stationnarité. "
            "Installez les dépendances du projet avant d'exécuter cette fonction."
        ) from exc

    try:
        from statsmodels.tsa.stattools import phillips_perron as _phillips_perron
    except ImportError:  # pragma: no cover - dépend de la version de statsmodels
        _phillips_perron = None

    return _adfuller, _kpss, _phillips_perron


def _prepare_series(series: pd.Series) -> pd.Series:
    """Nettoie la série IPC et la ramène sur un index mensuel ordonné."""
    if not isinstance(series, pd.Series):
        raise TypeError("La série IPC doit être un objet pandas.Series.")

    cleaned = series.dropna().copy().sort_index()
    if not isinstance(cleaned.index, pd.DatetimeIndex):
        raise TypeError("La série IPC doit avoir un DatetimeIndex.")

    cleaned = cleaned.asfreq("MS")
    return cleaned


def _run_adf(series: pd.Series) -> dict:
    """Retourne les résultats du test ADF sous forme de dictionnaire."""
    adfuller, _, _ = _get_stattools()
    result = adfuller(series.dropna(), autolag="AIC")
    return {
        "statistic": float(result[0]),
        "p_value": float(result[1]),
    }


def _run_kpss(series: pd.Series) -> dict:
    """Retourne les résultats du test KPSS sous forme de dictionnaire."""
    _, kpss, _ = _get_stattools()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = kpss(series.dropna(), regression="c", nlags="auto")
    return {
        "statistic": float(result[0]),
        "p_value": float(result[1]),
    }


def _run_pp(series: pd.Series) -> dict | None:
    """Retourne les résultats du test Phillips-Perron si disponible."""
    _, _, phillips_perron = _get_stattools()
    if phillips_perron is None:
        return None

    result = phillips_perron(series.dropna())
    return {
        "statistic": float(result[0]),
        "p_value": float(result[1]),
    }


def _decision(test_name: str, p_value: float, alpha: float) -> str:
    """Traduit la p-value en décision de stationnarité."""
    stationary_tests = {"ADF", "Phillips-Perron"}
    if test_name in stationary_tests:
        return "Stationnaire" if p_value < alpha else "Non-stationnaire"
    return "Stationnaire" if p_value > alpha else "Non-stationnaire"


def _interpretation(test_name: str, variant: str, p_value: float, alpha: float) -> str:
    """Retourne un commentaire court pour l'interprétation du tableau."""
    if test_name == "ADF":
        if p_value < alpha:
            return f"{variant} : rejet de H0, racine unitaire rejetée."
        return f"{variant} : H0 non rejetée, la série reste non stationnaire."

    if test_name == "KPSS":
        if p_value > alpha:
            return f"{variant} : H0 non rejetée, stationnarité compatible avec les données."
        return f"{variant} : rejet de H0, la série n'est pas stationnaire autour d'une constante."

    if p_value < alpha:
        return f"{variant} : rejet de H0, comportement compatible avec une série stationnaire."
    return f"{variant} : H0 non rejetée, pas d'évidence suffisante de stationnarité."


def _build_rows(series: pd.Series, variant: str, alpha: float, include_pp: bool) -> list[dict]:
    """Construit les lignes du tableau pour une version de la série."""
    rows: list[dict] = []

    tests = [
        ("ADF", _run_adf(series)),
        ("KPSS", _run_kpss(series)),
    ]

    if include_pp:
        pp_result = _run_pp(series)
        if pp_result is not None:
            tests.append(("Phillips-Perron", pp_result))

    for test_name, result in tests:
        p_value = result["p_value"]
        rows.append(
            {
                "series_variant": variant,
                "test_name": test_name,
                "statistic": result["statistic"],
                "p_value": p_value,
                "stationarity_decision": _decision(test_name, p_value, alpha),
                "interpretation_comment": _interpretation(test_name, variant, p_value, alpha),
            }
        )

    return rows


def analyze_ipc_stationarity(
    ipc_series: pd.Series,
    alpha: float = 0.05,
    include_pp: bool = True,
    save_csv: bool = True,
    output_name: str = "ipc_stationarity_summary.csv",
) -> pd.DataFrame:
    """
    Analyse la stationnarité de l'IPC au niveau et en première différence.

    Paramètres
    ----------
    ipc_series : pd.Series
        Série mensuelle de l'IPC avec DatetimeIndex.
    alpha : float
        Seuil de décision pour les tests statistiques.
    include_pp : bool
        Inclut le test de Phillips-Perron si disponible dans statsmodels.
    save_csv : bool
        Sauvegarde le tableau résumé dans outputs/reports/.
    output_name : str
        Nom du fichier CSV de sortie.

    Retour
    ------
    pd.DataFrame
        Tableau comparatif contenant le nom du test, la statistique,
        la p-value, la décision et un commentaire d'interprétation.
    """
    level_series = _prepare_series(ipc_series)
    diff_series = level_series.diff().dropna()

    rows = []
    rows.extend(_build_rows(level_series, "IPC level", alpha, include_pp))
    rows.extend(_build_rows(diff_series, "First difference", alpha, include_pp))

    summary = pd.DataFrame(rows)
    summary = summary[
        [
            "series_variant",
            "test_name",
            "statistic",
            "p_value",
            "stationarity_decision",
            "interpretation_comment",
        ]
    ].copy()
    summary["series_variant"] = pd.Categorical(
        summary["series_variant"],
        categories=["IPC level", "First difference"],
        ordered=True,
    )
    summary["test_name"] = pd.Categorical(
        summary["test_name"],
        categories=["ADF", "KPSS", "Phillips-Perron"],
        ordered=True,
    )
    summary = summary.sort_values(["series_variant", "test_name"], kind="stable").reset_index(drop=True)
    summary["series_variant"] = summary["series_variant"].astype(str)
    summary["test_name"] = summary["test_name"].astype(str)
    summary[["statistic", "p_value"]] = summary[["statistic", "p_value"]].round(4)

    if save_csv:
        output_path = REPORT_DIR / output_name
        summary.to_csv(output_path, index=False, encoding="utf-8")

    return summary


__all__ = ["analyze_ipc_stationarity"]