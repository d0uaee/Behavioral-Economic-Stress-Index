"""
src/analysis/causal_validation.py — Tests de causalité et validation rigoureuse

RÉPOND AUX QUESTIONS MÉTHODOLOGIQUES :

1. GRANGER BIDIRECTIONNEL
   Question : est-ce BESI → inflation, ou inflation → BESI ?
   Si l'inflation cause les recherches (pas l'inverse) →
   BESI est un indicateur de réaction, pas d'anticipation.
   C'est honnête et toujours utile (diagnostic de régime).

2. CORRÉLATION SUR RÉSIDUS STL (vs corrélation brute)
   Question : "prix huile" corrèle avec l'inflation à cause du Ramadan
   ou à cause des vraies hausses de prix ?
   → Corrélation sur résidus STL = corrélation nette du biais saisonnier.

3. TEST D'ÉVÉNEMENTS (Event Study)
   Question : nos keywords ont-ils spikés aux bons moments ?
   → Vérifier que les pics Trends coïncident avec les chocs connus :
     mars 2022 (Ukraine), avril 2020 (COVID), pic Ramadan 2023 (viande).

4. TEST PLACEBO
   Question : nos keywords seraient-ils aussi corrélés avec n'importe
   quel indicateur macroéconomique (pas seulement l'inflation) ?
   → Tester la corrélation avec le taux de change et le taux directeur BAM.
   → Si BESI corrèle aussi fort avec ces indicateurs → signal non spécifique.

Output :
    outputs/reports/causal_validation_v4.csv
    outputs/reports/event_study_v4.csv
    outputs/figures/granger_bidirectionnel_v4.png
    outputs/figures/stl_residuals_correlation_v4.png
"""

import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parent.parent.parent
SILVER   = ROOT / "data" / "silver"
GOLD     = ROOT / "data" / "gold"
REPORTS  = ROOT / "outputs" / "reports"
FIGURES  = ROOT / "outputs" / "figures"
REPORTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


# ── 1. Granger bidirectionnel ─────────────────────────────────────────────────

def test_granger_bidirectionnel(
    besi: pd.Series,
    inflation: pd.Series,
    max_lag: int = 4,
) -> pd.DataFrame:
    """
    Teste la causalité de Granger dans les DEUX sens :
        Direction A : BESI(t-k) → inflation(t)   [H1 : BESI prédit l'inflation]
        Direction B : inflation(t-k) → BESI(t)   [H0 : l'inflation provoque les recherches]

    Interprétation :
    - Si A significatif, B non → BESI est prédicteur avancé (ideal)
    - Si B significatif, A non → BESI est indicateur de réaction (honnête, utile)
    - Si les deux → boucle de rétroaction (économie ↔ comportement)
    - Si aucun   → relation non-linéaire (notre cas actuel)
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    # Aligner les séries
    df = pd.DataFrame({"besi": besi, "inflation": inflation}).dropna()
    records = []

    for direction, data in [
        ("BESI -> inflation", df[["inflation", "besi"]]),
        ("inflation -> BESI", df[["besi", "inflation"]]),
    ]:
        try:
            results = grangercausalitytests(data, maxlag=max_lag, verbose=False)
            for lag in range(1, max_lag + 1):
                f_stat = results[lag][0]["ssr_ftest"][0]
                p_val  = results[lag][0]["ssr_ftest"][1]
                records.append({
                    "direction": direction,
                    "lag":       lag,
                    "f_stat":    round(f_stat, 3),
                    "p_value":   round(p_val, 4),
                    "significant_05": p_val < 0.05,
                    "significant_10": p_val < 0.10,
                })
        except Exception as e:
            logger.warning(f"Granger échoué pour '{direction}' : {e}")

    df_granger = pd.DataFrame(records)

    # Résumé lisible
    logger.info("[Granger bidirectionnel]")
    for direction in df_granger["direction"].unique():
        sub = df_granger[df_granger["direction"] == direction]
        min_p = sub["p_value"].min()
        sig = "SIGNIFICATIF (*)" if min_p < 0.05 else "non significatif"
        logger.info(f"  {direction} : p_min={min_p:.4f} -> {sig}")

    df_granger.to_csv(REPORTS / "granger_bidirectionnel_v4.csv", index=False)
    _plot_granger_bidirectionnel(df_granger)
    return df_granger


def _plot_granger_bidirectionnel(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    directions = df["direction"].unique()
    colors = {"BESI -> inflation": "#e74c3c", "inflation -> BESI": "#3498db"}

    for ax, direction in zip(axes, directions):
        sub = df[df["direction"] == direction]
        c = colors.get(direction, "gray")
        ax.bar(sub["lag"], sub["p_value"], color=c, alpha=0.75)
        ax.axhline(0.05, color="orange", ls="--", lw=1.5, label="p=0.05")
        ax.axhline(0.10, color="gray",   ls=":",  lw=1.0, label="p=0.10")
        ax.set_title(direction, fontsize=10, fontweight="bold")
        ax.set_xlabel("Lag (mois)")
        ax.set_ylabel("p-value")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("Test de Granger bidirectionnel — BESI vs Inflation",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES / "granger_bidirectionnel_v4.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Figure : granger_bidirectionnel_v4.png")


# ── 2. Corrélation résidus STL ────────────────────────────────────────────────

def _stl_decompose(series: pd.Series, period: int = 12) -> Dict[str, pd.Series]:
    """Décompose une série STL et retourne trend, saisonnier, résidu."""
    from statsmodels.tsa.seasonal import STL
    series_clean = series.interpolate(limit=3).ffill().bfill()
    stl = STL(series_clean, period=period, robust=True)
    res = stl.fit()
    return {
        "trend":    pd.Series(res.trend,    index=series.index),
        "seasonal": pd.Series(res.seasonal, index=series.index),
        "resid":    pd.Series(res.resid,    index=series.index),
    }


def compare_raw_vs_stl_correlation(
    besi: pd.Series,
    inflation: pd.Series,
    signal_name: str = "BESI behavioral",
) -> pd.DataFrame:
    """
    Compare la corrélation brute vs corrélation sur résidus STL.

    Résultat attendu :
    - Corrélation brute = forte (mais contaminée par la saisonnalité commune)
    - Corrélation résidus = plus faible mais honnête (signal non-saisonnier)

    Exemple pédagogique :
    Si "prix huile" corrèle à r=0.80 brut et r=0.35 sur résidus →
    une grande partie de la corrélation est due au Ramadan (saisonnier),
    pas à la vraie variation des prix.
    """
    from scipy.stats import pearsonr

    df = pd.DataFrame({
        "besi":      besi,
        "inflation": inflation,
    }).dropna()

    # Décomposition STL
    besi_stl  = _stl_decompose(df["besi"])
    infl_stl  = _stl_decompose(df["inflation"])

    records = []
    for lag in range(0, 4):
        besi_shifted = df["besi"].shift(lag)
        besi_resid_shifted = besi_stl["resid"].shift(lag)

        mask = besi_shifted.notna() & df["inflation"].notna()
        if mask.sum() < 20:
            continue

        # Corrélation brute
        r_raw, p_raw = pearsonr(besi_shifted[mask], df["inflation"][mask])

        # Corrélation saisonnière seulement (pour quantifier le biais)
        mask_r = besi_resid_shifted.notna() & infl_stl["resid"].notna()
        r_resid, p_resid = (0.0, 1.0)
        if mask_r.sum() >= 20:
            r_resid, p_resid = pearsonr(
                besi_resid_shifted[mask_r],
                infl_stl["resid"][mask_r]
            )

        records.append({
            "signal":          signal_name,
            "lag":             lag,
            "r_brut":          round(r_raw, 3),
            "p_brut":          round(p_raw, 4),
            "r_residu_stl":    round(r_resid, 3),
            "p_residu_stl":    round(p_resid, 4),
            "biais_saisonnier": round(abs(r_raw) - abs(r_resid), 3),
        })

    df_corr = pd.DataFrame(records)

    logger.info(f"[Corrélation brute vs STL — {signal_name}]")
    for _, row in df_corr.iterrows():
        logger.info(
            f"  lag={row['lag']} | r_brut={row['r_brut']:.3f} | "
            f"r_STL={row['r_residu_stl']:.3f} | biais={row['biais_saisonnier']:.3f}"
        )

    df_corr.to_csv(REPORTS / "stl_correlation_v4.csv", index=False)
    _plot_stl_residuals(df, besi_stl, infl_stl, signal_name)
    return df_corr


def _plot_stl_residuals(df, besi_stl, infl_stl, signal_name):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f"Décomposition STL — {signal_name} vs Inflation YoY",
                 fontsize=12, fontweight="bold")

    pairs = [
        ("Série brute",    df["besi"],         df["inflation"],
         "Brut (biaisé par saisonnalité)"),
        ("Composante Trend",  besi_stl["trend"],  infl_stl["trend"],  "Tendance"),
        ("Résidu (signal net)", besi_stl["resid"], infl_stl["resid"],
         "Résidu STL (signal économique pur)"),
    ]

    colors = [("#8e44ad", "#e74c3c"), ("#2c3e50", "#e67e22"), ("#27ae60", "#e74c3c")]

    for row_idx, (label, besi_s, infl_s, subtitle) in enumerate(pairs):
        # Graphe temporel
        ax = axes[row_idx, 0]
        ax2 = ax.twinx()
        common = besi_s.index.intersection(infl_s.index)
        ax.plot(common, besi_s.reindex(common),
                color=colors[row_idx][0], lw=1.5, label=signal_name)
        ax2.plot(common, infl_s.reindex(common),
                 color=colors[row_idx][1], lw=1.5, ls="--", label="Inflation YoY")
        ax.set_title(subtitle, fontsize=9)
        ax.set_ylabel(signal_name, color=colors[row_idx][0], fontsize=8)
        ax2.set_ylabel("Inflation %", color=colors[row_idx][1], fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(alpha=0.2)

        # Scatter
        ax_sc = axes[row_idx, 1]
        mask = besi_s.notna() & infl_s.notna()
        bx = besi_s[mask].values
        iy = infl_s[mask].values
        if len(bx) > 5:
            from scipy.stats import pearsonr
            r, p = pearsonr(bx, iy)
            ax_sc.scatter(bx, iy, alpha=0.6, s=25, color=colors[row_idx][0])
            z = np.polyfit(bx, iy, 1)
            xf = np.linspace(bx.min(), bx.max(), 50)
            ax_sc.plot(xf, np.polyval(z, xf), "k--", lw=1.2,
                       label=f"r={r:.3f} {'*' if p<0.05 else '(ns)'}")
            ax_sc.legend(fontsize=8)
        ax_sc.set_xlabel(signal_name, fontsize=8)
        ax_sc.set_ylabel("Inflation YoY %", fontsize=8)
        ax_sc.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(FIGURES / "stl_residuals_correlation_v4.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Figure : stl_residuals_correlation_v4.png")


# ── 3. Event Study ────────────────────────────────────────────────────────────

KNOWN_EVENTS = {
    "Ukraine/inflation": ("2022-02-01", "2022-05-01"),
    "COVID-lockdown":    ("2020-03-01", "2020-06-01"),
    "Ramadan-2023":      ("2023-03-01", "2023-04-01"),
    "Ramadan-2022":      ("2022-04-01", "2022-05-01"),
    "Pre-COVID-stable":  ("2018-01-01", "2019-12-01"),  # période contrôle
}


def event_study(
    besi: pd.Series,
    inflation: pd.Series,
) -> pd.DataFrame:
    """
    Vérifie que le BESI spikait aux bons moments économiques.

    Logique :
    - Si BESI est élevé pendant les chocs économiques connus → signal valide
    - Si BESI est aussi élevé en période calme → faux positifs → signal bruité
    - La période contrôle (pré-COVID stable) doit avoir BESI bas

    Ceci remplace l'argument subjectif "nos keywords semblent bons"
    par un test objectif sur des événements datés précisément.
    """
    records = []
    for event_name, (start, end) in KNOWN_EVENTS.items():
        mask = (besi.index >= start) & (besi.index <= end)
        besi_event = besi[mask]
        infl_event = inflation.reindex(besi.index)[mask]

        if besi_event.empty:
            continue

        records.append({
            "evenement":        event_name,
            "periode":          f"{start[:7]} -> {end[:7]}",
            "BESI_moyen":       round(besi_event.mean(), 3),
            "BESI_max":         round(besi_event.max(), 3),
            "inflation_moyenne":round(infl_event.mean(), 3),
            "type": "choc" if "stable" not in event_name else "controle",
        })

    df_events = pd.DataFrame(records)

    # Vérification : BESI choc > BESI contrôle ?
    besi_choc     = df_events[df_events["type"] == "choc"]["BESI_moyen"].mean()
    besi_controle = df_events[df_events["type"] == "controle"]["BESI_moyen"].mean()
    ratio = besi_choc / besi_controle if besi_controle > 0 else float("inf")

    logger.info("[Event Study]")
    for _, row in df_events.iterrows():
        logger.info(f"  {row['evenement']:<25} | BESI={row['BESI_moyen']:.3f} "
                    f"| Inflation={row['inflation_moyenne']:.2f}%")
    logger.info(f"  Ratio BESI(chocs) / BESI(controle) = {ratio:.2f}")
    logger.info(f"  {'[OK] BESI plus eleve pendant les chocs' if ratio > 1.2 else '[??] BESI non discriminant'}")

    df_events.to_csv(REPORTS / "event_study_v4.csv", index=False)
    return df_events


# ── 4. Test Placebo ───────────────────────────────────────────────────────────

def placebo_test(
    besi: pd.Series,
    inflation: pd.Series,
    fx_series: pd.Series = None,
) -> pd.DataFrame:
    """
    Test placebo : si BESI corrèle autant avec le taux de change qu'avec l'inflation
    → le signal n'est pas spécifique à l'inflation → problème de validité discriminante.

    Un bon BESI doit :
    - Avoir r(BESI, inflation) > r(BESI, indicateur_non_cible)
    """
    from scipy.stats import pearsonr

    records = []
    df_base = pd.DataFrame({"besi": besi, "inflation": inflation}).dropna()

    # Corrélation principale
    r_infl, p_infl = pearsonr(df_base["besi"], df_base["inflation"])
    records.append({
        "cible": "inflation_yoy (variable cible)",
        "r": round(r_infl, 3),
        "p": round(p_infl, 4),
        "est_cible": True,
    })

    # Placebo : taux de change MAD/EUR
    if fx_series is not None:
        df_fx = pd.DataFrame({"besi": besi, "fx": fx_series}).dropna()
        if len(df_fx) > 20:
            r_fx, p_fx = pearsonr(df_fx["besi"], df_fx["fx"])
            records.append({
                "cible": "fx_yoy MAD/EUR (placebo)",
                "r": round(r_fx, 3),
                "p": round(p_fx, 4),
                "est_cible": False,
            })

    # Placebo : tendance temporelle pure (index numérique)
    time_idx = pd.Series(range(len(besi)), index=besi.index)
    df_time = pd.DataFrame({"besi": besi, "time": time_idx}).dropna()
    r_time, p_time = pearsonr(df_time["besi"], df_time["time"])
    records.append({
        "cible": "tendance_lineaire (placebo)",
        "r": round(r_time, 3),
        "p": round(p_time, 4),
        "est_cible": False,
    })

    df_placebo = pd.DataFrame(records)

    # Verdict
    r_cible   = df_placebo[df_placebo["est_cible"]]["r"].iloc[0]
    r_placebo = df_placebo[~df_placebo["est_cible"]]["r"].abs().max()
    discriminant = r_cible > r_placebo

    logger.info("[Test placebo]")
    for _, row in df_placebo.iterrows():
        tag = "[cible]" if row["est_cible"] else "[placebo]"
        logger.info(f"  {tag} {row['cible']:<40} r={row['r']:.3f} p={row['p']:.4f}")
    logger.info(f"  Validite discriminante : {'OK' if discriminant else 'PROBLEME — BESI non specifique'}")

    df_placebo.to_csv(REPORTS / "placebo_test_v4.csv", index=False)
    return df_placebo


# ── Pipeline principal ────────────────────────────────────────────────────────

def run_causal_validation(gold_df: pd.DataFrame = None) -> Dict[str, pd.DataFrame]:
    """
    Exécute tous les tests de validation causale.
    Charge les données depuis le Gold dataset si non fourni.
    """
    if gold_df is None:
        gold_path = GOLD / "model_dataset_monthly.csv"
        gold_df = pd.read_csv(gold_path, parse_dates=["month"], index_col="month")

    besi      = gold_df["behavioral_index_pure"].dropna()
    inflation = gold_df["inflation_yoy"].dropna()
    fx        = gold_df.get("fx_yoy", None)
    if fx is not None:
        fx = fx.dropna()

    results = {}

    # 1. Granger bidirectionnel
    logger.info("\n=== Test Granger bidirectionnel ===")
    results["granger"] = test_granger_bidirectionnel(besi, inflation)

    # 2. Corrélation brute vs STL
    logger.info("\n=== Corrélation brute vs résidus STL ===")
    results["stl_corr"] = compare_raw_vs_stl_correlation(besi, inflation)

    # 3. Event Study
    logger.info("\n=== Event Study ===")
    results["event_study"] = event_study(besi, inflation)

    # 4. Test Placebo
    logger.info("\n=== Test Placebo ===")
    results["placebo"] = placebo_test(besi, inflation, fx)

    return results
