"""
src/analysis/chow_test_besi.py — Test de Chow propre pour BESI → Inflation

OBJECTIF :
    Tester la rupture structurelle de mars 2022 dans la relation entre
    le BESI (et les keywords individuels) et l'inflation YoY.

    Le Chow test classique (Chow 1960) est un F-test qui compare :
      - Le RSS du modèle poolé (données entières, mêmes paramètres pré et post)
      - La somme des RSS des deux modèles séparés (pré et post rupture)

    F = [(RSS_poolé - (RSS₁ + RSS₂)) / k] / [(RSS₁ + RSS₂) / (n₁ + n₂ - 2k)]

    où k = nombre de paramètres (constante + pente), n₁ et n₂ = tailles sous-périodes

    La p-value : p = 1 - F.cdf(F_stat, dfn=k, dfd=n₁+n₂-2k)

    Note : cette formule utilise OLS sur inflation_yoy (variable stationnaire)
    pour éviter les régressions fallacieuses sur les niveaux.

TESTS COMPLÉMENTAIRES :
    - CUSUM (statsmodels) : détecte les changements graduels de paramètres
    - Ruptures (Pelt) : détecte automatiquement les points de rupture sans
      spécifier la date à l'avance

Outputs :
    results/chow_test_besi_proper.csv   — F-stat et p-value par relation
    results/cusum_test_results.csv      — résultats CUSUM
    results/ruptures_breakpoints.csv    — points de rupture détectés

Usage :
    python -m src.analysis.chow_test_besi
"""

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parent.parent.parent
GOLD_DIR = ROOT / "data" / "gold"
OUT_DIR  = ROOT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Point de rupture : mars 2022
BREAK_DATE = pd.Timestamp("2022-03-01")

# Colonnes à tester
BESI_COL   = "behavioral_index_pure_lag1"
TARGET_COL = "inflation_yoy"                 # Variable stationnaire (YoY %)
TREND_COLS = [
    "trends_prix_alim",
    "trends_inflation",
    "trends_carburant",
    "trends_subvention",
    "trends_composite",
]


# ─── Chargement des données ───────────────────────────────────────────────────

def load_gold() -> pd.DataFrame:
    path = GOLD_DIR / "model_dataset_monthly.csv"
    df = pd.read_csv(path, parse_dates=["month"])
    df = df.set_index("month").sort_index()
    return df


# ─── Chow Test classique (F-test, Chow 1960) ─────────────────────────────────

def chow_test(y: pd.Series, x: pd.Series, break_date: pd.Timestamp) -> dict:
    """
    Chow test classique pour la relation y ~ x.

    Paramètres
    ----------
    y          : variable dépendante (stationnaire)
    x          : variable explicative
    break_date : date de rupture supposée

    Retourne un dictionnaire avec :
        n1, n2, k, rss_pooled, rss1, rss2, f_stat, p_value, verdict
    """
    df = pd.DataFrame({"y": y, "x": x}).dropna()

    # Sous-périodes
    pre  = df[df.index < break_date]
    post = df[df.index >= break_date]

    n1 = len(pre)
    n2 = len(post)
    k  = 2  # constante + pente

    # Vérification des degrés de liberté
    if n1 < k + 2 or n2 < k + 2:
        logger.warning(f"Sous-période trop petite (n1={n1}, n2={n2}, k={k}). "
                       f"Test non fiable.")
        return {
            "n1": n1, "n2": n2, "k": k,
            "rss_pooled": np.nan, "rss1": np.nan, "rss2": np.nan,
            "f_stat": np.nan, "p_value": np.nan,
            "verdict": f"NON FIABLE (n1={n1} ou n2={n2} trop petit)"
        }

    # Modèle poolé (données entières)
    X_pooled = sm.add_constant(df["x"])
    m_pooled = sm.OLS(df["y"], X_pooled).fit()
    rss_pooled = float(m_pooled.ssr)

    # Modèle pré-rupture
    X_pre = sm.add_constant(pre["x"])
    m_pre = sm.OLS(pre["y"], X_pre).fit()
    rss1  = float(m_pre.ssr)

    # Modèle post-rupture
    X_post = sm.add_constant(post["x"])
    m_post = sm.OLS(post["y"], X_post).fit()
    rss2   = float(m_post.ssr)

    # Statistique F
    numerator   = (rss_pooled - (rss1 + rss2)) / k
    denominator = (rss1 + rss2) / (n1 + n2 - 2 * k)

    if denominator <= 0:
        return {
            "n1": n1, "n2": n2, "k": k,
            "rss_pooled": rss_pooled, "rss1": rss1, "rss2": rss2,
            "f_stat": np.nan, "p_value": np.nan,
            "verdict": "ERREUR (dénominateur nul)"
        }

    f_stat  = numerator / denominator
    dfd     = n1 + n2 - 2 * k
    p_value = float(1 - stats.f.cdf(f_stat, dfn=k, dfd=dfd))

    # Validation : p-value doit être dans [0, 1]
    assert 0 <= p_value <= 1, f"p-value hors limites : {p_value}"

    if p_value < 0.05:
        verdict = "RUPTURE SIGNIFICATIVE (p<0.05)"
    elif p_value < 0.10:
        verdict = "RUPTURE MARGINALE (0.05≤p<0.10)"
    elif p_value < 0.20:
        verdict = "TENDANCE (0.10≤p<0.20)"
    else:
        verdict = "NON SIGNIFICATIF (p≥0.20)"

    coef_pre  = float(m_pre.params["x"])
    coef_post = float(m_post.params["x"])

    return {
        "n1": n1, "n2": n2, "k": k,
        "rss_pooled": round(rss_pooled, 4),
        "rss1": round(rss1, 4),
        "rss2": round(rss2, 4),
        "f_stat":  round(f_stat,  4),
        "p_value": round(p_value, 4),
        "coef_pre":  round(coef_pre,  4),
        "coef_post": round(coef_post, 4),
        "delta_coef": round(coef_post - coef_pre, 4),
        "verdict": verdict,
    }


# ─── CUSUM Test (statsmodels) ─────────────────────────────────────────────────

def cusum_test(y: pd.Series, x: pd.Series, label: str) -> dict:
    """
    Test CUSUM des résidus OLS récursifs (Brown, Durbin, Evans 1975).
    H0 : paramètres stables dans le temps.
    Retourne : statistic, p_value, verdict.
    """
    from statsmodels.stats.diagnostic import breaks_cusumolsresid

    df = pd.DataFrame({"y": y, "x": x}).dropna()
    if len(df) < 20:
        return {"label": label, "cusum_stat": np.nan,
                "cusum_p": np.nan, "cusum_verdict": "DONNÉES INSUFFISANTES"}

    X = sm.add_constant(df["x"])
    m = sm.OLS(df["y"], X).fit()

    try:
        cusum_stat, p_value, _ = breaks_cusumolsresid(m.resid)
        verdict = "INSTABILITÉ DÉTECTÉE" if p_value < 0.05 else "STABLE"
        return {
            "label": label,
            "cusum_stat": round(float(cusum_stat), 4),
            "cusum_p":    round(float(p_value), 4),
            "cusum_verdict": verdict,
        }
    except Exception as e:
        return {"label": label, "cusum_stat": np.nan,
                "cusum_p": np.nan, "cusum_verdict": f"ERREUR: {e}"}


# ─── Ruptures (Pelt) : détection automatique ─────────────────────────────────

def detect_breakpoints(y: pd.Series, x: pd.Series, label: str) -> dict:
    """
    Détecte automatiquement les points de rupture dans les résidus OLS
    via l'algorithme PELT (Killick et al. 2012) du package 'ruptures'.
    Ne nécessite pas de spécifier la date de rupture à l'avance.
    """
    try:
        import ruptures as rpt
    except ImportError:
        return {"label": label, "breakpoints": "ruptures non installé"}

    df = pd.DataFrame({"y": y, "x": x}).dropna()
    if len(df) < 20:
        return {"label": label, "breakpoints": "DONNÉES INSUFFISANTES"}

    X = sm.add_constant(df["x"])
    m = sm.OLS(df["y"], X).fit()
    resid = m.resid.values.reshape(-1, 1)

    # PELT avec coût L2 et pénalité automatique (règle BIC)
    model_rpt = rpt.Pelt(model="rbf").fit(resid)
    try:
        bkps = model_rpt.predict(pen=np.log(len(resid)) * resid.var())
        # Convertir indices en dates
        dates_idx = df.index.tolist()
        bkp_dates = [str(dates_idx[b - 1].date()) for b in bkps if b < len(dates_idx)]
    except Exception:
        bkp_dates = []

    return {
        "label": label,
        "n_breakpoints": len(bkp_dates),
        "breakpoint_dates": ", ".join(bkp_dates) if bkp_dates else "aucun",
    }


# ─── Runner principal ─────────────────────────────────────────────────────────

def main():
    logger.info("=" * 70)
    logger.info("  CHOW TEST PROPRE (F-statistic) — BESI et Keywords → Inflation YoY")
    logger.info("=" * 70)

    gold = load_gold()

    y_full = gold[TARGET_COL]

    # Variables à tester : BESI + keywords individuels
    variables = {BESI_COL: "BESI (behavioral_index_pure_lag1)"}
    for col in TREND_COLS:
        if col in gold.columns:
            variables[col] = col

    # ── Chow Tests ────────────────────────────────────────────────────────────
    logger.info(f"\nPoint de rupture testé : {BREAK_DATE.date()}")
    chow_rows = []
    for col, label in variables.items():
        if col not in gold.columns:
            logger.warning(f"  Colonne absente : {col}")
            continue
        x = gold[col]
        res = chow_test(y_full, x, BREAK_DATE)
        res["relation"]  = f"{label} → {TARGET_COL}"
        res["signal"]    = col
        chow_rows.append(res)

        # Affichage
        fstr  = f"F={res['f_stat']:.3f}" if not np.isnan(res.get('f_stat', np.nan)) else "F=NaN"
        pstr  = f"p={res['p_value']:.4f}" if not np.isnan(res.get('p_value', np.nan)) else "p=NaN"
        dc    = f"Δcoef={res.get('delta_coef', np.nan):+.3f}" if not np.isnan(res.get('delta_coef', np.nan)) else ""
        logger.info(f"  {label:<35} {fstr:>10}  {pstr:>10}  {res['verdict']}")
        if dc:
            logger.info(f"    coef pré={res.get('coef_pre', np.nan):.3f}  "
                        f"coef post={res.get('coef_post', np.nan):.3f}  {dc}")

    chow_df = pd.DataFrame(chow_rows)
    chow_df.to_csv(OUT_DIR / "chow_test_besi_proper.csv", index=False)
    logger.info(f"\n  CSV sauvegardé : {OUT_DIR / 'chow_test_besi_proper.csv'}")

    # ── CUSUM Tests ───────────────────────────────────────────────────────────
    logger.info("\n" + "─" * 70)
    logger.info("  TEST CUSUM (Brown-Durbin-Evans) — instabilité des paramètres")
    logger.info("─" * 70)
    cusum_rows = []
    for col, label in variables.items():
        if col not in gold.columns:
            continue
        res = cusum_test(y_full, gold[col], label)
        cusum_rows.append(res)
        pstr = f"p={res['cusum_p']:.4f}" if not np.isnan(res.get('cusum_p', np.nan)) else "p=NaN"
        logger.info(f"  {label:<35} {pstr:>10}  {res['cusum_verdict']}")

    cusum_df = pd.DataFrame(cusum_rows)
    cusum_df.to_csv(OUT_DIR / "cusum_test_results.csv", index=False)
    logger.info(f"\n  CSV sauvegardé : {OUT_DIR / 'cusum_test_results.csv'}")

    # ── Ruptures (Pelt) ───────────────────────────────────────────────────────
    logger.info("\n" + "─" * 70)
    logger.info("  DÉTECTION AUTOMATIQUE (ruptures PELT) — sans date imposée")
    logger.info("─" * 70)
    rpt_rows = []
    for col, label in variables.items():
        if col not in gold.columns:
            continue
        res = detect_breakpoints(y_full, gold[col], label)
        rpt_rows.append(res)
        logger.info(f"  {label:<35} bkps={res.get('n_breakpoints', '?')}  "
                    f"dates={res.get('breakpoint_dates', '?')}")

    rpt_df = pd.DataFrame(rpt_rows)
    rpt_df.to_csv(OUT_DIR / "ruptures_breakpoints.csv", index=False)
    logger.info(f"\n  CSV sauvegardé : {OUT_DIR / 'ruptures_breakpoints.csv'}")

    # ── Résumé Chow ───────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("  RÉSUMÉ — CHOW TEST F-STATISTIC")
    logger.info("=" * 70)
    logger.info(f"  {'Relation':<40} {'F-stat':>8}  {'p-value':>8}  {'n1':>4}  {'n2':>4}  Verdict")
    logger.info("  " + "-" * 90)
    for _, row in chow_df.iterrows():
        f   = f"{row['f_stat']:.3f}" if not np.isnan(row['f_stat']) else "  NaN"
        p   = f"{row['p_value']:.4f}" if not np.isnan(row['p_value']) else "   NaN"
        rel = str(row['relation'])[:40]
        logger.info(f"  {rel:<40} {f:>8}  {p:>8}  {int(row['n1']):>4}  {int(row['n2']):>4}  {row['verdict']}")

    logger.info("=" * 70)

    return chow_df, cusum_df, rpt_df


if __name__ == "__main__":
    main()
