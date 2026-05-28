"""
Intégration d'un signal presse/NLP au BESI existant sans modifier le pipeline v3.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
GOLD_PATH = ROOT / "data" / "gold" / "model_dataset_monthly.csv"
SILVER_DIR = ROOT / "data" / "silver"
REPORTS_DIR = ROOT / "outputs" / "reports"
RESULTS_DIR = ROOT / "results"
SILVER_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _fit_lasso_weights(train_df: pd.DataFrame, feature_cols: list[str], target_col: str) -> dict[str, float]:
    try:
        from sklearn.linear_model import LassoCV
    except ImportError as exc:  # pragma: no cover
        raise ImportError("scikit-learn requis pour le calibrage Lasso.") from exc

    X = train_df[feature_cols].fillna(0.0).values
    y = train_df[target_col].fillna(0.0).values
    model = LassoCV(cv=5, random_state=42, positive=True, max_iter=5000)
    model.fit(X, y)
    raw = {col: coef for col, coef in zip(feature_cols, model.coef_)}
    total = sum(abs(v) for v in raw.values())
    if total == 0:
        return {feature_cols[0]: 1.0, feature_cols[1]: 0.0}
    return {col: abs(val) / total for col, val in raw.items()}


def _evaluate_signal(
    df: pd.DataFrame,
    signal_col: str,
    label: str,
    train_end: str = "2021-12-01",
    test_start: str = "2022-01-01",
    test_end: str = "2024-12-01",
) -> dict:
    from src.evaluation.backtest import _sarima_fit, _walk_forward_predict, _rmse
    from src.evaluation.warning_metrics import (
        _calibrate_threshold,
        _metrics_at_threshold,
        _compute_pr,
        _compute_ap,
    )

    work = df.copy()
    lag_col = f"{signal_col}_lag1"
    work[lag_col] = work[signal_col].shift(1)

    train_mask = (work.index >= pd.Timestamp("2017-01-01")) & (work.index <= pd.Timestamp(train_end))
    test_mask = (work.index >= pd.Timestamp(test_start)) & (work.index <= pd.Timestamp(test_end))

    train_series = work.loc[train_mask, "ipc_level"].dropna()
    exog_train = work.loc[train_series.index, lag_col].fillna(0.0).to_frame(lag_col)
    fit = _sarima_fit(train_series, exog_train=exog_train, simple=False)
    if fit is None:
        fit = _sarima_fit(train_series, exog_train=exog_train, simple=True)
    aic = float(fit.aic) if fit is not None else float("nan")

    y_test = work.loc[test_mask, "ipc_level"].dropna()
    pred = _walk_forward_predict(
        work, "ipc_level", pd.Timestamp("2017-01-01"), y_test.index, "sarimax", exog_col=lag_col
    )
    rmse_block_b = _rmse(y_test, pred)

    y_train_cls = work.loc[train_mask, "target_high_inflation_regime_t1"].dropna().astype(int)
    scores_train = work.loc[y_train_cls.index, lag_col].fillna(0.0).values
    threshold = _calibrate_threshold(y_train_cls.values, scores_train)

    y_test_cls = work.loc[test_mask, "target_high_inflation_regime_t1"].dropna().astype(int)
    scores_test = work.loc[y_test_cls.index, lag_col].fillna(0.0).values
    m = _metrics_at_threshold(y_test_cls.values, scores_test, threshold)
    precs, recs, _ = _compute_pr(y_test_cls.values, scores_test)
    ap = _compute_ap(precs, recs)

    return {
        "model": label,
        "signal_col": signal_col,
        "aic": aic,
        "rmse_bloc_b": rmse_block_b,
        "recall_bloc_b": float(m["recall"]),
        "precision_bloc_b": float(m["precision"]),
        "f1_bloc_b": float(m["f1"]),
        "ap_bloc_b": float(ap),
        "threshold_signal": float(threshold),
        "tp": m["tp"],
        "fp": m["fp"],
        "fn": m["fn"],
    }


def build_besi_v2(
    gold_path: str | Path | None = None,
    sentiment_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gold_path = Path(gold_path or GOLD_PATH)
    sentiment_path = Path(sentiment_path or (SILVER_DIR / "sentiment_monthly.csv"))

    gold = pd.read_csv(gold_path, parse_dates=["month"], index_col="month")
    sentiment = pd.read_csv(sentiment_path, parse_dates=["date"]).rename(columns={"date": "month"})
    sentiment = sentiment.set_index("month")

    merged = gold.join(sentiment[["score_lexical_norm", "coverage_flag", "n_texts"]], how="left")
    merged["score_lexical_norm"] = merged["score_lexical_norm"].fillna(0.0)
    merged["coverage_flag_nlp"] = merged["coverage_flag"].fillna(0).astype(int)
    merged["n_texts_nlp"] = merged["n_texts"].fillna(0).astype(int)

    merged["besi_v2a_fixed"] = 0.7 * merged["behavioral_index_pure"] + 0.3 * merged["score_lexical_norm"]
    merged["besi_v2c_nlp_only"] = merged["score_lexical_norm"]

    train_mask = (merged.index >= pd.Timestamp("2017-01-01")) & (merged.index <= pd.Timestamp("2021-12-01"))
    train_df = merged.loc[train_mask].dropna(subset=["target_inflation_yoy_t1"])
    weights = _fit_lasso_weights(
        train_df,
        ["behavioral_index_pure", "score_lexical_norm"],
        "target_inflation_yoy_t1",
    )
    alpha = weights["behavioral_index_pure"]
    beta = weights["score_lexical_norm"]
    merged["besi_v2b_lasso"] = alpha * merged["behavioral_index_pure"] + beta * merged["score_lexical_norm"]

    variants_path = SILVER_DIR / "besi_v2_variants_monthly.csv"
    merged.reset_index().to_csv(SILVER_DIR / "model_dataset_monthly_with_nlp.csv", index=False, encoding="utf-8")
    merged[["behavioral_index_pure", "score_lexical_norm", "besi_v2a_fixed", "besi_v2b_lasso", "besi_v2c_nlp_only"]].reset_index().to_csv(
        variants_path, index=False, encoding="utf-8"
    )

    results = [
        _evaluate_signal(merged, "behavioral_index_pure", "SARIMAX + BESI v1 (Trends)"),
        _evaluate_signal(merged, "besi_v2a_fixed", "SARIMAX + BESI v2a (Fixe)"),
        _evaluate_signal(merged, "besi_v2b_lasso", "SARIMAX + BESI v2b (Lasso)"),
        _evaluate_signal(merged, "besi_v2c_nlp_only", "SARIMAX + NLP seul"),
    ]
    results_df = pd.DataFrame(results)

    baseline_aic = results_df.loc[results_df["model"] == "SARIMAX + BESI v1 (Trends)", "aic"].iloc[0]
    results_df["aic_delta_vs_v1"] = results_df["aic"] - baseline_aic
    results_df.to_csv(REPORTS_DIR / "nlp_besi_comparison.csv", index=False, encoding="utf-8")

    weights_df = pd.DataFrame(
        [{"feature": "behavioral_index_pure", "weight": alpha}, {"feature": "score_lexical_norm", "weight": beta}]
    )
    weights_df.to_csv(REPORTS_DIR / "nlp_lasso_weights.csv", index=False, encoding="utf-8")

    verdict = "B"
    if beta == 0:
        verdict = "C"
    else:
        v2 = results_df[results_df["model"] == "SARIMAX + BESI v2b (Lasso)"].iloc[0]
        v1 = results_df[results_df["model"] == "SARIMAX + BESI v1 (Trends)"].iloc[0]
        if (v2["aic"] < v1["aic"]) and (v2["recall_bloc_b"] >= v1["recall_bloc_b"]):
            verdict = "A"

    report_path = RESULTS_DIR / "NLP_RESULTS.md"
    _write_results_md(report_path, merged, results_df, alpha, beta, verdict)

    logger.info("BESI v2 sauvegardé : %s", variants_path)
    logger.info("Comparaison NLP sauvegardée : %s", REPORTS_DIR / "nlp_besi_comparison.csv")
    logger.info("Verdict NLP : CAS %s", verdict)
    return merged, results_df


def _write_results_md(
    output_path: Path,
    merged: pd.DataFrame,
    results_df: pd.DataFrame,
    alpha: float,
    beta: float,
    verdict: str,
) -> None:
    yearly = (
        merged.groupby(merged.index.year)["n_texts_nlp"]
        .sum()
        .rename_axis("year")
        .to_string()
    )
    imputed_months = int((merged["coverage_flag_nlp"] == 0).sum())
    results_table = results_df.round(4).to_string(index=False)
    top = []
    top_terms_path = REPORTS_DIR / "hespress_top_lexical_terms.csv"
    if top_terms_path.exists():
        top = pd.read_csv(top_terms_path).head(10).to_dict(orient="records")

    if verdict == "A":
        verdict_text = (
            "Le signal presse apporte une information complémentaire aux Trends. "
            "BESI v2 améliore le fit in-sample et conserve une détection compétitive."
        )
    elif verdict == "C":
        verdict_text = (
            "Le Lasso assigne un poids nul au signal NLP. Le signal presse n'apporte "
            "pas d'information conditionnelle supplémentaire au-delà des Trends."
        )
    else:
        verdict_text = (
            "Le signal presse lexical n'apporte pas d'amélioration claire au-delà des "
            "Google Trends sur cette période. Le résultat reste scientifiquement valide."
        )

    lines = [
        "# NLP_RESULTS",
        "",
        "## 1. Couverture des données",
        "",
        yearly,
        "",
        f"- Mois imputés (coverage_flag=0) : {imputed_months}",
        f"- Source primaire du signal : post_excerpt / RSS fallback si nécessaire",
        f"- Nature du signal : presse editoriale (pas commentaires lecteurs)",
        "",
        "## 2. Composition du lexique",
        "",
        f"- Poids Lasso : alpha={alpha:.3f}, beta={beta:.3f}",
        "",
        "Top 10 termes les plus fréquents :",
    ]
    for row in top:
        lines.append(f"- {row['term']}: {row['count']}")

    lines += [
        "",
        "## 3. Performance comparative BESI v1 vs BESI v2",
        "",
        "```text",
        results_table,
        "```",
        "",
        "## 4. Verdict honnête",
        "",
        verdict_text,
        "",
    ]
    if verdict == "A":
        oral = "Le signal presse apporte une information complémentaire mesurable aux Trends dans BESI v2."
    elif verdict == "C":
        oral = "Le signal presse editorial a ete teste proprement, mais le Lasso lui attribue un poids nul face aux Trends."
    else:
        oral = "Le signal presse enrichit l'interprétation, mais n'améliore pas clairement BESI au-delà des Trends sur cet échantillon."
    lines += ["## 5. Phrase pour l'oral", "", oral, ""]

    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    build_besi_v2()
