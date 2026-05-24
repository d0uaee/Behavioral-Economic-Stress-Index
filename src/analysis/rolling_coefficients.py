"""
src/analysis/rolling_coefficients.py

Rolling Lasso coefficients for the 7 Google Trends keywords and
rolling SARIMAX coefficient for the behavioral BESI exogenous signal.

Saves:
 - results/rolling_coefficients.csv
 - results/figures/rolling_coefficients.png
 - results/chow_test_results.csv

Usage: run as a script from project root (uses project virtualenv).
"""
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
GOLD = ROOT / "data" / "gold" / "model_dataset_monthly.csv"
OUT_DIR = ROOT / "results"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

BREAK_DATE = pd.Timestamp("2022-03-01")
WINDOW = 36
STEP = 1

DEFAULT_KEYWORDS = [
    "inflation maroc", "prix huile", "hausse prix",
    "credit consommation", "chomage maroc", "prix alimentaires",
    "pouvoir achat",
]


def infer_trend_columns(gold: pd.DataFrame, keywords: list[str]) -> list[str]:
    # Try exact matches first, then fuzzy contains
    cols = []
    for kw in keywords:
        if kw in gold.columns:
            cols.append(kw)
    if cols:
        return cols

    # fuzzy match: column contains all tokens of keyword
    for col in gold.columns:
        low = col.lower()
        for kw in keywords:
            tokens = [t for t in kw.split() if t]
            if all(tok in low for tok in tokens):
                cols.append(col)
                break
    if cols:
        return cols

    # Last resort: pick columns starting with trends_ (excluding lagged names)
    trend_candidates = [c for c in gold.columns if c.lower().startswith("trends_") and "_lag" not in c.lower()]
    # exclude columns that are entirely missing
    trend_candidates = [c for c in trend_candidates if gold[c].notna().sum() > 0]
    if trend_candidates:
        return trend_candidates

    return cols


def rolling_lasso_coefficients(gold: pd.DataFrame, trend_cols: list[str]) -> pd.DataFrame:
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    idx = gold.index.sort_values()
    centers = []
    records = []

    for start_idx in range(0, len(idx) - WINDOW + 1, STEP):
        window_idx = idx[start_idx:start_idx + WINDOW]
        center = window_idx[WINDOW // 2]
        centers.append(center)

        X = gold.loc[window_idx, trend_cols].copy()
        y = gold.loc[window_idx, "ipc_level"].copy()
        # Combine and impute exogenous NaNs by forward/back fill, keep only rows with target
        df = pd.concat([y, X], axis=1)
        # keep rows where target exists
        df = df[df["ipc_level"].notna()].copy()
        # impute trends (safer than dropping entire windows)
        df[trend_cols] = df[trend_cols].ffill().bfill()
        # if still missing too many values, skip
        if df[trend_cols].isna().any(axis=None):
            coeffs = {c: np.nan for c in trend_cols}
            rec = {"center": center, "window_start": window_idx[0], "window_end": window_idx[-1]}
            rec.update(coeffs)
            records.append(rec)
            continue
        if df.shape[0] < 12:
            coeffs = {c: np.nan for c in trend_cols}
        else:
            pipeline = make_pipeline(StandardScaler(), LassoCV(cv=5, n_jobs=-1, random_state=42, max_iter=5000))
            try:
                pipeline.fit(df[trend_cols].values, df["ipc_level"].values)
                lasso = pipeline.named_steps["lassocv"]
                coef = lasso.coef_
                coeffs = {col: float(coef[i]) for i, col in enumerate(trend_cols)}
            except Exception as e:
                logger.debug(f"Lasso failed on window {center}: {e}")
                coeffs = {c: np.nan for c in trend_cols}

        rec = {"center": center, "window_start": window_idx[0], "window_end": window_idx[-1]}
        rec.update(coeffs)
        records.append(rec)

    return pd.DataFrame(records).set_index("center")


def chow_test_per_coeff(gold: pd.DataFrame, trend_cols: list[str], break_date: pd.Timestamp):
    import statsmodels.api as sm
    from scipy import stats

    results = []

    for col in trend_cols:
        pre = gold.loc[gold.index < break_date, ["ipc_level", col]].dropna()
        post = gold.loc[gold.index >= break_date, ["ipc_level", col]].dropna()
        if pre.shape[0] < 12 or post.shape[0] < 12:
            logger.debug(f"Insufficient data for {col}: pre={pre.shape[0]}, post={post.shape[0]}")
            results.append({"keyword": col, "pre_coef": np.nan, "post_coef": np.nan, "tstat": np.nan, "pvalue": np.nan})
            continue

        Xpre = sm.add_constant(pre[[col]])
        Xpost = sm.add_constant(post[[col]])
        ypre = pre["ipc_level"]
        ypost = post["ipc_level"]

        mpre = sm.OLS(ypre, Xpre).fit()
        mpost = sm.OLS(ypost, Xpost).fit()

        b1 = mpre.params.get(col, np.nan)
        b2 = mpost.params.get(col, np.nan)
        se1 = mpre.bse.get(col, np.nan)
        se2 = mpost.bse.get(col, np.nan)

        if pd.isna(b1) or pd.isna(b2) or pd.isna(se1) or pd.isna(se2):
            results.append({"keyword": col, "pre_coef": np.nan, "post_coef": np.nan, "tstat": np.nan, "pvalue": np.nan})
            continue

        denom = np.sqrt(se1 ** 2 + se2 ** 2)
        if denom == 0:
            tstat = np.nan
            pval = np.nan
        else:
            tstat = (b1 - b2) / denom
            df = pre.shape[0] + post.shape[0] - 4
            pval = 2 * (1 - stats.t.cdf(abs(tstat), df))

        results.append({"keyword": col, "pre_coef": float(b1), "post_coef": float(b2), "tstat": float(tstat) if not pd.isna(tstat) else np.nan, "pvalue": float(pval) if not pd.isna(pval) else np.nan})

    return pd.DataFrame(results).set_index("keyword")


def rolling_sarimax_exog_coeff(gold: pd.DataFrame, exog_col: str, window: int = WINDOW) -> pd.DataFrame:
    """Rolling SARIMAX to capture exog coefficient per window. Returns series indexed by center date."""
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except Exception:
        logger.warning("statsmodels unavailable — skipping rolling SARIMAX")
        return pd.DataFrame()

    idx = gold.index.sort_values()
    records = []
    for start_idx in range(0, len(idx) - window + 1, STEP):
        window_idx = idx[start_idx:start_idx + window]
        center = window_idx[window // 2]
        df = gold.loc[window_idx].dropna(subset=["ipc_level", exog_col])
        if df.shape[0] < 24:
            records.append({"center": center, "sarimax_exog_coef": np.nan})
            continue
        try:
            model = SARIMAX(df["ipc_level"], exog=df[[exog_col]], order=(1,1,0), seasonal_order=(0,1,0,12), enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False)
            # try to extract param by name
            pnames = res.param_names
            coef = np.nan
            for name, val in zip(pnames, res.params):
                if exog_col in name or name.lower().startswith("beta"):
                    coef = float(val)
                    break
            records.append({"center": center, "sarimax_exog_coef": coef})
        except Exception as e:
            logger.debug(f"SARIMAX window failed {center}: {e}")
            records.append({"center": center, "sarimax_exog_coef": np.nan})

    return pd.DataFrame(records).set_index("center")


def plot_rolling_coeffs(df_coeffs: pd.DataFrame, trend_cols: list[str], sarimax_series: pd.Series | None = None):
    fig, axes = plt.subplots(len(trend_cols), 1, figsize=(12, 2.6 * len(trend_cols)), sharex=True)
    if len(trend_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, trend_cols):
        ax.plot(df_coeffs.index, df_coeffs[col], marker="o", linestyle="-", label=col)
        ax.axvline(BREAK_DATE, color="red", linestyle="--", label="Mar 2022")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel(col)
        ax.grid(alpha=0.3)

    if sarimax_series is not None and not sarimax_series.empty:
        ax = axes[-1]
        ax2 = ax.twinx()
        ax2.plot(sarimax_series.index, sarimax_series.values, color="purple", linestyle="-.", label="SARIMAX exog coef")
        ax2.set_ylabel("SARIMAX exog coef", color="purple")

    axes[-1].set_xlabel("Center date")
    plt.suptitle("Rolling coefficients (Lasso) for Trends — center of 36-month windows")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out = FIG_DIR / "rolling_coefficients.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"Figure saved: {out}")


def main(gold_path: Path | None = None):
    if gold_path is None:
        gold_path = GOLD
    gold = pd.read_csv(gold_path, parse_dates=["month"], index_col="month")

    trend_cols = infer_trend_columns(gold, DEFAULT_KEYWORDS)
    if not trend_cols:
        raise RuntimeError("No trend columns inferred — inspect Gold dataset column names")
    logger.info(f"Using trend columns: {trend_cols}")

    df_roll = rolling_lasso_coefficients(gold, trend_cols)
    df_roll.to_csv(OUT_DIR / "rolling_coefficients.csv")
    logger.info(f"Rolling Lasso coefficients saved: {OUT_DIR / 'rolling_coefficients.csv'}")

    # Chow / two-sample test per coefficient
    chow = chow_test_per_coeff(gold, trend_cols, BREAK_DATE)
    chow.to_csv(OUT_DIR / "chow_test_results.csv")
    logger.info(f"Chow test results saved: {OUT_DIR / 'chow_test_results.csv'}")

    # Rolling SARIMAX for BESI exog if available
    exog_col = None
    for c in gold.columns:
        if "behavioral" in c.lower():
            exog_col = c
            break

    sarimax_df = None
    if exog_col:
        logger.info(f"Found exog for SARIMAX: {exog_col} — computing rolling SARIMAX coefficients")
        sarimax_df = rolling_sarimax_exog_coeff(gold, exog_col)
        sarimax_df.to_csv(OUT_DIR / "rolling_sarimax_exog_coeff.csv")
        logger.info(f"Rolling SARIMAX exog coeffs saved: {OUT_DIR / 'rolling_sarimax_exog_coeff.csv'}")

    # Merge sarimax series into df_roll for plotting convenience
    sar_series = sarimax_df["sarimax_exog_coef"] if sarimax_df is not None and not sarimax_df.empty else None
    plot_rolling_coeffs(df_roll, trend_cols, sar_series)

    logger.info("Done.")


if __name__ == "__main__":
    main()
