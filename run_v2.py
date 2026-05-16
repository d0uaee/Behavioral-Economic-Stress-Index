"""
run_v2.py — Pipeline complet BESI Maroc (v2 — réponse critiques prof)

Améliorations v2.1 :
  - Sous-indices Trends thématiques : prix / inflation / stress ménages
  - BESI_enrichi soumis à validation walk-forward formelle
  - Deep Learning étendu : LSTM / GRU / BiLSTM / XGBoost TS
  - Prophet maintenu comme benchmark alternatif

Usage :
    python run_v2.py

Exports produits :
  outputs/reports/model_comparison_v2.csv        ← SARIMA/SARIMAX (8 modèles)
  outputs/reports/period_performance_v2.csv
  outputs/reports/dl_comparison.csv              ← LSTM/GRU/BiLSTM/XGBoost
  outputs/reports/prophet_results.csv
  outputs/figures/compare_all_predictions_v2.png
  outputs/figures/dl_comparison_rmse.png
  outputs/figures/dl_predictions_all.png
"""

import warnings
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # mode non-interactif — pas de fenêtre popup

np.random.seed(42)
warnings.filterwarnings("ignore")

from models import compare_models_v2
from deep_learning import compare_all_dl_models


if __name__ == "__main__":

    # ════════════════════════════════════════════════════════════════════════
    # PARTIE 1 — SARIMA / SARIMAX (sous-indices Trends + BESI_enrichi)
    # ════════════════════════════════════════════════════════════════════════
    print("=" * 68)
    print("  BESI Maroc v2.1 — Comparaison modèles")
    print("  Sous-indices Trends thématiques + BESI_enrichi + DL étendu")
    print("=" * 68)

    df_cmp, df_period = compare_models_v2(
        series        = None,          # chargé automatiquement
        master_df     = None,
        train_start   = "2015-01-01",
        train_end     = "2021-12-01",
        test_end      = "2024-12-01",
        horizons      = [1],
        try_worldbank = True,
        save_fig      = True,
    )

    # ════════════════════════════════════════════════════════════════════════
    # PARTIE 2 — DEEP LEARNING ÉTENDU
    # ════════════════════════════════════════════════════════════════════════
    master_path = ROOT / "data" / "processed" / "master_dataset.csv"
    df_dl = None

    if master_path.exists():
        master = pd.read_csv(master_path, parse_dates=["date"], index_col="date")
        try:
            master.index = pd.DatetimeIndex(master.index, freq="MS")
        except Exception:
            master.index = pd.DatetimeIndex(master.index)
            master = master.asfreq("MS")

        ipc_series = master["ipc"]

        # Exog : besi_trends (signal le plus propre, 100% réel)
        exog_col = "besi_trends" if "besi_trends" in master.columns else "besi"
        exog_dl  = master[[exog_col]] if exog_col in master.columns else None

        print("\n" + "=" * 68)
        print("  DEEP LEARNING — LSTM / GRU / BiLSTM / XGBoost TS")
        print(f"  Exog utilisé : {exog_col if exog_dl is not None else 'aucun'}")
        print("=" * 68)

        df_dl = compare_all_dl_models(
            series    = ipc_series,
            exog      = exog_dl,
            look_back = 12,
            train_end = "2021-12-01",
            epochs    = 50,
            save_fig  = True,
        )
    else:
        print(f"\n[WARNING] master_dataset.csv manquant : {master_path}")
        print("  Lancer d'abord : python src/data_pipeline.py")

    # ════════════════════════════════════════════════════════════════════════
    # PARTIE 3 — PROPHET (benchmark alternatif)
    # ════════════════════════════════════════════════════════════════════════
    prophet_results = None
    try:
        from prophet_model import train_prophet
        print("\n" + "=" * 68)
        print("  PROPHET — Benchmark alternatif")
        print("=" * 68)
        if master_path.exists():
            prophet_results = train_prophet(master, train_end="2021-12-01")
            if prophet_results:
                print(f"  RMSE  : {prophet_results['rmse']:.5f}")
                print(f"  MAE   : {prophet_results['mae']:.5f}")
                print(f"  MAPE  : {prophet_results['mape']:.2f}%")
    except Exception as e:
        print(f"  [WARN] Prophet non disponible : {e}")

    # ════════════════════════════════════════════════════════════════════════
    # RÉSUMÉ FINAL
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 68)
    print("  RÉSUMÉ FINAL COMPLET")
    print("=" * 68)

    # — SARIMA/SARIMAX ─────────────────────────────────────────────────────
    h1_cols = [c for c in df_cmp.columns if "h1" in c.lower()]
    print("\n[ SARIMA / SARIMAX — Métriques h=1 ]")
    print(df_cmp[["AIC", "BIC"] + h1_cols].to_string())

    rmse_col = "RMSE_h1"
    if rmse_col in df_cmp.columns:
        best_sarimax      = df_cmp[rmse_col].idxmin()
        best_rmse_sarimax = df_cmp.loc[best_sarimax, rmse_col]
        sarima_rmse = df_cmp.loc["SARIMA", rmse_col] if "SARIMA" in df_cmp.index else None
        print(f"\n  Meilleur SARIMAX : {best_sarimax}  (RMSE={best_rmse_sarimax:.5f})")
        if sarima_rmse:
            gain = (sarima_rmse - best_rmse_sarimax) / sarima_rmse * 100
            print(f"  Gain vs SARIMA   : {gain:+.1f}%")

    # — Deep Learning ─────────────────────────────────────────────────────
    if df_dl is not None:
        print("\n[ DEEP LEARNING — Résumé ]")
        print(df_dl[["Modele", "RMSE", "MAE", "MAPE%", "Temps_s"]].to_string(index=False))
        best_dl = df_dl.iloc[0]
        print(f"\n  Meilleur DL : {best_dl['Modele']}  (RMSE={best_dl['RMSE']:.5f})")
        if sarima_rmse:
            gain_dl = (sarima_rmse - best_dl["RMSE"]) / sarima_rmse * 100
            sign = "meilleur" if gain_dl > 0 else "moins bon"
            print(f"  vs SARIMA    : {gain_dl:+.1f}%  ({sign})")

    # — Prophet ───────────────────────────────────────────────────────────
    if prophet_results:
        print(f"\n[ PROPHET ]  RMSE={prophet_results['rmse']:.5f}  "
              f"MAE={prophet_results['mae']:.5f}  "
              f"MAPE={prophet_results['mape']:.2f}%")

    # — Vérification fichiers exportés ────────────────────────────────────
    print("\n[ Fichiers exportés ]")
    rep_dir = ROOT / "outputs" / "reports"
    fig_dir = ROOT / "outputs" / "figures"

    report_files = [
        "model_comparison_v2.csv",
        "period_performance_v2.csv",
        "dl_comparison.csv",
        "prophet_results.csv",
    ]
    figure_files = [
        "compare_all_predictions_v2.png",
        "period_performance_v2.png",
        "gain_vs_sarima_v2.png",
        "dl_comparison_rmse.png",
        "dl_predictions_all.png",
        "lstm_predictions.png",
        "gru_predictions.png",
        "bilstm_predictions.png",
        "xgboost_ts_predictions.png",
        "prophet_forecast.png",
    ]

    for f in report_files:
        p = rep_dir / f
        print(f"  [{'OK' if p.exists() else 'MANQUANT':>8}]  reports/{f}")
    for f in figure_files:
        p = fig_dir / f
        print(f"  [{'OK' if p.exists() else 'MANQUANT':>8}]  figures/{f}")

    print("\n  Terminé — run_v2.py v2.1")
