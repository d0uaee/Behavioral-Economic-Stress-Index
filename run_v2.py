"""
run_v2.py — Script de lancement de la comparaison v2 (sans Reddit/YouTube simulés)

Usage :
    python run_v2.py

Ce script :
1. Tente de télécharger l'IPC réel depuis la Banque Mondiale (FP.CPI.TOTL, MA)
2. Calcule BESI_trends = 0.70*Trends + 0.30*|IPC_change|
3. Lance compare_models_v2() : Naif / SARIMA / SARIMAX_T / SARIMAX_BT
4. Sauvegarde :
   - outputs/reports/model_comparison_v2.csv
   - outputs/reports/period_performance_v2.csv
   - outputs/reports/data_sources.txt
   - outputs/figures/compare_all_predictions_v2.png
   - outputs/figures/period_performance_v2.png
   - outputs/figures/gain_vs_sarima_v2.png
"""

import warnings
import sys
from pathlib import Path

# Ajouter le dossier src au path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # mode non-interactif pour sauvegarde propre

np.random.seed(42)
warnings.filterwarnings("ignore")

from models import compare_models_v2
from deep_learning import build_lstm, compare_window_sizes

if __name__ == "__main__":
    print("=" * 65)
    print("  BESI Maroc — Comparaison modèles v2")
    print("  Correction : Google Trends uniquement (sans Reddit/YouTube)")
    print("=" * 65)

    df_cmp, df_period = compare_models_v2(
        series       = None,     # chargé automatiquement depuis data/processed/
        master_df    = None,
        train_start  = "2015-01-01",
        train_end    = "2021-12-01",
        test_end     = "2024-12-01",
        horizons     = [1],
        try_worldbank = True,
        save_fig     = True,
    )

    # ── Charger les données master pour LSTM ──────────────────────────────────
    master_path = ROOT / "data" / "processed" / "master_dataset.csv"
    if master_path.exists():
        master = pd.read_csv(master_path, parse_dates=["date"], index_col="date")
        master.index.freq = "MS"
        ipc_series = master["ipc"]
        besi_series = master[["besi"]]

        print("\n" + "=" * 65)
        print("  LSTM — Entraînement de base")
        print("=" * 65)

        # LSTM sans exogènes
        print("\n>>> LSTM (IPC seul)")
        lstm_ipc = build_lstm(
            series=ipc_series,
            exog=None,
            look_back=12,
            train_end="2021-12-01",
            epochs=50,
            save_fig=True,
        )

        # LSTM avec BESI
        print("\n>>> LSTM + BESI")
        lstm_besi = build_lstm(
            series=ipc_series,
            exog=besi_series,
            look_back=12,
            train_end="2021-12-01",
            epochs=50,
            save_fig=False,
        )

        # Comparaison des tailles de fenêtre glissante
        print("\n" + "=" * 65)
        print("  LSTM — Comparaison des tailles de fenêtre")
        print("=" * 65)

        df_window_comp = compare_window_sizes(
            series=ipc_series,
            exog=besi_series,
            window_sizes=[6, 12, 18, 24],
            train_end="2021-12-01",
            epochs=50,
            save_fig=True,
        )
    else:
        print(f"\n[WARNING] Fichier master_dataset.csv manquant : {master_path}")
        lstm_ipc = lstm_besi = df_window_comp = None

    # ── Prophet ────────────────────────────────────────────────────────────────
    from prophet_model import train_prophet
    print("\n" + "=" * 65)
    print("  PROPHET — Prévision IPC Maroc")
    print("=" * 65)
    prophet_results = train_prophet(master, train_end="2021-12-01")
    if prophet_results:
        print(f"  RMSE : {prophet_results['rmse']:.5f}")
        print(f"  MAE  : {prophet_results['mae']:.5f}")
        print(f"  MAPE : {prophet_results['mape']:.2f}%")

    print("\n" + "=" * 65)
    print("  RÉSUMÉ FINAL — Comparaison v2")
    print("=" * 65)

    # Tableau de synthèse
    h1_cols = [c for c in df_cmp.columns if "h1" in c]
    print("\nMétriques globales (h=1) :")
    print(df_cmp[["AIC", "BIC"] + h1_cols].to_string())

    # Meilleur modèle
    rmse_col = "RMSE_h1"
    if rmse_col in df_cmp.columns:
        best = df_cmp[rmse_col].idxmin()
        best_rmse = df_cmp.loc[best, rmse_col]
        sarima_rmse = df_cmp.loc["SARIMA", rmse_col] if "SARIMA" in df_cmp.index else None
        print(f"\n  Meilleur modele : {best}  (RMSE = {best_rmse:.5f})")
        if sarima_rmse:
            gain = (sarima_rmse - best_rmse) / sarima_rmse * 100
            print(f"  Gain vs SARIMA  : {gain:+.1f}%")

    # Tableau sous-périodes
    print("\nPerformances par sous-période :")
    print(df_period.to_string(index=False))

    # ── Résumé LSTM ──────────────────────────────────────────────────────────
    if lstm_ipc and lstm_besi:
        print("\n" + "=" * 65)
        print("  RÉSUMÉ LSTM")
        print("=" * 65)
        print(f"  LSTM (IPC seul)  : RMSE={lstm_ipc['rmse']:.5f}  MAE={lstm_ipc['mae']:.5f}  MAPE={lstm_ipc['mape']:.2f}%")
        print(f"  LSTM + BESI      : RMSE={lstm_besi['rmse']:.5f}  MAE={lstm_besi['mae']:.5f}  MAPE={lstm_besi['mape']:.2f}%")

        if df_window_comp is not None:
            print("\n  Comparaison fenêtres glissantes :")
            for _, row in df_window_comp.iterrows():
                typ = "avec BESI" if row["type"] == "avec_exog" else "sans exog"
                print(f"    {int(row['window_size']):2d} mois ({typ:<10}) : RMSE={row['rmse']:.5f}")

    # ── Résumé Prophet ───────────────────────────────────────────────────────
    if prophet_results:
        print("\n" + "=" * 65)
        print("  RÉSUMÉ PROPHET")
        print("=" * 65)
        print(f"  Prophet : RMSE={prophet_results['rmse']:.5f}  MAE={prophet_results['mae']:.5f}  MAPE={prophet_results['mape']:.2f}%")

    rep_dir = ROOT / "outputs" / "reports"
    print(f"\n  Fichiers sauvegardés dans {rep_dir}/")
    for f in ["model_comparison_v2.csv", "period_performance_v2.csv", "data_sources.txt"]:
        p = rep_dir / f
        status = "OK" if p.exists() else "MANQUANT"
        print(f"    [{status}]  {f}")

    # Fichiers LSTM
    if lstm_ipc:
        lstm_files = ["lstm_window_comparison.csv"]
        for f in lstm_files:
            p = rep_dir / f
            status = "OK" if p.exists() else "MANQUANT"
            print(f"    [{status}]  {f}")

    fig_dir = ROOT / "outputs" / "figures"
    for f in ["compare_all_predictions_v2.png", "period_performance_v2.png",
              "gain_vs_sarima_v2.png"]:
        p = fig_dir / f
        status = "OK" if p.exists() else "MANQUANT"
        print(f"    [{status}]  figures/{f}")

    # Figures LSTM
    if lstm_ipc:
        lstm_figs = ["lstm_predictions.png", "lstm_window_comparison.png"]
        for f in lstm_figs:
            p = fig_dir / f
            status = "OK" if p.exists() else "MANQUANT"
            print(f"    [{status}]  figures/{f}")

    # Figures Prophet
    if prophet_results:
        prophet_figs = ["prophet_forecast.png"]
        for f in prophet_figs:
            p = fig_dir / f
            status = "OK" if p.exists() else "MANQUANT"
            print(f"    [{status}]  figures/{f}")

    print("\n  Terminé.")
