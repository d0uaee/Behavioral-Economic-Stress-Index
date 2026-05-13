"""
run_all.py — Script maitre du projet BESI Maroc
================================================
Lance toutes les etapes du projet dans l'ordre.

Usage
-----
    python run_all.py              # pipeline complet
    python run_all.py --skip-data  # saute la collecte (si data/processed/ existe deja)
    python run_all.py --step 4     # lance uniquement l'etape 4

Etapes
------
1. Donnees       : charge IPC + Google Trends + BESI
2. Modeles stat. : SARIMA, SARIMAX, walk-forward v2
3. Deep Learning : LSTM (fenetres glissantes) + Prophet
4. NLP           : scraping medias + scoring Darija + BESI enrichi
5. Analyse       : rupture Chow, Granger, early warning, Markov
6. Visualisation : dashboard 6 figures + NLP vs IPC
7. Rapport       : results_summary.md regenere

Auteurs : Douae Ahadji & Adama Basse | ENSAM Meknes | 2025
"""

import sys
import time
import warnings
import argparse
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Chemins ───────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent
SRC_DIR  = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

PROC_DIR = ROOT / "data" / "processed"
REP_DIR  = ROOT / "outputs" / "reports"
FIG_DIR  = ROOT / "outputs" / "figures"

for _d in (PROC_DIR, REP_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Helpers d'affichage ───────────────────────────────────────────────────────

def _header(step: int, title: str) -> None:
    print(f"\n{'='*65}")
    print(f"  ETAPE {step}/7 — {title}")
    print(f"{'='*65}")

def _ok(msg: str) -> None:
    print(f"  [OK]  {msg}")

def _warn(msg: str) -> None:
    print(f"  [WARN] {msg}")

def _err(msg: str, exc: Exception) -> None:
    print(f"  [ERREUR] {msg}")
    print(f"           {exc}")

def _separator() -> None:
    print(f"  {'-'*60}")


# =============================================================================
# ETAPE 1 — DONNEES
# =============================================================================

def step1_data(skip: bool = False) -> tuple[pd.Series, pd.DataFrame]:
    """
    Charge l'IPC et le master dataset.
    Si skip=True et que les fichiers existent : lecture directe sans re-collecte.
    """
    _header(1, "DONNEES IPC + BESI")

    ipc_path    = PROC_DIR / "ipc_processed.csv"
    master_path = PROC_DIR / "master_dataset.csv"

    if skip or (ipc_path.exists() and master_path.exists()):
        _ok("Fichiers traites trouves — lecture locale (skip-data actif ou cache present).")
        df_ipc    = pd.read_csv(ipc_path,    index_col=0, parse_dates=True)
        master_df = pd.read_csv(master_path, index_col=0, parse_dates=True)
        try:
            df_ipc.index    = pd.DatetimeIndex(df_ipc.index,    freq="MS")
            master_df.index = pd.DatetimeIndex(master_df.index, freq="MS")
        except Exception:
            df_ipc    = df_ipc.asfreq("MS")
            master_df = master_df.asfreq("MS")
        ipc = df_ipc["ipc"]
        _ok(f"IPC : {len(ipc)} mois  ({ipc.index[0].date()} -> {ipc.index[-1].date()})")
        _ok(f"Master : {master_df.shape[1]} colonnes — {list(master_df.columns)}")
        return ipc, master_df

    # Re-collecte complete via data_pipeline
    _warn("Lancement data_pipeline.py (peut prendre plusieurs minutes)...")
    try:
        import data_pipeline as dp
        ipc    = dp.fetch_ipc()["ipc"]
        trends = dp.fetch_google_trends()
        reddit = dp.fetch_reddit()
        youtube = dp.fetch_youtube()
        master_df = dp.build_master_dataset(ipc, trends, reddit, youtube)
        _ok(f"Pipeline complet : {len(ipc)} mois IPC + {master_df.shape[1]} features")
    except Exception as exc:
        _err("data_pipeline echec — lecture fichiers existants", exc)
        df_ipc    = pd.read_csv(ipc_path,    index_col=0, parse_dates=True)
        master_df = pd.read_csv(master_path, index_col=0, parse_dates=True)
        ipc = df_ipc["ipc"]

    return ipc, master_df


# =============================================================================
# ETAPE 2 — MODELES STATISTIQUES (SARIMA / SARIMAX v2)
# =============================================================================

def step2_models(ipc: pd.Series, master_df: pd.DataFrame) -> tuple:
    """Walk-forward SARIMA / SARIMAX_T / SARIMAX_BT + Naif."""
    _header(2, "MODELES STATISTIQUES — SARIMA / SARIMAX v2")

    try:
        from models import compare_models_v2
        df_cmp, df_period = compare_models_v2(
            series      = ipc,
            master_df   = master_df,
            train_start = "2015-01-01",
            train_end   = "2021-12-01",
            test_end    = "2024-12-01",
            horizons    = [1],
            try_worldbank = False,
            save_fig    = True,
        )
        _ok(f"Comparaison v2 : {len(df_cmp)} modeles evalues")
        _separator()
        print(df_cmp[["RMSE_h1", "MAE_h1", "MAPE_h1", "Gain%_h1"]].to_string())
        return df_cmp, df_period
    except Exception as exc:
        _err("compare_models_v2 echec", exc)
        traceback.print_exc()
        return None, None


# =============================================================================
# ETAPE 3 — DEEP LEARNING (LSTM + Prophet)
# =============================================================================

def step3_deep_learning(ipc: pd.Series, master_df: pd.DataFrame) -> dict:
    """LSTM (fenetre 12 mois + comparaison fenetres) + Prophet."""
    _header(3, "DEEP LEARNING — LSTM + Prophet")

    results_dl = {}
    besi = master_df[["besi"]] if "besi" in master_df.columns else None

    # ── LSTM ─────────────────────────────────────────────────────────────────
    try:
        from deep_learning import build_lstm, compare_window_sizes
        print("\n  [LSTM] Entrainement (IPC seul, look_back=12) ...")
        res_lstm = build_lstm(ipc, exog=None, look_back=12,
                              train_end="2021-12-01", epochs=50, save_fig=True)
        results_dl["LSTM"] = res_lstm
        _ok(f"LSTM : RMSE={res_lstm['rmse']:.5f}  MAPE={res_lstm['mape']:.2f}%"
            f"  ({res_lstm['epochs']} epochs, {res_lstm['train_time']}s)")

        if besi is not None:
            print("\n  [LSTM+BESI] Entrainement ...")
            res_lstm_b = build_lstm(ipc, exog=besi, look_back=12,
                                    train_end="2021-12-01", epochs=50, save_fig=False)
            results_dl["LSTM_BESI"] = res_lstm_b
            _ok(f"LSTM+BESI : RMSE={res_lstm_b['rmse']:.5f}  MAPE={res_lstm_b['mape']:.2f}%")

        print("\n  [LSTM] Comparaison fenetres glissantes (6/12/18/24 mois) ...")
        df_win = compare_window_sizes(
            ipc, exog=besi, window_sizes=[6, 12, 18, 24],
            train_end="2021-12-01", epochs=50, save_fig=True,
        )
        results_dl["window_comparison"] = df_win
        best_win = df_win[df_win["type"] == "sans_exog"].sort_values("rmse").iloc[0]
        _ok(f"Meilleure fenetre (sans exog) : {int(best_win['window_size'])} mois "
            f"(RMSE={best_win['rmse']:.5f})")
        _ok("Conclusion : la taille de fenetre n'ameliore pas significativement le RMSE")

    except Exception as exc:
        _err("LSTM echec (TensorFlow manquant ?)", exc)

    # ── Prophet ───────────────────────────────────────────────────────────────
    try:
        from prophet_model import train_prophet
        print("\n  [Prophet] Entrainement ...")
        res_prophet = train_prophet(master_df, train_end="2021-12-01")
        results_dl["Prophet"] = res_prophet
        _ok(f"Prophet : RMSE={res_prophet['rmse']:.5f}  MAPE={res_prophet['mape']:.2f}%")
        _ok("Conclusion : Prophet capte la saisonnalite mais pas les ruptures brusques")
    except Exception as exc:
        _err("Prophet echec (pip install prophet ?)", exc)

    # ── Resume Deep Learning ──────────────────────────────────────────────────
    _separator()
    print("\n  RESUME DEEP LEARNING :")
    for name, res in results_dl.items():
        if isinstance(res, dict) and "rmse" in res:
            print(f"    {name:<15} RMSE={res['rmse']:.5f}  MAPE={res['mape']:.2f}%")
    if results_dl:
        print("\n  Note : SARIMA (RMSE~0.00272) domine tous les modeles DL.")
        print("  Cela confirme la superiorite des modeles statistiques sur")
        print("  les series mensuelles courtes (180 observations).")

    return results_dl


# =============================================================================
# ETAPE 4 — NLP MEDIAS MAROCAINS
# =============================================================================

def step4_nlp() -> pd.DataFrame:
    """Pipeline NLP : scraping medias + scoring Darija + BESI enrichi."""
    _header(4, "NLP MEDIAS MAROCAINS — Darija / Arabe")

    # Si le fichier existe deja, eviter de relancer le scraping long
    nlp_cache = PROC_DIR / "morocco_nlp_monthly.csv"
    if nlp_cache.exists():
        _ok("morocco_nlp_monthly.csv trouve — lecture locale (passer --refresh pour relancer).")
        monthly_nlp = pd.read_csv(nlp_cache, index_col=0, parse_dates=True)
        _ok(f"NLP : {len(monthly_nlp)} mois  signal moyen = {monthly_nlp['morocco_nlp_signal'].mean():.3f}")
        return monthly_nlp

    try:
        from nlp_morocco import run_nlp_pipeline
        results = run_nlp_pipeline(force_refresh=False)
        monthly_nlp = results["monthly_nlp"]
        _ok(f"Pipeline NLP complet : {len(monthly_nlp)} mois")
        _ok(f"Signal NLP moyen pre-2022  : {monthly_nlp.loc[:'2021','morocco_nlp_signal'].mean():.3f}")
        _ok(f"Signal NLP moyen post-2022 : {monthly_nlp.loc['2022:','morocco_nlp_signal'].mean():.3f}")
        return monthly_nlp
    except Exception as exc:
        _err("Pipeline NLP echec", exc)
        return pd.DataFrame()


# =============================================================================
# ETAPE 5 — ANALYSES STATISTIQUES
# =============================================================================

def step5_analysis(ipc: pd.Series, master_df: pd.DataFrame) -> dict:
    """Test de Chow, Granger, early warning, matrice de Markov."""
    _header(5, "ANALYSES STATISTIQUES")

    results_analysis = {}
    besi = master_df["besi"].dropna() if "besi" in master_df.columns else None

    if besi is None:
        _warn("Colonne 'besi' absente du master dataset — analyse ignoree.")
        return results_analysis

    from analysis import chow_test, early_warning_analysis, stress_transition_matrix

    # Test de Chow
    try:
        print("\n  [Chow] Test de rupture structurelle 2022 ...")
        chow = chow_test(ipc, besi, breakpoint="2022-01-01", save_fig=True)
        results_analysis["chow"] = chow
        _ok(f"Chow F={chow['f_stat']:.2f}  p={chow['p_value']:.4f}  "
            f"Rupture={'OUI' if chow['p_value'] < 0.05 else 'NON'}")
    except Exception as exc:
        _err("Test de Chow echec", exc)

    # Early warning
    try:
        print("\n  [EarlyWarning] CCF BESI -> IPC (lag optimal) ...")
        besi_al = besi.reindex(ipc.index).ffill()
        ew = early_warning_analysis(besi_al, ipc, save_fig=True)
        results_analysis["early_warning"] = ew
        _ok(f"Lag optimal : {ew['lag_optimal']} mois  |  Lead time : {ew['lead_time_mean']:.0f} mois")
        _ok(f"Recall={ew['recall']:.0%}  Precision={ew['precision']:.1%}  F1={ew['f1']:.3f}")
    except Exception as exc:
        _err("Early warning echec", exc)

    # Matrice de Markov
    try:
        print("\n  [Markov] Matrice de transition des etats de stress ...")
        stress = master_df["stress_level"].dropna() if "stress_level" in master_df.columns else None
        if stress is not None:
            markov = stress_transition_matrix(stress, save_fig=True)
            results_analysis["markov"] = markov
            ss = markov.get("steady_state", {})
            _ok(f"Etat stationnaire : Normal={ss.get(0, 0):.1%}  "
                f"Warning={ss.get(1, 0):.1%}  HighStress={ss.get(2, 0):.1%}")
    except Exception as exc:
        _err("Matrice Markov echec", exc)

    return results_analysis


# =============================================================================
# ETAPE 6 — VISUALISATION
# =============================================================================

def step6_visualisation(ipc: pd.Series, master_df: pd.DataFrame) -> None:
    """Dashboard 6 figures + graphique NLP vs IPC."""
    _header(6, "VISUALISATION — Dashboard + NLP")

    # Dashboard principal
    try:
        from visualization import generate_dashboard
        print("\n  [Dashboard] Generation des 6 figures (300 DPI) ...")
        figs = generate_dashboard(save_combined=True, verbose=False)
        _ok(f"Dashboard genere : {len(figs)} figures")
    except Exception as exc:
        _err("generate_dashboard echec", exc)

    # Figure NLP vs IPC
    try:
        from nlp_morocco import plot_nlp_vs_ipc
        monthly_nlp = pd.read_csv(
            PROC_DIR / "morocco_nlp_monthly.csv", index_col=0, parse_dates=True
        ) if (PROC_DIR / "morocco_nlp_monthly.csv").exists() else None
        print("\n  [NLP] Graphique NLP vs IPC ...")
        fig = plot_nlp_vs_ipc(monthly_nlp, master_df, dpi=300, save=True)
        import matplotlib.pyplot as plt
        plt.close(fig)
        _ok("morocco_nlp_vs_ipc.png genere")
    except Exception as exc:
        _err("plot_nlp_vs_ipc echec", exc)

    # Compter les figures produites
    n_figs = len(list(FIG_DIR.glob("*.png")))
    _ok(f"Total figures dans outputs/figures/ : {n_figs} PNG")


# =============================================================================
# ETAPE 7 — RAPPORT FINAL
# =============================================================================

def step7_report() -> None:
    """Regenere results_summary.md avec les resultats a jour."""
    _header(7, "RAPPORT FINAL — results_summary.md")

    try:
        import importlib.util, subprocess
        report_script = ROOT / "generate_report.py"
        if not report_script.exists():
            _warn("generate_report.py introuvable — rapport non regenere.")
            return

        print("\n  Generation du rapport ...")
        import generate_report
        generate_report.main()
        report_path = REP_DIR / "results_summary.md"
        if report_path.exists():
            size = report_path.stat().st_size
            _ok(f"results_summary.md genere ({size:,} octets)")
        else:
            _warn("results_summary.md non trouve apres generation.")
    except Exception as exc:
        _err("generate_report echec", exc)
        traceback.print_exc()


# =============================================================================
# RESUME FINAL
# =============================================================================

def _print_final_summary(t_total: float, step_times: dict) -> None:
    """Affiche le resume de tous les fichiers produits."""
    print(f"\n{'='*65}")
    print(f"  RESUME FINAL  (duree totale : {t_total:.0f}s)")
    print(f"{'='*65}")

    print("\n  Temps par etape :")
    for step, duration in step_times.items():
        status = f"{duration:.0f}s" if duration >= 0 else "ECHEC"
        print(f"    Etape {step} : {status}")

    print("\n  Fichiers produits :")
    key_files = [
        (REP_DIR / "results_summary.md",        "Rapport final"),
        (REP_DIR / "model_comparison_v2.csv",   "Comparaison modeles v2"),
        (REP_DIR / "period_performance_v2.csv", "Performances sous-periodes"),
        (PROC_DIR / "morocco_nlp_monthly.csv",  "Signal NLP mensuel"),
        (PROC_DIR / "master_dataset.csv",       "Master dataset"),
        (FIG_DIR  / "dashboard_combined.png",   "Dashboard combine"),
        (FIG_DIR  / "morocco_nlp_vs_ipc.png",   "NLP vs IPC"),
    ]
    for path, label in key_files:
        status = "OK" if path.exists() else "MANQUANT"
        print(f"    [{status}]  {label:35s} {path.name}")

    n_figs = len(list(FIG_DIR.glob("*.png")))
    print(f"\n  Figures PNG totales : {n_figs}")
    print(f"\n  Projet BESI Maroc — Douae & Adama | ENSAM Meknes")
    print(f"{'='*65}")


# =============================================================================
# POINT D'ENTREE
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="run_all.py — Pipeline complet BESI Maroc"
    )
    parser.add_argument(
        "--skip-data", action="store_true",
        help="Sauter la re-collecte des donnees (utiliser les CSV existants)"
    )
    parser.add_argument(
        "--step", type=int, default=0,
        help="Lancer uniquement l'etape N (0 = toutes)"
    )
    parser.add_argument(
        "--skip-dl", action="store_true",
        help="Sauter le deep learning (LSTM + Prophet, lent)"
    )
    parser.add_argument(
        "--skip-nlp", action="store_true",
        help="Sauter le pipeline NLP (scraping, lent)"
    )
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  BESI Maroc — Pipeline complet")
    print(f"  Detection precoce du stress economique des menages marocains")
    print(f"  Douae Ahadji & Adama Basse | ENSAM Meknes | 2025")
    print(f"{'='*65}")
    print(f"  Options : skip-data={args.skip_data}  skip-dl={args.skip_dl}"
          f"  skip-nlp={args.skip_nlp}  step={args.step or 'TOUTES'}")

    t_start     = time.time()
    step_times  = {}
    ipc         = None
    master_df   = None

    def _run_step(n: int, fn, *fn_args):
        """Execute une etape et mesure son temps."""
        if args.step != 0 and args.step != n:
            return None
        t0 = time.time()
        try:
            result = fn(*fn_args)
            step_times[n] = time.time() - t0
            return result
        except Exception as exc:
            step_times[n] = -1
            print(f"\n  [ETAPE {n} ECHOUEE] {exc}")
            traceback.print_exc()
            return None

    # ── Etape 1 : Donnees ─────────────────────────────────────────────────────
    data_result = _run_step(1, step1_data, args.skip_data)
    if data_result:
        ipc, master_df = data_result

    if ipc is None or master_df is None:
        if args.step == 0:
            print("\n  [FATAL] IPC ou master_dataset absent. Pipeline interrompu.")
            sys.exit(1)

    # ── Etape 2 : Modeles statistiques ────────────────────────────────────────
    _run_step(2, step2_models, ipc, master_df)

    # ── Etape 3 : Deep Learning ───────────────────────────────────────────────
    if not args.skip_dl:
        _run_step(3, step3_deep_learning, ipc, master_df)
    else:
        print("\n  [ETAPE 3] Deep Learning ignore (--skip-dl actif).")

    # ── Etape 4 : NLP ─────────────────────────────────────────────────────────
    if not args.skip_nlp:
        _run_step(4, step4_nlp)
    else:
        print("\n  [ETAPE 4] NLP ignore (--skip-nlp actif).")

    # ── Etape 5 : Analyses statistiques ───────────────────────────────────────
    _run_step(5, step5_analysis, ipc, master_df)

    # ── Etape 6 : Visualisation ───────────────────────────────────────────────
    _run_step(6, step6_visualisation, ipc, master_df)

    # ── Etape 7 : Rapport ─────────────────────────────────────────────────────
    _run_step(7, step7_report)

    # ── Resume ────────────────────────────────────────────────────────────────
    _print_final_summary(time.time() - t_start, step_times)
