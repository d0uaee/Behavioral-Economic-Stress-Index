"""
run_v3.py — Pipeline complet BESI Maroc V3

Orchestrateur principal : exécute toutes les étapes dans l'ordre correct.

Usage :
    python run_v3.py                          # pipeline complet (auto-détection plage)
    python run_v3.py --step ingest            # ingestion uniquement
    python run_v3.py --step transform         # transforms bronze → silver
    python run_v3.py --step indexes           # construction indices BESI v3
    python run_v3.py --step gold              # assemblage Gold dataset
    python run_v3.py --step backtest          # backtest walk-forward
    python run_v3.py --step warnings          # métriques d'alerte précoce
    python run_v3.py --step all               # tout (défaut)
    python run_v3.py --skip-ingest            # saute l'ingestion (données déjà présentes)
    python run_v3.py --start-date 2017-01-01  # forcer le début de plage (données partielles)

Mode données partielles (IPC 2017-2024 uniquement) :
    python run_v3.py --skip-ingest --start-date 2017-01-01
    → Active automatiquement SHORT_EVAL_WINDOWS (2 blocs : COVID + inflation)
    → Fonctionne sans données 2010-2016

Architecture des données :
    Bronze : data/bronze/   ← raw, jamais modifié
    Silver : data/silver/   ← nettoyé, standardisé
    Gold   : data/gold/     ← prêt modélisation (lags, targets, split labels)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt = "%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("run_v3.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("run_v3")

ROOT       = Path(__file__).resolve().parent
BRONZE_DIR = ROOT / "data" / "bronze"
SILVER_DIR = ROOT / "data" / "silver"
GOLD_DIR   = ROOT / "data" / "gold"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    bar = "─" * 60
    logger.info(f"\n{bar}\n  {title}\n{bar}")


def _ok(msg: str) -> None:
    logger.info(f"  ✓ {msg}")


def _warn(msg: str) -> None:
    logger.warning(f"  ⚠ {msg}")


def _fail(step: str, exc: Exception, fatal: bool = True) -> None:
    logger.error(f"  ✗ {step} ÉCHOUÉ : {exc}")
    if fatal:
        sys.exit(1)


# ─── Étapes ───────────────────────────────────────────────────────────────────

def step_ingest(skip_existing: bool = True) -> None:
    """Télécharge / vérifie les données Bronze."""
    _section("ÉTAPE 1 — INGESTION Bronze")

    # ── IPC HCP ──────────────────────────────────────────────────────────────
    cpi_bronze = BRONZE_DIR / "cpi_hcp_monthly_raw.csv"
    if cpi_bronze.exists() and skip_existing:
        _ok(f"IPC HCP bronze déjà présent : {cpi_bronze}")
    else:
        logger.info("  Chargement IPC HCP depuis fichier CSV local...")
        logger.warning(
            "  !! IPC HCP doit être téléchargé manuellement depuis hcp.ma\n"
            f"     → Placer le CSV dans : {cpi_bronze}\n"
            "     Format attendu : colonnes 'date' (YYYY-MM-01) + 'ipc_level'"
        )

    # ── FAO Food Price Index ──────────────────────────────────────────────────
    fao_bronze = BRONZE_DIR / "fao_food_price_raw.csv"
    if fao_bronze.exists() and skip_existing:
        _ok(f"FAO bronze déjà présent : {fao_bronze}")
    else:
        logger.info("  Téléchargement FAO Food Price Index...")
        try:
            from src.ingestion.fao import ingest_fao_fpi
            df_fao = ingest_fao_fpi(output_path=fao_bronze)
            _ok(f"FAO téléchargé : {len(df_fao)} mois")
        except Exception as e:
            _fail("FAO ingest", e, fatal=False)
            _warn("FAO non disponible — colonnes macro FAO seront NaN")

    # ── BAM FX ───────────────────────────────────────────────────────────────
    fx_bronze = BRONZE_DIR / "bam_fx_raw.csv"
    if fx_bronze.exists() and skip_existing:
        _ok(f"BAM FX bronze déjà présent : {fx_bronze}")
    else:
        logger.info("  Chargement taux de change MAD/EUR...")
        try:
            from src.ingestion.bam_fx import ingest_bam_fx
            df_fx = ingest_bam_fx(output_path=fx_bronze)
            _ok(f"BAM FX chargé : {len(df_fx)} mois")
        except Exception as e:
            _fail("BAM FX ingest", e, fatal=False)
            _warn("FX non disponible — fx_yoy sera NaN")

    # ── Google Trends v3 ─────────────────────────────────────────────────────
    trends_bronze = BRONZE_DIR / "google_trends_raw_v3.csv"
    if trends_bronze.exists() and skip_existing:
        _ok(f"Trends v3 bronze déjà présent : {trends_bronze}")
    else:
        logger.info("  Téléchargement Google Trends v3...")
        try:
            from src.ingestion.google_trends_v3 import ingest_google_trends_v3
            df_tr = ingest_google_trends_v3(output_path=trends_bronze)
            _ok(f"Trends v3 téléchargé : {len(df_tr)} mois")
        except Exception as e:
            _fail("Google Trends v3 ingest", e, fatal=False)
            _warn("Trends non disponible — indices comportementaux seront NaN")

    _ok("Ingestion terminée")


def step_transform() -> None:
    """Transforme Bronze → Silver."""
    _section("ÉTAPE 2 — TRANSFORM Bronze → Silver")

    # ── CPI Silver ────────────────────────────────────────────────────────────
    # Chercher d'abord le fichier bronze standardisé, puis le fichier brut HCP
    cpi_bronze = BRONZE_DIR / "cpi_hcp_monthly_raw.csv"
    cpi_raw    = BRONZE_DIR / "ipc_hcp_raw.csv"

    if not cpi_bronze.exists() and cpi_raw.exists():
        # Ingestion du fichier brut HCP → bronze standardisé
        logger.info("  Ingestion du fichier IPC HCP brut...")
        try:
            from src.ingestion.cpi_hcp import ingest_cpi_hcp
            ingest_cpi_hcp(filepath=cpi_raw, output_path=cpi_bronze)
            _ok(f"Fichier HCP ingéré → {cpi_bronze.name}")
        except Exception as e:
            _fail("CPI HCP ingest", e)
            return

    if not cpi_bronze.exists():
        _fail("CPI transform", FileNotFoundError(
            f"Bronze IPC introuvable : {cpi_bronze}\n"
            f"  Option 1 : placer votre CSV HCP dans data/bronze/ipc_hcp_raw.csv\n"
            f"  Option 2 : placer un fichier déjà standardisé dans {cpi_bronze}"
        ))
        return

    try:
        from src.transforms.cpi import transform_cpi
        df_cpi = transform_cpi()
        _ok(f"CPI Silver : {len(df_cpi)} mois, colonnes={list(df_cpi.columns)}")
    except Exception as e:
        _fail("CPI transform", e)

    # ── Trends Silver ─────────────────────────────────────────────────────────
    try:
        from src.transforms.trends import transform_trends
        df_tr = transform_trends()
        _ok(f"Trends Silver : {len(df_tr)} mois, sous-indices={[c for c in df_tr.columns if c.startswith('trends_')]}")
    except Exception as e:
        _fail("Trends transform", e, fatal=False)
        _warn("Trends Silver absent — BESI comportemental indisponible")

    # ── Macro Silver ──────────────────────────────────────────────────────────
    try:
        from src.transforms.macro import transform_macro
        df_mac = transform_macro()
        _ok(f"Macro Silver : {len(df_mac)} mois, colonnes={list(df_mac.columns)}")
    except Exception as e:
        _fail("Macro transform", e, fatal=False)
        _warn("Macro Silver absent — FAO/FX indisponibles")

    _ok("Transforms terminés")


def step_indexes() -> None:
    """Construit les indices BESI v3 (behavioral_pure + hybrid_macro)."""
    _section("ÉTAPE 3 — INDICES BESI v3")

    try:
        import pandas as pd
        from src.features.indexes import build_behavioral_index_pure, build_hybrid_macro_index

        # Charger les silver
        trends_path = SILVER_DIR / "google_trends_monthly.csv"
        macro_path  = SILVER_DIR / "macro_signals_monthly.csv"
        cpi_path    = SILVER_DIR / "cpi_monthly.csv"

        if not trends_path.exists():
            raise FileNotFoundError(f"Trends Silver absent : {trends_path}")

        trends_df = pd.read_csv(trends_path,  parse_dates=["month"], index_col="month")
        cpi_df    = pd.read_csv(cpi_path,     parse_dates=["month"], index_col="month") \
                    if cpi_path.exists() else None

        # behavioral_index_pure
        method = "lasso" if cpi_df is not None else "equal"
        beh = build_behavioral_index_pure(
            trends_df  = trends_df,
            cpi_silver = cpi_df,
            method     = method,
        )
        # Sauvegarder
        beh.to_frame().to_csv(SILVER_DIR / "behavioral_index_pure.csv")
        _ok(f"behavioral_index_pure ({method}) : mean={beh.mean():.3f}  std={beh.std():.3f}")

        # hybrid_macro_index
        if macro_path.exists():
            macro_df = pd.read_csv(macro_path, parse_dates=["month"], index_col="month")
            hyb = build_hybrid_macro_index(
                behavioral_series = beh,
                macro_df          = macro_df,
                cpi_silver        = cpi_df,
                method            = method,
            )
            hyb.to_frame().to_csv(SILVER_DIR / "hybrid_macro_index.csv")
            _ok(f"hybrid_macro_index ({method}) : mean={hyb.mean():.3f}  std={hyb.std():.3f}")
        else:
            _warn("Macro Silver absent — hybrid_macro_index non construit")

    except Exception as e:
        _fail("BESI indexes", e)

    _ok("Indices BESI v3 construits")


def step_gold(start_date: str = "2010-01-01", end_date: str = "2024-12-01") -> None:
    """Assemble le Gold dataset (lags + targets + split labels)."""
    _section("ÉTAPE 4 — GOLD Dataset")

    try:
        from src.gold.build_model_dataset import build_gold_dataset
        df = build_gold_dataset(start_date=start_date, end_date=end_date)
        _ok(f"Gold dataset : {df.shape[0]} mois × {df.shape[1]} colonnes")
        _ok(f"Targets : {[c for c in df.columns if 'target' in c]}")
        _ok(f"Split distribution :\n{df['split_label'].value_counts().to_string()}")
    except Exception as e:
        _fail("Gold dataset", e)


def step_backtest() -> None:
    """Lance le backtest walk-forward sur 3 blocs."""
    _section("ÉTAPE 5 — BACKTEST Walk-Forward")

    try:
        from src.evaluation.backtest import run_backtest
        df = run_backtest()
        _ok(f"Backtest terminé : {len(df)} lignes de résultats")
    except Exception as e:
        _fail("Backtest", e)


def step_warnings() -> None:
    """Calcule les métriques d'alerte précoce (ROC-AUC, F1, lead-time)."""
    _section("ÉTAPE 6 — MÉTRIQUES D'ALERTE PRÉCOCE")

    try:
        from src.evaluation.warning_metrics import compute_warning_metrics
        df = compute_warning_metrics()
        _ok(f"Warning metrics terminés : {len(df)} lignes")
    except Exception as e:
        _fail("Warning metrics", e)


# ─── Pipeline complet ──────────────────────────────────────────────────────────

def run_all(skip_ingest: bool = False, start_date: str = "2010-01-01") -> None:
    t0 = time.time()
    logger.info("\n" + "═" * 60)
    logger.info("  BESI MAROC V3 — PIPELINE COMPLET")
    mode = "FULL (2010-2024)" if start_date <= "2012-01-01" else f"SHORT ({start_date[:4]}-2024)"
    logger.info(f"  Mode données : {mode}")
    logger.info("═" * 60)

    if not skip_ingest:
        step_ingest()

    step_transform()
    step_indexes()
    step_gold(start_date=start_date)
    step_backtest()
    step_warnings()

    elapsed = time.time() - t0
    logger.info(f"\n{'═'*60}")
    logger.info(f"  Pipeline V3 terminé en {elapsed:.1f}s")
    logger.info(f"  Résultats → outputs/reports/ et outputs/figures/")
    logger.info(f"{'═'*60}\n")

    _print_final_checklist()


def _print_final_checklist() -> None:
    """Vérifie que tous les outputs attendus ont bien été produits."""
    expected = {
        "Gold dataset":           GOLD_DIR / "model_dataset_monthly.csv",
        "BESI behavioral index":  SILVER_DIR / "behavioral_index_pure.csv",
        "BESI hybrid index":      SILVER_DIR / "hybrid_macro_index.csv",
        "Backtest results":       ROOT / "outputs" / "reports" / "backtest_v3_results.csv",
        "Backtest summary":       ROOT / "outputs" / "reports" / "backtest_v3_summary.csv",
        "Warning metrics":        ROOT / "outputs" / "reports" / "warning_metrics_v3.csv",
        "ROC curves":             ROOT / "outputs" / "figures" / "roc_curves_v3.png",
        "PR curves":              ROOT / "outputs" / "figures" / "precision_recall_v3.png",
        "Threshold analysis":     ROOT / "outputs" / "figures" / "threshold_analysis_v3.png",
        "Backtest predictions":   ROOT / "outputs" / "figures" / "backtest_v3_predictions.png",
        "BESI weights (beh)":     ROOT / "outputs" / "reports" / "besi_v3_behavioral_weights.csv",
        "BESI weights (hyb)":     ROOT / "outputs" / "reports" / "besi_v3_hybrid_weights.csv",
    }

    print("\n" + "─" * 55)
    print("CHECKLIST OUTPUTS V3")
    print("─" * 55)
    all_ok = True
    for name, path in expected.items():
        status = "✓" if path.exists() else "✗"
        if not path.exists():
            all_ok = False
        print(f"  {status}  {name:<35} {'OK' if path.exists() else 'MANQUANT'}")

    print("─" * 55)
    if all_ok:
        print("  Tous les outputs présents — V3 complet ✓")
    else:
        print("  Certains outputs manquants — vérifier les logs ci-dessus.")
    print()


# ─── Point d'entrée ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BESI Maroc V3 — Pipeline de prédiction inflation"
    )
    parser.add_argument(
        "--step",
        choices=["ingest", "transform", "indexes", "gold", "backtest", "warnings", "all"],
        default="all",
        help="Étape à exécuter (défaut : all)",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Sauter l'ingestion (données bronze déjà présentes)",
    )
    parser.add_argument(
        "--start-date",
        default="2010-01-01",
        metavar="YYYY-MM-DD",
        help=(
            "Date de début de la plage d'analyse (défaut : 2010-01-01). "
            "Utiliser 2017-01-01 si les données IPC HCP ne sont disponibles "
            "que depuis 2017 — active automatiquement SHORT_EVAL_WINDOWS."
        ),
    )
    args = parser.parse_args()

    dispatch = {
        "ingest":    lambda: step_ingest(skip_existing=True),
        "transform": step_transform,
        "indexes":   step_indexes,
        "gold":      lambda: step_gold(start_date=args.start_date),
        "backtest":  step_backtest,
        "warnings":  step_warnings,
        "all":       lambda: run_all(skip_ingest=args.skip_ingest, start_date=args.start_date),
    }

    dispatch[args.step]()


if __name__ == "__main__":
    main()
