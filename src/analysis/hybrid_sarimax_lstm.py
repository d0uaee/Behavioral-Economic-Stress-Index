"""
src/analysis/hybrid_sarimax_lstm.py
====================================
POC : Modèle hybride SARIMAX + LSTM sur les résidus

Référence théorique :
    Zhang, G. P. (2003). Time series forecasting using a hybrid ARIMA
    and neural network model. Neurocomputing, 50, 159-175.
    https://doi.org/10.1016/S0925-2312(01)00702-0

Architecture (Zhang 2003) :
    Postulat : y_t = L_t + N_t
        L_t = composante linéaire  → captée par SARIMAX+BESI
        N_t = composante non-linéaire (résidus) → captée par LSTM
    Prédiction finale : ŷ_t = ŷ_SARIMAX(t) + ŷ_LSTM(résidu_t)

LIMITES IMPORTANTES (à citer à l'oral) :
    1. n=96 (60 train + 36 test) est très court pour un LSTM.
       La littérature recommande >= 500 observations (Zhang, 2003, §4).
    2. Risque de surapprentissage élevé sur les résidus SARIMAX.
    3. Le LSTM nécessite que les résidus soient non-linéaires — à vérifier
       par un test ARCH (Engle, 1982) avant d'appliquer ce pipeline.
    4. L'AIC n'est pas défini pour la composante LSTM :
       la comparaison AIC reste celle du SARIMAX seul.
    5. Perspective de perspective : transfer learning depuis d'autres
       pays MENA (Égypte, Tunisie) avec séries IPC similaires pourrait
       pré-entraîner le LSTM et atténuer le problème de taille d'échantillon.

Usage :
    python -m src.analysis.hybrid_sarimax_lstm

Outputs :
    results/hybrid_sarimax_lstm_results.csv
    results/figures/hybrid_comparison.png
"""

import sys
import time
import warnings
import logging
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

# ── Reproductibilité ─────────────────────────────────────────────────────────
np.random.seed(42)

# ── Chemins ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent.parent
GOLD_CSV    = ROOT / "data" / "gold" / "model_dataset_monthly.csv"
OUT_DIR     = ROOT / "results"
FIG_DIR     = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparamètres ──────────────────────────────────────────────────────────
SARIMA_ORDER    = (1, 1, 1)
SARIMA_SEASONAL = (1, 0, 1, 12)
TARGET_COL      = "ipc_level"
BESI_COL        = "behavioral_index_pure_lag1"

LOOK_BACK  = 12    # fenêtre glissante LSTM (mois)
LSTM_UNITS = 32    # unités couche LSTM
EPOCHS     = 80    # max epochs (early stopping actif)
PATIENCE   = 12    # early stopping patience
BATCH_SIZE = 8
N_SPLITS   = 5     # TimeSeriesSplit folds

# Référence BESI v1 (issue du rapport principal)
REF_RMSE_BESI = 1.976   # RMSE Bloc B SARIMAX+BESI (backtest_v3)
REF_AIC_BESI  = 57.09   # AIC SARIMAX+BESI (notebook 02_modeling_v3)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT DES DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    df = pd.read_csv(GOLD_CSV, parse_dates=["month"])
    df = df.set_index("month").sort_index()

    # Remplir le seul NaN du BESI (lag1 → premier mois sans valeur)
    df[BESI_COL] = df[BESI_COL].ffill().bfill()

    train_mask = df["split_label"].str.contains("train_B", na=False)
    test_mask  = df["split_label"].str.contains("test_B",  na=False)

    train = df[train_mask][[TARGET_COL, BESI_COL]].dropna()
    test  = df[test_mask ][[TARGET_COL, BESI_COL]].dropna()

    logger.info(f"Train B : {len(train)} obs  "
                f"({train.index.min().date()} → {train.index.max().date()})")
    logger.info(f"Test  B : {len(test)} obs  "
                f"({test.index.min().date()} → {test.index.max().date()})")
    return train, test


# ═══════════════════════════════════════════════════════════════════════════════
# 2. COMPOSANTE LINÉAIRE — SARIMAX
# ═══════════════════════════════════════════════════════════════════════════════

def fit_sarimax(y_train: pd.Series, exog_train: pd.Series):
    """Ajuste SARIMAX(1,1,1)(1,0,1)[12] + BESI sur le train.
    Retourne (modèle ajusté, résidus in-sample, AIC)."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(
        y_train,
        exog           = exog_train,
        order          = SARIMA_ORDER,
        seasonal_order = SARIMA_SEASONAL,
        enforce_stationarity = True,
        enforce_invertibility = True,
    ).fit(disp=False, maxiter=300)

    residuals = model.resid   # résidus in-sample : y_t - ŷ_SARIMAX(t)
    logger.info(f"  SARIMAX ajusté   AIC={model.aic:.2f}   "
                f"résidus std={residuals.std():.4f}")
    return model, residuals, model.aic


def predict_sarimax(model, y_hist: pd.Series, exog_hist: pd.Series,
                    y_test: pd.Series, exog_test: pd.Series) -> np.ndarray:
    """
    Prédictions SARIMAX 1-step-ahead (walk-forward) sur la période test.

    PRINCIPE (prévision as-of-date) :
        A chaque pas t, le modèle voit tous les y RÉELS jusqu'à t-1.
        C'est la convention standard du backtest walk-forward.
        → Donne RMSE comparable à la référence du rapport (≈ 1.976).

    ⚠️  Ne pas confondre avec la prévision récursive (iterated multi-step)
        où les prédictions remplacent les vraies valeurs : cette approche
        accumule les erreurs et produit RMSE >> 10 sur 36 mois.
    """
    preds = []
    y_so_far    = list(y_hist.values)
    exog_so_far = list(exog_hist.values)

    from statsmodels.tsa.statespace.sarimax import SARIMAX

    for i in range(len(exog_test)):
        res_i = SARIMAX(
            np.array(y_so_far),
            exog           = np.array(exog_so_far).reshape(-1, 1),
            order          = SARIMA_ORDER,
            seasonal_order = SARIMA_SEASONAL,
            enforce_stationarity  = True,
            enforce_invertibility = True,
        ).fit(disp=False, maxiter=200, start_params=model.params)

        fc = res_i.forecast(steps=1,
                            exog=np.array([[exog_test.iloc[i]]]))
        fc_val = fc.iloc[0] if hasattr(fc, "iloc") else float(fc[0])
        preds.append(float(fc_val))

        # ← Utilise la VRAIE valeur observée (1-step-ahead propre)
        y_so_far.append(float(y_test.iloc[i]))
        exog_so_far.append(float(exog_test.iloc[i]))

    return np.array(preds)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. COMPOSANTE NON-LINÉAIRE — LSTM SUR LES RÉSIDUS
# ═══════════════════════════════════════════════════════════════════════════════

def build_sequences(series: np.ndarray, look_back: int):
    """Construit les séquences (X, y) pour le LSTM.
    X.shape = (n_samples, look_back, 1)
    y.shape = (n_samples,)
    """
    X, y = [], []
    for i in range(len(series) - look_back):
        X.append(series[i : i + look_back])
        y.append(series[i + look_back])
    return np.array(X).reshape(-1, look_back, 1), np.array(y)


def build_lstm_model(look_back: int):
    """Architecture minimale : LSTM(32) → Dense(1).
    Choix délibérément simple pour n=60 (Zhang 2003, §4 : éviter le
    surapprentissage avec peu de données)."""
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
    except ImportError:
        raise ImportError("TensorFlow requis : pip install tensorflow")

    model = Sequential([
        Input(shape=(look_back, 1)),
        LSTM(LSTM_UNITS, return_sequences=False),
        Dropout(0.20),    # régularisation légère
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model


def fit_lstm_on_residuals(residuals: np.ndarray, look_back: int):
    """
    Entraîne le LSTM sur les résidus SARIMAX.

    LIMITE : avec n_residuals ≈ 48 (60 obs - 12 look_back),
    le LSTM dispose d'un jeu d'entraînement très réduit.
    L'early stopping est critique pour éviter le surapprentissage.
    """
    from tensorflow.keras.callbacks import EarlyStopping

    scaler = MinMaxScaler(feature_range=(-1, 1))
    res_scaled = scaler.fit_transform(residuals.reshape(-1, 1)).flatten()

    X, y = build_sequences(res_scaled, look_back)

    # ⚠️  LIMITE : validation_split=0.15 sur ~48 obs = ~7 points de validation
    model = build_lstm_model(look_back)
    es = EarlyStopping(monitor="val_loss", patience=PATIENCE,
                       restore_best_weights=True, verbose=0)
    model.fit(
        X, y,
        epochs          = EPOCHS,
        batch_size      = BATCH_SIZE,
        validation_split = 0.15,
        callbacks       = [es],
        verbose         = 0,
    )
    return model, scaler


def predict_lstm_residuals(lstm_model, scaler, residuals_hist: np.ndarray,
                           n_steps: int, look_back: int) -> np.ndarray:
    """
    Prédictions récursives du LSTM sur n_steps pas de temps.
    Utilise la fenêtre des `look_back` derniers résidus (ou résidus prédits).
    """
    res_scaled = scaler.transform(
        residuals_hist.reshape(-1, 1)
    ).flatten()

    window    = list(res_scaled[-look_back:])   # fenêtre initiale
    lstm_preds = []

    for _ in range(n_steps):
        x_in = np.array(window[-look_back:]).reshape(1, look_back, 1)
        pred = float(lstm_model.predict(x_in, verbose=0)[0, 0])
        lstm_preds.append(pred)
        window.append(pred)   # prédiction récursive (accumule les erreurs)

    # Inverse-transform
    return scaler.inverse_transform(
        np.array(lstm_preds).reshape(-1, 1)
    ).flatten()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. VALIDATION CROISÉE — TimeSeriesSplit (5 folds)
# ═══════════════════════════════════════════════════════════════════════════════

def cross_validate_hybrid(train: pd.DataFrame) -> pd.DataFrame:
    """
    TimeSeriesSplit(n_splits=5) sur le bloc train B (60 obs).
    Pour chaque fold :
        - SARIMAX in-sample sur le sous-train
        - LSTM sur résidus SARIMAX
        - RMSE fold : hybride vs SARIMAX seul

    ⚠️  LIMITE : avec 60 obs et 5 folds, chaque sous-test a ~10 points.
    Ces RMSE par fold sont indicatifs, pas conclusifs.
    """
    from tensorflow.keras.callbacks import EarlyStopping

    tss    = TimeSeriesSplit(n_splits=N_SPLITS, gap=0)
    y_all  = train[TARGET_COL].values
    ex_all = train[BESI_COL].values
    idx    = train.index

    fold_results = []

    for fold, (tr_idx, te_idx) in enumerate(tss.split(y_all)):
        n_tr = len(tr_idx)
        if n_tr < LOOK_BACK + 12:   # besoin d'au moins 24 obs train
            logger.debug(f"  Fold {fold+1} : train trop court ({n_tr}), skip")
            continue

        y_tr  = pd.Series(y_all[tr_idx],  index=idx[tr_idx])
        ex_tr = pd.Series(ex_all[tr_idx], index=idx[tr_idx])
        y_te  = y_all[te_idx]
        ex_te = pd.Series(ex_all[te_idx], index=idx[te_idx])

        try:
            # ── Étape 1 : SARIMAX ────────────────────────────────────────────
            from statsmodels.tsa.statespace.sarimax import SARIMAX

            sar = SARIMAX(
                y_tr, exog=ex_tr,
                order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL,
                enforce_stationarity=True, enforce_invertibility=True,
            ).fit(disp=False, maxiter=200)

            residuals_tr = sar.resid.values

            # Prédictions SARIMAX sur le test (1-step, rapide)
            fc = sar.forecast(steps=len(y_te), exog=ex_te)
            sar_preds = fc.values

            # ── Étape 2 : LSTM sur résidus ───────────────────────────────────
            if len(residuals_tr) <= LOOK_BACK + 4:
                fold_results.append({
                    "fold": fold + 1, "n_train": n_tr, "n_test": len(y_te),
                    "rmse_sarimax": np.sqrt(np.mean((sar_preds - y_te)**2)),
                    "rmse_hybrid": np.nan,
                    "note": "résidus insuffisants pour LSTM",
                })
                continue

            lstm_m, scaler = fit_lstm_on_residuals(residuals_tr, LOOK_BACK)
            lstm_res_preds = predict_lstm_residuals(
                lstm_m, scaler, residuals_tr, len(y_te), LOOK_BACK
            )

            hybrid_preds = sar_preds + lstm_res_preds

            rmse_sar    = float(np.sqrt(np.mean((sar_preds - y_te)**2)))
            rmse_hybrid = float(np.sqrt(np.mean((hybrid_preds - y_te)**2)))

            logger.info(f"  Fold {fold+1}  (n_train={n_tr}, n_test={len(y_te)}) | "
                        f"RMSE SARIMAX={rmse_sar:.4f}  RMSE Hybride={rmse_hybrid:.4f}  "
                        f"Gain={rmse_sar - rmse_hybrid:+.4f}")

            fold_results.append({
                "fold": fold + 1, "n_train": n_tr, "n_test": len(y_te),
                "rmse_sarimax": rmse_sar,
                "rmse_hybrid":  rmse_hybrid,
                "gain":         rmse_sar - rmse_hybrid,
                "note": "ok",
            })

        except Exception as e:
            logger.warning(f"  Fold {fold+1} échoué : {e}")
            fold_results.append({"fold": fold+1, "note": str(e)})

    return pd.DataFrame(fold_results)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ÉVALUATION FINALE — Bloc B complet
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_bloc_b(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """
    Évaluation finale sur le Bloc B test (36 mois, 2022-2024).
    Étape 1 : SARIMAX ajusté sur tout le train B (60 obs)
    Étape 2 : LSTM sur les résidus in-sample
    Étape 3 : prédictions hybrides sur test
    """
    logger.info("\n── Étape 1 : SARIMAX sur Train B complet ──")
    t0 = time.time()
    sar_model, residuals, aic = fit_sarimax(
        train[TARGET_COL],
        train[BESI_COL],
    )

    logger.info("\n── Étape 2 : Prédictions SARIMAX 1-step-ahead sur Test B ──")
    logger.info("  (prédictions récursives pas-à-pas — peut prendre 1-2 min)")
    sar_preds = predict_sarimax(
        sar_model,
        y_hist    = train[TARGET_COL],
        exog_hist = train[BESI_COL],
        y_test    = test[TARGET_COL],   # valeurs réelles observées (walk-forward propre)
        exog_test = test[BESI_COL],
    )
    y_test = test[TARGET_COL].values

    rmse_sar = float(np.sqrt(np.mean((sar_preds - y_test)**2)))
    logger.info(f"  RMSE SARIMAX seul = {rmse_sar:.4f}")

    logger.info("\n── Étape 3 : LSTM sur résidus in-sample ──")
    logger.info(f"  ⚠️  n_résidus={len(residuals)} (très court pour LSTM)")
    lstm_model, scaler = fit_lstm_on_residuals(residuals.values, LOOK_BACK)

    logger.info("\n── Étape 4 : Prédictions LSTM résidus (récursif, 36 pas) ──")
    logger.info("  ⚠️  Mode récursif : erreurs s'accumulent sur 36 mois")
    lstm_res_preds = predict_lstm_residuals(
        lstm_model, scaler, residuals.values, len(y_test), LOOK_BACK
    )

    hybrid_preds = sar_preds + lstm_res_preds
    rmse_hybrid  = float(np.sqrt(np.mean((hybrid_preds - y_test)**2)))
    elapsed      = time.time() - t0

    logger.info(f"\n  RMSE SARIMAX+BESI seul = {rmse_sar:.4f}")
    logger.info(f"  RMSE Hybride (+ LSTM)  = {rmse_hybrid:.4f}")
    logger.info(f"  Gain absolu            = {rmse_sar - rmse_hybrid:+.4f}")
    logger.info(f"  Temps total            = {elapsed:.0f}s")

    return {
        "sar_preds":    sar_preds,
        "hybrid_preds": hybrid_preds,
        "lstm_resids":  lstm_res_preds,
        "y_test":       y_test,
        "test_index":   test.index,
        "residuals":    residuals,
        "rmse_sar":     rmse_sar,
        "rmse_hybrid":  rmse_hybrid,
        "aic":          aic,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

def plot_results(eval_res: dict, cv_df: pd.DataFrame):
    """Deux panneaux :
       (A) Prédictions vs réel Bloc B
       (B) RMSE par fold (validation croisée)
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        "Hybride SARIMAX+BESI + LSTM résidus (Zhang 2003)\n"
        "⚠️  POC — n=60 train, résultats indicatifs uniquement",
        fontsize=12, fontweight="bold", color="#333333"
    )

    # ── Panneau A : prédictions ──────────────────────────────────────────────
    ax = axes[0]
    idx = eval_res["test_index"]

    ax.plot(idx, eval_res["y_test"],      color="#2C2C2C", lw=2.0,
            label="IPC réel", zorder=5)
    ax.plot(idx, eval_res["sar_preds"],   color="#ff7f0e", lw=1.6,
            ls="--", label=f"SARIMAX+BESI  (RMSE={eval_res['rmse_sar']:.3f})")
    ax.plot(idx, eval_res["hybrid_preds"],color="#2ca02c", lw=1.8,
            ls="-",  label=f"Hybride+LSTM  (RMSE={eval_res['rmse_hybrid']:.3f})")

    # Zone 2022
    ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"),
               alpha=0.08, color="red", label="Choc 2022")
    ax.set_title("Bloc B Test (2022-2024)\nPrédictions 1-step-ahead", fontsize=10)
    ax.set_xlabel("Date"); ax.set_ylabel("IPC niveau")
    ax.legend(fontsize=8.5)
    ax.grid(True, alpha=0.3)

    # ── Panneau B : RMSE par fold ────────────────────────────────────────────
    ax2 = axes[1]
    cv_ok = cv_df[cv_df["note"] == "ok"].copy()

    if not cv_ok.empty:
        x = np.arange(len(cv_ok))
        w = 0.35
        ax2.bar(x - w/2, cv_ok["rmse_sarimax"], width=w, label="SARIMAX+BESI",
                color="#ff7f0e", alpha=0.85, edgecolor="white")
        ax2.bar(x + w/2, cv_ok["rmse_hybrid"],  width=w, label="Hybride+LSTM",
                color="#2ca02c", alpha=0.85, edgecolor="white")
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"Fold {int(r)}" for r in cv_ok["fold"]], fontsize=9)
        ax2.set_ylabel("RMSE")
        ax2.set_title(
            f"TimeSeriesSplit ({N_SPLITS} folds) — Train B\n"
            "⚠️  ~10 obs/fold → intervalles larges attendus",
            fontsize=10
        )
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis="y")
    else:
        ax2.text(0.5, 0.5, "CV insuffisant\n(folds trop courts)",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=12)

    plt.tight_layout()
    path = FIG_DIR / "hybrid_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    logger.info(f"\n  Figure : {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. TABLEAU COMPARATIF + RAPPORT
# ═══════════════════════════════════════════════════════════════════════════════

def print_comparison(eval_res: dict, cv_df: pd.DataFrame):
    rmse_h = eval_res["rmse_hybrid"]
    rmse_s = eval_res["rmse_sar"]
    aic_s  = eval_res["aic"]

    print("\n" + "=" * 72)
    print("  COMPARAISON DES MODÈLES — Bloc B Test (n=36, 2022-2024)")
    print("=" * 72)
    print(f"  {'Modèle':<35} {'RMSE':>8}  {'Δ RMSE':>9}  {'AIC':>8}")
    print("  " + "-" * 65)
    print(f"  {'Référence rapport (SARIMAX+BESI)':<35} {REF_RMSE_BESI:>8.3f}  {'—':>9}  {REF_AIC_BESI:>8.2f}")
    print(f"  {'SARIMAX+BESI (ce script)':<35} {rmse_s:>8.3f}  {'—':>9}  {aic_s:>8.2f}")
    print(f"  {'Hybride SARIMAX+BESI + LSTM':<35} {rmse_h:>8.3f}  {rmse_h-rmse_s:>+9.3f}  {'N/A (*)':>8}")
    print("=" * 72)
    print("  (*) L'AIC n'est pas défini pour la composante LSTM")
    print("      Le gain RMSE doit être interprété avec précaution (n=36)")

    if cv_df is not None and "rmse_hybrid" in cv_df.columns:
        cv_ok = cv_df[cv_df["note"] == "ok"]
        if not cv_ok.empty:
            mean_gain = cv_ok["gain"].mean()
            print(f"\n  Gain moyen cross-validation ({N_SPLITS} folds) : {mean_gain:+.4f} RMSE")
            print("  (valeurs indicatives — ~10 obs/fold)")

    print("\n  LIMITES CRITIQUES (Zhang 2003 §4 + contexte BESI) :")
    print("  1. n=60 train est très en dessous du minimum recommandé pour LSTM")
    print("     → risque de surapprentissage sur les résidus")
    print("  2. Prédiction récursive LSTM sur 36 mois : accumulation d'erreurs")
    print("  3. L'AIC du SARIMAX intègre k=5 paramètres ; le LSTM en ajoute")
    print("     ~1500 (LSTM_32 + Dense_1) sans critère de sélection comparable")
    print("  4. Si ΔRMSE > 0 : le LSTM amplifie les erreurs (surapprentissage)")
    print("  5. Perspective : pré-entraîner le LSTM sur séries IPC MENA")
    print("     (Maroc, Tunisie, Égypte) → transfer learning atténuerait")
    print("     le problème de taille d'échantillon")
    print("=" * 72)


def save_results(eval_res: dict, cv_df: pd.DataFrame):
    rows = [
        {"modele": "Référence rapport (SARIMAX+BESI)",
         "rmse": REF_RMSE_BESI, "delta_rmse": 0.0,
         "aic": REF_AIC_BESI,   "note": "rapport principal"},
        {"modele": "SARIMAX+BESI (ce script)",
         "rmse": eval_res["rmse_sar"], "delta_rmse": 0.0,
         "aic": eval_res["aic"],       "note": "1-step-ahead Bloc B"},
        {"modele": "Hybride SARIMAX+BESI + LSTM résidus",
         "rmse": eval_res["rmse_hybrid"],
         "delta_rmse": eval_res["rmse_hybrid"] - eval_res["rmse_sar"],
         "aic": float("nan"),
         "note": "AIC non defini pour LSTM"},
    ]
    df_out = pd.DataFrame(rows)
    path   = OUT_DIR / "hybrid_sarimax_lstm_results.csv"
    df_out.to_csv(path, index=False)
    logger.info(f"  CSV : {path}")

    if cv_df is not None:
        cv_path = OUT_DIR / "hybrid_cv_results.csv"
        cv_df.to_csv(cv_path, index=False)
        logger.info(f"  CV  : {cv_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 72)
    print("  POC — Modèle Hybride SARIMAX + LSTM Résidus (Zhang 2003)")
    print("  ⚠️  Code prototype — non destiné à la production")
    print("=" * 72 + "\n")

    # 0. Données
    train, test = load_data()

    # 1. Validation croisée
    logger.info("\n[1/2] Validation croisée TimeSeriesSplit (5 folds)...")
    logger.info("  ⚠️  Sur n=60, chaque fold a ~10 obs test — indicatif uniquement")
    cv_df = cross_validate_hybrid(train)

    # 2. Évaluation finale Bloc B
    logger.info("\n[2/2] Évaluation finale — Bloc B test complet (n=36)...")
    eval_res = evaluate_bloc_b(train, test)

    # 3. Rapport
    print_comparison(eval_res, cv_df)
    save_results(eval_res, cv_df)
    plot_results(eval_res, cv_df)

    print("\n  Référence théorique :")
    print("  Zhang, G. P. (2003). Time series forecasting using a hybrid")
    print("  ARIMA and neural network model. Neurocomputing, 50, 159-175.")
    print("  https://doi.org/10.1016/S0925-2312(01)00702-0\n")


if __name__ == "__main__":
    main()
