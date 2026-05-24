"""
LSTM pour la prevision de l'IPC Maroc — Comparaison avec SARIMA/SARIMAX
Session 7-8 : deep_learning.py

But : comparaison equitable avec les modeles statistiques, pas optimisation LSTM.
      Le LSTM tourne sur CPU en moins de 2 minutes.

Fonctions principales
---------------------
build_lstm(series, exog, look_back=12)  -> dict
    Entraine un LSTM 2 couches + Dense, evalue sur le meme split que SARIMA.
compare_all_models(results_dict)        -> pd.DataFrame
    Tableau final SARIMA / SARIMAX / LSTM / Naif avec interpretabilite et temps.
"""

import time
import itertools
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

np.random.seed(42)

# ── Chemins ───────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "outputs" / "figures"
MOD_DIR = ROOT / "outputs" / "models"
REP_DIR = ROOT / "outputs" / "reports"
for _d in (FIG_DIR, MOD_DIR, REP_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Palette coherente avec models.py / analysis.py
_COL_REAL    = "#2C2C2C"
_COL_SARIMA  = "#2C5F8A"
_COL_SARIMAX = "#E07B39"
_COL_LSTM    = "#9467BD"
_COL_NAIVE   = "#8C8C8C"
_COL_TRAIN   = "#2CA02C"


# ── Helpers metriques ─────────────────────────────────────────────────────────

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))

def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    mask   = yt != 0
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100)

def _model_color(name: str) -> str:
    """Couleur coherente avec la palette du projet."""
    nl = name.lower()
    if "lstm" in nl:    return _COL_LSTM
    if "naive" in nl:   return _COL_NAIVE
    if "sarimax" in nl: return _COL_SARIMAX
    return _COL_SARIMA


# =============================================================================
# 1. LSTM
# =============================================================================

def _build_single_lstm(data_arr, s, train_end, look_back, lstm_units_1,
                       lstm_units_2, dropout, learning_rate, batch_size,
                       n_features, scaler, tf_imports):
    try:
        Sequential    = tf_imports['Sequential']
        Input         = tf_imports['Input']
        LSTM          = tf_imports['LSTM']
        Dropout_layer = tf_imports['Dropout']
        Dense         = tf_imports['Dense']
        EarlyStopping = tf_imports['EarlyStopping']
        Adam          = tf_imports['Adam']

        te         = pd.Timestamp(train_end)
        X_all, y_all, dates_all = [], [], []
        for i in range(len(data_arr) - look_back):
            X_all.append(data_arr[i : i + look_back])
            y_all.append(data_arr[i + look_back, 0])
            dates_all.append(s.index[i + look_back])
        X_all     = np.array(X_all)
        y_all     = np.array(y_all)
        dates_all = pd.DatetimeIndex(dates_all)

        train_mask = dates_all <= te
        test_mask  = dates_all >  te
        if train_mask.sum() < max(look_back + 3, 15) or test_mask.sum() < 5:
            return None

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test,  y_test  = X_all[test_mask],  y_all[test_mask]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = Sequential([
                Input(shape=(look_back, n_features)),
                LSTM(lstm_units_1, return_sequences=True),
                Dropout_layer(dropout),
                LSTM(lstm_units_2),
                Dense(1)
            ])
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")

            early_stop = EarlyStopping(monitor="val_loss", patience=10,
                                       restore_best_weights=True, verbose=0)
            t0 = time.time()
            history = model.fit(X_train, y_train, epochs=150, batch_size=batch_size,
                                validation_split=0.15, callbacks=[early_stop],
                                shuffle=False, verbose=0)
            train_time = time.time() - t0

        def _inv(norm_vals):
            dummy = np.zeros((len(norm_vals), n_features))
            dummy[:, 0] = norm_vals
            return scaler.inverse_transform(dummy)[:, 0]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_pred_inv = _inv(model.predict(X_test, verbose=0).ravel())
        y_true_inv = _inv(y_test)

        # Vérifier les NaN
        if np.isnan(y_pred_inv).any() or np.isnan(y_true_inv).any():
            return None
        
        rmse = _rmse(y_true_inv, y_pred_inv)
        mae  = _mae(y_true_inv,  y_pred_inv)
        mape = _mape(y_true_inv, y_pred_inv)
        
        # Vérifier si les métriques sont valides
        if np.isnan(rmse) or np.isinf(rmse) or rmse > 100:
            return None

        return {
            "rmse": rmse,
            "mae":  mae,
            "mape": mape,
            "y_true": y_true_inv, "y_pred": y_pred_inv,
            "test_dates": dates_all[test_mask],
            "epochs_run": len(history.history["loss"]),
            "history": history.history,
            "model": model,
            "look_back": look_back,
            "n_features": n_features,
            "train_time": round(train_time, 1),
        }
    except Exception as e:
        return None


def build_lstm(
    series,
    exog=None,
    train_end="2021-12-01",
    epochs=150,
    save_fig=True,
) -> dict:
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.optimizers import Adam
        from sklearn.preprocessing import MinMaxScaler
        tf.random.set_seed(42)
        tf_imports = dict(Sequential=Sequential, Input=Input, LSTM=LSTM,
                          Dropout=Dropout, Dense=Dense,
                          EarlyStopping=EarlyStopping, Adam=Adam)
    except ImportError as exc:
        raise ImportError(
            "TensorFlow et scikit-learn sont requis.\n"
            "Installation : pip install tensorflow scikit-learn\n"
            f"Erreur : {exc}"
        ) from exc

    # Preparation donnees
    s = series.dropna().copy()
    month_sin = pd.Series(np.sin(2*np.pi*s.index.month/12), index=s.index)
    month_cos = pd.Series(np.cos(2*np.pi*s.index.month/12), index=s.index)

    if exog is not None:
        exog_al  = exog.reindex(s.index).ffill().bfill()
        common   = s.index.intersection(exog_al.dropna().index)
        s        = s.loc[common]
        exog_al  = exog_al.loc[common]
        month_sin = month_sin.loc[common]
        month_cos = month_cos.loc[common]
        data_arr = np.column_stack([s.values, exog_al.values,
                                    month_sin.values, month_cos.values])
    else:
        data_arr = np.column_stack([s.values, month_sin.values, month_cos.values])

    n_features = data_arr.shape[1]

    # Normalisation fit sur train uniquement (pas de leakage)
    te        = pd.Timestamp(train_end)
    train_idx = s.index <= te
    scaler    = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data_arr[train_idx])
    data_norm = scaler.transform(data_arr)

    # Grille de recherche
    param_grid = {
        "look_back"    : [6, 12, 18, 24],
        "lstm_units_1" : [32, 64, 128],
        "lstm_units_2" : [16, 32, 64],
        "dropout"      : [0.1, 0.2, 0.3],
        "learning_rate": [0.001, 0.0005],
        "batch_size"   : [8, 16, 32],
    }
    combos = list(itertools.product(*param_grid.values()))
    total  = len(combos)
    print(f"\nGridSearch LSTM : {total} combinaisons")
    print(f"Features        : {n_features} (IPC + month_sin/cos"
          f"{' + exog' if exog is not None else ''})")
    print(f"Scaler          : fit sur train uniquement (no leakage)")

    gs_results = []
    best_rmse  = np.inf
    best_res   = None
    best_combo = None

    for i, (lb, u1, u2, dr, lr, bs) in enumerate(combos):
        res = _build_single_lstm(data_norm, s, train_end, lb,
                                 u1, u2, dr, lr, bs,
                                 n_features, scaler, tf_imports)
        if res is None:
            continue

        gs_results.append({
            "look_back": lb, "lstm_units_1": u1, "lstm_units_2": u2,
            "dropout": dr, "learning_rate": lr, "batch_size": bs,
            "rmse": res["rmse"], "mae": res["mae"], "mape": res["mape"],
            "epochs_run": res["epochs_run"],
            "train_time_s": res["train_time"],
        })

        if res["rmse"] < best_rmse:
            best_rmse  = res["rmse"]
            best_res   = res
            best_combo = {"look_back": lb, "lstm_units_1": u1,
                          "lstm_units_2": u2, "dropout": dr,
                          "learning_rate": lr, "batch_size": bs}

        print(f"  [{i+1:4d}/{total}] lb={lb:2d} u=({u1:3d},{u2:2d}) "
              f"dr={dr} lr={lr} bs={bs:2d} "
              f"-> RMSE={res['rmse']:.5f} ep={res['epochs_run']:3d} "
              f"[BEST={best_rmse:.5f}]")

    df_gs = pd.DataFrame(gs_results).sort_values("rmse").reset_index(drop=True)

    sep = "=" * 70
    print(f"\n{sep}")
    print("  TOP 10 — GridSearch LSTM")
    print(sep)
    print(df_gs.head(10).to_string(index=False))
    print(sep)

    csv_path = REP_DIR / "gridsearch_lstm_results.csv"
    df_gs.to_csv(csv_path, index=False)
    print(f"  CSV : {csv_path}")

    print(f"\n  MEILLEUR LSTM")
    for k, v in best_combo.items():
        print(f"    {k:<16}: {v}")
    print(f"    {'rmse':<16}: {best_res['rmse']:.5f}")
    print(f"    {'mae':<16}: {best_res['mae']:.5f}")
    print(f"    {'mape':<16}: {best_res['mape']:.2f}%")

    try:
        best_res["model"].save(str(MOD_DIR / "lstm_best.keras"))
        print(f"  Modele sauvegarde : {MOD_DIR / 'lstm_best.keras'}")
    except Exception as e:
        print(f"  [WARN] {e}")

    return {
        "rmse": best_res["rmse"], "mae": best_res["mae"],
        "mape": best_res["mape"], "y_true": best_res["y_true"],
        "y_pred": best_res["y_pred"], "test_dates": best_res["test_dates"],
        "train_time": best_res["train_time"],
        "n_params": best_res["model"].count_params(),
        "epochs": best_res["epochs_run"], "history": best_res["history"],
        "model": best_res["model"], "look_back": best_combo["look_back"],
        "n_features": n_features, "best_params": best_combo,
        "gridsearch_df": df_gs,
    }


def plot_gridsearch_results(build_lstm_result, series, train_end="2021-12-01"):
    best_res   = build_lstm_result
    best_combo = build_lstm_result["best_params"]
    df_gs      = build_lstm_result["gridsearch_df"]
    te         = pd.Timestamp(train_end)

    # Figure 1 : Predictions meilleur modele
    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.52, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    s       = series.dropna()
    s_train = s[s.index <= te]
    s_test  = s[s.index >  te]

    ax1.plot(s_train.index, s_train.values, color="lightgray", lw=1.2,
             label="IPC train")
    ax1.plot(best_res["test_dates"], best_res["y_true"],
             color=_COL_REAL, lw=2.0, label="IPC reel (test)")
    ax1.plot(best_res["test_dates"], best_res["y_pred"],
             color=_COL_LSTM, lw=1.8, ls="--",
             label=f"LSTM predit  RMSE={best_res['rmse']:.5f}")
    ax1.axvline(te, color="red", lw=1.2, ls="--", alpha=0.6,
                label=f"Coupure {te.date()}")
    ax1.set_title(
        f"Meilleur LSTM — lb={best_combo['look_back']} "
        f"u=({best_combo['lstm_units_1']},{best_combo['lstm_units_2']}) "
        f"dr={best_combo['dropout']} lr={best_combo['learning_rate']} "
        f"bs={best_combo['batch_size']}",
        fontsize=9, fontweight="bold"
    )
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    ax2.plot(best_res["history"]["loss"],
             color=_COL_LSTM, lw=1.8, label="Loss train")
    ax2.plot(best_res["history"]["val_loss"],
             color="#D62728", lw=1.5, ls="--", label="Loss val")
    ax2.set_title("Courbe apprentissage", fontsize=9)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    lims = (min(best_res["y_true"].min(), best_res["y_pred"].min()) * 0.999,
            max(best_res["y_true"].max(), best_res["y_pred"].max()) * 1.001)
    ax3.scatter(best_res["y_true"], best_res["y_pred"],
                color=_COL_LSTM, alpha=0.75, s=30)
    ax3.plot(lims, lims, color="black", lw=1.0, ls="--", alpha=0.5,
             label="Prediction parfaite")
    ax3.set_xlim(*lims); ax3.set_ylim(*lims)
    ax3.set_xlabel("IPC reel"); ax3.set_ylabel("IPC predit")
    ax3.set_title("Pred vs Reel (test)", fontsize=9)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    fig.suptitle("GridSearch LSTM — Meilleur modele", fontsize=11, fontweight="bold")
    p1 = FIG_DIR / "lstm_gridsearch_best.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    print(f"  Figure 1 : {p1}")
    plt.close(fig)

    # Figure 2 : Distribution RMSE
    naive_rmse = _rmse(s_test.values[1:], s_test.values[:-1])
    fig2, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df_gs["rmse"], bins=30, color=_COL_LSTM, alpha=0.7, edgecolor="white")
    ax.axvline(best_res["rmse"], color="red", lw=2,
               label=f"Meilleur RMSE = {best_res['rmse']:.5f}")
    ax.axvline(naive_rmse, color="gray", lw=1.5, ls="--",
               label=f"Naif RMSE = {naive_rmse:.5f}")
    ax.set_xlabel("RMSE")
    ax.set_ylabel("Nombre de combinaisons")
    ax.set_title(f"Distribution RMSE — GridSearch LSTM ({len(df_gs)} combinaisons)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    p2 = FIG_DIR / "gridsearch_rmse_distribution.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    print(f"  Figure 2 : {p2}")
    plt.close(fig2)

    # Figure 3 : Heatmap look_back vs units
    top100 = df_gs.head(100)
    fig3, ax3b = plt.subplots(figsize=(10, 6))
    sc = ax3b.scatter(
        top100["look_back"], top100["lstm_units_1"],
        s=top100["lstm_units_2"] * 3,
        c=top100["rmse"], cmap="viridis_r", alpha=0.75
    )
    plt.colorbar(sc, ax=ax3b, label="RMSE")
    ax3b.set_xlabel("look_back (mois)")
    ax3b.set_ylabel("lstm_units_1")
    ax3b.set_title("GridSearch — look_back vs units_1 (taille=units_2, couleur=RMSE)",
                   fontsize=10, fontweight="bold")
    ax3b.grid(True, alpha=0.3)
    p3 = FIG_DIR / "gridsearch_heatmap.png"
    fig3.savefig(p3, dpi=150, bbox_inches="tight")
    print(f"  Figure 3 : {p3}")
    plt.close(fig3)


# =============================================================================
# 2. COMPARAISON FINALE DE TOUS LES MODELES
# =============================================================================

# Table d'interpretabilite et de complexite par type de modele
_INTERPRET_MAP = {
    "lstm":    "Faible",
    "naive":   "Haute",
    "sarimax": "Haute",
    "sarima":  "Haute",
}
_COMPLEXITY_MAP = {
    "lstm":    "Haute",
    "naive":   "Tres faible",
    "sarimax": "Moyenne",
    "sarima":  "Moyenne",
}


def _classify(name: str, lookup: dict, default: str = "Moyenne") -> str:
    nl = name.lower()
    for key, val in lookup.items():
        if key in nl:
            return val
    return default


def compare_all_models(
    results_dict: dict,
    series:    "pd.Series | None" = None,
    train_end: str  = "2021-12-01",
    save_fig:  bool = True,
) -> pd.DataFrame:
    """
    Tableau recapitulatif de tous les modeles testes sur le projet.

    Ajoute automatiquement :
    - Le modele naif (Random Walk : IPC_pred[t] = IPC[t-1]) si 'series' est fourni
    - La colonne 'Interpretabilite' : Haute / Moyenne / Faible
    - La colonne 'Complexite'       : Tres faible / Moyenne / Haute
    - Le gain relatif de chaque modele vs le naif (en RMSE)

    Parametres
    ----------
    results_dict : dict {nom_modele: dict_resultats}
        Chaque dict doit contenir au moins : rmse, mae, mape
        Optionnel : aic, train_time, y_true, y_pred, test_dates, n_params
        Exemple :
            {
              "SARIMA":        {"rmse": 0.00206, "mae": ..., "mape": ..., "aic": -1091},
              "SARIMAX_BESI":  {"rmse": 0.00252, ...},
              "LSTM":          build_lstm(...),   # dict retourne par build_lstm()
            }
    series    : serie IPC pour calculer le modele naif (si absent de results_dict)
    train_end : coupure train/test (doit correspondre aux autres modeles)
    save_fig  : sauvegarder 3 figures (barres metriques, predictions, radar)

    Retourne
    --------
    DataFrame trie par RMSE croissant.
    Sauvegarde : outputs/reports/model_comparison_final.csv
    """

    # ── Construire la table des metriques ─────────────────────────────────────
    rows: dict = {}
    for name, res in results_dict.items():
        rows[name] = {
            "RMSE":    res.get("rmse",       np.nan),
            "MAE":     res.get("mae",        np.nan),
            "MAPE":    res.get("mape",       np.nan),
            "AIC":     res.get("aic",        np.nan),
            "Temps_s": res.get("train_time", np.nan),
        }

    # ── Modele naif (Random Walk) ─────────────────────────────────────────────
    naive_key = next((k for k in rows if "naive" in k.lower()), None)
    if naive_key is None and series is not None:
        te    = pd.Timestamp(train_end)
        s     = series.dropna()
        test_s = s[s.index > te]
        if len(test_s) > 1:
            yt_naive = test_s.values[1:]
            yp_naive = test_s.values[:-1]
            rows["Naive (RW)"] = {
                "RMSE":    _rmse(yt_naive, yp_naive),
                "MAE":     _mae(yt_naive,  yp_naive),
                "MAPE":    _mape(yt_naive, yp_naive),
                "AIC":     np.nan,
                "Temps_s": 0.0,
            }
            naive_key = "Naive (RW)"
            # Ajouter les predictions pour la figure
            results_dict["Naive (RW)"] = {
                **rows["Naive (RW)"],
                "y_true":     yt_naive,
                "y_pred":     yp_naive,
                "test_dates": test_s.index[1:],
            }

    # ── DataFrame ─────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows).T.copy()
    df.index.name = "Modele"
    for col in ["RMSE", "MAE", "MAPE", "AIC", "Temps_s"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Interpretabilite"] = [
        _classify(n, _INTERPRET_MAP) for n in df.index
    ]
    df["Complexite"] = [
        _classify(n, _COMPLEXITY_MAP) for n in df.index
    ]

    # Trier par RMSE croissant
    df = df.sort_values("RMSE", na_position="last")

    # ── Affichage console ─────────────────────────────────────────────────────
    sep = "=" * 82
    print(f"\n{sep}")
    print("  COMPARAISON FINALE -- SARIMA / SARIMAX / LSTM / NAIF")
    print(sep)
    print(
        f"  {'Modele':<20}  {'RMSE':>9}  {'MAE':>9}  {'MAPE%':>7}  "
        f"{'AIC':>10}  {'Temps(s)':>9}  {'Interpret.':>12}  {'Complexite':>12}"
    )
    print(f"  {'-'*78}")
    for name, row in df.iterrows():
        def _fmt(v, fmt):
            return f"{v:{fmt}}" if not (isinstance(v, float) and np.isnan(v)) else "  n/a  "
        print(
            f"  {name:<20}  "
            f"{_fmt(row['RMSE'],    '.5f'):>9}  "
            f"{_fmt(row['MAE'],     '.5f'):>9}  "
            f"{_fmt(row['MAPE'],    '.2f'):>7}  "
            f"{_fmt(row['AIC'],     '.1f'):>10}  "
            f"{_fmt(row['Temps_s'], '.1f') + 's':>9}  "
            f"{row['Interpretabilite']:>12}  "
            f"{row['Complexite']:>12}"
        )
    print(sep)

    # Meilleur modele
    best_name = df["RMSE"].idxmin()
    best_rmse = df.loc[best_name, "RMSE"]
    print(f"\n  Meilleur modele (RMSE) : {best_name}  (RMSE = {best_rmse:.5f})")

    # Gain vs naif
    if naive_key and naive_key in df.index:
        naive_rmse = df.loc[naive_key, "RMSE"]
        print(f"\n  Gain RMSE vs modele naif ({naive_key},  RMSE={naive_rmse:.5f}) :")
        for name, row in df.iterrows():
            if "naive" not in name.lower() and not np.isnan(row["RMSE"]):
                gain = (naive_rmse - row["RMSE"]) / naive_rmse * 100
                sign = "(meilleur)" if gain > 0 else "(moins bon)"
                print(f"    {name:<22} : {gain:+.1f}%  {sign}")
    print(sep)

    # CSV
    csv_path = REP_DIR / "model_comparison_final.csv"
    df.to_csv(csv_path)
    print(f"\n  CSV sauvegarde : {csv_path}")

    # ── Figures ───────────────────────────────────────────────────────────────
    if save_fig:
        model_names = list(df.index)
        colors      = [_model_color(n) for n in model_names]

        # Figure 1 : barres RMSE / MAE / MAPE ─────────────────────────────────
        fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
        fig1.suptitle(
            "Comparaison finale : SARIMA vs SARIMAX vs LSTM vs Naif",
            fontsize=10, fontweight="bold",
        )
        for ax, (metric, col, fmt) in zip(
            axes1,
            [("RMSE", "RMSE", ".5f"), ("MAE", "MAE", ".5f"), ("MAPE (%)", "MAPE", ".2f")],
        ):
            vals = [df.loc[n, col] for n in model_names]
            bars = ax.bar(range(len(model_names)), vals, color=colors, alpha=0.85)
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        v * 1.012,
                        f"{v:{fmt}}",
                        ha="center", fontsize=6.5, fontweight="bold",
                    )
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=7.5)
            ax.set_title(metric, fontsize=10)
            ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        p1 = FIG_DIR / "compare_all_models.png"
        fig1.savefig(p1, dpi=150, bbox_inches="tight")
        print(f"  Figure sauvegardee : {p1}")
        plt.close(fig1)

        # Figure 2 : predictions superposees sur le test ───────────────────────
        preds_available = [
            (name, results_dict[name])
            for name in model_names
            if name in results_dict
            and "y_pred" in results_dict[name]
            and results_dict[name].get("y_pred") is not None
            and "test_dates" in results_dict[name]
        ]
        if preds_available:
            fig2, ax2 = plt.subplots(figsize=(13, 5))
            fig2.suptitle(
                "Previsions superposees sur la periode test",
                fontsize=10, fontweight="bold",
            )
            # Valeurs reelles (premier modele)
            first_name, first_res = preds_available[0]
            ax2.plot(
                first_res["test_dates"],
                first_res["y_true"],
                color=_COL_REAL, lw=2.2, zorder=5, label="IPC reel",
            )
            ls_cycle = ["--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]
            for i, (name, res) in enumerate(preds_available):
                c   = _model_color(name)
                ls  = ls_cycle[i % len(ls_cycle)]
                r   = res.get("rmse", np.nan)
                lbl = f"{name}  RMSE={r:.5f}" if not np.isnan(r) else name
                ax2.plot(
                    res["test_dates"], res["y_pred"],
                    color=c, lw=1.4, ls=ls, alpha=0.85, label=lbl,
                )
            ax2.legend(fontsize=7.5, ncol=2)
            ax2.set_xlabel("Date", fontsize=9)
            ax2.set_ylabel("IPC", fontsize=9)
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            p2 = FIG_DIR / "compare_all_predictions.png"
            fig2.savefig(p2, dpi=150, bbox_inches="tight")
            print(f"  Figure sauvegardee : {p2}")
            plt.close(fig2)

        # Figure 3 : radar chart multi-criteres ───────────────────────────────
        categories = [
            "Precision\n(RMSE norme)",
            "Interpretabilite",
            "Simplicite",
            "Rapidite\nentrainement",
        ]
        n_cat  = len(categories)
        angles = np.linspace(0, 2 * np.pi, n_cat, endpoint=False).tolist()
        angles += angles[:1]

        fig3, ax3 = plt.subplots(figsize=(7, 7),
                                  subplot_kw=dict(projection="polar"))
        fig3.suptitle(
            "Profil multi-criteres des modeles",
            fontsize=10, fontweight="bold",
        )

        rmse_vals = df["RMSE"].dropna()
        rmse_min, rmse_max = rmse_vals.min(), rmse_vals.max()

        def _norm_rmse(r: float) -> float:
            if np.isnan(r): return 0.0
            if rmse_max == rmse_min: return 1.0
            return 1.0 - (r - rmse_min) / (rmse_max - rmse_min)

        interp_num = {"Haute": 1.0, "Moyenne": 0.6, "Faible": 0.3}
        compl_num  = {"Tres faible": 1.0, "Moyenne": 0.6, "Haute": 0.3}

        t_max = df["Temps_s"].replace(0, np.nan).max(skipna=True)
        if np.isnan(t_max) or t_max == 0:
            t_max = 1.0

        for name, row in df.iterrows():
            prec   = _norm_rmse(float(row["RMSE"]))
            interp = interp_num.get(row["Interpretabilite"], 0.5)
            simpl  = compl_num.get(row["Complexite"], 0.5)
            t_s    = float(row["Temps_s"]) if not np.isnan(row["Temps_s"]) else t_max
            rapid  = 1.0 - min(t_s / t_max, 1.0)

            vals  = [prec, interp, simpl, rapid]
            vals += vals[:1]
            c = _model_color(name)
            ax3.plot(angles, vals, color=c, lw=1.8, label=name)
            ax3.fill(angles, vals, color=c, alpha=0.07)

        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(categories, fontsize=8)
        ax3.set_ylim(0, 1)
        ax3.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax3.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=6)
        ax3.legend(
            loc="upper right",
            bbox_to_anchor=(1.40, 1.14),
            fontsize=8,
        )
        ax3.grid(True, alpha=0.4)
        plt.tight_layout()
        p3 = FIG_DIR / "compare_all_radar.png"
        fig3.savefig(p3, dpi=150, bbox_inches="tight")
        print(f"  Figure sauvegardee : {p3}")
        plt.close(fig3)

    return df


# =============================================================================
# 3. ENTRAINEMENT LSTM AVEC FENETRES GLISSANTES
# =============================================================================


def train_lstm_sliding_window(
    series: pd.Series,
    exog: "pd.Series | pd.DataFrame | None" = None,
    window_sizes: list[int] = [6, 12, 18, 24],
    train_end: str = "2021-12-01",
    epochs: int = 50,
    batch_size: int = 32,
    lstm_units: list[int] = [50, 25],
    dropout: float = 0.2,
) -> pd.DataFrame:
    """
    Entrainement d'un LSTM pour plusieurs tailles de fenetre glissante.

    Parametres
    ----------
    series      : serie temporelle cible (IPC)
    exog        : variables exogenes (optionnel, ex: BESI)
    window_sizes: liste des tailles de fenetre a tester
    train_end   : date de coupure train/test
    epochs      : nombre max d'epochs par entrainement
    batch_size  : taille des batches
    lstm_units  : unites des couches LSTM [unites1, unites2]
    dropout     : taux de dropout

    Retourne
    --------
    DataFrame avec metriques pour chaque fenetre :
        window_size, rmse, mae, mape, train_time, n_params, epochs
    """
    results = []

    for ws in window_sizes:
        print(f"\n>>> Entrainement LSTM avec fenetre glissante = {ws} mois")
        try:
            res = build_lstm(
                series=series,
                exog=exog,
                look_back=ws,
                train_end=train_end,
                epochs=epochs,
                batch_size=batch_size,
                lstm_units=lstm_units,
                dropout=dropout,
                save_fig=False,  # pas de figures individuelles
            )
            results.append({
                "window_size": ws,
                "rmse":        res["rmse"],
                "mae":         res["mae"],
                "mape":        res["mape"],
                "train_time":  res["train_time"],
                "n_params":    res["n_params"],
                "epochs":      res["epochs"],
            })
        except Exception as e:
            print(f"  [ERREUR] Fenetre {ws} : {e}")
            results.append({
                "window_size": ws,
                "rmse":        np.nan,
                "mae":         np.nan,
                "mape":        np.nan,
                "train_time":  np.nan,
                "n_params":    np.nan,
                "epochs":      np.nan,
            })

    df = pd.DataFrame(results)
    df = df.sort_values("rmse", na_position="last")
    return df


def compare_window_sizes(
    series: pd.Series,
    exog: "pd.Series | pd.DataFrame | None" = None,
    window_sizes: list[int] = [6, 12, 18, 24],
    train_end: str = "2021-12-01",
    epochs: int = 50,
    batch_size: int = 32,
    lstm_units: list[int] = [50, 25],
    dropout: float = 0.2,
    save_fig: bool = True,
) -> pd.DataFrame:
    """
    Comparaison des performances LSTM pour differentes tailles de fenetre.

    Teste avec et sans variables exogenes, trace un graphique a barres des RMSE,
    sauvegarde figure et CSV, imprime la meilleure fenetre.

    Parametres
    ----------
    series      : serie temporelle cible (IPC)
    exog        : variables exogenes (optionnel, ex: BESI)
    window_sizes: liste des tailles de fenetre a tester
    train_end   : date de coupure train/test
    epochs      : nombre max d'epochs par entrainement
    batch_size  : taille des batches
    lstm_units  : unites des couches LSTM [unites1, unites2]
    dropout     : taux de dropout
    save_fig    : sauvegarder la figure de comparaison

    Retourne
    --------
    DataFrame concatene avec colonne 'type' ('sans_exog' ou 'avec_exog')
    Sauvegarde : outputs/figures/lstm_window_comparison.png
                 outputs/reports/lstm_window_comparison.csv
    """
    # ── Entrainement sans exog ────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  COMPARAISON FENETRES GLISSANTES -- LSTM SANS EXOG")
    print("="*70)
    df_no_exog = train_lstm_sliding_window(
        series=series,
        exog=None,
        window_sizes=window_sizes,
        train_end=train_end,
        epochs=epochs,
        batch_size=batch_size,
        lstm_units=lstm_units,
        dropout=dropout,
    )
    df_no_exog["type"] = "sans_exog"

    # ── Entrainement avec exog ────────────────────────────────────────────────
    if exog is not None:
        print("\n" + "="*70)
        print("  COMPARAISON FENETRES GLISSANTES -- LSTM AVEC EXOG")
        print("="*70)
        df_with_exog = train_lstm_sliding_window(
            series=series,
            exog=exog,
            window_sizes=window_sizes,
            train_end=train_end,
            epochs=epochs,
            batch_size=batch_size,
            lstm_units=lstm_units,
            dropout=dropout,
        )
        df_with_exog["type"] = "avec_exog"
        df_all = pd.concat([df_no_exog, df_with_exog], ignore_index=True)
    else:
        df_all = df_no_exog.copy()

    # ── Affichage console ─────────────────────────────────────────────────────
    sep = "=" * 82
    print(f"\n{sep}")
    print("  RESULTATS COMPARAISON FENETRES GLISSANTES")
    print(sep)
    print(
        f"  {'Type':<12}  {'Fenetre':>7}  {'RMSE':>9}  {'MAE':>9}  {'MAPE%':>7}  "
        f"{'Temps(s)':>9}  {'Params':>8}  {'Epochs':>6}"
    )
    print(f"  {'-'*78}")
    for _, row in df_all.iterrows():
        def _fmt(v, fmt):
            return f"{v:{fmt}}" if not (isinstance(v, float) and np.isnan(v)) else "  n/a  "
        print(
            f"  {row['type']:<12}  "
            f"{int(row['window_size']):>7}  "
            f"{_fmt(row['rmse'],    '.5f'):>9}  "
            f"{_fmt(row['mae'],     '.5f'):>9}  "
            f"{_fmt(row['mape'],    '.2f'):>7}  "
            f"{_fmt(row['train_time'], '.1f') + 's':>9}  "
            f"{_fmt(row['n_params'], ','):>8}  "
            f"{_fmt(row['epochs'],   'd'):>6}"
        )
    print(sep)

    # Meilleures fenetres
    for typ in df_all["type"].unique():
        sub = df_all[df_all["type"] == typ]
        best = sub.loc[sub["rmse"].idxmin()]
        print(f"\n  Meilleure fenetre pour {typ} : {int(best['window_size'])} mois  "
              f"(RMSE = {best['rmse']:.5f})")

    # ── Figure barres RMSE ────────────────────────────────────────────────────
    if save_fig:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle(
            "Comparaison RMSE LSTM par taille de fenetre glissante",
            fontsize=11, fontweight="bold",
        )

        types = df_all["type"].unique()
        x = np.arange(len(window_sizes))
        width = 0.35

        for i, typ in enumerate(types):
            sub = df_all[df_all["type"] == typ].set_index("window_size").reindex(window_sizes)
            rmse_vals = sub["rmse"].values
            bars = ax.bar(
                x + i * width - width/2 * (len(types)-1),
                rmse_vals,
                width,
                label=typ.replace("_", " ").title(),
                color=_COL_LSTM if typ == "sans_exog" else "#FF7F0E",
                alpha=0.8,
            )
            for bar, v in zip(bars, rmse_vals):
                if not np.isnan(v):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        v + max(rmse_vals) * 0.01,
                        f"{v:.5f}",
                        ha="center", fontsize=7, fontweight="bold",
                    )

        ax.set_xlabel("Taille de fenetre (mois)", fontsize=9)
        ax.set_ylabel("RMSE", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(window_sizes)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        path = FIG_DIR / "lstm_window_comparison.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"\n  Figure sauvegardee : {path}")
        plt.close(fig)

    # ── CSV sauvegarde ────────────────────────────────────────────────────────
    csv_path = REP_DIR / "lstm_window_comparison.csv"
    df_all.to_csv(csv_path, index=False)
    print(f"  CSV sauvegarde : {csv_path}")

    return df_all


# =============================================================================
# 4. GRU
# =============================================================================

def build_gru(
    series:     pd.Series,
    exog:       "pd.DataFrame | None" = None,
    look_back:  int   = 12,
    train_end:  str   = "2021-12-01",
    epochs:     int   = 50,
    batch_size: int   = 16,
    gru_units:  tuple = (64, 32),
    dropout:    float = 0.10,
    save_fig:   bool  = True,
) -> dict:
    """
    Reseau GRU (Gated Recurrent Unit) — architecture identique au LSTM
    mais avec des cellules GRU. Plus rapide a entrainer et souvent plus
    efficace sur des series courtes.

    Architecture : Input -> GRU(64, return_seq) -> Dropout -> GRU(32) -> Dense(1)
    Meme split train/test que SARIMA pour une comparaison equitable.

    Retourne le meme format de dict que build_lstm().
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import GRU, Dense, Dropout, Input
        from tensorflow.keras.callbacks import EarlyStopping
        from sklearn.preprocessing import MinMaxScaler
        tf.random.set_seed(42)
    except ImportError as exc:
        raise ImportError(f"TensorFlow requis pour build_gru(). {exc}") from exc

    # ── Preparation identique a build_lstm ────────────────────────────────────
    s = series.dropna().copy()
    if exog is not None:
        exog_al  = exog.reindex(s.index).ffill().bfill()
        common   = s.index.intersection(exog_al.dropna().index)
        s        = s.loc[common]
        exog_al  = exog_al.loc[common]
        data_arr = np.column_stack([s.values, exog_al.values])
    else:
        exog_al  = None
        data_arr = s.values.reshape(-1, 1)

    n_features = data_arr.shape[1]
    scaler     = MinMaxScaler(feature_range=(0, 1))
    data_norm  = scaler.fit_transform(data_arr)

    X_all, y_all, dates_all = [], [], []
    for i in range(len(data_norm) - look_back):
        X_all.append(data_norm[i : i + look_back])
        y_all.append(data_norm[i + look_back, 0])
        dates_all.append(s.index[i + look_back])

    X_all     = np.array(X_all)
    y_all     = np.array(y_all)
    dates_all = pd.DatetimeIndex(dates_all)

    te         = pd.Timestamp(train_end)
    train_mask = dates_all <= te
    test_mask  = dates_all >  te

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test,  y_test  = X_all[test_mask],  y_all[test_mask]
    dates_test       = dates_all[test_mask]

    sep = "=" * 62
    print(f"\n{sep}")
    print("  GRU -- PREVISION IPC MAROC")
    print(f"  Features  : {n_features}  |  Look-back : {look_back} mois")
    print(f"  Train     : {dates_all[train_mask][0].date()} -> {te.date()}")
    print(f"  Test      : {dates_test[0].date()} -> {dates_test[-1].date()}")
    print(f"  Archi     : Input({look_back},{n_features}) -> GRU({gru_units[0]}) "
          f"-> Dropout({dropout}) -> GRU({gru_units[1]}) -> Dense(1)")
    print(sep)

    # ── Construction du modele ────────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Sequential(
            [
                Input(shape=(look_back, n_features)),
                GRU(gru_units[0], return_sequences=True),
                Dropout(dropout),
                GRU(gru_units[1]),
                Dense(1),
            ],
            name="IPC_GRU",
        )
        model.compile(optimizer="adam", loss="mse")

    n_params   = model.count_params()
    early_stop = EarlyStopping(monitor="val_loss", patience=8,
                               restore_best_weights=True, verbose=0)
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        history = model.fit(
            X_train, y_train,
            epochs=epochs, batch_size=batch_size,
            validation_split=0.15, callbacks=[early_stop],
            verbose=0, shuffle=False,
        )
    train_time    = time.time() - t0
    actual_epochs = len(history.history["loss"])
    print(f"  Entrainement : {actual_epochs} epochs  ({train_time:.1f}s)")

    # ── Predictions ───────────────────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred_norm = model.predict(X_test, verbose=0).ravel()

    def _inv(norm_vals):
        dummy = np.zeros((len(norm_vals), n_features))
        dummy[:, 0] = norm_vals
        return scaler.inverse_transform(dummy)[:, 0]

    y_pred_inv = _inv(y_pred_norm)
    y_true_inv = _inv(y_test)

    rmse = _rmse(y_true_inv, y_pred_inv)
    mae  = _mae(y_true_inv,  y_pred_inv)
    mape = _mape(y_true_inv, y_pred_inv)

    print(f"  RMSE={rmse:.5f}  MAE={mae:.5f}  MAPE={mape:.2f}%")

    # ── Sauvegarde modele ─────────────────────────────────────────────────────
    model_path = MOD_DIR / "gru_ipc.keras"
    try:
        model.save(str(model_path))
    except Exception:
        pass

    # ── Figure ────────────────────────────────────────────────────────────────
    if save_fig:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(dates_test, y_true_inv,
                color=_COL_REAL, lw=2.0, label="IPC reel")
        ax.plot(dates_test, y_pred_inv,
                color="#17BECF", lw=1.8, ls="--",
                label=f"GRU pred  RMSE={rmse:.5f}")
        ax.axvline(te, color="red", lw=1.0, ls="--", alpha=0.5, label="Coupure")
        ax.set_title(f"GRU IPC Maroc — RMSE={rmse:.5f}  epochs={actual_epochs}",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = FIG_DIR / "gru_predictions.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Figure sauvegardee : {path}")
        plt.close(fig)

    return {
        "rmse": rmse, "mae": mae, "mape": mape,
        "y_true": y_true_inv, "y_pred": y_pred_inv,
        "test_dates": dates_test,
        "train_time": round(train_time, 1),
        "n_params": n_params, "epochs": actual_epochs,
        "history": history.history, "model": model,
        "look_back": look_back, "n_features": n_features,
    }


# =============================================================================
# 5. BIDIRECTIONAL LSTM
# =============================================================================

def build_bilstm(
    series:     pd.Series,
    exog:       "pd.DataFrame | None" = None,
    look_back:  int   = 12,
    train_end:  str   = "2021-12-01",
    epochs:     int   = 50,
    batch_size: int   = 16,
    lstm_units: tuple = (64, 32),
    dropout:    float = 0.10,
    save_fig:   bool  = True,
) -> dict:
    """
    BiLSTM (Bidirectionnel) — lit la sequence dans les deux sens.
    Particulierement utile quand les dependances passees ET futures
    d'une fenetre ont de l'importance (ex. phenomenes saisonniers).

    Architecture : Input -> Bidirectionnel(LSTM(64)) -> Dropout
                -> LSTM(32) -> Dense(1)

    Retourne le meme format de dict que build_lstm().
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Input
        from tensorflow.keras.callbacks import EarlyStopping
        from sklearn.preprocessing import MinMaxScaler
        tf.random.set_seed(42)
    except ImportError as exc:
        raise ImportError(f"TensorFlow requis pour build_bilstm(). {exc}") from exc

    # ── Preparation ───────────────────────────────────────────────────────────
    s = series.dropna().copy()
    if exog is not None:
        exog_al  = exog.reindex(s.index).ffill().bfill()
        common   = s.index.intersection(exog_al.dropna().index)
        s        = s.loc[common]
        exog_al  = exog_al.loc[common]
        data_arr = np.column_stack([s.values, exog_al.values])
    else:
        exog_al  = None
        data_arr = s.values.reshape(-1, 1)

    n_features = data_arr.shape[1]
    scaler     = MinMaxScaler(feature_range=(0, 1))
    data_norm  = scaler.fit_transform(data_arr)

    X_all, y_all, dates_all = [], [], []
    for i in range(len(data_norm) - look_back):
        X_all.append(data_norm[i : i + look_back])
        y_all.append(data_norm[i + look_back, 0])
        dates_all.append(s.index[i + look_back])

    X_all     = np.array(X_all)
    y_all     = np.array(y_all)
    dates_all = pd.DatetimeIndex(dates_all)

    te         = pd.Timestamp(train_end)
    train_mask = dates_all <= te
    test_mask  = dates_all >  te

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test,  y_test  = X_all[test_mask],  y_all[test_mask]
    dates_test       = dates_all[test_mask]

    sep = "=" * 62
    print(f"\n{sep}")
    print("  BiLSTM -- PREVISION IPC MAROC")
    print(f"  Features  : {n_features}  |  Look-back : {look_back} mois")
    print(f"  Train     : {dates_all[train_mask][0].date()} -> {te.date()}")
    print(f"  Test      : {dates_test[0].date()} -> {dates_test[-1].date()}")
    print(f"  Archi     : Input -> Bidirectionnel(LSTM({lstm_units[0]})) "
          f"-> Dropout -> LSTM({lstm_units[1]}) -> Dense(1)")
    print(sep)

    # ── Modele ────────────────────────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Sequential(
            [
                Input(shape=(look_back, n_features)),
                Bidirectional(LSTM(lstm_units[0], return_sequences=True)),
                Dropout(dropout),
                LSTM(lstm_units[1]),
                Dense(1),
            ],
            name="IPC_BiLSTM",
        )
        model.compile(optimizer="adam", loss="mse")

    n_params   = model.count_params()
    early_stop = EarlyStopping(monitor="val_loss", patience=8,
                               restore_best_weights=True, verbose=0)
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        history = model.fit(
            X_train, y_train,
            epochs=epochs, batch_size=batch_size,
            validation_split=0.15, callbacks=[early_stop],
            verbose=0, shuffle=False,
        )
    train_time    = time.time() - t0
    actual_epochs = len(history.history["loss"])
    print(f"  Entrainement : {actual_epochs} epochs  ({train_time:.1f}s)")

    # ── Predictions ───────────────────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred_norm = model.predict(X_test, verbose=0).ravel()

    def _inv(norm_vals):
        dummy = np.zeros((len(norm_vals), n_features))
        dummy[:, 0] = norm_vals
        return scaler.inverse_transform(dummy)[:, 0]

    y_pred_inv = _inv(y_pred_norm)
    y_true_inv = _inv(y_test)

    rmse = _rmse(y_true_inv, y_pred_inv)
    mae  = _mae(y_true_inv,  y_pred_inv)
    mape = _mape(y_true_inv, y_pred_inv)

    print(f"  RMSE={rmse:.5f}  MAE={mae:.5f}  MAPE={mape:.2f}%")

    model_path = MOD_DIR / "bilstm_ipc.keras"
    try:
        model.save(str(model_path))
    except Exception:
        pass

    if save_fig:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(dates_test, y_true_inv,
                color=_COL_REAL, lw=2.0, label="IPC reel")
        ax.plot(dates_test, y_pred_inv,
                color="#9467BD", lw=1.8, ls="--",
                label=f"BiLSTM pred  RMSE={rmse:.5f}")
        ax.axvline(te, color="red", lw=1.0, ls="--", alpha=0.5, label="Coupure")
        ax.set_title(f"BiLSTM IPC Maroc — RMSE={rmse:.5f}  epochs={actual_epochs}",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = FIG_DIR / "bilstm_predictions.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Figure sauvegardee : {path}")
        plt.close(fig)

    return {
        "rmse": rmse, "mae": mae, "mape": mape,
        "y_true": y_true_inv, "y_pred": y_pred_inv,
        "test_dates": dates_test,
        "train_time": round(train_time, 1),
        "n_params": n_params, "epochs": actual_epochs,
        "history": history.history, "model": model,
        "look_back": look_back, "n_features": n_features,
    }


# =============================================================================
# 6. XGBOOST TIME-SERIES
# =============================================================================

def build_xgboost_ts(
    series:    pd.Series,
    exog:      "pd.DataFrame | None" = None,
    look_back: int  = 12,
    train_end: str  = "2021-12-01",
    n_estimators: int   = 300,
    max_depth:    int   = 4,
    learning_rate: float = 0.05,
    subsample:    float = 0.8,
    save_fig:  bool = True,
) -> dict:
    """
    XGBoost applique a la prevision de series temporelles.

    Strategie : features = lags 1..look_back de l'IPC + exog alignees.
    Pas de sequences 3D — XGBoost est un modele tabular.
    Permet une comparaison interpretabilite/performance vs deep learning.

    Necessite : pip install xgboost scikit-learn

    Retourne le meme format de dict que build_lstm().
    """
    try:
        import xgboost as xgb
        from sklearn.preprocessing import MinMaxScaler
    except ImportError as exc:
        raise ImportError(
            "xgboost et scikit-learn sont requis pour build_xgboost_ts().\n"
            f"Installation : pip install xgboost scikit-learn\n{exc}"
        ) from exc

    s = series.dropna().copy()

    # ── Construction des features lag ─────────────────────────────────────────
    # X[t] = [IPC_{t-1}, ..., IPC_{t-look_back}] + exog_{t}
    feat_dict: dict = {}
    for lag in range(1, look_back + 1):
        feat_dict[f"ipc_lag{lag}"] = s.shift(lag)

    if exog is not None:
        exog_al = exog.reindex(s.index).ffill().bfill()
        for col in exog_al.columns:
            feat_dict[col] = exog_al[col]

    feat_df = pd.DataFrame(feat_dict, index=s.index)
    feat_df["target"] = s.values

    feat_df = feat_df.dropna()
    X = feat_df.drop(columns=["target"]).values
    y = feat_df["target"].values
    dates = feat_df.index

    # ── Split ────────────────────────────────────────────────────────────────
    te         = pd.Timestamp(train_end)
    train_mask = dates <= te
    test_mask  = dates >  te

    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]
    dates_test       = dates[test_mask]

    sep = "=" * 62
    print(f"\n{sep}")
    print("  XGBoost TS -- PREVISION IPC MAROC")
    print(f"  Features   : {X.shape[1]} ({look_back} lags IPC"
          f"{' + ' + str(exog.shape[1]) + ' exog' if exog is not None else ''})")
    print(f"  Train      : {dates[train_mask][0].date()} -> {te.date()}")
    print(f"  Test       : {dates_test[0].date()} -> {dates_test[-1].date()}")
    print(f"  Hyperparams: n_estimators={n_estimators}  max_depth={max_depth}  "
          f"lr={learning_rate}  subsample={subsample}")
    print(sep)

    t0 = time.time()
    regressor = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    regressor.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)
    train_time = time.time() - t0

    y_pred = regressor.predict(X_test)

    rmse = _rmse(y_test, y_pred)
    mae  = _mae(y_test,  y_pred)
    mape = _mape(y_test, y_pred)

    print(f"  Entrainement : {train_time:.1f}s")
    print(f"  RMSE={rmse:.5f}  MAE={mae:.5f}  MAPE={mape:.2f}%")
    print(sep)

    # Importance des features
    fi = dict(zip(feat_df.drop(columns=["target"]).columns,
                  regressor.feature_importances_))
    fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)
    print("  Top 5 features XGBoost :")
    for feat, imp in fi_sorted[:5]:
        print(f"    {feat:<20} : {imp:.4f}")

    if save_fig:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"XGBoost TS IPC Maroc — RMSE={rmse:.5f}",
                     fontsize=10, fontweight="bold")

        # Previsions
        ax = axes[0]
        ax.plot(dates_test, y_test, color=_COL_REAL, lw=2.0, label="IPC reel")
        ax.plot(dates_test, y_pred, color="#FF7F0E", lw=1.8, ls="--",
                label=f"XGBoost pred")
        ax.axvline(te, color="red", lw=1.0, ls="--", alpha=0.5)
        ax.set_title("Previsions vs valeurs reelles", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Importance
        ax2 = axes[1]
        top_feats = [f for f, _ in fi_sorted[:10]]
        top_imps  = [fi[f] for f in top_feats]
        ax2.barh(top_feats[::-1], top_imps[::-1],
                 color="#FF7F0E", alpha=0.8)
        ax2.set_title("Importance des features (top 10)", fontsize=9)
        ax2.set_xlabel("Importance")
        ax2.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        path = FIG_DIR / "xgboost_ts_predictions.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Figure sauvegardee : {path}")
        plt.close(fig)

    return {
        "rmse": rmse, "mae": mae, "mape": mape,
        "y_true": y_test, "y_pred": y_pred,
        "test_dates": dates_test,
        "train_time": round(train_time, 1),
        "n_params": int(regressor.get_params()["n_estimators"]),
        "feature_importance": fi_sorted,
        "model": regressor,
        "look_back": look_back,
    }


# =============================================================================
# 7. COMPARAISON COMPLETE DEEP LEARNING
# =============================================================================

def compare_all_dl_models(
    series:    pd.Series,
    exog:      "pd.DataFrame | None" = None,
    look_back: int  = 12,
    train_end: str  = "2021-12-01",
    epochs:    int  = 50,
    save_fig:  bool = True,
) -> pd.DataFrame:
    """
    Lance et compare tous les modeles DL du projet :
    LSTM, GRU, BiLSTM, XGBoost TS (avec et sans exog si fourni).

    Parametres
    ----------
    series    : serie IPC mensuelle
    exog      : variables exogenes (BESI ou sous-indices Trends) — optionnel
    look_back : fenetre d'entree en mois
    train_end : coupure train/test
    epochs    : epochs max pour les reseaux neuronaux
    save_fig  : sauvegarder les figures de comparaison

    Retourne
    --------
    pd.DataFrame trie par RMSE croissant
    Exports : outputs/reports/dl_comparison.csv
              outputs/figures/dl_comparison_rmse.png
    """
    results: dict = {}

    # ── 1. LSTM ───────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  [1/4]  LSTM")
    print("="*65)
    try:
        results["LSTM"] = build_lstm(
            series, exog=None, look_back=look_back,
            train_end=train_end, epochs=epochs, save_fig=save_fig,
        )
    except Exception as e:
        print(f"  [ERREUR] LSTM : {e}")

    if exog is not None:
        try:
            results["LSTM+exog"] = build_lstm(
                series, exog=exog, look_back=look_back,
                train_end=train_end, epochs=epochs, save_fig=False,
            )
        except Exception as e:
            print(f"  [ERREUR] LSTM+exog : {e}")

    # ── 2. GRU ────────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  [2/4]  GRU")
    print("="*65)
    try:
        results["GRU"] = build_gru(
            series, exog=None, look_back=look_back,
            train_end=train_end, epochs=epochs, save_fig=save_fig,
        )
    except Exception as e:
        print(f"  [ERREUR] GRU : {e}")

    if exog is not None:
        try:
            results["GRU+exog"] = build_gru(
                series, exog=exog, look_back=look_back,
                train_end=train_end, epochs=epochs, save_fig=False,
            )
        except Exception as e:
            print(f"  [ERREUR] GRU+exog : {e}")

    # ── 3. BiLSTM ─────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  [3/4]  BiLSTM")
    print("="*65)
    try:
        results["BiLSTM"] = build_bilstm(
            series, exog=None, look_back=look_back,
            train_end=train_end, epochs=epochs, save_fig=save_fig,
        )
    except Exception as e:
        print(f"  [ERREUR] BiLSTM : {e}")

    if exog is not None:
        try:
            results["BiLSTM+exog"] = build_bilstm(
                series, exog=exog, look_back=look_back,
                train_end=train_end, epochs=epochs, save_fig=False,
            )
        except Exception as e:
            print(f"  [ERREUR] BiLSTM+exog : {e}")

    # ── 4. XGBoost ────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  [4/4]  XGBoost TS")
    print("="*65)
    try:
        results["XGBoost"] = build_xgboost_ts(
            series, exog=None, look_back=look_back,
            train_end=train_end, save_fig=save_fig,
        )
    except Exception as e:
        print(f"  [ERREUR] XGBoost : {e}")

    if exog is not None:
        try:
            results["XGBoost+exog"] = build_xgboost_ts(
                series, exog=exog, look_back=look_back,
                train_end=train_end, save_fig=False,
            )
        except Exception as e:
            print(f"  [ERREUR] XGBoost+exog : {e}")

    # ── Tableau recapitulatif ─────────────────────────────────────────────────
    rows = []
    for name, res in results.items():
        rows.append({
            "Modele":      name,
            "RMSE":        round(res.get("rmse",       np.nan), 5),
            "MAE":         round(res.get("mae",        np.nan), 5),
            "MAPE%":       round(res.get("mape",       np.nan), 2),
            "Temps_s":     res.get("train_time", np.nan),
            "N_params":    res.get("n_params",   np.nan),
            "Look_back":   res.get("look_back",  look_back),
        })

    df_dl = pd.DataFrame(rows).sort_values("RMSE", na_position="last").reset_index(drop=True)

    sep = "=" * 75
    print(f"\n{sep}")
    print("  COMPARAISON DL COMPLETE — LSTM / GRU / BiLSTM / XGBoost")
    print(sep)
    print(df_dl.to_string(index=False))
    print(sep)

    best = df_dl.iloc[0]
    print(f"\n  Meilleur modele DL : {best['Modele']}  (RMSE={best['RMSE']:.5f})")

    # ── Sauvegarde CSV ────────────────────────────────────────────────────────
    csv_path = REP_DIR / "dl_comparison.csv"
    df_dl.to_csv(csv_path, index=False)
    print(f"  CSV sauvegarde : {csv_path}")

    # ── Figure barres RMSE ────────────────────────────────────────────────────
    if save_fig and len(df_dl) > 0:
        _col_map = {
            "LSTM":         _COL_LSTM,
            "LSTM+exog":    "#7B4EA8",
            "GRU":          "#17BECF",
            "GRU+exog":     "#0E8F99",
            "BiLSTM":       "#9467BD",
            "BiLSTM+exog":  "#6E4E9B",
            "XGBoost":      "#FF7F0E",
            "XGBoost+exog": "#D45F00",
        }
        colors = [_col_map.get(n, "#888888") for n in df_dl["Modele"]]

        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        fig.suptitle(
            f"Comparaison modeles DL — look_back={look_back} mois  "
            f"train_end={train_end}",
            fontsize=10, fontweight="bold",
        )
        for ax, metric in zip(axes, ["RMSE", "MAE", "MAPE%"]):
            vals = df_dl[metric].values
            bars = ax.bar(df_dl["Modele"], vals, color=colors, alpha=0.85)
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            v * 1.015, f"{v:.5f}" if metric != "MAPE%" else f"{v:.2f}",
                            ha="center", fontsize=6, fontweight="bold")
            ax.set_title(metric, fontsize=10)
            ax.set_xticklabels(df_dl["Modele"], rotation=40, ha="right", fontsize=7)
            ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        path = FIG_DIR / "dl_comparison_rmse.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Figure sauvegardee : {path}")
        plt.close(fig)

        # Figure predictions superposees
        preds_ok = [(n, r) for n, r in results.items()
                    if "y_pred" in r and "test_dates" in r]
        if preds_ok:
            fig2, ax2 = plt.subplots(figsize=(14, 6))
            fig2.suptitle("Previsions DL superposees — periode test",
                          fontsize=10, fontweight="bold")
            y_true = preds_ok[0][1]["y_true"]
            dates_t = preds_ok[0][1]["test_dates"]
            ax2.plot(dates_t, y_true, color=_COL_REAL, lw=2.5,
                     zorder=5, label="IPC reel")
            ls_cycle = ["--", "-.", ":", (0, (3,1,1,1)), (0,(5,2)), "--", "-."]
            for i, (name, res) in enumerate(preds_ok):
                c   = _col_map.get(name, "#888888")
                ls  = ls_cycle[i % len(ls_cycle)]
                r   = res.get("rmse", np.nan)
                ax2.plot(res["test_dates"], res["y_pred"],
                         color=c, lw=1.4, ls=ls, alpha=0.85,
                         label=f"{name}  RMSE={r:.5f}")
            ax2.legend(fontsize=7.5, ncol=2)
            ax2.set_xlabel("Date", fontsize=9)
            ax2.set_ylabel("IPC", fontsize=9)
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            path2 = FIG_DIR / "dl_predictions_all.png"
            fig2.savefig(path2, dpi=150, bbox_inches="tight")
            print(f"  Figure sauvegardee : {path2}")
            plt.close(fig2)

    return df_dl

def _plot_blocs(best_by_bloc: dict):
    """Figure comparative des deux blocs cote a cote."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "GridSearch LSTM — Resultats par bloc\n"
        "Bloc A : COVID (2020-2021)  |  Bloc B : Inflation (2022-2024)",
        fontsize=12, fontweight="bold"
    )

    for col, (bloc_name, data) in enumerate(best_by_bloc.items()):
        res      = data["res"]
        combo    = data["combo"]
        ipc      = data["ipc"]
        train_end = pd.Timestamp(data["train_end"])

        # Panel haut : predictions vs reel
        ax_top = axes[0, col]
        ipc_train = ipc[ipc.index <= train_end]
        ax_top.plot(ipc_train.index, ipc_train.values,
                    color="lightgray", lw=1.2, label="Train")
        ax_top.plot(res["test_dates"], res["y_true"],
                    color=_COL_REAL, lw=2.0, label="IPC reel (test)")
        ax_top.plot(res["test_dates"], res["y_pred"],
                    color=_COL_LSTM, lw=1.8, ls="--",
                    label=f"LSTM  RMSE={res['rmse']:.4f}")
        ax_top.axvline(train_end, color="red", lw=1.2,
                       ls="--", alpha=0.6, label="Fin train")
        ax_top.set_title(
            f"Bloc {bloc_name} — RMSE={res['rmse']:.4f}\n"
            f"lb={combo['look_back']} u=({combo['lstm_units_1']},"
            f"{combo['lstm_units_2']}) dr={combo['dropout']} "
            f"lr={combo['learning_rate']}",
            fontsize=9
        )
        ax_top.legend(fontsize=7.5)
        ax_top.grid(True, alpha=0.3)
        ax_top.set_ylabel("IPC")

        # Panel bas : distribution RMSE du gridsearch
        ax_bot = axes[1, col]
        df_gs  = data["df_gs"]
        ax_bot.hist(df_gs["rmse"], bins=25,
                    color=_COL_LSTM, alpha=0.7, edgecolor="white")
        ax_bot.axvline(res["rmse"], color="red", lw=2,
                       label=f"Best={res['rmse']:.4f}")
        # Ligne baseline selon le bloc
        baseline = 1.609 if bloc_name == "A" else 1.891
        ax_bot.axvline(baseline, color="gray", lw=1.5, ls="--",
                       label=f"Baseline={baseline}")
        ax_bot.set_title(
            f"Bloc {bloc_name} — Distribution RMSE "
            f"({len(df_gs)} combos)",
            fontsize=9
        )
        ax_bot.set_xlabel("RMSE")
        ax_bot.set_ylabel("Nombre combinaisons")
        ax_bot.legend(fontsize=8)
        ax_bot.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIG_DIR / "lstm_gridsearch_blocs.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Figure blocs sauvegardee : {path}")
    plt.close(fig)


def run_gridsearch_blocs(
    df,
    ipc_col="ipc_level",
    exog_cols=None,
    save_fig=True,
) -> pd.DataFrame:
    """
    Lance le GridSearch LSTM sur les deux blocs definis par la binome :
      Bloc A : train 2017-2019  ->  test 2020-2021  (24 pas)
      Bloc B : train 2017-2021  ->  test 2022-2024  (36 pas)
    Retourne un DataFrame avec colonnes : bloc, model, rmse, mae, mape, n_test
    Compatible avec backtest_v3_results.csv de la binome.
    """
    BLOCS = {
        "A": {
            "train_start": "2017-01-01",
            "train_end"  : "2019-12-01",
            "test_start" : "2020-01-01",
            "test_end"   : "2021-12-01",
            "n_test"     : 24,
        },
        "B": {
            "train_start": "2017-01-01",
            "train_end"  : "2021-12-01",
            "test_start" : "2022-01-01",
            "test_end"   : "2024-12-01",
            "n_test"     : 36,
        },
    }

    # Grille de recherche
    param_grid = {
    "look_back"    : [6, 12, 24],
    "lstm_units_1" : [32, 64],
    "lstm_units_2" : [16, 32],
    "dropout"      : [0.1, 0.2],
    "learning_rate": [0.001, 0.0005],
    "batch_size"   : [16],
}

    # Import TensorFlow
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.optimizers import Adam
        from sklearn.preprocessing import MinMaxScaler
        tf.random.set_seed(42)
        tf_imports = dict(Sequential=Sequential, Input=Input, LSTM=LSTM,
                          Dropout=Dropout, Dense=Dense,
                          EarlyStopping=EarlyStopping, Adam=Adam)
    except ImportError as exc:
        raise ImportError(f"pip install tensorflow scikit-learn\n{exc}") from exc

    all_rows    = []
    best_by_bloc = {}

    for bloc_name, bloc in BLOCS.items():
        print(f"\n{'='*65}")
        print(f"  BLOC {bloc_name} : train {bloc['train_start']} -> "
              f"{bloc['train_end']}  |  test {bloc['test_start']} -> "
              f"{bloc['test_end']}")
        print(f"{'='*65}")

        # Extraire la periode du bloc (train + test)
        mask_bloc = (
            (df.index >= bloc["train_start"]) &
            (df.index <= bloc["test_end"])
        )
        df_bloc = df.loc[mask_bloc].copy()

        ipc_bloc = df_bloc[ipc_col].dropna()

        # Construction data_arr propre
        # Step 1 : colonnes de base
        cols_data = {"ipc": ipc_bloc.values}

        # Step 2 : exog disponibles — ffill/bfill par colonne individuellement
        if exog_cols:
            for col in exog_cols:
                series_col = df_bloc[col].reindex(ipc_bloc.index)
                series_col = series_col.ffill().bfill()
                if series_col.isna().sum() == 0:
                    cols_data[col] = series_col.values
                else:
                    print(f"  [WARN] {col} encore NaN apres ffill/bfill — ignore")

        # Step 3 : encodage cyclique mois (toujours present)
        cols_data["month_sin"] = np.sin(2 * np.pi * ipc_bloc.index.month / 12)
        cols_data["month_cos"] = np.cos(2 * np.pi * ipc_bloc.index.month / 12)

        # Step 4 : assembler en array
        data_arr   = np.column_stack(list(cols_data.values()))
        n_features = data_arr.shape[1]
        col_names  = list(cols_data.keys())

        # Step 5 : verifier qu'il ne reste aucun NaN
        nan_count = np.isnan(data_arr).sum()
        if nan_count > 0:
            print(f"  [WARN] {nan_count} NaN dans data_arr — lignes supprimees")
            nan_rows  = np.isnan(data_arr).any(axis=1)
            data_arr  = data_arr[~nan_rows]
            ipc_bloc  = ipc_bloc[~nan_rows]

        print(f"  data_arr final : {data_arr.shape}  colonnes={col_names}")

        # Verifier qu'on a assez de donnees train
        train_idx = ipc_bloc.index <= bloc["train_end"]
        n_train   = train_idx.sum()
        n_test    = (~train_idx).sum()
        print(f"  Train : {n_train}  |  Test : {n_test}")

        if n_train < 10:
            print(f"  [ERREUR] Pas assez de donnees train ({n_train}) — bloc saute")
            continue
        if n_test == 0:
            print(f"  [ERREUR] Pas de donnees test — bloc saute")
            continue

        # Normalisation FIT sur train uniquement
        train_idx  = ipc_bloc.index <= bloc["train_end"]
        scaler     = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(data_arr[train_idx])
        data_norm  = scaler.transform(data_arr)

        # GridSearch
        combos = list(itertools.product(*param_grid.values()))
        total  = len(combos)
        print(f"  {total} combinaisons  |  {n_features} features")

        gs_results  = []
        best_rmse   = np.inf
        best_res    = None
        best_combo  = None

        for i, (lb, u1, u2, dr, lr, bs) in enumerate(combos):
            res = _build_single_lstm(
                data_norm, ipc_bloc, bloc["train_end"],
                lb, u1, u2, dr, lr, bs,
                n_features, scaler, tf_imports
            )
            if res is None:
                continue

            gs_results.append({
                "look_back": lb, "lstm_units_1": u1, "lstm_units_2": u2,
                "dropout": dr, "learning_rate": lr, "batch_size": bs,
                "rmse": res["rmse"], "mae": res["mae"], "mape": res["mape"],
                "epochs_run": res["epochs_run"],
            })

            if res["rmse"] < best_rmse:
                best_rmse  = res["rmse"]
                best_res   = res
                best_combo = {
                    "look_back": lb, "lstm_units_1": u1,
                    "lstm_units_2": u2, "dropout": dr,
                    "learning_rate": lr, "batch_size": bs,
                }

            print(f"  [{i+1:4d}/{total}] Bloc {bloc_name} | "
                  f"lb={lb:2d} u=({u1:3d},{u2:2d}) "
                  f"dr={dr} lr={lr} bs={bs:2d} "
                  f"-> RMSE={res['rmse']:.4f} "
                  f"[BEST={best_rmse:.4f}]")

        # Sauvegarder CSV du bloc
        df_gs_bloc = pd.DataFrame(gs_results).sort_values("rmse").reset_index(drop=True)
        bloc_csv   = REP_DIR / f"gridsearch_lstm_bloc{bloc_name}.csv"
        df_gs_bloc.to_csv(bloc_csv, index=False)
        
        # Sauvegarder aussi une copie avec le nom du scaler utilise
        bloc_csv_minmax = REP_DIR / f"gridsearch_lstm_bloc{bloc_name}_minmax.csv"
        df_gs_bloc.to_csv(bloc_csv_minmax, index=False)
        
        print(f"\n  Top 5 Bloc {bloc_name} :")
        print(df_gs_bloc.head(5).to_string(index=False))

        # Stocker pour figures
        best_by_bloc[bloc_name] = {
            "res": best_res, "combo": best_combo,
            "ipc": ipc_bloc, "df_gs": df_gs_bloc,
            "train_end": bloc["train_end"],
        }

        # Ligne pour le tableau final (format binome)
        if best_res is not None:
            all_rows.append({
                "bloc" : f"Bloc_{bloc_name}",
                "model": "LSTM_GridSearch",
                "rmse" : round(best_res["rmse"], 4),
                "mae"  : round(best_res["mae"],  4),
                "mape" : round(best_res["mape"], 2),
                "n_test": bloc["n_test"],
            })
        else:
            print(f"\n  [WARN] Bloc {bloc_name} : pas assez de donnees pour le GridSearch")
            all_rows.append({
                "bloc" : f"Bloc_{bloc_name}",
                "model": "LSTM_GridSearch",
                "rmse" : np.nan, "mae": np.nan, "mape": np.nan,
                "n_test": bloc["n_test"],
            })

    # Ligne globale (moyenne ponderee par n_test)
    total_n  = sum(r["n_test"] for r in all_rows)
    
    if total_n == 0 or all(np.isnan(r["rmse"]) for r in all_rows):
        print(f"\n  [WARN] Aucun modele valide trouvé. Vérifier les données ou paramètres.")
        df_results = pd.DataFrame(all_rows)
        return df_results
    
    # Calculer les moyennes uniquement sur les modèles valides (non-NaN)
    valid_rows = [r for r in all_rows if not np.isnan(r.get("rmse", np.nan))]
    if valid_rows:
        valid_n  = sum(r["n_test"] for r in valid_rows)
        rmse_gl  = sum(r["rmse"] * r["n_test"] for r in valid_rows) / valid_n if valid_n > 0 else np.nan
        mae_gl   = sum(r["mae"]  * r["n_test"] for r in valid_rows) / valid_n if valid_n > 0 else np.nan
        mape_gl  = sum(r["mape"] * r["n_test"] for r in valid_rows) / valid_n if valid_n > 0 else np.nan
    else:
        rmse_gl = mae_gl = mape_gl = np.nan
        valid_n = 0
    
    all_rows.append({
        "bloc": "Global", "model": "LSTM_GridSearch",
        "rmse": rmse_gl if np.isnan(rmse_gl) else round(rmse_gl, 4),
        "mae": mae_gl if np.isnan(mae_gl) else round(mae_gl, 4),
        "mape": mape_gl if np.isnan(mape_gl) else round(mape_gl, 2),
        "n_test": valid_n,
    })

    df_results = pd.DataFrame(all_rows)

    # Afficher baseline binome pour comparaison
    print(f"\n{'='*55}")
    print(f"  BASELINE A BATTRE (resultats binome)")
    print(f"  Naif        RMSE global = 1.609")
    print(f"  SARIMAX+BESI RMSE global = 1.891  <- objectif")
    if not np.isnan(rmse_gl):
        print(f"  LSTM_GS     RMSE global = {rmse_gl:.4f}")
        if rmse_gl < 1.609:
            print(f"  -> EXCELLENT : tu bats meme le modele naif !")
        elif rmse_gl < 1.891:
            print(f"  -> BON : tu bats SARIMAX+BESI")
        else:
            print(f"  -> A ameliorer : SARIMAX reste meilleur")
    else:
        print(f"  LSTM_GS     RMSE global = NaN (aucun modele valide)")
    print(f"{'='*55}")

    # Figures si save_fig
    if save_fig and best_by_bloc:
        _plot_blocs(best_by_bloc)

    return df_results


# ── Point d'entree ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    GOLD = ROOT / "data" / "gold" / "model_dataset_monthly.csv"

    if not GOLD.exists():
        print("Fichier manquant : data/gold/model_dataset_monthly.csv")
        print("Demander a la binome ce fichier.")
    else:
        df = pd.read_csv(GOLD, parse_dates=["month"], index_col="month")
        df.index.name = "date"

        # Ne PAS faire dropna global — ca supprime des lignes et casse les blocs
        # Remplacer seulement les chaines vides par NaN
        df = df.replace('', np.nan)

        print(f"Gold dataset charge : {df.shape}")
        print(f"Periode : {df.index[0].date()} -> {df.index[-1].date()}")

        # Features voulues — on garde seulement celles presentes ET non-vides
        FEATURES_WANTED = [
            "behavioral_index_pure_lag1",
            "trends_prix_alim",
            "fao_oils_yoy",
            "fx_yoy",
        ]
        # trends_carburant retire car 100% NaN dans ce dataset

        exog_cols = []
        for c in FEATURES_WANTED:
            if c not in df.columns:
                print(f"  [WARN] colonne absente : {c}")
                continue
            pct_nan = df[c].isna().mean() * 100
            if pct_nan > 50:
                print(f"  [WARN] colonne ignoree (trop de NaN : {pct_nan:.0f}%) : {c}")
                continue
            exog_cols.append(c)
            print(f"  [OK]  {c}  ({pct_nan:.1f}% NaN)")

        print(f"\nFeatures retenues : {exog_cols}")
        print(f"IPC level disponible : {df['ipc_level'].notna().sum()} mois")

        print(f"\n>>> GridSearch LSTM — Bloc A et Bloc B")
        df_all_results = run_gridsearch_blocs(
            df=df,
            ipc_col="ipc_level",
            exog_cols=exog_cols if exog_cols else None,
        )

        print("\n=== RESULTATS FINAUX PAR BLOC ===")
        print(df_all_results.to_string(index=False))

        # Comparaison avec resultats RobustScaler precedents
        csv_robust = REP_DIR / 'lstm_results_robust.csv'
        if csv_robust.exists():
            df_robust = pd.read_csv(csv_robust)
            print('\n=== COMPARAISON MinMaxScaler vs RobustScaler ===')
            df_compare = pd.merge(
                df_all_results[['bloc','rmse','mae','mape']].rename(
                    columns={'rmse':'rmse_minmax','mae':'mae_minmax','mape':'mape_minmax'}),
                df_robust[['bloc','rmse','mae','mape']].rename(
                    columns={'rmse':'rmse_robust','mae':'mae_robust','mape':'mape_robust'}),
                on='bloc'
            )
            df_compare['gain_rmse%'] = (
                (df_compare['rmse_robust'] - df_compare['rmse_minmax'])
                / df_compare['rmse_robust'] * 100
            ).round(1)
            print(df_compare.to_string(index=False))
            # Sauvegarder la comparaison
            df_compare.to_csv(REP_DIR / 'lstm_scaler_comparison.csv', index=False)
            print(f"Comparaison sauvegardee : {REP_DIR / 'lstm_scaler_comparison.csv'}")

        csv_final = REP_DIR / "lstm_results.csv"
        df_all_results.to_csv(csv_final, index=False)
        
        # Copie avec label RobustScaler pour comparaison future
        csv_robust = REP_DIR / "lstm_results_robust.csv"
        df_all_results.copy().assign(scaler="RobustScaler").to_csv(csv_robust, index=False)
        print(f"Resultats RobustScaler sauvegardes : {csv_robust}")
        
        print(f"\nCSV sauvegarde : {csv_final}")
