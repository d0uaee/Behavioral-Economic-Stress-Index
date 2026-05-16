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

def build_lstm(
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
    Entraine un reseau LSTM pour la prevision h=1 de l'IPC Maroc.

    Architecture
    ------------
    Input  : sequences de look_back mois (IPC + exog eventuellement)
    Couche 1 : LSTM(lstm_units[0], return_sequences=True)
    Dropout  : dropout
    Couche 2 : LSTM(lstm_units[1])
    Sortie   : Dense(1)
    Optimizer : Adam(lr=0.001)   Loss : MSE
    Early stopping : patience=8 sur val_loss

    Note methodologique
    -------------------
    Meme coupure train/test que SARIMA (train_end='2021-12-01').
    Normalisation MinMaxScaler 0-1, invertion avant les metriques.
    Pas d'optimisation hyperparametres -- but = comparaison juste.
    shuffle=False : la serie temporelle ne doit pas etre melangee.

    Parametres
    ----------
    series     : serie IPC mensuelle (valeurs absolues, freq='MS')
    exog       : variables exogenes (BESI, trends...) -- optionnel
    look_back  : taille de la fenetre d'entree en mois (defaut 12)
    train_end  : date de fin d'entrainement (defaut '2021-12-01')
    epochs     : nombre maximal d'epochs (early stopping actif)
    batch_size : taille des mini-lots
    lstm_units : (unites_couche1, unites_couche2)
    dropout    : taux de dropout entre les deux couches LSTM
    save_fig   : sauvegarder les 3 graphiques

    Retourne
    --------
    dict : rmse, mae, mape, y_true, y_pred, test_dates,
           train_time, n_params, epochs, history
    """
    # Import TensorFlow -- optionnel (ne bloque pas les autres modules)
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
        from tensorflow.keras.callbacks import EarlyStopping
        from sklearn.preprocessing import MinMaxScaler
        tf.random.set_seed(42)
    except ImportError as exc:
        raise ImportError(
            "TensorFlow et scikit-learn sont requis pour build_lstm().\n"
            "Installation : pip install tensorflow scikit-learn\n"
            f"Erreur originale : {exc}"
        ) from exc

    # ── Preparation des donnees ───────────────────────────────────────────────
    s = series.dropna().copy()

    if exog is not None:
        exog_al = exog.reindex(s.index).ffill().bfill()
        common  = s.index.intersection(exog_al.dropna().index)
        s       = s.loc[common]
        exog_al = exog_al.loc[common]
        data_arr = np.column_stack([s.values, exog_al.values])
    else:
        exog_al  = None
        data_arr = s.values.reshape(-1, 1)

    n_features = data_arr.shape[1]

    # Normalisation 0-1 (toutes les colonnes ensemble)
    scaler    = MinMaxScaler(feature_range=(0, 1))
    data_norm = scaler.fit_transform(data_arr)

    # ── Sequences glissantes ──────────────────────────────────────────────────
    # X[i] = data_norm[i : i+look_back]   forme : (n, look_back, n_features)
    # y[i] = data_norm[i+look_back, 0]    cible : IPC uniquement
    X_all, y_all, dates_all = [], [], []
    for i in range(len(data_norm) - look_back):
        X_all.append(data_norm[i : i + look_back])
        y_all.append(data_norm[i + look_back, 0])
        dates_all.append(s.index[i + look_back])

    X_all      = np.array(X_all)
    y_all      = np.array(y_all)
    dates_all  = pd.DatetimeIndex(dates_all)

    # Split train / test identique a SARIMA
    te         = pd.Timestamp(train_end)
    train_mask = dates_all <= te
    test_mask  = dates_all >  te

    if train_mask.sum() < look_back + 1:
        raise ValueError(
            f"Pas assez de donnees d'entrainement ({train_mask.sum()} sequences). "
            f"Reduire look_back ou avancer train_end."
        )
    if test_mask.sum() == 0:
        raise ValueError(f"Aucune donnee de test apres {train_end}.")

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test,  y_test  = X_all[test_mask],  y_all[test_mask]
    dates_test       = dates_all[test_mask]

    sep = "=" * 62
    print(f"\n{sep}")
    print("  LSTM -- PREVISION IPC MAROC")
    print(f"  Serie        : {s.index[0].date()} -> {s.index[-1].date()}"
          f"  ({len(s)} obs)")
    print(f"  Features     : {n_features}"
          f"  (IPC{' + exog' if exog is not None else ''})")
    print(f"  Look-back    : {look_back} mois")
    print(f"  Train        : {dates_all[train_mask][0].date()} -> "
          f"{te.date()}  ({train_mask.sum()} sequences)")
    print(f"  Test         : {dates_test[0].date()} -> "
          f"{dates_test[-1].date()}  ({test_mask.sum()} sequences)")
    print(f"  Architecture : Input({look_back},{n_features}) -> "
          f"LSTM({lstm_units[0]}) -> Dropout({dropout}) -> "
          f"LSTM({lstm_units[1]}) -> Dense(1)")
    print(f"  Epochs max   : {epochs}  |  Batch : {batch_size}")
    print(sep)

    # ── Construction du modele ────────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Sequential(
            [
                Input(shape=(look_back, n_features)),
                LSTM(lstm_units[0], return_sequences=True),
                Dropout(dropout),
                LSTM(lstm_units[1]),
                Dense(1),
            ],
            name="IPC_LSTM",
        )
        model.compile(optimizer="adam", loss="mse")

    n_params = model.count_params()
    print(f"\n  Parametres trainables : {n_params:,}")

    # ── Entrainement ──────────────────────────────────────────────────────────
    early_stop = EarlyStopping(
        monitor="val_loss", patience=8,
        restore_best_weights=True, verbose=0,
    )

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.15,
            callbacks=[early_stop],
            verbose=0,
            shuffle=False,   # serie temporelle -- ne pas melanger
        )
    train_time    = time.time() - t0
    actual_epochs = len(history.history["loss"])

    print(f"  Entrainement : {actual_epochs} epochs  ({train_time:.1f}s)")
    print(f"  Loss finale  train : {history.history['loss'][-1]:.6f}")
    print(f"  Loss finale  val   : {history.history['val_loss'][-1]:.6f}")

    # ── Predictions et inverse transform ─────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred_norm = model.predict(X_test, verbose=0).ravel()

    def _inverse_ipc(norm_vals: np.ndarray) -> np.ndarray:
        """Inverse MinMaxScaler sur la colonne IPC (col 0) seulement."""
        dummy = np.zeros((len(norm_vals), n_features))
        dummy[:, 0] = norm_vals
        return scaler.inverse_transform(dummy)[:, 0]

    y_pred_inv = _inverse_ipc(y_pred_norm)
    y_true_inv = _inverse_ipc(y_test)

    # ── Metriques ─────────────────────────────────────────────────────────────
    rmse = _rmse(y_true_inv, y_pred_inv)
    mae  = _mae(y_true_inv, y_pred_inv)
    mape = _mape(y_true_inv, y_pred_inv)

    print(f"\n  Metriques test ({test_mask.sum()} points) :")
    print(f"    RMSE  : {rmse:.5f}")
    print(f"    MAE   : {mae:.5f}")
    print(f"    MAPE  : {mape:.2f}%")
    print(sep)

    # ── Sauvegarde du modele ──────────────────────────────────────────────────
    model_path = MOD_DIR / "lstm_ipc.keras"
    try:
        model.save(str(model_path))
        print(f"  Modele sauvegarde : {model_path}")
    except Exception as e:
        print(f"  [WARN] Sauvegarde modele echouee : {e}")

    # ── Figures ───────────────────────────────────────────────────────────────
    if save_fig:
        # Predictions sur le train pour la figure
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_train_pred_norm = model.predict(X_train, verbose=0).ravel()
        y_train_pred_inv = _inverse_ipc(y_train_pred_norm)
        y_train_true_inv = _inverse_ipc(y_train)
        dates_train      = dates_all[train_mask]

        fig = plt.figure(figsize=(14, 9))
        gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.52, wspace=0.35)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])

        fig.suptitle(
            f"LSTM IPC Maroc -- RMSE={rmse:.5f}  MAE={mae:.5f}  MAPE={mape:.2f}%\n"
            f"LSTM({lstm_units[0]})-LSTM({lstm_units[1]})-Dense(1)  "
            f"look_back={look_back}  epochs={actual_epochs}  "
            f"params={n_params:,}  temps={train_time:.1f}s",
            fontsize=9, fontweight="bold",
        )

        # Panneau 1 : previsions vs reel ──────────────────────────────────────
        ax1.plot(dates_train, y_train_true_inv,
                 color="lightgray", lw=1.0, label="IPC (entrainement)")
        ax1.plot(dates_train, y_train_pred_inv,
                 color=_COL_TRAIN, lw=0.9, ls=":", alpha=0.7,
                 label="LSTM fit (train)")
        ax1.plot(dates_test, y_true_inv,
                 color=_COL_REAL, lw=2.0, label="IPC reel (test)", zorder=5)
        ax1.plot(dates_test, y_pred_inv,
                 color=_COL_LSTM, lw=1.8, ls="--",
                 label=f"LSTM pred (test)  RMSE={rmse:.5f}", zorder=4)
        ax1.axvline(te, color="red", lw=1.2, ls="--", alpha=0.55,
                    label=f"Coupure {te.date()}")
        ax1.set_title("LSTM : previsions vs valeurs reelles", fontsize=9)
        ax1.legend(fontsize=7.5, ncol=3)
        ax1.set_ylabel("IPC", fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Panneau 2 : courbe d'apprentissage ──────────────────────────────────
        ax2.plot(history.history["loss"],
                 color=_COL_LSTM, lw=1.8, label="Loss train (MSE)")
        ax2.plot(history.history["val_loss"],
                 color="#D62728", lw=1.5, ls="--", label="Loss validation")
        ax2.set_xlabel("Epoch", fontsize=8)
        ax2.set_ylabel("MSE", fontsize=8)
        ax2.set_title("Courbe d'apprentissage LSTM", fontsize=9)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Panneau 3 : scatter pred vs reel ────────────────────────────────────
        lims = (
            min(y_true_inv.min(), y_pred_inv.min()) * 0.9985,
            max(y_true_inv.max(), y_pred_inv.max()) * 1.0015,
        )
        ax3.scatter(y_true_inv, y_pred_inv,
                    color=_COL_LSTM, alpha=0.75, s=30, zorder=3)
        ax3.plot(lims, lims, color="black", lw=1.0, ls="--", alpha=0.45,
                 label="Prediction parfaite")
        ax3.set_xlim(*lims)
        ax3.set_ylim(*lims)
        ax3.set_xlabel("IPC reel", fontsize=8)
        ax3.set_ylabel("IPC predit", fontsize=8)
        ax3.set_title("Previsions vs valeurs reelles (test)", fontsize=9)
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        path = FIG_DIR / "lstm_predictions.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Figure sauvegardee : {path}")
        plt.close(fig)

    return {
        "rmse":       rmse,
        "mae":        mae,
        "mape":       mape,
        "y_true":     y_true_inv,
        "y_pred":     y_pred_inv,
        "test_dates": dates_test,
        "train_time": round(train_time, 1),
        "n_params":   n_params,
        "epochs":     actual_epochs,
        "history":    history.history,
        "model":      model,
        "look_back":  look_back,
        "n_features": n_features,
    }


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


# ── Point d'entree ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from pathlib import Path as _P

    _root   = _P(__file__).resolve().parent.parent
    _ipc    = _root / "data" / "processed" / "ipc_processed.csv"
    _master = _root / "data" / "processed" / "master_dataset.csv"

    if not (_ipc.exists() and _master.exists()):
        print("Fichiers manquants -- lancer d'abord : python src/data_pipeline.py")
    else:
        df_ipc    = pd.read_csv(_ipc,    parse_dates=["date"], index_col="date")
        df_master = pd.read_csv(_master, parse_dates=["date"], index_col="date")
        df_ipc.index.freq    = "MS"
        df_master.index.freq = "MS"

        ipc  = df_ipc["ipc"]
        exog = df_master[["besi"]]

        # ── LSTM sans exog ────────────────────────────────────────────────────
        print("\n>>> LSTM (IPC seul)")
        res_lstm = build_lstm(
            ipc, exog=None, look_back=12,
            train_end="2021-12-01", epochs=50, save_fig=True,
        )

        # ── LSTM avec BESI ────────────────────────────────────────────────────
        print("\n>>> LSTM + BESI")
        res_lstm_besi = build_lstm(
            ipc, exog=exog, look_back=12,
            train_end="2021-12-01", epochs=50, save_fig=False,
        )

        # ── Comparaison finale ────────────────────────────────────────────────
        # Charger les resultats SARIMA/SARIMAX si dispo (CSV)
        results = {
            "LSTM":           res_lstm,
            "LSTM_BESI":      res_lstm_besi,
        }

        # Essayer de recuperer les metriques SARIMA depuis le CSV de comparaison
        comp_csv = _root / "outputs" / "reports" / "model_comparison.csv"
        if comp_csv.exists():
            try:
                df_prev = pd.read_csv(comp_csv, index_col=0)
                for mname in df_prev.index:
                    if "RMSE_h1" in df_prev.columns:
                        results[mname] = {
                            "rmse": float(df_prev.loc[mname, "RMSE_h1"]),
                            "mae":  float(df_prev.loc[mname, "MAE_h1"])
                                    if "MAE_h1" in df_prev.columns else np.nan,
                            "mape": float(df_prev.loc[mname, "MAPE_h1"])
                                    if "MAPE_h1" in df_prev.columns else np.nan,
                        }
                print(f"  Metriques precedentes chargees depuis {comp_csv}")
            except Exception as e:
                print(f"  [WARN] Lecture model_comparison.csv echouee : {e}")

        print("\n>>> Comparaison finale de tous les modeles")
        df_final = compare_all_models(results, series=ipc,
                                       train_end="2021-12-01", save_fig=True)
        print(df_final.to_string())
