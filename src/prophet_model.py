# prophet_model.py — Modèle Prophet pour prévision IPC Maroc

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Chemins
ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Couleurs cohérentes
_COL_REAL = "#1F77B4"
_COL_PRED = "#FF7F0E"
_COL_TRAIN = "#2CA02C"

# Fonctions métriques
def _rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def _mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def _mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def train_prophet(master_df: pd.DataFrame, train_end: str = "2021-12-01") -> dict:
    """
    Entraîne un modèle Prophet pour la prévision de l'IPC Maroc.

    Paramètres
    ----------
    master_df : DataFrame avec colonne 'ipc' et index date
    train_end : date de fin d'entraînement (début des prédictions)

    Retourne
    --------
    dict : rmse, mae, mape, y_true, y_pred, forecast_df
    """
    try:
        from prophet import Prophet
    except ImportError as exc:
        raise ImportError(
            "Prophet n'est pas installe. Installez-le avant execution avec: pip install prophet"
        ) from exc

    # Préparation des données
    df_prophet = master_df[['ipc']].reset_index()
    df_prophet.columns = ['ds', 'y']
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

    # Split train/test
    train_data = df_prophet[df_prophet['ds'] <= pd.Timestamp(train_end)]
    test_data = df_prophet[df_prophet['ds'] > pd.Timestamp(train_end)]

    if len(test_data) == 0:
        raise ValueError(f"Aucune donnée de test après {train_end}")

    print(f"Train: {train_data['ds'].min().date()} -> {train_data['ds'].max().date()} ({len(train_data)} obs)")
    print(f"Test:  {test_data['ds'].min().date()} -> {test_data['ds'].max().date()} ({len(test_data)} obs)")

    # Modèle Prophet
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )

    # Entraînement
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(train_data)

    # Prédictions
    future = model.make_future_dataframe(periods=len(test_data), freq='MS')
    forecast = model.predict(future)

    # Extraire les prédictions pour la période test
    forecast_test = forecast[forecast['ds'] > pd.Timestamp(train_end)]
    y_pred = forecast_test['yhat'].values
    y_true = test_data['y'].values

    # Métriques
    rmse = _rmse(y_true, y_pred)
    mae = _mae(y_true, y_pred)
    mape = _mape(y_true, y_pred)

    print("Métriques Prophet:")
    print(f"  RMSE: {rmse:.5f}")
    print(f"  MAE:  {mae:.5f}")
    print(f"  MAPE: {mape:.2f}%")
    # Figure
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Prophet — Prévision IPC Maroc", fontsize=12, fontweight="bold")

    # Données d'entraînement
    ax.plot(train_data['ds'], train_data['y'],
            color=_COL_TRAIN, lw=2, label="IPC entraînement")

    # Données de test réelles
    ax.plot(test_data['ds'], test_data['y'],
            color=_COL_REAL, lw=2, label="IPC réel (test)")

    # Prédictions
    ax.plot(forecast_test['ds'], y_pred,
            color=_COL_PRED, lw=2, ls='--', label=f"Prophet prédit (RMSE={rmse:.5f})")

    # Intervalle de confiance
    ax.fill_between(forecast_test['ds'],
                    forecast_test['yhat_lower'],
                    forecast_test['yhat_upper'],
                    color=_COL_PRED, alpha=0.2, label="Intervalle 95%")

    ax.axvline(pd.Timestamp(train_end), color="red", lw=1.5, ls="--", alpha=0.7,
               label=f"Coupure {pd.Timestamp(train_end).date()}")

    ax.set_xlabel("Date")
    ax.set_ylabel("IPC")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = FIG_DIR / "prophet_forecast.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Figure sauvegardee: {fig_path}")
    plt.close()

    # Sauvegarder les métriques en CSV (utilisé par results.ipynb)
    rep_dir = ROOT / "outputs" / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"rmse": rmse, "mae": mae, "mape": mape,
                   "n_train": len(train_data), "n_test": len(test_data),
                   "train_end": train_end}]).to_csv(
        rep_dir / "prophet_results.csv", index=False)
    print(f"Resultats sauvegardes : {rep_dir / 'prophet_results.csv'}")

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "y_true": y_true,
        "y_pred": y_pred,
        "forecast_df": forecast,
        "model": model
    }


# Point d'entrée pour test
if __name__ == "__main__":
    master_path = ROOT / "data" / "processed" / "master_dataset.csv"
    if master_path.exists():
        master = pd.read_csv(master_path, parse_dates=["date"], index_col="date")
        master.index.freq = "MS"

        result = train_prophet(master, train_end="2021-12-01")
        print(f"\nRésultat: RMSE={result['rmse']:.5f}")
    else:
        print(f"Fichier manquant: {master_path}")
