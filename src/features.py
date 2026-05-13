"""
Sélection de features — Projet BESI Maroc
Fonctions principales :
1) Analyse de corrélation par lag BESI -> IPC
2) Sélection des variables par causalité de Granger
3) Importance des features (Pearson + Spearman)
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import grangercausalitytests

np.random.seed(42)
warnings.filterwarnings("ignore")


# Chemins du projet
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FIGURES = ROOT / "outputs" / "figures"
OUTPUT_REPORTS = ROOT / "outputs" / "reports"
OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
OUTPUT_REPORTS.mkdir(parents=True, exist_ok=True)


def _to_series(x: pd.Series, name: str) -> pd.Series:
    """Convertit l'entrée en Series pandas et lui assigne un nom."""
    if isinstance(x, pd.Series):
        s = x.copy()
    else:
        s = pd.Series(x)
    s.name = name
    return s


def lag_correlation_analysis(besi_series, ipc_series, max_lag: int = 6) -> pd.DataFrame:
    """
    Calcule la corrélation de Pearson pour chaque lag (0..max_lag).

    Interprétation du lag k : corr(BESI_t, IPC_{t+k})
    => si la corrélation est forte pour k>0, BESI peut agir comme signal avancé.

    Paramètres
    ----------
    besi_series : array-like ou pd.Series
        Série BESI.
    ipc_series : array-like ou pd.Series
        Série IPC cible.
    max_lag : int, default=6
        Lag maximal en mois.

    Retour
    ------
    pd.DataFrame
        Colonnes : lag, correlation, p-value, n_obs
    """
    besi = _to_series(besi_series, "besi")
    ipc = _to_series(ipc_series, "ipc")

    if len(besi) != len(ipc):
        raise ValueError("besi_series et ipc_series doivent avoir la même longueur.")
    if max_lag < 0:
        raise ValueError("max_lag doit être >= 0.")

    results = []

    for lag in range(max_lag + 1):
        # Pour un lag k, on corrèle BESI_t avec IPC_{t+k}
        tmp = pd.concat([besi, ipc.shift(-lag)], axis=1).dropna()

        if len(tmp) < 3:
            corr = np.nan
            pval = np.nan
            n_obs = len(tmp)
        else:
            corr, pval = pearsonr(tmp["besi"], tmp["ipc"])
            n_obs = len(tmp)

        results.append(
            {
                "lag": lag,
                "correlation": corr,
                "p-value": pval,
                "n_obs": n_obs,
            }
        )

    results_df = pd.DataFrame(results)

    # Visualisation de l'analyse des lags
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        results_df["lag"],
        results_df["correlation"],
        color="#2E86AB",
        alpha=0.85,
        edgecolor="black",
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Corrélation BESI → IPC par Lag (Pearson)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Lag (mois)")
    ax.set_ylabel("Corrélation")
    ax.set_xticks(results_df["lag"])
    ax.grid(axis="y", alpha=0.3)

    # Annoter la valeur sur chaque barre pour lecture rapide
    for bar, value in zip(bars, results_df["correlation"]):
        if pd.notna(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + (0.02 if value >= 0 else -0.05),
                f"{value:.2f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES / "lag_correlation_besi_ipc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return results_df


def granger_feature_selection(
    features_df: pd.DataFrame,
    target_series,
    max_lag: int = 4,
    alpha: float = 0.05,
) -> list:
    """
    Sélectionne les features significatives avec un test de causalité de Granger.

    Pour chaque feature X, on teste : X cause-t-elle IPC ?
    Règle de sélection : min p-value (sur les lags 1..max_lag) < alpha.

    Paramètres
    ----------
    features_df : pd.DataFrame
        Features candidates (colonnes numériques).
    target_series : array-like ou pd.Series
        Série cible IPC.
    max_lag : int, default=4
        Lag maximal du test de Granger.
    alpha : float, default=0.05
        Seuil de significativité.

    Retour
    ------
    list
        Liste des noms de features significatives.
    """
    if not isinstance(features_df, pd.DataFrame):
        raise TypeError("features_df doit être un DataFrame pandas.")
    if max_lag < 1:
        raise ValueError("max_lag doit être >= 1.")

    target = _to_series(target_series, "target")
    significant_features = []

    for col in features_df.columns:
        # Garder uniquement les lignes complètes pour le test
        tmp = pd.concat([target, features_df[col]], axis=1).dropna()
        tmp.columns = ["target", "feature"]

        # Taille minimale approximative pour éviter les tests instables
        if len(tmp) <= (max_lag + 5):
            continue

        try:
            # Convention statsmodels : la 2e colonne "cause" la 1re
            test_result = grangercausalitytests(
                tmp[["target", "feature"]],
                maxlag=max_lag,
                verbose=False,
            )

            p_values = []
            for lag in range(1, max_lag + 1):
                pval = test_result[lag][0]["ssr_ftest"][1]
                p_values.append(pval)

            min_p = np.min(p_values)
            if min_p < alpha:
                significant_features.append(col)

        except Exception:
            # Si un test échoue pour une feature, on passe à la suivante
            continue

    return significant_features


def compute_feature_importance(features_df: pd.DataFrame, target_series) -> pd.DataFrame:
    """
    Calcule l'importance des features via corrélations Pearson + Spearman.

    Score d'importance retenu :
    importance_score = moyenne(|pearson_corr|, |spearman_corr|)

    Paramètres
    ----------
    features_df : pd.DataFrame
        Features candidates.
    target_series : array-like ou pd.Series
        Série cible IPC.

    Retour
    ------
    pd.DataFrame
        DataFrame trié par importance décroissante.
    """
    if not isinstance(features_df, pd.DataFrame):
        raise TypeError("features_df doit être un DataFrame pandas.")

    target = _to_series(target_series, "target")
    rows = []

    for col in features_df.columns:
        tmp = pd.concat([features_df[col], target], axis=1).dropna()
        tmp.columns = ["feature", "target"]

        if len(tmp) < 3:
            rows.append(
                {
                    "feature": col,
                    "pearson_corr": np.nan,
                    "pearson_p-value": np.nan,
                    "spearman_corr": np.nan,
                    "spearman_p-value": np.nan,
                    "importance_score": np.nan,
                    "n_obs": len(tmp),
                }
            )
            continue

        p_corr, p_pval = pearsonr(tmp["feature"], tmp["target"])
        s_corr, s_pval = spearmanr(tmp["feature"], tmp["target"])
        importance = np.mean([abs(p_corr), abs(s_corr)])

        rows.append(
            {
                "feature": col,
                "pearson_corr": p_corr,
                "pearson_p-value": p_pval,
                "spearman_corr": s_corr,
                "spearman_p-value": s_pval,
                "importance_score": importance,
                "n_obs": len(tmp),
            }
        )

    out = pd.DataFrame(rows).sort_values("importance_score", ascending=False).reset_index(drop=True)
    return out


def run_feature_selection_pipeline(
    dataset_path: Path | None = None,
    lag_max: int = 6,
    granger_max_lag: int = 4,
    alpha: float = 0.05,
) -> dict:
    """
    Exécute le pipeline complet de feature selection et exporte les résultats.

    Exports produits dans outputs/reports :
    - lag_correlation_results.csv
    - granger_significant_features.csv
    - feature_importance.csv
    - features_summary.txt
    """
    if dataset_path is None:
        dataset_path = ROOT / "data" / "processed" / "master_dataset.csv"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {dataset_path}")

    df = pd.read_csv(dataset_path)

    required_cols = {"besi", "ipc"}
    missing_required = required_cols - set(df.columns)
    if missing_required:
        raise ValueError(f"Colonnes manquantes dans dataset: {sorted(missing_required)}")

    lag_df = lag_correlation_analysis(df["besi"], df["ipc"], max_lag=lag_max)

    candidate_cols = [
        c
        for c in [
            "trends_composite",
            "reddit_composite",
            "youtube_composite",
            "besi",
            "ipc_change",
            "ipc_mom",
            "ipc_yoy",
        ]
        if c in df.columns
    ]

    features = df[candidate_cols]
    target = df["ipc"]

    significant = granger_feature_selection(
        features_df=features,
        target_series=target,
        max_lag=granger_max_lag,
        alpha=alpha,
    )
    importance_df = compute_feature_importance(features, target)

    # Exports CSV
    lag_path = OUTPUT_REPORTS / "lag_correlation_results.csv"
    granger_path = OUTPUT_REPORTS / "granger_significant_features.csv"
    importance_path = OUTPUT_REPORTS / "feature_importance.csv"
    summary_path = OUTPUT_REPORTS / "features_summary.txt"

    lag_df.to_csv(lag_path, index=False)
    pd.DataFrame({"feature": significant}).to_csv(granger_path, index=False)
    importance_df.to_csv(importance_path, index=False)

    # Résumé texte court pour lecture rapide
    best_lag_row = lag_df.loc[lag_df["correlation"].abs().idxmax()] if not lag_df.empty else None
    top_feature = (
        importance_df.iloc[0]["feature"]
        if not importance_df.empty and pd.notna(importance_df.iloc[0]["importance_score"])
        else "N/A"
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Résumé Feature Selection - Projet BESI Maroc\n")
        f.write("=" * 60 + "\n")
        f.write(f"Dataset: {dataset_path}\n")
        f.write(f"Observations: {len(df)}\n")
        f.write(f"Features testées: {len(candidate_cols)}\n\n")

        if best_lag_row is not None:
            f.write("Lag correlation BESI -> IPC\n")
            f.write(
                f"Meilleur lag (|corr| max): {int(best_lag_row['lag'])} | "
                f"corr={best_lag_row['correlation']:.4f} | "
                f"p-value={best_lag_row['p-value']:.4g}\n\n"
            )

        f.write("Granger (p < 0.05)\n")
        if significant:
            f.write("Features significatives: " + ", ".join(significant) + "\n\n")
        else:
            f.write("Aucune feature significative au seuil choisi.\n\n")

        f.write("Importance (Pearson + Spearman)\n")
        f.write(f"Top feature: {top_feature}\n")
        f.write("Fichiers exportés:\n")
        f.write(f"- {lag_path}\n")
        f.write(f"- {granger_path}\n")
        f.write(f"- {importance_path}\n")
        f.write(f"- {summary_path}\n")

    return {
        "lag_results": lag_df,
        "significant_features": significant,
        "importance": importance_df,
        "exports": {
            "lag_csv": lag_path,
            "granger_csv": granger_path,
            "importance_csv": importance_path,
            "summary_txt": summary_path,
            "lag_plot": OUTPUT_FIGURES / "lag_correlation_besi_ipc.png",
        },
    }


if __name__ == "__main__":
    outputs = run_feature_selection_pipeline()

    print("\n=== Lag Correlation BESI -> IPC ===")
    print(outputs["lag_results"])

    print("\n=== Features significatives (Granger, p<0.05) ===")
    print(outputs["significant_features"])

    print("\n=== Feature Importance (Pearson + Spearman) ===")
    print(outputs["importance"])

    print("\n=== Fichiers exportés ===")
    for key, path in outputs["exports"].items():
        print(f"{key}: {path}")
