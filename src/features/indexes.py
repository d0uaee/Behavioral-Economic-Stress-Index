"""
src/features/indexes.py — Construction BESI v3

Deux indices séparés, sans data leakage :

    behavioral_index_pure :
        Uniquement Google Trends — 100% comportemental, 0% IPC
        Poids calibrés par LassoCV sur train (2010-2018)

    hybrid_macro_index :
        Trends + FAO FPI + taux de change MAD/EUR
        Poids calibrés par LassoCV sur train (2010-2018)

Règle fondamentale : ipc_level, inflation_mom, inflation_yoy, ipc_change
sont INTERDITS dans les deux indices. Ce sont les cibles.
"""

import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parent.parent.parent
SILVER_DIR = ROOT / "data" / "silver"
REPORTS    = ROOT / "outputs" / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

# Période d'entraînement pour la calibration des poids
LASSO_TRAIN_END = "2018-12-01"


def _normalise_0_1(s: pd.Series) -> pd.Series:
    mn, mx = s.dropna().min(), s.dropna().max()
    if mx == mn:
        return pd.Series(0.0, index=s.index, name=s.name)
    return (s - mn) / (mx - mn)


def build_behavioral_index_pure(
    trends_df:   pd.DataFrame,
    train_end:   str = LASSO_TRAIN_END,
    target_col:  str = "inflation_yoy",
    cpi_silver:  "pd.DataFrame | None" = None,
    method:      str = "lasso",   # "lasso" | "equal" | "pca"
) -> pd.Series:
    """
    Construit le behavioral_index_pure à partir des signaux Google Trends uniquement.

    Paramètres
    ----------
    trends_df   : silver/google_trends_monthly.csv (sous-indices thématiques)
    train_end   : date de fin de la période de calibration des poids
    target_col  : variable cible pour calibration Lasso (doit être dans cpi_silver)
    cpi_silver  : silver/cpi_monthly.csv (nécessaire si method='lasso')
    method      : 'lasso' (calibré), 'equal' (moyenne équipondérée), 'pca' (1er facteur)

    Retourne
    --------
    pd.Series 0-1, index MS, name='behavioral_index_pure'
    """
    # Colonnes des sous-indices thématiques (pas le composite)
    feature_cols = [c for c in trends_df.columns
                    if c.startswith("trends_") and c != "trends_composite"
                    and "n_keywords" not in c]

    if not feature_cols:
        raise ValueError(
            "Aucun sous-indice trends_* trouvé dans trends_df. "
            f"Colonnes disponibles : {list(trends_df.columns)}"
        )

    X = trends_df[feature_cols].copy()
    X = X.ffill().bfill()

    if method == "equal":
        raw = X.mean(axis=1)
        weights = {c: 1.0 / len(feature_cols) for c in feature_cols}

    elif method == "lasso":
        if cpi_silver is None:
            logger.warning("cpi_silver non fourni pour Lasso → fallback equal weights")
            return build_behavioral_index_pure(trends_df, train_end, target_col,
                                               cpi_silver=None, method="equal")
        try:
            from sklearn.linear_model import LassoCV
            from sklearn.preprocessing import StandardScaler

            # Aligner trends et cible sur la période de train
            common_train = X.index.intersection(cpi_silver.index)
            common_train = common_train[common_train <= pd.Timestamp(train_end)]

            X_train = X.loc[common_train].values
            y_train = cpi_silver.loc[common_train, target_col].values

            # Retirer les NaN
            mask = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
            X_train, y_train = X_train[mask], y_train[mask]

            if len(X_train) < 12:
                logger.warning(f"Trop peu d'observations ({len(X_train)}) pour Lasso → equal")
                return build_behavioral_index_pure(trends_df, train_end, target_col,
                                                   cpi_silver=None, method="equal")

            scaler  = StandardScaler()
            X_sc    = scaler.fit_transform(X_train)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lasso = LassoCV(cv=5, max_iter=5000, random_state=42, positive=True)
                lasso.fit(X_sc, y_train)

            raw_weights = dict(zip(feature_cols, lasso.coef_))
            total = sum(abs(v) for v in raw_weights.values())

            if total == 0:
                logger.warning("Lasso → tous coefficients nuls, fallback equal")
                weights = {c: 1.0/len(feature_cols) for c in feature_cols}
            else:
                weights = {c: abs(v)/total for c, v in raw_weights.items()}

            # Appliquer les poids sur toutes les données (pas seulement train)
            X_all_sc = scaler.transform(X.fillna(0).values)
            raw = pd.Series(X_all_sc @ lasso.coef_, index=X.index)

        except ImportError:
            logger.warning("scikit-learn requis pour Lasso → fallback equal")
            return build_behavioral_index_pure(trends_df, train_end, target_col,
                                               cpi_silver=None, method="equal")

    elif method == "pca":
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            common_train = X.index.intersection(cpi_silver.index) if cpi_silver is not None else X.index
            common_train = common_train[common_train <= pd.Timestamp(train_end)]

            X_filled = X.fillna(X.mean())
            X_train  = X_filled.loc[common_train].values
            if len(X_train) < 12:
                logger.warning("Trop peu d'observations pour PCA → fallback equal")
                return build_behavioral_index_pure(trends_df, train_end, target_col,
                                                   cpi_silver=None, method="equal")

            scaler   = StandardScaler()
            X_sc     = scaler.fit_transform(X_train)
            pca      = PCA(n_components=1, random_state=42)
            pca.fit(X_sc)
            raw      = pd.Series(pca.transform(scaler.transform(X_filled.values))[:, 0], index=X.index)
            weights  = {c: float(v) for c, v in zip(feature_cols, pca.components_[0])}
        except ImportError:
            logger.warning("scikit-learn requis pour PCA → fallback equal")
            return build_behavioral_index_pure(trends_df, train_end, target_col,
                                               cpi_silver=None, method="equal")
    else:
        raise ValueError(f"method doit être 'lasso', 'equal' ou 'pca'. Reçu : {method}")

    # Normalisation 0-1 finale
    idx = build_behavioral_index_pure  # just to avoid redefinition
    result = _normalise_0_1(raw).rename("behavioral_index_pure")

    # Log des poids
    logger.info(f"behavioral_index_pure ({method}) :")
    for col, w in sorted(weights.items(), key=lambda x: -abs(x[1])):
        logger.info(f"  {col:<30} : {w:.4f}")
    logger.info(f"  mean={result.mean():.3f}  std={result.std():.3f}")

    # Sauvegarder les poids
    w_df = pd.DataFrame(
        list(weights.items()), columns=["feature", "weight"]
    ).sort_values("weight", ascending=False)
    w_df["method"] = method
    w_df["train_end"] = train_end
    w_df.to_csv(REPORTS / "besi_v3_behavioral_weights.csv", index=False)

    return result


def build_hybrid_macro_index(
    behavioral_series: pd.Series,
    macro_df:          pd.DataFrame,
    train_end:         str = LASSO_TRAIN_END,
    target_col:        str = "inflation_yoy",
    cpi_silver:        "pd.DataFrame | None" = None,
    method:            str = "lasso",
) -> pd.Series:
    """
    Construit le hybrid_macro_index en combinant signaux comportementaux
    et variables macro (FAO FPI, taux de change).

    Paramètres
    ----------
    behavioral_series : résultat de build_behavioral_index_pure()
    macro_df          : silver/macro_signals_monthly.csv
    (autres params identiques à build_behavioral_index_pure)

    Retourne
    --------
    pd.Series 0-1, index MS, name='hybrid_macro_index'
    """
    # Features macro pertinentes (variation annuelle — mêmes unités)
    macro_feature_cols = [c for c in macro_df.columns
                          if any(kw in c for kw in ["yoy", "fx_yoy"])
                          and c not in ["inflation_yoy", "inflation_mom"]]

    if not macro_feature_cols:
        logger.warning("Aucune feature macro YoY trouvée → retour behavioral_index_pure")
        return behavioral_series.rename("hybrid_macro_index")

    # Assembler : behavioral + macro
    combined = pd.DataFrame({"behavioral_pure": behavioral_series})
    for col in macro_feature_cols:
        if col in macro_df.columns:
            combined[col] = _normalise_0_1(macro_df[col].reindex(combined.index).ffill())

    combined = combined.ffill().bfill()
    feature_cols = list(combined.columns)

    if method == "equal":
        raw     = combined.mean(axis=1)
        weights = {c: 1.0/len(feature_cols) for c in feature_cols}

    elif method == "lasso":
        if cpi_silver is None:
            logger.warning("cpi_silver non fourni → fallback equal")
            return build_hybrid_macro_index(behavioral_series, macro_df, train_end,
                                            target_col, cpi_silver=None, method="equal")
        try:
            from sklearn.linear_model import LassoCV
            from sklearn.preprocessing import StandardScaler

            common_train = combined.index.intersection(cpi_silver.index)
            common_train = common_train[common_train <= pd.Timestamp(train_end)]

            X_train = combined.loc[common_train].values
            y_train = cpi_silver.loc[common_train, target_col].values
            mask    = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
            X_train, y_train = X_train[mask], y_train[mask]

            if len(X_train) < 12:
                logger.warning("Trop peu d'obs → equal")
                return build_hybrid_macro_index(behavioral_series, macro_df, train_end,
                                                target_col, cpi_silver=None, method="equal")

            scaler = StandardScaler()
            X_sc   = scaler.fit_transform(X_train)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lasso = LassoCV(cv=5, max_iter=5000, random_state=42, positive=True)
                lasso.fit(X_sc, y_train)

            raw_w  = dict(zip(feature_cols, lasso.coef_))
            total  = sum(abs(v) for v in raw_w.values())
            weights = {c: abs(v)/total for c, v in raw_w.items()} if total > 0 \
                      else {c: 1.0/len(feature_cols) for c in feature_cols}

            X_all_sc = scaler.transform(combined.fillna(0).values)
            raw      = pd.Series(X_all_sc @ lasso.coef_, index=combined.index)

        except ImportError:
            logger.warning("scikit-learn requis → fallback equal")
            return build_hybrid_macro_index(behavioral_series, macro_df, train_end,
                                            target_col, cpi_silver=None, method="equal")
    else:
        raise ValueError(f"method invalide : {method}")

    result = _normalise_0_1(raw).rename("hybrid_macro_index")

    logger.info(f"hybrid_macro_index ({method}) :")
    for col, w in sorted(weights.items(), key=lambda x: -abs(x[1])):
        logger.info(f"  {col:<35} : {w:.4f}")
    logger.info(f"  mean={result.mean():.3f}  std={result.std():.3f}")

    # Sauvegarder les poids
    w_df = pd.DataFrame(list(weights.items()), columns=["feature", "weight"])
    w_df["method"] = method
    w_df["train_end"] = train_end
    w_df.to_csv(REPORTS / "besi_v3_hybrid_weights.csv", index=False)

    return result
