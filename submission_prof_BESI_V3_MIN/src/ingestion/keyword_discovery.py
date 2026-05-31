"""
src/ingestion/keyword_discovery.py — Découverte et validation rigoureuse des keywords

OBJECTIF : répondre à la critique "pourquoi ces 7 mots-clés ?"
en remplaçant la sélection manuelle par une sélection automatique et validée.

PIPELINE EN 4 ÉTAPES :
    1. Expansion    : related_queries() → 50-100 keywords candidats
    2. Collecte     : télécharger les séries Trends pour tous les candidats
    3. Filtrage STL : décomposer la saisonnalité, corrélation sur résidus
    4. Clustering   : K-Means sur les séries → 1 représentant par cluster

SORTIE :
    data/silver/validated_keywords.csv   — liste des keywords retenus + scores
    data/silver/keyword_candidates.csv   — tous les candidats avec leurs métriques
    data/bronze/trends_candidates_raw.csv — séries brutes Trends

Usage :
    from src.ingestion.keyword_discovery import run_keyword_discovery
    validated = run_keyword_discovery()
"""

import logging
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

ROOT       = Path(__file__).resolve().parent.parent.parent
BRONZE_DIR = ROOT / "data" / "bronze"
SILVER_DIR = ROOT / "data" / "silver"
REPORTS    = ROOT / "outputs" / "reports"
BRONZE_DIR.mkdir(parents=True, exist_ok=True)
SILVER_DIR.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

# ── Paramètres ────────────────────────────────────────────────────────────────
GEO            = "MA"
TIMEFRAME      = "2017-01-01 2024-12-31"
CORR_THRESHOLD = 0.25      # corrélation minimale (résidu STL) pour garder un keyword
MAX_LAG        = 2         # lag max pour la corrélation croisée (signal avancé)
N_CLUSTERS     = 7         # nombre de groupes thématiques finaux
DELAY_BETWEEN  = 65        # secondes entre requêtes pytrends (anti-ban)

# Keywords seeds : points de départ de la découverte
SEED_KEYWORDS = [
    "inflation maroc",       # anchor principal
    "prix alimentaires",     # panier IPC alimentaire (40%)
    "pouvoir d achat",       # concept économique clé
    "cherté",                # terme populaire marocain
    "prix carburant",        # énergie (9% panier IPC)
]


# ── Étape 1 : Expansion via related_queries ───────────────────────────────────

def _get_related_queries(seed: str) -> pd.DataFrame:
    """
    Appelle pytrends.related_queries() pour un seed.
    Retourne un DataFrame avec les keywords candidats et leur score de pertinence.
    """
    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl="fr-MA", geo=GEO, timeout=(10, 25))
        pytrends.build_payload([seed], geo=GEO, timeframe=TIMEFRAME)
        time.sleep(3)
        related = pytrends.related_queries()

        results = []
        for query_type in ["top", "rising"]:
            df = related.get(seed, {}).get(query_type)
            if df is not None and not df.empty:
                df = df.copy()
                df["seed"]  = seed
                df["type"]  = query_type
                df.columns  = [c if c not in ["query", "value"]
                                else ("keyword" if c == "query" else "score")
                                for c in df.columns]
                results.append(df)

        if results:
            return pd.concat(results, ignore_index=True)
    except Exception as e:
        logger.warning(f"related_queries failed pour '{seed}' : {e}")
    return pd.DataFrame(columns=["keyword", "score", "seed", "type"])


def discover_candidate_keywords(seeds: List[str] = None) -> List[str]:
    """
    Étape 1 : Génère une liste de keywords candidats à partir des seeds.
    Retourne une liste dédupliquée de keywords en français (filtre les requêtes
    trop génériques ou hors-sujet).

    Justification de l'exclusion :
    - Keywords < 3 caractères : trop génériques
    - Keywords sans rapport économique direct : exclus par filtre thématique
    """
    if seeds is None:
        seeds = SEED_KEYWORDS

    # Mots non économiques à exclure (bruit)
    EXCLUDE_TERMS = {
        "recette", "cuisine", "video", "youtube", "facebook",
        "تحميل", "مشاهدة", "film", "musique", "chanson",
        "meteo", "weather", "football", "sport",
    }

    all_candidates = []
    for seed in seeds:
        logger.info(f"  related_queries pour : '{seed}'")
        df = _get_related_queries(seed)
        if not df.empty and "keyword" in df.columns:
            candidates = df["keyword"].dropna().tolist()
            # Filtrer : longueur minimale + pas de bruit
            candidates = [
                kw for kw in candidates
                if len(kw) >= 3 and
                not any(excl in kw.lower() for excl in EXCLUDE_TERMS)
            ]
            all_candidates.extend(candidates)
            logger.info(f"    -> {len(candidates)} candidats ajoutés")
        time.sleep(DELAY_BETWEEN)

    # Dédupliquer + ajouter les seeds originaux
    all_candidates.extend(seeds)
    unique_candidates = list(dict.fromkeys(all_candidates))  # préserve l'ordre
    logger.info(f"  Total candidats uniques : {len(unique_candidates)}")
    return unique_candidates


# ── Étape 2 : Collecte des séries temporelles ─────────────────────────────────

def _fetch_trends_batch(keywords: List[str], anchor: str) -> pd.DataFrame:
    """
    Télécharge les séries Trends pour un batch de keywords.
    Utilise l'anchor pour rendre les séries comparables entre batches.
    pytrends limite à 5 keywords par requête.
    """
    from pytrends.request import TrendReq
    pytrends = TrendReq(hl="fr-MA", geo=GEO, timeout=(10, 25))

    # Toujours inclure l'anchor pour la normalisation
    batch = [anchor] + [kw for kw in keywords if kw != anchor][:4]

    try:
        pytrends.build_payload(batch, geo=GEO, timeframe=TIMEFRAME, gprop="")
        time.sleep(4)
        df = pytrends.interest_over_time()
        if df.empty:
            return pd.DataFrame()
        df = df.drop(columns=["isPartial"], errors="ignore")
        df.index = pd.to_datetime(df.index)
        df.index = df.index.to_period("M").to_timestamp("MS")
        return df
    except Exception as e:
        logger.warning(f"  fetch_trends_batch échoué pour {batch} : {e}")
        return pd.DataFrame()


def collect_trends_for_candidates(
    candidates: List[str],
    anchor: str = "inflation maroc",
    out_path: Path = None,
) -> pd.DataFrame:
    """
    Étape 2 : Télécharge les séries Trends pour tous les candidats en batches de 4.
    L'anchor est toujours présent pour assurer la comparabilité.
    """
    if out_path is None:
        out_path = BRONZE_DIR / "trends_candidates_raw.csv"

    # Si déjà téléchargé → réutiliser
    if out_path.exists():
        logger.info(f"  Chargement depuis cache : {out_path.name}")
        df = pd.read_csv(out_path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        return df

    all_series = {}
    batches = [candidates[i:i+4] for i in range(0, len(candidates), 4)]

    for i, batch in enumerate(batches):
        logger.info(f"  Batch {i+1}/{len(batches)} : {batch}")
        df_batch = _fetch_trends_batch(batch, anchor)
        if not df_batch.empty:
            for col in df_batch.columns:
                if col not in all_series:
                    all_series[col] = df_batch[col]
        time.sleep(DELAY_BETWEEN)

    if not all_series:
        logger.error("Aucune série téléchargée")
        return pd.DataFrame()

    result = pd.DataFrame(all_series)
    result.index.name = "month"
    result.to_csv(out_path)
    logger.info(f"  Séries sauvegardées : {result.shape} -> {out_path.name}")
    return result


# ── Étape 3 : Filtrage par corrélation sur résidus STL ───────────────────────

def _stl_residual(series: pd.Series, period: int = 12) -> pd.Series:
    """
    Décompose une série temporelle (STL) et retourne le résidu non-saisonnier.
    C'est ce résidu qui est corrélé avec l'inflation — pas la série brute.

    Pourquoi les résidus et pas la série brute ?
    → La série brute de "prix huile" contient un pic chaque Ramadan.
      Ce pic n'est pas un signal d'inflation, c'est un signal calendaire.
      Le résidu STL supprime ce biais saisonnier systématique.
    """
    from statsmodels.tsa.seasonal import STL
    try:
        series_clean = series.interpolate(limit=3).fillna(method="bfill")
        stl = STL(series_clean, period=period, robust=True)
        result = stl.fit()
        return pd.Series(result.resid, index=series.index, name=series.name)
    except Exception:
        # Fallback : différence à 12 mois (élimination saisonnalité)
        return series.diff(12).fillna(0)


def compute_residual_correlations(
    trends_df: pd.DataFrame,
    inflation_series: pd.Series,
    max_lag: int = MAX_LAG,
) -> pd.DataFrame:
    """
    Étape 3a : Calcule la corrélation entre chaque keyword (résidu STL)
    et l'inflation_yoy (résidu STL) à différents lags.

    Retourne un DataFrame avec :
    - keyword : nom du keyword
    - best_lag : lag (0, 1 ou 2) avec la meilleure corrélation absolue
    - best_r   : corrélation de Pearson au meilleur lag (résidus)
    - raw_r    : corrélation brute (sans STL, pour comparaison)
    - keep     : True si |best_r| > CORR_THRESHOLD
    """
    # Résidu STL de l'inflation
    infl_resid = _stl_residual(inflation_series.reindex(trends_df.index).interpolate())

    records = []
    for col in trends_df.columns:
        series = trends_df[col].dropna()
        if len(series) < 24:
            continue

        # Résidu STL du keyword
        kw_resid = _stl_residual(series.reindex(trends_df.index).interpolate())

        # Corrélation brute (sans STL) — pour montrer le biais
        raw_r = float(series.corr(inflation_series.reindex(series.index)))

        # Corrélation sur résidus à différents lags
        best_r, best_lag = 0.0, 0
        for lag in range(0, max_lag + 1):
            # lag > 0 : le keyword précède l'inflation → signal avancé
            shifted_infl = infl_resid.shift(-lag)
            mask = kw_resid.notna() & shifted_infl.notna()
            if mask.sum() < 20:
                continue
            r = float(kw_resid[mask].corr(shifted_infl[mask]))
            if abs(r) > abs(best_r):
                best_r, best_lag = r, lag

        records.append({
            "keyword":  col,
            "best_lag": best_lag,
            "best_r":   round(best_r, 3),
            "raw_r":    round(raw_r, 3),
            "bias_stl": round(abs(raw_r) - abs(best_r), 3),  # biais supprimé par STL
            "keep":     abs(best_r) >= CORR_THRESHOLD,
        })

    df_scores = pd.DataFrame(records).sort_values("best_r", ascending=False, key=abs)
    return df_scores


# ── Étape 4 : Clustering pour éliminer la redondance ─────────────────────────

def cluster_and_select(
    trends_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    n_clusters: int = N_CLUSTERS,
) -> Dict[int, str]:
    """
    Étape 4 : Regroupe les keywords retenus en clusters thématiques.
    Sélectionne 1 représentant par cluster (celui avec la meilleure corrélation).

    Pourquoi le clustering ?
    → "prix huile" et "prix huile maroc" ont des séries quasi-identiques.
      Les deux dans le BESI = double comptage du même signal.
      Le clustering garantit que chaque groupe thématique contribue une fois.

    Retourne : dict {cluster_id: keyword_représentant}
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    valid_kws = scores_df[scores_df["keep"]]["keyword"].tolist()
    valid_kws = [kw for kw in valid_kws if kw in trends_df.columns]

    if len(valid_kws) < n_clusters:
        logger.warning(f"Seulement {len(valid_kws)} keywords valides < {n_clusters} clusters")
        n_clusters = max(2, len(valid_kws))

    X = trends_df[valid_kws].fillna(0).T  # shape : (n_keywords, n_mois)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Pour chaque cluster, garder le keyword avec la meilleure corrélation
    cluster_map = {}
    for cluster_id in range(n_clusters):
        kws_in_cluster = [kw for kw, lbl in zip(valid_kws, labels) if lbl == cluster_id]
        if not kws_in_cluster:
            continue
        # Trier par |best_r| décroissant → prendre le meilleur
        best_kw = (scores_df[scores_df["keyword"].isin(kws_in_cluster)]
                   .sort_values("best_r", ascending=False, key=abs)
                   .iloc[0]["keyword"])
        cluster_map[cluster_id] = best_kw

    return cluster_map


# ── Pipeline principal ────────────────────────────────────────────────────────

def run_keyword_discovery(
    use_cached_candidates: bool = True,
    use_cached_trends: bool = True,
) -> pd.DataFrame:
    """
    Pipeline complet de découverte et validation des keywords.

    Arguments
    ---------
    use_cached_candidates : si True, charge la liste de candidats depuis le cache
                            (évite les appels pytrends coûteux)
    use_cached_trends     : si True, charge les séries Trends depuis le cache

    Retourne
    --------
    pd.DataFrame avec les keywords validés et leurs métriques
    (sauvegardé dans data/silver/validated_keywords.csv)
    """
    out_validated = SILVER_DIR / "validated_keywords.csv"
    out_candidates = SILVER_DIR / "keyword_candidates.csv"

    # Charger l'IPC pour la corrélation
    cpi_path = SILVER_DIR / "cpi_monthly.csv"
    if not cpi_path.exists():
        raise FileNotFoundError("cpi_monthly.csv manquant — exécuter d'abord la transform CPI")
    cpi = pd.read_csv(cpi_path, parse_dates=["month"], index_col="month")
    inflation = cpi["inflation_yoy"].dropna()

    # ── Étape 1 : Découverte des candidats ──────────────────────────────────
    candidates_path = BRONZE_DIR / "keyword_candidates_list.txt"
    if use_cached_candidates and candidates_path.exists():
        with open(candidates_path, "r", encoding="utf-8") as f:
            candidates = [line.strip() for line in f if line.strip()]
        logger.info(f"[Étape 1] Candidats chargés depuis cache : {len(candidates)}")
    else:
        logger.info("[Étape 1] Découverte des keywords via related_queries...")
        candidates = discover_candidate_keywords()
        with open(candidates_path, "w", encoding="utf-8") as f:
            f.write("\n".join(candidates))
        logger.info(f"  {len(candidates)} candidats découverts et sauvegardés")

    # ── Étape 2 : Collecte des séries Trends ────────────────────────────────
    logger.info("[Étape 2] Collecte des séries temporelles pour les candidats...")
    trends_raw = collect_trends_for_candidates(
        candidates,
        anchor="inflation maroc",
        out_path=BRONZE_DIR / "trends_candidates_raw.csv",
    )

    if trends_raw.empty:
        logger.error("Aucune série Trends disponible")
        return pd.DataFrame()

    # ── Étape 3 : Filtrage par corrélation sur résidus STL ──────────────────
    logger.info("[Étape 3] Filtrage par corrélation résidus STL...")
    scores = compute_residual_correlations(
        trends_raw,
        inflation,
        max_lag=MAX_LAG,
    )

    n_kept    = scores["keep"].sum()
    n_total   = len(scores)
    bias_mean = scores["bias_stl"].abs().mean()
    logger.info(f"  Keywords retenus : {n_kept}/{n_total} (seuil r > {CORR_THRESHOLD})")
    logger.info(f"  Biais saisonnier moyen supprimé par STL : {bias_mean:.3f}")
    scores.to_csv(out_candidates, index=False)

    # ── Étape 4 : Clustering et sélection des représentants ─────────────────
    logger.info(f"[Étape 4] Clustering K-Means en {N_CLUSTERS} groupes thématiques...")
    cluster_map = cluster_and_select(trends_raw, scores, n_clusters=N_CLUSTERS)

    # Construire le DataFrame final
    validated_rows = []
    for cluster_id, kw in cluster_map.items():
        row = scores[scores["keyword"] == kw].iloc[0].to_dict()
        row["cluster"] = cluster_id
        validated_rows.append(row)

    validated = pd.DataFrame(validated_rows).sort_values("best_r", ascending=False, key=abs)
    validated.to_csv(out_validated, index=False)

    logger.info("[RÉSULTAT] Keywords validés finaux :")
    for _, row in validated.iterrows():
        logger.info(f"  Cluster {int(row['cluster'])} | '{row['keyword']}' "
                    f"| r={row['best_r']:.3f} @ lag={int(row['best_lag'])} "
                    f"| biais_STL={row['bias_stl']:.3f}")

    return validated
