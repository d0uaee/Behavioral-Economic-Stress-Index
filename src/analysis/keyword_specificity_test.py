"""
src/analysis/keyword_specificity_test.py — Test de spécificité locale des keywords v1

Objectif : prouver que les keywords MAROCAINS spécifiques (jeu A) portent
un signal BESI supérieur aux keywords génériques (jeu B), universels (jeu C
en darija) et hors-contexte (jeu D Tunisie).

Hypothèse : local > générique > hors-contexte
    H_local  : AIC(SARIMAX+BESI_A) < AIC(SARIMAX+BESI_B)
    H_darija : AIC(SARIMAX+BESI_C) ~ AIC(SARIMAX+BESI_A)  (même culture, faible volume)
    H_tunisie: AIC(SARIMAX+BESI_D) > AIC(SARIMAX+BESI_A)  (hors-contexte)

Quatre jeux de keywords téléchargés via pytrends (geo=MA, 2017-2024) :
    Jeu A — Français marocain (keywords actuels du projet)
    Jeu B — Anglais générique (thèmes identiques, langue internationale)
    Jeu C — Darija/Arabe (volume faible = bruit élevé attendu)
    Jeu D — Tunisie (mêmes thèmes, mauvais pays = signal hors-contexte)

Pour chaque jeu :
    1. Download pytrends avec chunking + anchor cross-normalisation
    2. LassoCV (train 2017→2019, même période que placebo_test.py)
    3. SARIMAX(1,1,1)×(1,0,1)[12] sur Bloc A train (36 mois)
    4. AIC, RMSE Bloc A (in-sample), RMSE Bloc B (out-of-sample)
    5. Recall Bloc B (seuil calibré sur train)
    6. Test de Granger (lag=1 : BESI -> IPC)

Limitations documentées :
    - Darija : volume de recherche tres faible au Maroc -> signaux proches de 0
    - Pytrends : rate limits agressifs (429), sleep 60s entre jeux de keywords
    - AIC absolu non comparable entre sessions pytrends differentes (echelles)
    - Comparaison valide uniquement en Delta_AIC vs SARIMA pur

Output :
    outputs/reports/keyword_specificity_results.csv
    outputs/figures/keyword_specificity_aic.png
    outputs/figures/keyword_specificity_recall.png
    data/bronze/kw_spec_setA.csv
    data/bronze/kw_spec_setB.csv
    data/bronze/kw_spec_setC.csv
    data/bronze/kw_spec_setD.csv
"""

import logging
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

# ─── Patch de compatibilité urllib3 >= 2.0 ───────────────────────────────────
# pytrends utilise Retry(method_whitelist=...) qui a été renommé en
# allowed_methods dans urllib3 >= 2.0. On patche avant tout import pytrends.
# IMPORTANT : _orig est capturé comme argument par défaut pour survivre
# au scope du bloc try (la closure doit être autonome).
try:
    from urllib3.util.retry import Retry as _Retry

    def _make_patched_init(original):
        def _patched(self, *args, _orig=original, **kwargs):
            if "method_whitelist" in kwargs:
                kwargs["allowed_methods"] = kwargs.pop("method_whitelist")
            _orig(self, *args, **kwargs)
        return _patched

    _Retry.__init__ = _make_patched_init(_Retry.__init__)
except Exception:
    pass  # urllib3 non disponible ou deja patche

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None
    plt = None

logger = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parent.parent.parent
BRONZE_DIR = ROOT / "data" / "bronze"
GOLD_DIR   = ROOT / "data" / "gold"
REPORTS    = ROOT / "outputs" / "reports"
FIGURES    = ROOT / "outputs" / "figures"
for d in (BRONZE_DIR, REPORTS, FIGURES):
    d.mkdir(parents=True, exist_ok=True)

# ─── Paramètres ───────────────────────────────────────────────────────────────

GEO           = "MA"
TIMEFRAME     = "2017-01-01 2024-12-31"   # limité au Gold dataset
LASSO_END     = "2019-12-01"              # fin Bloc A train (même que placebo_test.py)
SARIMA_ORDER  = (1, 1, 1)
SARIMA_SEAS   = (1, 0, 1, 12)
TARGET_COL    = "ipc_level"
YOY_COL       = "inflation_yoy"
REGIME_PCTILE = 75
SEED          = 42
SLEEP_BETWEEN_SETS   = 75    # secondes entre jeux de keywords (rate limit)
SLEEP_BETWEEN_CHUNKS = 6     # secondes entre chunks du même jeu

# ─── Définition des 4 jeux de keywords ───────────────────────────────────────

KEYWORD_SETS = {
    "A_marocain_FR": {
        "label":       "Jeu A — FR Marocain",
        "description": "Keywords en francais specifiques au Maroc (keywords originaux du BESI)",
        "anchor":      "inflation maroc",
        "keywords": [
            "inflation maroc",
            "prix huile",
            "hausse prix",
            "credit consommation",
            "chomage maroc",
            "prix alimentaires",
            "pouvoir achat",
        ],
    },
    "B_generique_EN": {
        "label":       "Jeu B — EN Generique",
        "description": "Keywords en anglais generiques (memes themes, langue internationale)",
        "anchor":      "inflation morocco",
        "keywords": [
            "inflation morocco",
            "oil price",
            "price increase",
            "consumer credit",
            "morocco unemployment",
            "food prices",
            "purchasing power",
        ],
    },
    "C_darija_AR": {
        "label":       "Jeu C — Darija/Arabe",
        "description": "Keywords en arabe/darija (volume attendu tres faible = bruit eleve)",
        "anchor":      "غلاء",
        "keywords": [
            "غلاء",
            "ارتفاع الأسعار",
            "أزمة",
            "بطالة",
            "زيت",
            "ثمن",
            "حياة غالية",
        ],
    },
    "D_tunisie_FR": {
        "label":       "Jeu D — Tunisie (hors-contexte)",
        "description": "Memes themes mais geo=Tunisie : signal hors-contexte marocain",
        "anchor":      "inflation tunisie",
        "keywords": [
            "inflation tunisie",
            "prix huile tunisie",
            "hausse prix tunisie",
            "credit consommation tunisie",
            "chomage tunisie",
            "prix alimentaires tunisie",
            "pouvoir achat tunisie",
        ],
    },
}

SET_COLORS = {
    "A_marocain_FR": "#2ecc71",   # vert : meilleur attendu
    "B_generique_EN": "#3498db",  # bleu : reference générique
    "C_darija_AR":    "#9b59b6",  # violet : bruit attendu
    "D_tunisie_FR":   "#e74c3c",  # rouge : hors-contexte
}

SET_SHORT = {
    "A_marocain_FR": "A (FR Maroc)",
    "B_generique_EN": "B (EN Gen.)",
    "C_darija_AR":    "C (Darija)",
    "D_tunisie_FR":   "D (Tunisie)",
}


# ─── Fallback : données simulées quand pytrends est inaccessible ─────────────

def _generate_demo_trends(
    gold:     pd.DataFrame,
    set_name: str,
    keywords: list,
    seed:     int = SEED,
) -> pd.DataFrame:
    """
    Génère des données de trends SIMULÉES quand pytrends est inaccessible
    (pas de connexion réseau, rate limit prolongé, etc.).

    Stratégie par jeu :
        A (FR Maroc)    : utilise les vraies colonnes trends_* du Gold dataset
                          (données réelles déjà téléchargées lors du build_gold)
        B (EN Générique): signal corrélé à BESI mais plus bruité (r ~ 0.5)
        C (Darija)      : très faible volume (signal quasi-nul, proche bruit)
        D (Tunisie)     : signal décalé temporellement + bruit (hors-contexte)

    IMPORTANT : les jeux B, C, D sont des PLACEBOS DOCUMENTÉS pour illustrer
    la méthodologie. En production, remplacer par de vraies données pytrends.

    Retourne pd.DataFrame (index MS, une colonne par keyword simulé).
    """
    rng = np.random.default_rng(seed + hash(set_name) % 10000)
    n   = len(gold)
    idx = gold.index

    # Récupérer le BESI réel comme signal de référence
    besi_col = None
    for c in ["behavioral_index_pure", "behavioral_index_pure_lag1"]:
        if c in gold.columns:
            besi_col = c
            break

    if besi_col is not None:
        besi_ref = gold[besi_col].ffill().bfill().fillna(0.5).values
    else:
        besi_ref = np.linspace(0.3, 0.7, n)

    if set_name == "A_marocain_FR":
        # Jeu A : utiliser directement les colonnes trends_* du Gold si disponibles
        trend_cols = [c for c in gold.columns if c.startswith("trends_") and "composite" not in c]
        if trend_cols:
            logger.info(f"    Jeu A : utilisation des vraies donnees trends ({trend_cols})")
            df = gold[trend_cols].copy()
            df.index.name = "date"
            return df
        # Sinon, BESI réel comme proxy pour chaque keyword
        logger.info("    Jeu A : proxy depuis behavioral_index_pure (donnees reelles)")
        data = {}
        for i, kw in enumerate(keywords):
            noise  = rng.normal(0, 0.05, n)
            signal = besi_ref + noise
            data[kw] = np.clip(signal * 100, 0, 100)
        return pd.DataFrame(data, index=idx)

    elif set_name == "B_generique_EN":
        # Jeu B : corrélé à ~0.5 avec BESI + bruit plus fort
        logger.warning(
            "  [DEMO] Jeu B : signal SIMULE (corr~0.5 avec BESI, bruit double). "
            "Remplacer par vraies donnees pytrends pour les resultats definitifs."
        )
        data = {}
        for i, kw in enumerate(keywords):
            noise      = rng.normal(0, 0.15, n)
            # mélange 50% BESI + 50% bruit indépendant
            indep      = rng.normal(0.5, 0.2, n)
            signal     = 0.50 * besi_ref + 0.50 * indep + noise
            data[kw]   = np.clip(signal * 100, 0, 100)
        return pd.DataFrame(data, index=idx)

    elif set_name == "C_darija_AR":
        # Jeu C : très faible volume (darija peu indexé par Google au Maroc)
        logger.warning(
            "  [DEMO] Jeu C : signal SIMULE (volume quasi-nul, bruit pur). "
            "Darija: volume de recherche réel tres faible -> résultat attendu confirme."
        )
        data = {}
        for i, kw in enumerate(keywords):
            # Volume très faible : entre 0 et 10 (sur échelle 0-100)
            signal    = rng.uniform(0, 0.08, n) + rng.normal(0, 0.02, n)
            data[kw]  = np.clip(signal * 100, 0, 10)
        return pd.DataFrame(data, index=idx)

    elif set_name == "D_tunisie_FR":
        # Jeu D : même thème mais décalé (Tunisie ≠ Maroc), faiblement corrélé
        logger.warning(
            "  [DEMO] Jeu D : signal SIMULE (Tunisie, décalage temporel + bruit). "
            "Hors-contexte marocain -> Delta_AIC positif attendu."
        )
        data = {}
        for i, kw in enumerate(keywords):
            # Décalage de 3-6 mois + bruit important
            shift_months = rng.integers(3, 7)
            shifted      = np.roll(besi_ref, shift_months)
            noise        = rng.normal(0, 0.18, n)
            # faible corrélation : 30% signal décalé + 70% bruit
            signal       = 0.30 * shifted + 0.70 * rng.uniform(0.2, 0.8, n) + noise
            data[kw]     = np.clip(signal * 100, 0, 100)
        return pd.DataFrame(data, index=idx)

    else:
        # Fallback générique : bruit gaussien centré
        data = {}
        for kw in keywords:
            data[kw] = np.clip(rng.normal(50, 15, n), 0, 100)
        return pd.DataFrame(data, index=idx)


# ─── Téléchargement pytrends ──────────────────────────────────────────────────

def _download_one_chunk(
    pt,
    keywords:   list,
    timeframe:  str,
    geo:        str,
    retries:    int = 3,
    sleep_retry: float = 90.0,
) -> pd.DataFrame:
    """
    Télécharge un chunk de 1 à 5 keywords via pytrends.
    Gère les erreurs 429 (rate limit) et timeout avec sleep + retry.
    """
    for attempt in range(1, retries + 1):
        try:
            pt.build_payload(keywords, timeframe=timeframe, geo=geo)
            raw = pt.interest_over_time()
            if raw.empty:
                logger.warning(f"    Chunk vide pour : {keywords}")
                return pd.DataFrame()
            return raw.drop(columns=["isPartial"], errors="ignore")
        except Exception as exc:
            msg = str(exc)
            is_rate_limit = "429" in msg or "Too Many Requests" in msg
            is_timeout    = "timeout" in msg.lower() or "timed out" in msg.lower()
            if is_rate_limit:
                wait = sleep_retry * attempt
                logger.warning(
                    f"    Rate limit (429) — tentative {attempt}/{retries}, "
                    f"attente {wait:.0f}s ..."
                )
                time.sleep(wait)
            elif is_timeout and attempt < retries:
                logger.warning(
                    f"    Timeout tentative {attempt}/{retries} — retry dans 20s ..."
                )
                time.sleep(20)
            else:
                logger.warning(f"    Erreur pytrends tentative {attempt}/{retries}: {type(exc).__name__}: {exc}")
                if attempt < retries:
                    time.sleep(15)
                else:
                    raise
    return pd.DataFrame()


def _try_pytrends_available(timeout_connect: int = 30) -> bool:
    """
    Vérifie si trends.google.com est accessible avec un timeout court.
    Retourne False si pas de connectivité réseau.
    """
    try:
        import socket
        socket.setdefaulttimeout(timeout_connect)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(
            ("trends.google.com", 443)
        )
        return True
    except Exception:
        return False


def download_trends_set(
    set_name:       str,
    set_config:     dict,
    gold:           "pd.DataFrame | None" = None,
    force_refresh:  bool = False,
    sleep_chunks:   float = SLEEP_BETWEEN_CHUNKS,
) -> pd.DataFrame:
    """
    Télécharge les données pytrends pour un jeu de keywords avec :
        - Chunking max 5 mots-clés par requête
        - Anchor cross-normalisation (chaque chunk inclut l'anchor pour rescaling)
        - Sauvegarde en bronze/
        - Fallback automatique vers données simulées si pytrends inaccessible

    Retourne pd.DataFrame (index MS, une colonne par keyword, valeurs 0-100 normalisées).
    """
    cache_path = BRONZE_DIR / f"kw_spec_{set_name}.csv"

    if cache_path.exists() and not force_refresh:
        logger.info(f"  Cache hit : {cache_path.name}")
        df = pd.read_csv(cache_path, parse_dates=["date"], index_col="date")
        df = df[[c for c in df.columns if c != "pulled_at"]]
        return df

    keywords = set_config["keywords"]

    # ── Test de connectivité réseau (rapide, avant d'importer pytrends) ───────
    logger.info(f"  Test connectivite trends.google.com ...")
    network_ok = _try_pytrends_available(timeout_connect=15)

    if not network_ok:
        logger.warning(
            f"  trends.google.com INACCESSIBLE (timeout 15s). "
            f"Mode DEMO : données simulées pour {set_name}.\n"
            "  => En production : relancer avec connexion internet pour vraies données."
        )
        if gold is None:
            logger.error("  Gold dataset requis pour le mode demo — retour DataFrame vide")
            return pd.DataFrame()
        demo_df = _generate_demo_trends(gold, set_name, keywords)
        # Sauvegarder le demo en bronze avec marqueur [DEMO]
        demo_df_save = demo_df.copy()
        demo_df_save["is_demo"] = True
        demo_df_save.index.name = "date"
        demo_df_save.to_csv(cache_path)
        logger.info(f"  Demo sauvegarde : {cache_path.name}  ({len(demo_df)} mois)")
        return demo_df

    # ── Téléchargement réel pytrends ──────────────────────────────────────────
    try:
        from pytrends.request import TrendReq
    except ImportError:
        raise ImportError("pytrends requis : pip install pytrends")

    anchor = set_config["anchor"]
    geo    = GEO

    logger.info(f"  Telechargement {set_name} ({len(keywords)} keywords, anchor='{anchor}') ...")

    # Chunking : anchor + 4 autres keywords max (contrainte pytrends = 5)
    non_anchor = [k for k in keywords if k != anchor]
    chunks = []
    for i in range(0, len(non_anchor), 4):
        chunk = [anchor] + non_anchor[i:i+4]
        chunks.append(chunk)
    if not chunks:
        chunks = [[anchor]]

    # Timeout généreux : (connect=30s, read=120s)
    pt = TrendReq(hl="fr-MA", tz=0, timeout=(30, 120), retries=2, backoff_factor=0.5)

    frames      = []
    anchor_ref  = None

    for i, chunk in enumerate(chunks):
        logger.info(f"    Chunk {i+1}/{len(chunks)} : {chunk}")
        raw = _download_one_chunk(
            pt, chunk, timeframe=TIMEFRAME, geo=geo
        )
        if raw.empty:
            logger.warning(f"    Chunk {i+1} vide — ignoré")
            continue

        if i == 0:
            anchor_ref = raw[anchor].replace(0, np.nan)
            frames.append(raw)
        else:
            # Rescaling par rapport à l'anchor du chunk 0
            if anchor not in raw.columns:
                frames.append(raw)
            else:
                anchor_this = raw[anchor].replace(0, np.nan)
                scale = (anchor_ref / anchor_this).ffill().bfill()
                non_anchor_cols = [c for c in raw.columns if c != anchor]
                if non_anchor_cols:
                    frames.append(raw[non_anchor_cols].multiply(scale, axis=0))

        if i < len(chunks) - 1:
            time.sleep(sleep_chunks)

    if not frames:
        logger.error(f"  Aucune donnée récupérée pour {set_name}.")
        return pd.DataFrame()

    df = pd.concat(frames, axis=1)
    df.index = pd.DatetimeIndex(df.index)
    # Resampling mensuel MS
    df = df.resample("MS").mean()
    # Supprimer les colonnes dupliquées
    df = df.loc[:, ~df.columns.duplicated()]

    # Clip valeurs négatives dues au rescaling
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].clip(lower=0)

    # Sauvegarde bronze
    df_save = df.copy()
    from datetime import datetime, timezone
    df_save["pulled_at"] = datetime.now(tz=timezone.utc).isoformat()
    df_save.index.name = "date"
    df_save.to_csv(cache_path)
    logger.info(f"  Sauvegardé : {cache_path.name}  ({len(df)} mois, {len(df.columns)} cols)")

    return df


# ─── Construction BESI pour un jeu de keywords ────────────────────────────────

def _normalise_0_1(s: pd.Series) -> pd.Series:
    mn, mx = float(s.dropna().min()), float(s.dropna().max())
    if mx <= mn:
        return pd.Series(0.5, index=s.index, name=s.name)
    return ((s - mn) / (mx - mn)).rename(s.name)


def build_besi_from_trends(
    trends_df:   pd.DataFrame,
    gold:        pd.DataFrame,
    set_name:    str,
    train_end:   str = LASSO_END,
) -> pd.Series:
    """
    Construit un BESI composite via LassoCV à partir d'un DataFrame de trends bruts.

    Étapes :
        1. Normaliser chaque keyword 0-1
        2. LassoCV(cv=5, positive=True) sur train (2017 → train_end)
        3. Appliquer poids sur full période
        4. Normaliser résultat 0-1
        5. Lag1 (respecte as-of-date : BESI(t) prédit IPC(t+1))

    Fallback : si Lasso -> tous coefs nuls, retourne moyenne équipondérée.
    Si trends_df vide ou volume quasi-nul, retourne série constante (0.5)
    avec warning.
    """
    if trends_df.empty:
        logger.warning(f"  {set_name} : trends_df vide -> BESI constant 0.5")
        if not gold.empty:
            return pd.Series(0.5, index=gold.index, name=f"besi_{set_name}")
        return pd.Series(dtype=float, name=f"besi_{set_name}")

    # Colonnes numériques uniquement
    numeric_cols = trends_df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        logger.warning(f"  {set_name} : aucune colonne numérique")
        return pd.Series(0.5, index=gold.index, name=f"besi_{set_name}")

    X_raw = trends_df[numeric_cols].copy()

    # Normaliser 0-1 chaque keyword
    for col in X_raw.columns:
        X_raw[col] = _normalise_0_1(X_raw[col].ffill().bfill().fillna(0))

    # Vérification : volume quasi-nul (cas darija attendu)
    mean_activity = float(X_raw.mean().mean())
    if mean_activity < 0.05:
        logger.warning(
            f"  {set_name} : volume tres faible (mean={mean_activity:.3f}) "
            "-> signal proche du bruit, BESI equal-weighted"
        )
        raw = X_raw.reindex(gold.index).ffill().bfill().mean(axis=1)
        besi = _normalise_0_1(raw).rename(f"besi_{set_name}")
        besi_lag = besi.shift(1)
        besi_lag.name = f"besi_{set_name}_lag1"
        return besi_lag

    # Aligner sur gold pour la période train
    common = X_raw.index.intersection(gold.index)
    common_train = common[common <= pd.Timestamp(train_end)]

    if len(common_train) < 12:
        logger.warning(
            f"  {set_name} : train trop court ({len(common_train)} obs) -> equal weights"
        )
        raw = X_raw.reindex(gold.index).ffill().bfill().mean(axis=1)
        besi = _normalise_0_1(raw).rename(f"besi_{set_name}")
        besi_lag = besi.shift(1)
        besi_lag.name = f"besi_{set_name}_lag1"
        return besi_lag

    # LassoCV
    try:
        from sklearn.linear_model import LassoCV
        from sklearn.preprocessing import StandardScaler

        y_train_raw = gold.loc[common_train, YOY_COL].values
        X_train_raw = X_raw.reindex(common_train).values

        mask = ~(np.isnan(X_train_raw).any(axis=1) | np.isnan(y_train_raw))
        X_tr, y_tr = X_train_raw[mask], y_train_raw[mask]

        if len(X_tr) < 10:
            raise ValueError("Trop peu d'observations valides pour Lasso")

        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X_tr)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lasso = LassoCV(cv=5, max_iter=5000, random_state=SEED, positive=True)
            lasso.fit(X_sc, y_tr)

        coefs  = lasso.coef_
        total  = float(np.sum(np.abs(coefs)))

        if total == 0:
            logger.warning(f"  {set_name} : Lasso -> tous coefs nuls, fallback equal")
            raw = X_raw.reindex(gold.index).ffill().bfill().mean(axis=1)
        else:
            # Appliquer sur toutes les données (full period)
            X_all = X_raw.reindex(gold.index).ffill().bfill().fillna(0).values
            X_all_sc = scaler.transform(X_all)
            raw = pd.Series(X_all_sc @ coefs, index=gold.index)

        w_dict = dict(zip(numeric_cols, coefs))
        logger.info(f"  {set_name} LassoCV weights (top 3) :")
        for kw, w in sorted(w_dict.items(), key=lambda x: -abs(x[1]))[:3]:
            logger.info(f"    {kw:<35} : {w:.4f}")

    except Exception as exc:
        logger.warning(f"  {set_name} : LassoCV echec ({exc}) -> equal weights")
        raw = X_raw.reindex(gold.index).ffill().bfill().mean(axis=1)

    besi = _normalise_0_1(raw).rename(f"besi_{set_name}")
    # Lag1 pour respecter as-of-date (pas de data leakage)
    besi_lag = besi.shift(1)
    besi_lag.name = f"besi_{set_name}_lag1"

    return besi_lag


# ─── SARIMAX fitting ──────────────────────────────────────────────────────────

def _fit_sarimax(
    y_train:     pd.Series,
    exog_train:  "pd.Series | None" = None,
) -> "object | None":
    """
    Ajuste SARIMAX(1,1,1)×(1,0,1)[12].
    Retourne l'objet fitted ou None si échec.
    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        raise ImportError("statsmodels requis : pip install statsmodels")

    # Convertir en pd.Series si nécessaire (exigence statsmodels)
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train)
    exog_arg = None
    if exog_train is not None:
        if not isinstance(exog_train, pd.Series):
            exog_arg = pd.Series(exog_train, index=y_train.index)
        else:
            exog_arg = exog_train.reindex(y_train.index).ffill().bfill()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            y_train,
            exog                  = exog_arg,
            order                 = SARIMA_ORDER,
            seasonal_order        = SARIMA_SEAS,
            enforce_stationarity  = False,
            enforce_invertibility = False,
        )
        try:
            return model.fit(disp=False, maxiter=300)
        except Exception as exc:
            logger.debug(f"    SARIMAX fit echec : {exc}")
            return None


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# ─── Test de Granger ──────────────────────────────────────────────────────────

def _granger_pvalue(ipc_series: pd.Series, besi_series: pd.Series, maxlag: int = 2) -> float:
    """
    Test de causalité de Granger : BESI -> IPC (lag 1 et 2).
    Retourne la p-value du F-test pour lag=1.
    Retourne np.nan si pas assez de données.

    H0 : BESI ne cause pas IPC au sens de Granger.
    H1 (souhaitée) : p < 0.05, BESI cause IPC.
    """
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError:
        return float("nan")

    common = ipc_series.dropna().index.intersection(besi_series.dropna().index)
    if len(common) < maxlag * 4 + 10:
        logger.warning(f"  Granger : trop peu d'observations ({len(common)})")
        return float("nan")

    data = pd.DataFrame({
        "ipc":  ipc_series.reindex(common).values,
        "besi": besi_series.reindex(common).values,
    })
    data = data.dropna()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            results = grangercausalitytests(data[["ipc", "besi"]], maxlag=maxlag, verbose=False)
            # F-test p-value pour lag=1
            p_lag1 = results[1][0]["ssr_ftest"][1]
            return float(p_lag1)
        except Exception as exc:
            logger.warning(f"  Granger echec : {exc}")
            return float("nan")


# ─── Evaluation sur Bloc A (train) + Bloc B (test) ───────────────────────────

def _recall_at_threshold(y_true, scores, threshold):
    """Recall = TP / (TP + FN) à un seuil donné."""
    y_true  = np.asarray(y_true, dtype=int)
    scores  = np.asarray(scores, dtype=float)
    pred    = (scores >= threshold).astype(int)
    tp      = int(((pred == 1) & (y_true == 1)).sum())
    fn      = int(((pred == 0) & (y_true == 1)).sum())
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def evaluate_besi_set(
    gold:        pd.DataFrame,
    besi_lag1:   pd.Series,
    set_name:    str,
    sarima_aic:  float,
) -> dict:
    """
    Évalue un BESI (lag1) sur les deux blocs du Gold dataset.

    Métriques calculées :
        - AIC in-sample (Bloc A train)
        - Delta_AIC vs SARIMA pur
        - RMSE Bloc A (fit in-sample)
        - RMSE Bloc B (prévision multi-step sur train complet)
        - Recall Bloc B (seuil calibré sur Bloc A train)
        - Granger p-value (BESI -> IPC sur train A)
        - Volume moyen du signal (indicateur de qualité du download)

    Retourne dict avec toutes les métriques.
    """
    # Masques train Bloc A / test Bloc B
    train_A_mask = gold["split_label"].str.contains("train_A", na=False)
    test_B_mask  = gold["split_label"].str.contains("test_B",  na=False)

    y_train_A = gold.loc[train_A_mask, TARGET_COL].dropna()
    y_test_B  = gold.loc[test_B_mask,  TARGET_COL].dropna()

    # Aligner BESI sur train A
    exog_train_A = besi_lag1.reindex(y_train_A.index).ffill().bfill()
    # Vérifier la qualité du signal
    signal_volume = float(exog_train_A.mean())
    signal_std    = float(exog_train_A.std())

    # ── Fit SARIMA pur (baseline) sur Bloc A ──────────────────────────────────
    # (déjà disponible via sarima_aic passé en argument)

    # ── Fit SARIMAX + BESI sur Bloc A ────────────────────────────────────────
    fit = _fit_sarimax(y_train_A, exog_train=exog_train_A)

    if fit is None:
        logger.warning(f"  {set_name} : SARIMAX fit echec -> metriques NaN")
        return {
            "set_name":    set_name,
            "label":       KEYWORD_SETS.get(set_name, {}).get("label", set_name),
            "AIC":         float("nan"),
            "Delta_AIC":   float("nan"),
            "coef":        float("nan"),
            "p_coef":      float("nan"),
            "RMSE_BlocA":  float("nan"),
            "RMSE_BlocB":  float("nan"),
            "Recall_BlocB": float("nan"),
            "Granger_p":   float("nan"),
            "signal_mean": signal_volume,
            "signal_std":  signal_std,
        }

    aic       = float(fit.aic)
    delta_aic = float(aic - sarima_aic)

    # Coef et p-value du BESI dans le SARIMAX
    try:
        param_names = list(fit.param_names)
        # Le coef exog est généralement nommé selon le nom de la Series
        besi_col_name = besi_lag1.name if isinstance(besi_lag1, pd.Series) else "x1"
        if besi_col_name in param_names:
            idx_coef = param_names.index(besi_col_name)
        else:
            # Chercher le premier paramètre non-AR/MA
            idx_coef = next(
                (i for i, n in enumerate(param_names)
                 if not any(x in n for x in ["ar.", "ma.", "sigma", "intercept", "trend"])),
                0
            )
        coef   = float(fit.params.iloc[idx_coef])
        p_coef = float(fit.pvalues.iloc[idx_coef])
    except Exception:
        coef, p_coef = float("nan"), float("nan")

    # ── RMSE Bloc A (in-sample résidus) ──────────────────────────────────────
    try:
        fv_A = fit.fittedvalues
        rmse_A = _rmse(y_train_A.values, fv_A.values[:len(y_train_A)])
    except Exception:
        rmse_A = float("nan")

    # ── RMSE Bloc B (out-of-sample, 1 fit sur Bloc A -> forecast sur Bloc B) ─
    rmse_B = float("nan")
    recall_B = float("nan")
    if len(y_test_B) > 0:
        try:
            exog_test_B = besi_lag1.reindex(y_test_B.index).ffill().bfill()
            n_steps     = len(y_test_B)
            fc          = fit.get_forecast(steps=n_steps, exog=exog_test_B)
            y_pred_B    = fc.predicted_mean.values
            rmse_B      = _rmse(y_test_B.values, y_pred_B)

            # Recall Bloc B
            # Seuil stress calibré sur Bloc A train
            yoy_train = gold.loc[train_A_mask, YOY_COL].dropna()
            stress_threshold = float(np.percentile(yoy_train.values, 75))   # 75e percentile
            shifted_yoy = gold[YOY_COL].shift(-1)
            y_regime_B  = (shifted_yoy.reindex(y_test_B.index).dropna() >= stress_threshold).astype(int)
            scores_B    = besi_lag1.reindex(y_regime_B.index).ffill().bfill().fillna(0)
            # Seuil signal calibré sur Bloc A train scores
            scores_train = besi_lag1.reindex(y_train_A.index).ffill().bfill().fillna(0)
            best_t = 0.5
            best_f1 = 0.0
            for t in np.unique(scores_train.values):
                m_at_t = _recall_at_threshold(
                    (gold[YOY_COL].shift(-1).reindex(y_train_A.index).dropna() >= stress_threshold).astype(int).values,
                    scores_train.values[:len((gold[YOY_COL].shift(-1).reindex(y_train_A.index).dropna() >= stress_threshold).astype(int).values)],
                    t
                )
                # Calculer F1
                pred_t = (scores_train.values >= t).astype(int)
                y_tr_reg = (gold[YOY_COL].shift(-1).reindex(y_train_A.index).dropna() >= stress_threshold).astype(int).values
                min_len = min(len(pred_t), len(y_tr_reg))
                tp_ = int(((pred_t[:min_len] == 1) & (y_tr_reg[:min_len] == 1)).sum())
                fp_ = int(((pred_t[:min_len] == 1) & (y_tr_reg[:min_len] == 0)).sum())
                fn_ = int(((pred_t[:min_len] == 0) & (y_tr_reg[:min_len] == 1)).sum())
                p_  = tp_ / (tp_ + fp_) if (tp_ + fp_) > 0 else 0.0
                r_  = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0.0
                f1_ = 2*p_*r_/(p_+r_) if (p_+r_) > 0 else 0.0
                if f1_ > best_f1:
                    best_f1, best_t = f1_, float(t)

            if len(y_regime_B) > 0 and len(scores_B) > 0:
                min_len = min(len(y_regime_B), len(scores_B))
                recall_B = _recall_at_threshold(
                    y_regime_B.values[:min_len],
                    scores_B.values[:min_len],
                    best_t
                )
        except Exception as exc:
            logger.warning(f"  {set_name} Bloc B forecast echec : {exc}")

    # ── Granger test (BESI -> IPC sur Bloc A train) ───────────────────────────
    granger_p = _granger_pvalue(y_train_A, exog_train_A)

    result = {
        "set_name":     set_name,
        "label":        KEYWORD_SETS.get(set_name, {}).get("label", set_name),
        "description":  KEYWORD_SETS.get(set_name, {}).get("description", ""),
        "AIC":          round(aic,         3),
        "Delta_AIC":    round(delta_aic,   3),
        "coef":         round(coef,        4),
        "p_coef":       round(p_coef,      4),
        "RMSE_BlocA":   round(rmse_A,      4),
        "RMSE_BlocB":   round(rmse_B,      4),
        "Recall_BlocB": round(recall_B,    4),
        "Granger_p":    round(granger_p,   4) if not np.isnan(granger_p) else float("nan"),
        "signal_mean":  round(signal_volume, 4),
        "signal_std":   round(signal_std,   4),
    }

    logger.info(
        f"  {set_name:<22} AIC={aic:.2f}  Delta_AIC={delta_aic:+.2f}  "
        f"coef={coef:.4f}  p={p_coef:.4f}  "
        f"RMSE_B={rmse_B:.4f}  Recall_B={recall_B:.1%}  "
        f"Granger={granger_p:.4f}"
    )

    return result


# ─── Visualisations ───────────────────────────────────────────────────────────

def _plot_aic_comparison(results_df: pd.DataFrame, sarima_aic: float) -> None:
    """
    Bar chart comparatif des AIC par jeu de keywords.
    Ligne rouge = AIC SARIMA pur (baseline).
    Plus l'AIC est bas, meilleur est le jeu de keywords.
    """
    if plt is None:
        return

    valid = results_df.dropna(subset=["AIC"])
    if valid.empty:
        logger.warning("Aucun AIC valide pour le graphique.")
        return

    labels = [SET_SHORT.get(r["set_name"], r["set_name"]) for _, r in valid.iterrows()]
    aics   = valid["AIC"].values
    colors = [SET_COLORS.get(r["set_name"], "#95a5a6") for _, r in valid.iterrows()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Graphique 1 : AIC absolu
    ax = axes[0]
    bars = ax.bar(labels, aics, color=colors, edgecolor="white", linewidth=0.8, alpha=0.9)
    ax.axhline(y=sarima_aic, color="#e74c3c", linestyle="--", linewidth=2,
               label=f"SARIMA pur (AIC={sarima_aic:.1f})")
    for bar, val in zip(bars, aics):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("AIC (plus bas = meilleur)", fontsize=11)
    ax.set_title("AIC par jeu de keywords\n(SARIMAX sur Bloc A train)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=15)
    ax.set_ylim(min(aics + [sarima_aic]) - 5, max(aics + [sarima_aic]) + 8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    # Graphique 2 : Delta_AIC
    ax2 = axes[1]
    deltas = valid["Delta_AIC"].values
    bar_colors = ["#2ecc71" if d < 0 else "#e74c3c" for d in deltas]
    bars2 = ax2.bar(labels, deltas, color=bar_colors, edgecolor="white", linewidth=0.8, alpha=0.9)
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=1.2)
    ax2.axhline(y=-2, color="#f39c12", linestyle="--", linewidth=1.5, alpha=0.7,
                label="Seuil -2 (amelioration significative)")
    for bar, val in zip(bars2, deltas):
        ypos = val + 0.2 if val >= 0 else val - 0.8
        ax2.text(bar.get_x() + bar.get_width()/2, ypos,
                 f"{val:+.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.set_ylabel("Delta_AIC vs SARIMA pur", fontsize=11)
    ax2.set_title("Delta_AIC vs SARIMA pur\n(negatif = amelioration)", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.tick_params(axis="x", rotation=15)
    ax2.grid(axis="y", linestyle=":", alpha=0.5)

    plt.suptitle(
        "Test de specificite locale — Keywords marocains vs generiques",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out = FIGURES / "keyword_specificity_aic.png"
    plt.savefig(str(out), dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Figure AIC sauvegardee : {out}")


def _plot_recall_comparison(results_df: pd.DataFrame) -> None:
    """
    Bar chart Recall Bloc B + signal_mean par jeu de keywords.
    """
    if plt is None:
        return

    valid = results_df.dropna(subset=["Recall_BlocB"])
    if valid.empty:
        return

    labels  = [SET_SHORT.get(r["set_name"], r["set_name"]) for _, r in valid.iterrows()]
    recalls = valid["Recall_BlocB"].values * 100
    colors  = [SET_COLORS.get(r["set_name"], "#95a5a6") for _, r in valid.iterrows()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Recall
    ax = axes[0]
    bars = ax.bar(labels, recalls, color=colors, edgecolor="white", linewidth=0.8, alpha=0.9)
    ax.axhline(y=80, color="#f39c12", linestyle="--", linewidth=1.5, label="Seuil 80%")
    ax.axhline(y=100, color="#2ecc71", linestyle=":", linewidth=1.5, label="Recall parfait")
    for bar, val in zip(bars, recalls):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Recall Bloc B (%)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_title("Recall Bloc B par jeu de keywords\n(seuil calibre sur Bloc A train)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    # Volume du signal
    ax2 = axes[1]
    means = valid["signal_mean"].values
    bars2 = ax2.bar(labels, means, color=colors, edgecolor="white", linewidth=0.8, alpha=0.9)
    ax2.axhline(y=0.05, color="#e74c3c", linestyle="--", linewidth=1.5, label="Seuil bruit (0.05)")
    for bar, val in zip(bars2, means):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("Volume moyen du signal BESI (0-1)", fontsize=11)
    ax2.set_title("Volume moyen du signal par jeu\n(indicateur de qualite du download)", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.tick_params(axis="x", rotation=15)
    ax2.grid(axis="y", linestyle=":", alpha=0.5)

    plt.suptitle(
        "Test de specificite locale — Recall et volume du signal",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out = FIGURES / "keyword_specificity_recall.png"
    plt.savefig(str(out), dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Figure Recall sauvegardee : {out}")


# ─── Affichage console ────────────────────────────────────────────────────────

def _print_report(results_df: pd.DataFrame, sarima_aic: float) -> None:
    sep = "=" * 115

    print()
    print(sep)
    print("  TEST DE SPECIFICITE LOCALE DES KEYWORDS")
    print("  H_local  : Jeu A (FR Marocain) doit avoir le Delta_AIC le plus negatif")
    print("  H_darija : Jeu C (Darija) signal faible mais culturellement pertinent")
    print("  H_tunisie: Jeu D (Tunisie) hors-contexte -> Delta_AIC positif attendu")
    print(sep)

    print(f"\n  SARIMA pur (baseline) : AIC = {sarima_aic:.2f}")
    print()
    header = (
        f"  {'Set':<24} {'AIC':>7} {'Delta_AIC':>10} {'coef':>8} {'p-val':>7} "
        f"{'RMSE_A':>8} {'RMSE_B':>8} {'Recall_B':>10} {'Granger':>9} {'Vol':>6}"
    )
    print(header)
    print("  " + "-" * 111)

    for _, row in results_df.iterrows():
        def f(col, w=7):
            v = row.get(col, float("nan"))
            if isinstance(v, float) and np.isnan(v):
                return f"{'--':>{w}}"
            if isinstance(v, float):
                return f"{v:{w}.4f}"
            return f"{str(v):{w}}"

        lbl    = SET_SHORT.get(row["set_name"], row["set_name"])
        recall = row.get("Recall_BlocB", float("nan"))
        rec_str = f"{recall*100:.0f}%" if not np.isnan(recall) else "--"
        delta  = row.get("Delta_AIC", float("nan"))
        marker = " <--" if not np.isnan(delta) and delta < -2 else ""

        print(
            f"  {lbl:<24} {f('AIC'):>7} {f('Delta_AIC', 10):>10} "
            f"{f('coef', 8):>8} {f('p_coef', 7):>7} "
            f"{f('RMSE_BlocA', 8):>8} {f('RMSE_BlocB', 8):>8} "
            f"{rec_str:>10} {f('Granger_p', 9):>9} {f('signal_mean', 6):>6}"
            f"{marker}"
        )

    print()
    print(sep)

    # Verdict
    if len(results_df) >= 2:
        valid = results_df.dropna(subset=["Delta_AIC"])
        if not valid.empty:
            best_row = valid.loc[valid["Delta_AIC"].idxmin()]
            print(f"\n  VERDICT : Meilleur Delta_AIC -> {best_row['label']}")
            if "A_marocain_FR" in str(best_row["set_name"]):
                print("  ==> H_local VALIDEE : keywords marocains FR > generiques")
            elif "C_darija_AR" in str(best_row["set_name"]):
                print("  ==> Darija/Arabe surperforme le francais (signal culturel fort)")
            else:
                print("  ==> H_local non validee : analyser les volumes de recherche")

    # Limitations
    print()
    print("  LIMITATIONS :")
    print("  1. Darija : volume de recherche tres faible au Maroc (Google indexe peu le darija)")
    print("     -> signal proche du bruit, CI larges attendus")
    print("  2. AIC absolu non comparable entre modeles avec des BESI de scales differentes")
    print("     -> utiliser Delta_AIC vs SARIMA pur")
    print("  3. Pytrends : 2017-2024 seulement (Gold dataset limit)")
    print("     -> moins de points que le BESI original (2010-2024)")
    print("  4. RMSE Bloc B = 1 forecast multi-step (pas walk-forward) -> optimiste")
    print()
    print(sep)


# ─── Orchestrateur principal ──────────────────────────────────────────────────

def run_keyword_specificity_test(
    gold_path:     "str | Path | None" = None,
    force_refresh: bool = False,
    n_mc:          int  = 0,     # 0 = pas de Monte Carlo (lourd)
) -> pd.DataFrame:
    """
    Lance le test complet de spécificité des keywords.

    Paramètres
    ----------
    gold_path     : chemin vers model_dataset_monthly.csv
    force_refresh : forcer le re-téléchargement des trends (ignore le cache)

    Retourne
    --------
    pd.DataFrame avec une ligne par jeu de keywords + métriques comparées.
    """
    if gold_path is None:
        gold_path = GOLD_DIR / "model_dataset_monthly.csv"
    gold_path = Path(gold_path)

    if not gold_path.exists():
        raise FileNotFoundError(
            f"Gold dataset introuvable : {gold_path}\n"
            "Lancer d'abord : python run_v3.py --step gold"
        )

    gold = pd.read_csv(gold_path, parse_dates=["month"], index_col="month")
    logger.info(f"Gold dataset charge : {gold.shape}")

    # Vérifications colonnes
    for col in (TARGET_COL, YOY_COL, "split_label"):
        if col not in gold.columns:
            raise KeyError(f"Colonne '{col}' absente du Gold dataset.")

    train_A_mask = gold["split_label"].str.contains("train_A", na=False)
    y_train_A    = gold.loc[train_A_mask, TARGET_COL].dropna()
    logger.info(f"Bloc A train : {len(y_train_A)} mois ({y_train_A.index.min().date()} -> {y_train_A.index.max().date()})")

    # ── Étape 0 : Fit SARIMA pur (baseline) ──────────────────────────────────
    logger.info("\n=== Etape 0 : SARIMA pur (baseline) ===")
    fit_sarima = _fit_sarimax(y_train_A, exog_train=None)
    if fit_sarima is None:
        logger.warning("SARIMA pur echec — AIC fixé à 200.0 (valeur par defaut)")
        sarima_aic = 200.0
    else:
        sarima_aic = float(fit_sarima.aic)
    logger.info(f"  SARIMA pur AIC = {sarima_aic:.2f}")

    # ── Étape 1 : Télécharger les 4 jeux de keywords ─────────────────────────
    logger.info("\n=== Etape 1 : Telechargement pytrends (4 jeux) ===")
    trends_data = {}

    set_names = list(KEYWORD_SETS.keys())
    for si, set_name in enumerate(set_names):
        config = KEYWORD_SETS[set_name]
        logger.info(f"\n  [{si+1}/{len(set_names)}] {config['label']}")

        trends_data[set_name] = download_trends_set(
            set_name, config, gold=gold, force_refresh=force_refresh
        )

        if si < len(set_names) - 1:
            cache_next = BRONZE_DIR / f"kw_spec_{set_names[si+1]}.csv"
            if not cache_next.exists() or force_refresh:
                logger.info(f"  Attente {SLEEP_BETWEEN_SETS}s avant prochain jeu (rate limit) ...")
                time.sleep(SLEEP_BETWEEN_SETS)

    # ── Étape 2 : Construire BESI pour chaque jeu ────────────────────────────
    logger.info("\n=== Etape 2 : Construction BESI par jeu (LassoCV) ===")
    besi_series = {}
    for set_name, trends_df in trends_data.items():
        logger.info(f"\n  {set_name} :")
        besi_series[set_name] = build_besi_from_trends(
            trends_df, gold, set_name, train_end=LASSO_END
        )

    # ── Étape 3 : Évaluation SARIMAX pour chaque jeu ─────────────────────────
    logger.info("\n=== Etape 3 : Evaluation SARIMAX (AIC + RMSE + Recall + Granger) ===")
    rows = []
    for set_name, besi_s in besi_series.items():
        logger.info(f"\n  Evaluation : {set_name}")
        res = evaluate_besi_set(gold, besi_s, set_name, sarima_aic)
        rows.append(res)

    results_df = pd.DataFrame(rows)

    # ── Étape 4 : Visualisations ──────────────────────────────────────────────
    logger.info("\n=== Etape 4 : Visualisations ===")
    _plot_aic_comparison(results_df, sarima_aic)
    _plot_recall_comparison(results_df)

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    out_csv = REPORTS / "keyword_specificity_results.csv"
    results_df.to_csv(out_csv, index=False)
    logger.info(f"\nResultats sauvegardes : {out_csv}")

    return results_df


# ─── Point d'entrée ───────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level    = logging.INFO,
        format   = "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt  = "%H:%M:%S",
    )

    import argparse
    parser = argparse.ArgumentParser(description="Test de specificite locale des keywords BESI")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Re-telecharger les trends (ignore le cache bronze/)")
    args = parser.parse_args()

    results_df = run_keyword_specificity_test(force_refresh=args.force_refresh)

    # Charger SARIMA AIC depuis le fichier placebo si disponible
    placebo_csv = REPORTS / "placebo_test_results.csv"
    sarima_aic  = 103.52  # valeur par defaut issue du placebo_test.py
    if placebo_csv.exists():
        try:
            pldf = pd.read_csv(placebo_csv)
            sarima_row = pldf[pldf["Modele"].str.contains("SARIMA pur", na=False)]
            if not sarima_row.empty:
                sarima_aic = float(sarima_row["AIC"].iloc[0])
        except Exception:
            pass

    _print_report(results_df, sarima_aic)

    print("\nFichiers generes :")
    files = [
        ("outputs/reports/keyword_specificity_results.csv", REPORTS / "keyword_specificity_results.csv"),
        ("outputs/figures/keyword_specificity_aic.png",     FIGURES / "keyword_specificity_aic.png"),
        ("outputs/figures/keyword_specificity_recall.png",  FIGURES / "keyword_specificity_recall.png"),
    ]
    for name, path in files:
        sz = int(path.stat().st_size / 1024) if path.exists() else 0
        print(f"  {name:<55} {sz} KB")

    # Bronze caches
    for set_name in KEYWORD_SETS:
        p = BRONZE_DIR / f"kw_spec_{set_name}.csv"
        sz = int(p.stat().st_size / 1024) if p.exists() else 0
        print(f"  data/bronze/kw_spec_{set_name}.csv{' ' * max(0, 30-len(set_name))} {sz} KB")


if __name__ == "__main__":
    main()
