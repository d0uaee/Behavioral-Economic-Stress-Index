"""
build_cpi_hcp_monthly.py
========================
Construit : data/bronze/cpi_hcp_monthly_raw.csv
Plage     : 2010-01-01 -> 2024-12-01, trie croissant, base 2017=100

Sources :
  PRIMAIRE   -> HCP officiel  (2017-01 -> 2024-12)  — cpi_hcp_monthly_raw.csv
  SECONDAIRE -> IMF IFS mensuel (2010-01 -> 2016-12) — API IMF
  Chainage   -> facteur calcule sur les 12 mois de 2017 (ancrage)

Colonnes :
  date (YYYY-MM-01) | ipc_level | source

Usage :
  pip install requests pandas
  python build_cpi_hcp_monthly.py

Adapte HCP_CSV et OUTPUT_FILE si necessaire.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests

# -- Chemins -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "bronze"
OUTPUT_FILE = OUTPUT_DIR / "cpi_hcp_monthly_raw.csv"

# Place cpi_hcp_monthly_raw.csv a cote de ce script
HCP_CSV = PROJECT_ROOT / "cpi_hcp_monthly_raw.csv"

# -- IMF IFS params -----------------------------------------------------------
IMF_COUNTRY = "MA"       # Maroc
IMF_INDICATOR = "PCPI_IX"  # CPI Index (not YoY%)
IMF_FREQ = "M"           # Monthly
CHAIN_YEAR = 2017


def fetch_imf_cpi() -> pd.DataFrame:
    url = (
        f"https://dataservices.imf.org/REST/SDMX_JSON.svc"
        f"/CompactData/IFS/{IMF_FREQ}.{IMF_COUNTRY}.{IMF_INDICATOR}"
    )
    print(f"[IMF] GET {url}")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"[ERREUR] API IMF : {exc}")
        sys.exit(1)

    try:
        obs_raw = response.json()["CompactData"]["DataSet"]["Series"]["Obs"]
    except (KeyError, TypeError) as exc:
        print(f"[ERREUR] Structure JSON inattendue : {exc}")
        print(response.text[:500])
        sys.exit(1)

    records: list[dict[str, object]] = []
    for obs in obs_raw:
        try:
            records.append(
                {
                    "date": pd.to_datetime(obs["@TIME_PERIOD"] + "-01"),
                    "ipc_imf_raw": float(obs["@OBS_VALUE"]),
                }
            )
        except Exception:
            continue

    if not records:
        print("[ERREUR] Aucune observation IMF parsee.")
        sys.exit(1)

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    print(f"[IMF] {len(df)} obs  {df.date.min().date()} -> {df.date.max().date()}")
    return df


def load_hcp(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        print(f"[ERREUR] Introuvable : {csv_path}")
        print("  -> Place cpi_hcp_monthly_raw.csv a cote du script.")
        sys.exit(1)

    df = pd.read_csv(csv_path, parse_dates=["date"])
    required_cols = {"date", "ipc_level"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[ERREUR] Colonnes manquantes dans HCP : {sorted(missing)}")
        sys.exit(1)

    df = df[(df["date"] >= "2017-01-01") & (df["date"] <= "2024-12-01")].copy()
    if df.empty:
        print("[ERREUR] Le CSV HCP ne contient aucune ligne entre 2017-01 et 2024-12.")
        sys.exit(1)

    df["source"] = "HCP"
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    print(f"[HCP] {len(df)} lignes  {df.date.min().date()} -> {df.date.max().date()}")
    return df


def chain_factor(df_imf: pd.DataFrame, df_hcp: pd.DataFrame) -> float:
    merged = (
        df_imf[df_imf.date.dt.year == CHAIN_YEAR][["date", "ipc_imf_raw"]]
        .merge(df_hcp[df_hcp.date.dt.year == CHAIN_YEAR][["date", "ipc_level"]], on="date")
    )
    if len(merged) == 0:
        print("[ERREUR] Aucun mois commun en 2017 — chainage impossible.")
        sys.exit(1)

    factor = merged["ipc_level"].mean() / merged["ipc_imf_raw"].mean()
    print(
        f"[CHAINAGE] {len(merged)}/12 mois 2017  "
        f"| moy_HCP={merged['ipc_level'].mean():.3f}  "
        f"| moy_IMF={merged['ipc_imf_raw'].mean():.3f}  "
        f"| facteur={factor:.6f}"
    )
    return factor


def build_imf_segment(df_imf: pd.DataFrame, factor: float) -> pd.DataFrame:
    df = df_imf[(df_imf.date >= "2010-01-01") & (df_imf.date <= "2016-12-01")].copy()
    df["ipc_level"] = (df["ipc_imf_raw"] * factor).round(1)
    df["source"] = "IMF_rebased"
    df = df[["date", "ipc_level", "source"]]
    print(f"[IMF rebased] {len(df)} lignes  {df.date.min().date()} -> {df.date.max().date()}")
    return df


def verify_junction(df_imf_seg: pd.DataFrame, df_hcp: pd.DataFrame) -> None:
    last = df_imf_seg.iloc[-1]
    first = df_hcp.iloc[0]
    gap = abs(first.ipc_level - last.ipc_level)
    pct = gap / last.ipc_level * 100
    flag = "OK" if pct <= 2 else "ATTENTION >2%"
    print(
        f"\n[JONCTION] IMF {last.date.date()}={last.ipc_level:.1f}  "
        f"HCP {first.date.date()}={first.ipc_level:.1f}  "
        f"ecart={pct:.2f}%  {flag}"
    )


def main() -> None:
    print("=" * 60)
    print("  BUILD CPI HCP MONTHLY 2010-2024")
    print("=" * 60)

    df_hcp = load_hcp(HCP_CSV)
    df_imf = fetch_imf_cpi()
    factor = chain_factor(df_imf, df_hcp)
    df_imf_seg = build_imf_segment(df_imf, factor)

    verify_junction(df_imf_seg, df_hcp)

    df_final = (
        pd.concat([df_imf_seg, df_hcp[["date", "ipc_level", "source"]]])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )

    expected = pd.date_range("2010-01-01", "2024-12-01", freq="MS")
    missing = expected.difference(df_final.date)
    status = "serie complete" if len(missing) == 0 else f"{len(missing)} trous"
    print(f"\n[QUALITE] {len(df_final)} lignes  {status}")
    if len(missing) > 0:
        print("Mois manquants :", [str(m.date()) for m in missing])
    print(df_final.groupby("source").size().rename("n").to_string())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUTPUT_FILE, index=False, date_format="%Y-%m-%d")
    print(f"\nOK  {OUTPUT_FILE}")
    print(pd.concat([df_final.head(3), df_final.tail(3)]).to_string(index=False))


if __name__ == "__main__":
    main()
