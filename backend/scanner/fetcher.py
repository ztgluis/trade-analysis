"""
Finviz data fetcher with 15-minute pickle caching.

Uses the Custom screener view to get exactly the columns we need
(Ticker, Company, Sector, Price, Change, Volume, Avg Volume,
Rel Volume, Gap, Market Cap) in a single pass per index.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).parent.parent.parent / ".cache" / "scanner"
SCAN_CACHE = CACHE_DIR / "latest_scan.pkl"
SCAN_TTL_MINUTES = 15

# Custom screener column IDs (from finvizfinance.screener.get_custom_screener_columns)
# 1=Ticker, 2=Company, 3=Sector, 6=Market Cap, 61=Gap,
# 63=Average Volume, 64=Relative Volume, 65=Price, 66=Change, 67=Volume
_CUSTOM_COLUMNS = [1, 2, 3, 6, 61, 63, 64, 65, 66, 67]


def _fetch_from_finviz() -> pd.DataFrame:
    """
    Fetch screening data from Finviz for S&P 500 + NASDAQ 100.

    Uses the Custom screener view for each index, which gives us all
    needed columns in a single paginated fetch.
    """
    from finvizfinance.screener.custom import Custom

    all_frames: list[pd.DataFrame] = []

    for idx_label in ["S&P 500", "NASDAQ 100"]:
        print(f"[scanner] Fetching {idx_label} …")
        screener = Custom()
        screener.set_filter(filters_dict={"Index": idx_label})
        df = screener.screener_view(columns=_CUSTOM_COLUMNS)
        if df is not None and not df.empty:
            all_frames.append(df)

    if not all_frames:
        raise RuntimeError("Finviz returned no data. The site may be down or rate-limiting.")

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["Ticker"], keep="first")

    # ── Normalize types ──────────────────────────────────────────────────────
    # finvizfinance returns Change/Gap as float ratios (e.g. -0.0321 = -3.21%)
    for col in ["Change", "Gap"]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce").fillna(0)

    for col in ["Volume", "Average Volume"]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce").fillna(0).astype(int)

    # Rename "Average Volume" → "Avg Volume" and "Relative Volume" → "Rel Volume"
    rename_map = {}
    if "Average Volume" in combined.columns:
        rename_map["Average Volume"] = "Avg Volume"
    if "Relative Volume" in combined.columns:
        rename_map["Relative Volume"] = "Rel Volume"
    if "Market Cap." in combined.columns:
        rename_map["Market Cap."] = "Market Cap"
    if rename_map:
        combined = combined.rename(columns=rename_map)

    if "Rel Volume" in combined.columns:
        combined["Rel Volume"] = pd.to_numeric(combined["Rel Volume"], errors="coerce").fillna(0)

    if "Price" in combined.columns:
        combined["Price"] = pd.to_numeric(combined["Price"], errors="coerce")

    # Compute Rel Volume from Avg Volume if the column wasn't returned
    if "Rel Volume" not in combined.columns and "Avg Volume" in combined.columns:
        avg = combined["Avg Volume"].replace(0, 1)
        combined["Rel Volume"] = combined["Volume"] / avg

    print(f"[scanner] {len(combined)} tickers fetched from Finviz")
    return combined.reset_index(drop=True)


def fetch_scanner_data(force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch scanner data with 15-minute cache.
    Returns DataFrame with columns including:
        Ticker, Company, Sector, Price, Change, Volume,
        Avg Volume, Rel Volume, Gap, Market Cap
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not force_refresh and SCAN_CACHE.exists():
        age_min = (time.time() - SCAN_CACHE.stat().st_mtime) / 60
        if age_min < SCAN_TTL_MINUTES:
            df = pd.read_pickle(SCAN_CACHE)
            print(f"[scanner] {len(df)} tickers from cache "
                  f"({SCAN_TTL_MINUTES - age_min:.0f} min remaining)")
            return df

    df = _fetch_from_finviz()
    df.to_pickle(SCAN_CACHE)
    return df
