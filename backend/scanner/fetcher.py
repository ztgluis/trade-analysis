"""
Finviz data fetcher with 15-minute pickle caching.

Scrapes the Finviz screener for S&P 500 and NASDAQ-100 tickers,
returning a merged DataFrame with overview + technical columns.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).parent.parent.parent / ".cache" / "scanner"
SCAN_CACHE = CACHE_DIR / "latest_scan.pkl"
SCAN_TTL_MINUTES = 15


def _fetch_from_finviz() -> pd.DataFrame:
    """
    Fetch screening data from Finviz for S&P 500 + NASDAQ 100.

    Uses two screener views per index:
      - Overview  → Ticker, Company, Sector, Industry, Market Cap, P/E, Price, Change, Volume
      - Technical → Ticker, Gap, Avg Volume, Rel Volume  (and others)

    Merges on Ticker, deduplicates across indexes.
    """
    from finvizfinance.screener.overview import Overview
    from finvizfinance.screener.technical import Technical

    all_overview: list[pd.DataFrame] = []
    all_technical: list[pd.DataFrame] = []

    for idx_label in ["S&P 500", "NASDAQ 100"]:
        print(f"[scanner] Fetching {idx_label} overview …")
        ov = Overview()
        ov.set_filter(filters_dict={"Index": idx_label})
        df_ov = ov.screener_view()
        if df_ov is not None and not df_ov.empty:
            all_overview.append(df_ov)

        print(f"[scanner] Fetching {idx_label} technicals …")
        tech = Technical()
        tech.set_filter(filters_dict={"Index": idx_label})
        df_tech = tech.screener_view()
        if df_tech is not None and not df_tech.empty:
            all_technical.append(df_tech)

    # Merge overview data
    overview = pd.concat(all_overview, ignore_index=True) if all_overview else pd.DataFrame()
    technical = pd.concat(all_technical, ignore_index=True) if all_technical else pd.DataFrame()

    if overview.empty:
        raise RuntimeError("Finviz returned no data. The site may be down or rate-limiting.")

    # Deduplicate
    overview = overview.drop_duplicates(subset=["Ticker"], keep="first")
    technical = technical.drop_duplicates(subset=["Ticker"], keep="first")

    # Pick useful columns from technical (avoid duplicating Price/Change/Volume)
    tech_cols = ["Ticker"]
    for col in ["Gap", "Avg Volume", "Rel Volume", "SMA20", "SMA50", "SMA200", "RSI"]:
        if col in technical.columns:
            tech_cols.append(col)

    if len(tech_cols) > 1 and not technical.empty:
        combined = overview.merge(technical[tech_cols], on="Ticker", how="left")
    else:
        combined = overview

    # ── Normalize types ──────────────────────────────────────────────────────
    # Change: finvizfinance may return "5.23%" as string or 0.0523 as float
    if "Change" in combined.columns:
        combined["Change"] = _parse_pct(combined["Change"])

    if "Gap" in combined.columns:
        combined["Gap"] = _parse_pct(combined["Gap"])

    for col in ["Volume", "Avg Volume"]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce").fillna(0).astype(int)

    if "Rel Volume" in combined.columns:
        combined["Rel Volume"] = pd.to_numeric(combined["Rel Volume"], errors="coerce").fillna(0)

    if "Price" in combined.columns:
        combined["Price"] = pd.to_numeric(combined["Price"], errors="coerce")

    # Compute Rel Volume if missing but Avg Volume is available
    if "Rel Volume" not in combined.columns and "Avg Volume" in combined.columns:
        avg = combined["Avg Volume"].replace(0, 1)
        combined["Rel Volume"] = combined["Volume"] / avg

    print(f"[scanner] {len(combined)} tickers fetched from Finviz")
    return combined.reset_index(drop=True)


def _parse_pct(series: pd.Series) -> pd.Series:
    """Parse a column that may be '5.23%' strings or already float."""
    def _convert(val):
        if isinstance(val, (int, float)):
            # Already numeric — check if it looks like a ratio (< 5) or pct (> 5)
            # Finviz values like 0.0523 are ratios; 5.23 are percents
            return float(val) if abs(val) < 5 else float(val) / 100
        s = str(val).strip().replace("%", "")
        try:
            return float(s) / 100
        except ValueError:
            return 0.0
    return series.apply(_convert)


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
