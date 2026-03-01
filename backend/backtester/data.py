"""
Data fetching with local pickle caching.
Supports any Yahoo Finance ticker: BTC-USD, ABNB, NVDA, SPY, etc.
"""
import threading

import yfinance as yf
import pandas as pd
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent.parent / ".cache" / "ohlcv"

# yfinance is not thread-safe; serialize downloads
_download_lock = threading.Lock()


def fetch_ohlcv(
    ticker: str,
    period: str = "5y",
    interval: str = "1d",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch OHLCV data via yfinance with local pickle caching (8-hour TTL).

    Args:
        ticker:        Yahoo Finance symbol — e.g. "BTC-USD", "ABNB", "NVDA", "^GSPC"
        period:        History length: "1y" "2y" "5y" "10y" "max"
        interval:      Bar size: "1d" "1wk" "1mo"
        force_refresh: Bypass cache and re-download

    Returns:
        DataFrame(open, high, low, close, volume) — DatetimeIndex UTC
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = ticker.replace("/", "-").replace(":", "-").replace("^", "")
    cache_file = CACHE_DIR / f"{safe}_{period}_{interval}.pkl"

    # Return cached copy if fresh (< 8 hours old)
    if not force_refresh and cache_file.exists():
        age_h = (pd.Timestamp.utcnow().timestamp() - cache_file.stat().st_mtime) / 3600
        if age_h < 8:
            df = pd.read_pickle(cache_file)
            print(f"[data] {ticker}: {len(df)} bars from cache ({period} {interval})")
            return df

    print(f"[data] Downloading {ticker} {period} {interval} …")
    with _download_lock:
        raw = yf.download(ticker, period=period, interval=interval,
                          auto_adjust=True, progress=False, threads=False)

    if raw.empty:
        raise ValueError(f"No data for '{ticker}'. Check the symbol.")

    # yfinance sometimes returns MultiIndex columns
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index = pd.to_datetime(df.index, utc=True)
    df.dropna(inplace=True)
    df.to_pickle(cache_file)
    print(f"[data] {ticker}: {len(df)} bars downloaded")
    return df
