"""
Technical indicators — pure pandas/numpy, no external TA library.
Each function mirrors the equivalent Pine Script ta.* function exactly,
so Python backtests produce results comparable to TradingView.
"""
import pandas as pd
import numpy as np


# ── Moving Averages ───────────────────────────────────────────────────────────

def ema(series: pd.Series, length: int) -> pd.Series:
    """Exponential Moving Average. Matches Pine Script ta.ema()."""
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def sma(series: pd.Series, length: int) -> pd.Series:
    """Simple Moving Average. Matches Pine Script ta.sma()."""
    return series.rolling(window=length, min_periods=length).mean()


# ── RSI ───────────────────────────────────────────────────────────────────────

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """
    RSI using Wilder smoothing (RMA = EMA with alpha=1/length).
    Matches Pine Script ta.rsi() — NOT pandas_ta default (which uses SMA seed).
    """
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ── MACD ──────────────────────────────────────────────────────────────────────

def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD. Matches Pine Script ta.macd().
    Returns (macd_line, signal_line, histogram).
    """
    ema_fast    = series.ewm(span=fast,   adjust=False).mean()
    ema_slow    = series.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


# ── ATR ───────────────────────────────────────────────────────────────────────

def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
) -> pd.Series:
    """Average True Range — Wilder smoothing. Matches Pine Script ta.atr()."""
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


# ── Crossover / Crossunder ────────────────────────────────────────────────────

def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    """True on the bar where a crosses ABOVE b. Matches ta.crossover()."""
    return (a > b) & (a.shift(1) <= b.shift(1))


def crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    """True on the bar where a crosses BELOW b. Matches ta.crossunder()."""
    return (a < b) & (a.shift(1) >= b.shift(1))


# ── Volume ────────────────────────────────────────────────────────────────────

def avg_volume(volume: pd.Series, length: int = 20) -> pd.Series:
    """Rolling average volume (matches ta.sma(volume, length))."""
    return volume.rolling(window=length, min_periods=1).mean()


# ── RSI Divergence ────────────────────────────────────────────────────────────

def bull_divergence(close: pd.Series, rsi_series: pd.Series, lb: int = 5) -> pd.Series:
    """
    Bullish RSI divergence: price makes lower low, RSI makes higher low.
    Mirrors Pine Script: price_ll = close < ta.lowest(close[1], lb)
    """
    price_ll = close < close.shift(1).rolling(lb).min()
    rsi_hl   = rsi_series > rsi_series.shift(1).rolling(lb).min()
    return price_ll & rsi_hl


def bear_divergence(close: pd.Series, rsi_series: pd.Series, lb: int = 5) -> pd.Series:
    """Bearish RSI divergence: price makes higher high, RSI makes lower high."""
    price_hh = close > close.shift(1).rolling(lb).max()
    rsi_lh   = rsi_series < rsi_series.shift(1).rolling(lb).max()
    return price_hh & rsi_lh


# ── VWAP (Period-Resetting) ───────────────────────────────────────────────────

def vwap_periodic(
    df: pd.DataFrame,
    freq: str = "W",
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Cumulative VWAP that resets at the start of each calendar period.
    Mirrors Pine Script's manual weekly/monthly VWAP implementation exactly.

    Args:
        df:   DataFrame with columns open/high/low/close/volume, DatetimeIndex
        freq: 'W' = weekly reset, 'M' = monthly reset

    Returns:
        (vwap, upper_band, lower_band)
        upper/lower are ±1 population-StdDev of (hlc3 - vwap) within the period.
    """
    hlc3 = (df["high"] + df["low"] + df["close"]) / 3
    vol  = df["volume"]

    # to_period() requires a timezone-naive index
    tz        = df.index.tz
    idx_naive = df.index.tz_localize(None) if tz else df.index
    periods   = idx_naive.to_period(freq)

    vwap_arr  = np.full(len(df), np.nan)
    stdev_arr = np.full(len(df), np.nan)

    hlc3_v = hlc3.values
    vol_v  = vol.values

    for p in periods.unique():
        mask = np.asarray(periods == p)       # np.asarray() works in pandas 1.x and 2.x

        h = hlc3_v[mask]
        v = vol_v[mask]

        cum_vp = np.cumsum(h * v)
        cum_v  = np.cumsum(v)
        vw     = np.where(cum_v > 0, cum_vp / cum_v, np.nan)

        # Population StdDev of (hlc3 - vwap) — cumulative within period
        # Matches Pine Script: wk_sum_sq += (hlc3 - vwap)²; stdev = sqrt(sum_sq / n)
        sq     = (h - vw) ** 2
        n      = np.arange(1, len(h) + 1, dtype=float)
        sd     = np.sqrt(np.cumsum(sq) / n)

        vwap_arr[mask]  = vw
        stdev_arr[mask] = sd

    vwap  = pd.Series(vwap_arr,  index=df.index, name=f"vwap_{freq.lower()}")
    stdev = pd.Series(stdev_arr, index=df.index)
    return vwap, vwap + stdev, vwap - stdev


# ── Rolling Point of Control (Volume Profile approx.) ────────────────────────

def rolling_poc(df: pd.DataFrame, lookback: int = 50) -> pd.Series:
    """
    Rolling Point of Control: the HLC3 price of the highest-volume bar
    in the last `lookback` bars.  Approximates volume profile S/R without
    full bucket computation.

    Price > POC → buyers willing to pay above the most accepted level → bullish.
    Price < POC → rejected from the most popular level → bearish.

    Args:
        df:       DataFrame with columns high/low/close/volume, DatetimeIndex
        lookback: Number of bars to scan (default 50 ≈ 2.5 months on daily)

    Returns:
        pd.Series aligned to df.index
    """
    hlc3  = (df["high"] + df["low"] + df["close"]) / 3
    vol   = df["volume"].values
    hlc3v = hlc3.values
    result = np.full(len(df), np.nan)

    for i in range(lookback - 1, len(df)):
        window_vol = vol[i - lookback + 1 : i + 1]
        max_idx    = int(np.argmax(window_vol))
        result[i]  = hlc3v[i - lookback + 1 + max_idx]

    return pd.Series(result, index=df.index, name="poc")


# ── ADX (Average Directional Index) ──────────────────────────────────────────

def adx(
    high:   pd.Series,
    low:    pd.Series,
    close:  pd.Series,
    length: int = 14,
) -> pd.Series:
    """
    Average Directional Index — Wilder smoothing. Matches Pine Script ta.adx().

    ADX measures trend STRENGTH, not direction:
        < 20  →  weak / choppy market
        20-40 →  trending (usable signals)
        > 40  →  strong trend

    Returns the ADX series (0–100).
    """
    up   = high - high.shift(1)
    down = low.shift(1) - low

    plus_dm  = pd.Series(np.where((up > down) & (up > 0),   up,   0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)

    # True Range (same as atr() function above)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    alpha = 1.0 / length
    smooth_tr       = tr.ewm(alpha=alpha, adjust=False, min_periods=length).mean()
    smooth_plus_dm  = plus_dm.ewm(alpha=alpha, adjust=False, min_periods=length).mean()
    smooth_minus_dm = minus_dm.ewm(alpha=alpha, adjust=False, min_periods=length).mean()

    plus_di  = 100.0 * smooth_plus_dm  / smooth_tr.replace(0, np.nan)
    minus_di = 100.0 * smooth_minus_dm / smooth_tr.replace(0, np.nan)

    di_sum  = (plus_di + minus_di).replace(0, np.nan)
    dx      = 100.0 * (plus_di - minus_di).abs() / di_sum
    adx_ser = dx.ewm(alpha=alpha, adjust=False, min_periods=length).mean()

    return adx_ser.rename("adx")


# ── Fibonacci Retracement Near Level ─────────────────────────────────────────

def fib_near_level(
    high:          pd.Series,
    low:           pd.Series,
    close:         pd.Series,
    swing_len:     int             = 50,
    levels:        tuple[float, ...] = (0.382, 0.5, 0.618),
    threshold_pct: float           = 1.5,
) -> pd.Series:
    """
    Returns True on bars where close is within threshold_pct% of a Fibonacci
    retracement level computed over the last swing_len bars.

    Levels are measured from swing low upward:
        fib_price = swing_low + level * (swing_high - swing_low)

    Defaults to the three most-watched institutional levels: 38.2%, 50%, 61.8%.
    threshold_pct=1.5 means ±1.5% of the level counts as "near".

    Bullish context: price pulling back to a fib support → high-probability bounce.
    """
    swing_high  = high.rolling(swing_len, min_periods=swing_len).max()
    swing_low   = low.rolling(swing_len,  min_periods=swing_len).min()
    swing_range = swing_high - swing_low

    threshold = threshold_pct / 100.0
    near      = pd.Series(False, index=close.index)

    for lvl in levels:
        fib_price = swing_low + lvl * swing_range
        near     |= (close - fib_price).abs() / close.replace(0, np.nan) <= threshold

    return near.rename("near_fib")


# ── Weekly MTF (Multi-Timeframe) ──────────────────────────────────────────────

def weekly_mtf(
    close: pd.Series,
    rsi_len: int = 14,
    ema_len: int = 20,
) -> tuple[pd.Series, pd.Series]:
    """
    Compute weekly RSI and weekly EMA, forward-filled to daily frequency.
    Mirrors Pine Script request.security(ticker, 'W', ..., lookahead=off).

    Returns (weekly_rsi_daily, weekly_ema_daily) aligned to close.index.
    Each daily bar uses the most recently completed or in-progress weekly value —
    same behaviour as Pine Script's lookahead_off on a daily chart.
    """
    tz        = close.index.tz
    idx_naive = close.index.tz_localize(None) if tz else close.index

    s = close.copy()
    s.index = idx_naive

    weekly        = s.resample("W").last().dropna()
    weekly_rsi    = rsi(weekly, rsi_len)
    weekly_ema    = ema(weekly, ema_len)

    # Forward-fill: each day inherits the latest completed-week value
    w_rsi_d = weekly_rsi.reindex(idx_naive, method="ffill")
    w_ema_d = weekly_ema.reindex(idx_naive, method="ffill")

    if tz:
        w_rsi_d.index = w_rsi_d.index.tz_localize(tz)
        w_ema_d.index = w_ema_d.index.tz_localize(tz)

    return w_rsi_d, w_ema_d


# ── Generic Higher-Timeframe MTF ──────────────────────────────────────────────

def htf_mtf(
    close:   pd.Series,
    freq:    str = "W",
    rsi_len: int = 14,
    ema_len: int = 20,
) -> tuple[pd.Series, pd.Series]:
    """
    Generic higher-timeframe MTF: resample close to freq, compute RSI + EMA,
    then forward-fill back to the original index.

    Works for any pandas resample frequency:
        freq="W"  → weekly  (daily chart — equivalent to weekly_mtf())
        freq="D"  → daily   (1H chart — mirrors Pine Script request.security(.."D"..))
        freq="ME" → monthly

    Returns (htf_rsi, htf_ema) aligned to close.index.
    """
    tz        = close.index.tz
    idx_naive = close.index.tz_localize(None) if tz else close.index

    s = close.copy()
    s.index = idx_naive

    htf       = s.resample(freq).last().dropna()
    htf_rsi_s = rsi(htf, rsi_len)
    htf_ema_s = ema(htf, ema_len)

    htf_rsi_d = htf_rsi_s.reindex(idx_naive, method="ffill")
    htf_ema_d = htf_ema_s.reindex(idx_naive, method="ffill")

    if tz:
        htf_rsi_d.index = htf_rsi_d.index.tz_localize(tz)
        htf_ema_d.index = htf_ema_d.index.tz_localize(tz)

    return htf_rsi_d, htf_ema_d


# ── Daily SMA200 (for 1H regime detection) ───────────────────────────────────

def daily_sma200(
    close_1h: pd.Series,
    sma_len:  int = 200,
    shift_n:  int = 20,
) -> tuple[pd.Series, pd.Series]:
    """
    Compute SMA(sma_len) on daily-resampled bars from 1H data, plus a shifted
    copy for trend-direction detection, both forward-filled back to the 1H index.

    Mirrors Pine Script on 1H charts:
        d_sma200     = request.security(ticker, "D", ta.sma(close, 200), lookahead=off)
        d_sma200_ago = request.security(ticker, "D", ta.sma(close, 200)[shift_n], lookahead=off)

    The daily SMA200 requires ~200 trading days of data — about 1 year of 1H bars.
    Use warmup_bars ≈ 200 * 6.5 ≈ 1400 on 1H to skip the NaN warmup period.

    Returns (sma200_1h, sma200_ago_1h) aligned to close_1h.index.
    """
    tz        = close_1h.index.tz
    idx_naive = close_1h.index.tz_localize(None) if tz else close_1h.index

    s = close_1h.copy()
    s.index = idx_naive

    # Resample to business days, keep last bar of each day (≈ daily close)
    daily         = s.resample("B").last().dropna()
    sma200_d      = sma(daily, sma_len)
    sma200_ago_d  = sma200_d.shift(shift_n)

    sma200_1h     = sma200_d.reindex(idx_naive, method="ffill")
    sma200_ago_1h = sma200_ago_d.reindex(idx_naive, method="ffill")

    if tz:
        sma200_1h.index     = sma200_1h.index.tz_localize(tz)
        sma200_ago_1h.index = sma200_ago_1h.index.tz_localize(tz)

    return sma200_1h, sma200_ago_1h
