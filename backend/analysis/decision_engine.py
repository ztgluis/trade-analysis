"""
analysis/decision_engine.py  —  Core analysis logic, no print statements.

Returns structured dicts used by both:
  - run_decide_unified.py  (CLI rendering)
  - app.py                 (Streamlit UI rendering)

Main entry point:
    from analysis.decision_engine import analyze
    result = analyze("GOOG", horizon_td=21)
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy  as np
import pandas as pd
import yfinance as yf

# Ensure backend root is on path when imported from app.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtester.data   import fetch_ohlcv
from strategies.growth_signal_bot import GrowthSignalBot
from analysis.asset_profiles import get_profile


# ─────────────────────────────────────────────────────────────────────────────
# Primitive helpers
# ─────────────────────────────────────────────────────────────────────────────

def _regime(row: pd.Series) -> str:
    if bool(row.get("bull_regime")): return "bull"
    if bool(row.get("bear_regime")): return "bear"
    return "neutral"

def _macd_bull(row: pd.Series) -> bool:
    return float(row.get("macd_line", 0)) > float(row.get("signal_line", 0))

def _adx_ok(adx: float, thr: float) -> bool:
    return adx >= thr

def _score_bucket(s: int) -> str:
    return "strong" if s >= 11 else ("moderate" if s >= 8 else "weak")

def _bear_score_bucket(s: int) -> str:
    return "high" if s >= 9 else ("moderate" if s >= 6 else "low")

def bar_str(ts) -> str:
    try:    return pd.Timestamp(ts).strftime("%Y-%m-%d")
    except: return str(ts)


# ─────────────────────────────────────────────────────────────────────────────
# Earnings
# ─────────────────────────────────────────────────────────────────────────────

def get_earnings_info(ticker: str) -> dict | None:
    try:
        t   = yf.Ticker(ticker)
        cal = t.calendar
        dates = []
        if isinstance(cal, dict):
            raw = cal.get("Earnings Date", [])
            dates = list(raw) if raw else []
        elif cal is not None and hasattr(cal, "loc"):
            if "Earnings Date" in cal.index:
                dates = list(cal.loc["Earnings Date"].values)
        if not dates:
            return None
        now = pd.Timestamp.now(tz="UTC").normalize()
        upcoming = []
        for d in dates:
            try:
                d_ts = pd.Timestamp(d)
                if d_ts.tzinfo is None:
                    d_ts = d_ts.tz_localize("UTC")
                if d_ts >= now:
                    upcoming.append(d_ts)
            except Exception:
                continue
        if not upcoming:
            return None
        next_e = min(upcoming)
        cal_days = (next_e - now).days
        return {
            "date":          next_e,
            "date_str":      bar_str(next_e),
            "cal_days":      cal_days,
            "trading_days":  max(0, round(cal_days * 5 / 7)),
        }
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Macro context
# ─────────────────────────────────────────────────────────────────────────────

def get_macro_context(strat: GrowthSignalBot) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for tkr, lbl in [("SPY", "S&P 500"), ("QQQ", "Nasdaq 100")]:
        try:
            df = fetch_ohlcv(tkr, period="1y", interval="1d")
            if df is None or len(df) < 220:
                continue
            df_p  = strat.prepare(df.copy())
            last  = df_p.iloc[-1]
            close = float(last["close"])
            result[tkr] = {
                "label":         lbl,
                "close":         close,
                "regime":        _regime(last),
                "sma200_rising": float(last["sma200"]) > float(df_p.iloc[-2]["sma200"]),
                "adx":           float(last.get("adx", 0)),
                "rsi":           float(last.get("rsi", 0)),
                "ret_1m":        (close - float(df_p.iloc[-21]["close"])) / float(df_p.iloc[-21]["close"])
                                 if len(df_p) > 21 else None,
                "ret_3m":        (close - float(df_p.iloc[-63]["close"])) / float(df_p.iloc[-63]["close"])
                                 if len(df_p) > 63 else None,
            }
        except Exception:
            continue
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Base rates
# ─────────────────────────────────────────────────────────────────────────────

def compute_base_rates(
    df:             pd.DataFrame,
    bucket_key:     dict,
    horizons:       list[int],
    adx_thr:        float,
    direction:      str = "long",   # "long" | "short"
) -> dict:
    closes = df["close"].values
    max_h  = max(horizons)
    valid  = df.iloc[:-max_h] if len(df) > max_h else df.iloc[:0]

    match_mask = []
    for _, row in valid.iterrows():
        b = {
            "regime":     _regime(row),
            "bull_score": _score_bucket(int(row.get("bull_score", 0))),
            "bear_score": _bear_score_bucket(int(row.get("bear_score", 0))),
            "macd":       "bull" if _macd_bull(row) else "bear",
        }
        match_mask.append(b == bucket_key)

    idx_arr = np.where(match_mask)[0]
    results: dict = {"_n": len(idx_arr), "_direction": direction}

    for h in horizons:
        fwd = []
        for i in idx_arr:
            fi = i + h
            if fi < len(closes):
                fwd.append((closes[fi] - closes[i]) / closes[i])
        if not fwd:
            results[h] = {"n": 0}
            continue
        arr = np.array(fwd)
        win = float(np.mean(arr > 0)) if direction == "long" else float(np.mean(arr < 0))
        results[h] = {
            "n":      len(arr),
            "win":    win,
            "median": float(np.median(arr)),
            "p25":    float(np.percentile(arr, 25)),
            "p75":    float(np.percentile(arr, 75)),
            "best":   float(np.min(arr)) if direction == "short" else float(np.max(arr)),
            "worst":  float(np.max(arr)) if direction == "short" else float(np.min(arr)),
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def compute_long_score(
    regime, bull_score, adx_ok, rsi, macd_b, sma200_rising,
    bounce_sig, vwap_bounce, bull_div,
    rsi_bull_min, rsi_bull_max, adx_threshold,
) -> tuple[int, list[tuple[bool, str]]]:
    conds = []
    score = 0

    c = (regime == "bull")
    conds.append((c, "Bull regime (price > rising SMA200)"))
    if c: score += 3

    c = (bull_score >= 9)
    conds.append((c, f"Bull score ≥9  (currently {bull_score}/16)"))
    if c: score += 2

    c = adx_ok
    conds.append((c, f"ADX trending  ({adx_threshold:.0f}+ gate)"))
    if c: score += 1

    c = macd_b
    conds.append((c, "MACD above signal (bullish momentum)"))
    if c: score += 1

    c = (rsi_bull_min <= rsi <= rsi_bull_max)
    conds.append((c, f"RSI in bull zone {rsi_bull_min}–{rsi_bull_max}  (currently {rsi:.0f})"))
    if c: score += 1

    c = sma200_rising
    conds.append((c, "SMA200 rising (long-term uptrend intact)"))
    if c: score += 1

    c = (rsi < 38 and macd_b and (bounce_sig or vwap_bounce or bull_div))
    conds.append((c, f"Bounce setup: RSI {rsi:.0f} oversold + MACD turning + signal"))
    if c: score += 2

    return min(score, 10), conds


def compute_short_score(
    regime, bear_score, adx_ok, rsi, macd_b, sma200_rising, bear_div, bull_score,
) -> tuple[int, list[tuple[bool, str]]]:
    conds = []
    score = 0

    c = (regime == "bear")
    conds.append((c, "Bear regime (price < falling SMA200)"))
    if c: score += 3

    c = (bear_score >= 9)
    conds.append((c, f"Bear score ≥9  (currently {bear_score}/16)"))
    if c: score += 2

    c = adx_ok
    conds.append((c, "ADX trending (downtrend has momentum)"))
    if c: score += 1

    c = not macd_b
    conds.append((c, "MACD below signal (bearish momentum)"))
    if c: score += 1

    c = (30 < rsi < 60)
    conds.append((c, f"RSI not oversold / not overbought  ({rsi:.0f})"))
    if c: score += 1

    c = not sma200_rising
    conds.append((c, "SMA200 falling (structural downtrend)"))
    if c: score += 1

    c = (not bool(bear_div) and bull_score < 8)
    conds.append((c, "No bull divergence, low bull score (no counter-signals)"))
    if c: score += 1

    return min(score, 10), conds


def determine_verdict(
    long_score, short_score, regime, rsi, macd_b,
    bounce_sig, vwap_bounce, bull_div, earnings_td,
    bull_score, bear_score,
) -> tuple[str, str, str]:
    if earnings_td is not None and earnings_td <= 5:
        return (
            "⚠  WAIT — EARNINGS IMMINENT",
            "Do not enter new positions. Earnings gap risk in both directions.",
            "yellow",
        )
    is_bounce = rsi < 38 and macd_b and (bounce_sig or vwap_bounce or bull_div or rsi < 32)

    if long_score >= 8:
        return "🟢 STRONG LONG",  "Bot conditions strongly favor a long entry. Size normally.", "green"
    if long_score >= 6 and long_score > short_score + 1:
        if is_bounce:
            return ("🌊 BOUNCE PLAY",
                    "Oversold + bullish reversal signals. Long on bounce — smaller size, tight stop.",
                    "lime")
        return "🟢 LEAN LONG", "More long conditions met than bear. Can lean long, size conservatively.", "green"
    if is_bounce and long_score >= 4:
        return ("🌊 BOUNCE PLAY",
                "Oversold reversal signal. Long on bounce — reduced size, tight SL below recent low.",
                "lime")
    if short_score >= 8:
        return "🔴 STRONG SHORT", "Bear conditions dominant. Short entry favored — check RSI > 30.", "red"
    if short_score >= 6 and short_score > long_score + 1:
        return "🔴 LEAN SHORT", "More bear conditions met. Can lean short, size conservatively.", "red"
    if abs(long_score - short_score) <= 1:
        return ("⚪ WAIT — NO CLEAR BIAS",
                "Long and short conditions roughly equal. No edge. Wait for a signal to tip the balance.",
                "grey")
    return "⚪ WAIT — LOW CONVICTION", "Neither direction has enough conditions met. Stand aside.", "grey"


# ─────────────────────────────────────────────────────────────────────────────
# Main analyze() function
# ─────────────────────────────────────────────────────────────────────────────

def analyze(ticker: str, horizon_td: int = 21, profile_override: dict | None = None) -> dict:
    """
    Run full decision analysis for a ticker.
    Returns a structured dict — no print statements.
    Used by both run_decide_unified.py (CLI) and app.py (Streamlit).

    Args:
        ticker:           Yahoo Finance symbol
        horizon_td:       Target holding period in trading days
        profile_override: If provided, use this profile dict instead of the
                          auto-detected one. Useful for "what-if" comparisons.
                          Must have keys: rsi_bull_min, rsi_bull_max,
                          adx_threshold, sl_pct, tp_pct, category
    """
    result: dict = {
        "ticker":    ticker,
        "horizon":   horizon_td,
        "error":     None,
        "df":        None,
        "df_chart":  None,   # last 6mo slice for charting
    }

    # ── Fetch data ─────────────────────────────────────────────────────────
    df_long = fetch_ohlcv(ticker, period="max", interval="1d")
    if df_long is None or len(df_long) < 300:
        df_long = fetch_ohlcv(ticker, period="5y", interval="1d")
    if df_long is None or len(df_long) < 100:
        result["error"] = f"Not enough data for {ticker}"
        return result

    # ── Asset profile (per-ticker tuned params) ────────────────────────────
    profile  = profile_override if profile_override is not None else get_profile(ticker)

    # Macro uses default params (SPY/QQQ are _etf but default is fine for
    # regime detection — we just want the SMA200 cross signal)
    macro_strat = GrowthSignalBot()
    macro       = get_macro_context(macro_strat)
    earnings    = get_earnings_info(ticker)

    # Main ticker uses profile-adjusted strategy
    strat = GrowthSignalBot(
        rsi_bull_min  = profile["rsi_bull_min"],
        rsi_bull_max  = profile["rsi_bull_max"],
        adx_threshold = profile["adx_threshold"],
        sl_pct        = profile["sl_pct"],
        tp_pct        = profile["tp_pct"],
    )

    # ── Prepare indicators ─────────────────────────────────────────────────
    df   = strat.prepare(df_long.copy())
    last = df.iloc[-1]
    prev = df.iloc[-2]

    last_close    = float(last["close"])
    last_ts       = df.index[-1]
    regime        = _regime(last)
    bull_score    = int(last.get("bull_score", 0))
    bear_score    = int(last.get("bear_score", 0))
    adx           = float(last.get("adx", 0))
    adx_ok        = _adx_ok(adx, strat.adx_threshold)
    rsi           = float(last["rsi"])
    macd_b        = _macd_bull(last)
    macd_l        = float(last["macd_line"])
    sig_l         = float(last["signal_line"])
    sma200        = float(last["sma200"])
    sma50         = float(last["sma50"])
    sma200_rising = float(last["sma200"]) > float(prev["sma200"])
    ema20         = float(last["ema20"]) if "ema20" in df.columns else None
    vwap_w        = float(last["vwap_w"]) if "vwap_w" in df.columns else None
    poc           = float(last["poc"]) if "poc" in df.columns else None
    bounce_sig    = bool(last.get("bounce_signal", False))
    vwap_bounce   = bool(last.get("vwap_bounce", False))
    bull_div      = bool(last.get("bull_div", False))
    bear_div      = bool(last.get("bear_div", False))
    earnings_td   = earnings["trading_days"] if earnings else None

    # ── Scores & verdict ───────────────────────────────────────────────────
    long_score,  long_conds  = compute_long_score(
        regime, bull_score, adx_ok, rsi, macd_b, sma200_rising,
        bounce_sig, vwap_bounce, bull_div,
        strat.rsi_bull_min, strat.rsi_bull_max, strat.adx_threshold,
    )
    short_score, short_conds = compute_short_score(
        regime, bear_score, adx_ok, rsi, macd_b, sma200_rising, bear_div, bull_score,
    )
    verdict, action, color = determine_verdict(
        long_score, short_score, regime, rsi, macd_b,
        bounce_sig, vwap_bounce, bull_div, earnings_td,
        bull_score, bear_score,
    )

    # ── Base rates ─────────────────────────────────────────────────────────
    direction   = "long" if long_score >= short_score else "short"
    bucket_key  = {
        "regime":     regime,
        "bull_score": _score_bucket(bull_score),
        "bear_score": _bear_score_bucket(bear_score),
        "macd":       "bull" if macd_b else "bear",
    }
    horizons    = [5, 10, 21, 42, 63, 126]
    base_rates  = compute_base_rates(df, bucket_key, horizons, strat.adx_threshold, direction)

    # ── Recent signals ─────────────────────────────────────────────────────
    recent_signals: list[tuple] = []
    active_signals: list[str]   = []
    signal_map = [
        ("buy_signal",    "🔺 BUY"),
        ("bounce_signal", "⬤ BOUNCE"),
        ("vwap_bounce",   "◆ VWAP BOUNCE"),
        ("sell_signal",   "🔻 SELL"),
        ("bull_div",      "✕ Bull Div"),
        ("bear_div",      "✕ Bear Div"),
    ]
    for col, lbl in signal_map:
        if col not in df.columns:
            continue
        if last.get(col):
            active_signals.append(lbl)
        fired = df[df[col] == True]
        for ts, row in fired.iterrows():
            recent_signals.append((ts, lbl, float(row["close"])))
    if last.get("macd_cross"):
        active_signals.append("⚡ MACD crossed UP")
    if last.get("macd_crossunder"):
        active_signals.append("⚡ MACD crossed DOWN")
    recent_signals.sort(key=lambda x: x[0])

    # ── Key levels ─────────────────────────────────────────────────────────
    hi52 = float(df.iloc[-252:]["close"].max()) if len(df) >= 252 else float(df["close"].max())
    lo52 = float(df.iloc[-252:]["close"].min()) if len(df) >= 252 else float(df["close"].min())
    raw_levels = [
        ("52w High", hi52), ("SMA200", sma200), ("SMA50", sma50),
        ("VWAP(W)", vwap_w), ("EMA20", ema20), ("POC", poc),
        ("52w Low", lo52),
    ]
    levels = sorted([(n, v) for n, v in raw_levels if v is not None],
                    key=lambda x: x[1], reverse=True)

    # ── Action plan: SL / TP / R:R ────────────────────────────────────────
    sl_pct = profile["sl_pct"]
    tp_pct = profile["tp_pct"]

    if "LONG" in verdict:
        sl_price = last_close * (1 - sl_pct / 100)
        tp_price = last_close * (1 + tp_pct / 100)
        rr_ratio = tp_pct / sl_pct
        entry_lo = last_close * 0.99
        entry_hi = last_close * 1.01
    elif "SHORT" in verdict:
        resistances = sorted(v for _, v in levels if v > last_close * 1.005)
        sl_price  = resistances[0] if resistances else last_close * (1 + sl_pct / 100)
        tp_price  = last_close * 0.93
        sl_dist   = sl_price - last_close
        tp_dist   = last_close - tp_price
        rr_ratio  = (tp_dist / sl_dist) if sl_dist > 0 else None
        entry_lo  = last_close * 0.99
        entry_hi  = last_close * 1.005
    elif "BOUNCE" in verdict:
        ema20_val = ema20 or last_close * 1.04
        sl_price  = lo52
        tp_price  = ema20_val
        sl_dist   = last_close - lo52
        tp_dist   = ema20_val - last_close
        rr_ratio  = (tp_dist / sl_dist) if sl_dist > 0 and tp_dist > 0 else None
        entry_lo  = last_close * 0.995
        entry_hi  = last_close * 1.005
    else:
        sl_price = tp_price = rr_ratio = entry_lo = entry_hi = None

    # ── Alpha vs SPY ───────────────────────────────────────────────────────
    alpha_1m = None
    spy = macro.get("SPY")
    if spy and spy.get("ret_1m") is not None and len(df) > 21:
        ticker_1m = (last_close - float(df.iloc[-21]["close"])) / float(df.iloc[-21]["close"])
        alpha_1m  = ticker_1m - spy["ret_1m"]

    # ── Price & % changes ──────────────────────────────────────────────────
    price_changes = {}
    for bars, label in [(5, "1wk"), (21, "1mo"), (63, "3mo"), (126, "6mo")]:
        if len(df) > bars:
            past = float(df.iloc[-bars]["close"])
            price_changes[label] = (last_close - past) / past

    # ── Chart slice — last 6 months ────────────────────────────────────────
    df_chart = df.iloc[-126:].copy() if len(df) >= 126 else df.copy()

    # ── Pack result ────────────────────────────────────────────────────────
    result.update({
        # Identity
        "price":          last_close,
        "last_ts":        bar_str(last_ts),
        "data_start":     bar_str(df.index[0]),

        # Indicators
        "regime":         regime,
        "bull_score":     bull_score,
        "bear_score":     bear_score,
        "rsi":            rsi,
        "adx":            adx,
        "adx_ok":         adx_ok,
        "adx_threshold":  strat.adx_threshold,
        "macd_bull":      macd_b,
        "macd_line":      macd_l,
        "signal_line":    sig_l,
        "sma200":         sma200,
        "sma200_rising":  sma200_rising,
        "sma50":          sma50,
        "ema20":          ema20,
        "vwap_w":         vwap_w,
        "poc":            poc,
        "hi52":           hi52,
        "lo52":           lo52,

        # Scoring
        "long_score":      long_score,
        "short_score":     short_score,
        "long_conditions": long_conds,
        "short_conditions":short_conds,

        # Verdict
        "verdict":        verdict,
        "action":         action,
        "color":          color,

        # Asset profile
        "profile":        profile["category"],
        "sl_pct":         sl_pct,
        "tp_pct":         tp_pct,
        "sl_price":       sl_price,
        "tp_price":       tp_price,
        "rr_ratio":       rr_ratio,
        "entry_lo":       entry_lo,
        "entry_hi":       entry_hi,

        # Context
        "earnings":       earnings,
        "macro":          macro,
        "alpha_1m":       alpha_1m,
        "price_changes":  price_changes,

        # History
        "base_rates":     base_rates,
        "base_rates_dir": direction,
        "recent_signals": recent_signals,
        "active_signals": active_signals,

        # Levels
        "levels":         levels,

        # DataFrames
        "df":             df,
        "df_chart":       df_chart,
    })
    return result
