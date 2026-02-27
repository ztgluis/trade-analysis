#!/usr/bin/env python3
"""
run_decide_short.py  —  Trade Analysis SHORT Decision Engine

Answers the question: "Given the current market state for TICKER, does the
strategy's data support ENTERING or MAINTAINING a short position?"

The GrowthSignalBot bear signal stack mirrors the bull stack:
  bull_regime  →  bear_regime   (SMA200 falling + price below it)
  bull_score   →  bear_score    (up to ~12 bearish components)
  buy_signal   →  sell_signal   (when all bear conditions align)
  MACD cross   →  MACD crossunder

This engine re-frames the existing indicators around the SHORT decision:
  ✅ ENTER SHORT — bear regime confirmed, bear_score high, RSI NOT oversold
  ⏸ HOLD SHORT  — regime intact but RSI approaching oversold, or no new signal
  🔴 AVOID       — bull regime, RSI < 30 (exhausted), near earnings, near hard support

Sections:
  0. Earnings & Market Context  — gate checks (macro bears = tailwind for shorts)
  1. Current State               — bear score, regime, RSI oversold risk, MACD
  2. Short Base Rates            — in past instances with same bear state,
                                   how often did price fall? (short "win rate")
  3. Key Levels                  — resistance ABOVE (stop-loss zones)
                                   support BELOW (cover / profit targets)
  4. Short Decision Checklist    — enter / hold / cover / avoid conditions

Usage:
    python run_decide_short.py SNAP
    python run_decide_short.py TSLA PYPL RIVN
    python run_decide_short.py SNAP --horizon 21

⚠  NOT investment advice — objective indicator data only.
⚠  Shorting carries unlimited theoretical loss. Always use hard stop-losses.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy  as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent))

from backtester.data   import fetch_ohlcv
from strategies.growth_signal_bot import GrowthSignalBot


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

W = 66

def _sep(char="─"): return "  " + char * (W - 2)
def _hdr(title, char="═"): return f"\n  {char * (W - 2)}\n  {title}\n  {char * (W - 2)}"
def _h2(title): return f"\n  ── {title} {'─' * max(1, W - 6 - len(title))}"

def pct(v: float, decimals: int = 1) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v * 100:.{decimals}f}%"

def dol(v: float) -> str:
    return f"${v:>10,.2f}"

def bar_str(ts) -> str:
    try:
        return pd.Timestamp(ts).strftime("%Y-%m-%d")
    except Exception:
        return str(ts)

def _bar_chart(rate: float, width: int = 20, invert: bool = False) -> str:
    """Visual bar — if invert=True, fill represents SHORT win rate (price fell)."""
    filled = round(rate * width)
    empty  = width - filled
    fill_char = "▓" if invert else "█"
    return f"[{fill_char * filled}{'░' * empty}]"


# ─────────────────────────────────────────────────────────────────────────────
# Bear state bucketing (parallel to bull bucketing in run_decide.py)
# ─────────────────────────────────────────────────────────────────────────────

def _regime_bucket(row: pd.Series) -> str:
    if bool(row.get("bull_regime", False)):
        return "bull"
    if bool(row.get("bear_regime", False)):
        return "bear"
    return "neutral"

def _bear_score_bucket(score: int) -> str:
    """Bucket based on bear_score (mirrored from bull score thresholds)."""
    if score >= 9:  return "high"
    if score >= 6:  return "moderate"
    return "low"

def _adx_bucket(adx: float, threshold: float) -> str:
    return "trending" if adx >= threshold else "choppy"

def _macd_bucket(row: pd.Series) -> str:
    """'negative' = MACD below signal = bearish momentum."""
    return "positive" if float(row.get("macd_line", 0)) > float(row.get("signal_line", 0)) else "negative"

def get_bear_state_bucket(row: pd.Series, adx_threshold: float) -> dict:
    return {
        "regime":     _regime_bucket(row),
        "bear_score": _bear_score_bucket(int(row.get("bear_score", 0))),
        "adx":        _adx_bucket(float(row.get("adx", 0)), adx_threshold),
        "macd":       _macd_bucket(row),
    }

def bear_bucket_label(b: dict) -> str:
    return (f"regime={b['regime']} / bear_score={b['bear_score']} / "
            f"ADX={b['adx']} / MACD={b['macd']}")


def _rsi_short_zone(rsi: float) -> tuple[str, str]:
    """Returns (display label, short implication) based on RSI level."""
    if rsi > 70:
        return "🔴 overbought",        "AVOID short — bounce risk is extreme"
    if rsi > 60:
        return "🎯 rolling over",      "BEST entry zone — momentum just turning bearish"
    if rsi > 40:
        return "🟡 neutral/declining", "OK — established downtrend, not yet exhausted"
    if rsi > 30:
        return "⚠  approaching OS",   "CAUTION — reduce new short exposure"
    return     "🚨 oversold",          "AVOID new shorts / consider covering partial"


# ─────────────────────────────────────────────────────────────────────────────
# Short base rate computation
# Historical instances in same bear state → what % of the time did price FALL?
# ─────────────────────────────────────────────────────────────────────────────

def compute_short_base_rates(
    df:             pd.DataFrame,
    current_bucket: dict,
    horizons:       list[int],
    adx_threshold:  float,
    exact_match:    bool = True,
) -> dict:
    """
    Find all historical bars in the same BEAR state bucket and compute
    forward returns at each horizon.

    Returns the raw return distribution — the caller flips perspective for shorts:
      short_win_rate = fraction of instances where fwd_return < 0 (price fell)
    """
    closes  = df["close"].values
    max_h   = max(horizons)
    valid   = df.iloc[:-max_h] if len(df) > max_h else df.iloc[:0]

    match_mask = pd.Series(True, index=valid.index)
    for i, (ts, row) in enumerate(valid.iterrows()):
        b = get_bear_state_bucket(row, adx_threshold)
        if exact_match:
            match_mask.iloc[i] = (b == current_bucket)
        else:
            # loose: regime + bear_score only
            match_mask.iloc[i] = (
                b["regime"]     == current_bucket["regime"]
                and b["bear_score"] == current_bucket["bear_score"]
            )

    matching_idx = np.where(match_mask.values)[0]
    n_instances  = len(matching_idx)
    results = {"_n_instances": n_instances, "_exact": exact_match}

    for h in horizons:
        fwd_returns = []
        for idx in matching_idx:
            future_idx = idx + h
            if future_idx < len(closes):
                fwd_ret = (closes[future_idx] - closes[idx]) / closes[idx]
                fwd_returns.append(fwd_ret)

        if not fwd_returns:
            results[h] = {"n": 0}
            continue

        arr = np.array(fwd_returns)
        results[h] = {
            "n":              len(arr),
            "short_win_rate": float(np.mean(arr < 0)),   # price FELL = short wins
            "long_win_rate":  float(np.mean(arr > 0)),   # price rose = short loses
            "median":         float(np.median(arr)),      # negative = favorable for short
            "mean":           float(np.mean(arr)),
            "p25":            float(np.percentile(arr, 25)),
            "p75":            float(np.percentile(arr, 75)),
            "worst":          float(np.max(arr)),         # worst for short = biggest RISE
            "best":           float(np.min(arr)),         # best for short = biggest FALL
            "std":            float(np.std(arr)),
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Confidence grading (same as long engine)
# ─────────────────────────────────────────────────────────────────────────────

def _confidence_grade(n: int, spread_pct: float, exact_match: bool) -> tuple[str, str]:
    score = 0
    if n >= 60:    score += 3
    elif n >= 30:  score += 2
    elif n >= 15:  score += 1
    if spread_pct < 0.25:   score += 3
    elif spread_pct < 0.40: score += 2
    elif spread_pct < 0.60: score += 1
    if exact_match: score += 1
    if score >= 6:
        return "A", f"High confidence  (n={n}, spread={pct(spread_pct)})"
    if score >= 4:
        return "B", f"Moderate confidence  (n={n}, spread={pct(spread_pct)})"
    if score >= 2:
        return "C", f"Low confidence — wide variance  (n={n}, spread={pct(spread_pct)})"
    return "D", f"Very low confidence — treat as noise  (n={n}, spread={pct(spread_pct)})"


# ─────────────────────────────────────────────────────────────────────────────
# Earnings date (same as long engine)
# ─────────────────────────────────────────────────────────────────────────────

def get_earnings_info(ticker: str) -> dict | None:
    try:
        t   = yf.Ticker(ticker)
        cal = t.calendar
        dates = []
        if isinstance(cal, dict):
            raw = cal.get("Earnings Date", [])
            dates = list(raw) if raw is not None else []
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
        next_e       = min(upcoming)
        cal_days     = (next_e - now).days
        trading_days = max(0, round(cal_days * 5 / 7))
        return {"date": next_e, "cal_days": cal_days, "trading_days": trading_days}
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Macro overlay — for shorts we WANT SPY/QQQ in bear or neutral
# ─────────────────────────────────────────────────────────────────────────────

def get_macro_context() -> dict[str, dict]:
    result: dict[str, dict] = {}
    strat  = GrowthSignalBot()
    for mkt_ticker, label in [("SPY", "S&P 500 (SPY)"), ("QQQ", "Nasdaq 100 (QQQ)")]:
        try:
            df = fetch_ohlcv(mkt_ticker, period="1y", interval="1d")
            if df is None or len(df) < 220:
                continue
            df_p  = strat.prepare(df.copy())
            last  = df_p.iloc[-1]
            close = float(last["close"])
            result[mkt_ticker] = {
                "label":         label,
                "close":         close,
                "regime":        _regime_bucket(last),
                "bear_score":    int(last.get("bear_score", 0)),
                "bull_score":    int(last.get("bull_score", 0)),
                "adx":           float(last.get("adx", 0)),
                "sma200":        float(last["sma200"]),
                "sma200_rising": float(last["sma200"]) > float(df_p.iloc[-2]["sma200"]),
                "ret_1m":        (close - float(df_p.iloc[-21]["close"])) / float(df_p.iloc[-21]["close"]) if len(df_p) > 21 else None,
                "ret_3m":        (close - float(df_p.iloc[-63]["close"])) / float(df_p.iloc[-63]["close"]) if len(df_p) > 63 else None,
                "macd_positive": float(last.get("macd_line", 0)) > float(last.get("signal_line", 0)),
                "rsi":           float(last.get("rsi", 0)),
            }
        except Exception:
            continue
    return result


def _macro_regime_line(info: dict) -> str:
    regime_icons = {"bull": "🐂 BULL", "bear": "🐻 BEAR", "neutral": "🔀 NEUTRAL"}
    regime = regime_icons.get(info["regime"], info["regime"])
    sma200_arrow = "↑" if info["sma200_rising"] else "↓"
    ret_1m = pct(info["ret_1m"]) if info["ret_1m"] is not None else "n/a"
    ret_3m = pct(info["ret_3m"]) if info["ret_3m"] is not None else "n/a"
    return (f"  {info['label']:<20}  {regime:<16}  SMA200 {sma200_arrow}   "
            f"1mo {ret_1m:>7}   3mo {ret_3m:>7}   ADX {info['adx']:.0f}  RSI {info['rsi']:.0f}")


def _macro_short_note(macro: dict[str, dict]) -> str:
    """Plain-language note on whether macro regime is SHORT-FRIENDLY."""
    bear_count = sum(1 for v in macro.values() if v["regime"] == "bear")
    bull_count = sum(1 for v in macro.values() if v["regime"] == "bull")
    if bear_count == len(macro):
        return "🔴 Both SPY and QQQ in bear regime — strong macro tailwind for shorts"
    if bull_count == len(macro):
        return "🟢 Both SPY and QQQ in BULL regime — macro is a HEADWIND for shorts"
    if bear_count > 0:
        return "🟡 Mixed market regime — partial short tailwind, reduced conviction"
    return "🔀 Market transitioning — neutral macro context for shorts"


# ─────────────────────────────────────────────────────────────────────────────
# Main report
# ─────────────────────────────────────────────────────────────────────────────

def run_short_decision_report(ticker: str, horizon_td: int = 21) -> None:
    """Full short decision engine report for a single ticker."""

    print(_hdr(f"  SHORT DECISION ENGINE  ·  {ticker}  ·  {horizon_td}td horizon (~{horizon_td//5}wk)"))

    # ── Fetch data ─────────────────────────────────────────────────────────
    print(f"\n  Fetching data …", end="", flush=True)
    df_long = fetch_ohlcv(ticker, period="max", interval="1d")
    if df_long is None or len(df_long) < 300:
        df_long = fetch_ohlcv(ticker, period="5y", interval="1d")
    if df_long is None or len(df_long) < 100:
        print(f"\n  ⚠ Not enough data for {ticker}")
        return
    print(f" {len(df_long)} daily bars ({bar_str(df_long.index[0])} → {bar_str(df_long.index[-1])})")

    print(f"  Fetching macro (SPY/QQQ) + earnings …", end="", flush=True)
    macro    = get_macro_context()
    earnings = get_earnings_info(ticker)
    print(" done")

    # ── Run strategy indicators ────────────────────────────────────────────
    strat = GrowthSignalBot()
    df    = strat.prepare(df_long.copy())

    last       = df.iloc[-1]
    last_close = float(last["close"])
    last_ts    = df.index[-1]

    bear_bucket = get_bear_state_bucket(last, strat.adx_threshold)

    # ── ─────────────────────────────────────────────────────────────────────
    # SECTION 0: Earnings & Market Context
    # ── ─────────────────────────────────────────────────────────────────────
    print(_h2("0 · EARNINGS & MARKET CONTEXT"))

    if earnings:
        td  = earnings["trading_days"]
        cal = earnings["cal_days"]
        dt  = bar_str(earnings["date"])
        if td <= 3:
            flag = "🚨 IMMINENT"
        elif td <= 10:
            flag = "⚠  UPCOMING"
        else:
            flag = "📅"
        warn = " — ⚠ AVOID NEW SHORTS NEAR EARNINGS (gap-up risk)" if td <= 10 else ""
        print(f"\n  {flag}  Earnings: {dt}  "
              f"({cal} cal days / ~{td} trading days away){warn}")
    else:
        print(f"\n  📅  Earnings: not found / not scheduled in near term")

    print()
    if macro:
        for info in macro.values():
            print(_macro_regime_line(info))
        print(f"\n  {_macro_short_note(macro)}")

        # Relative performance vs SPY (underperformance = favorable for short thesis)
        spy = macro.get("SPY")
        if spy:
            for bars, label in [(21, "1mo"), (63, "3mo")]:
                if len(df) > bars:
                    ticker_ret = (last_close - float(df.iloc[-bars]["close"])) / float(df.iloc[-bars]["close"])
                    spy_ret    = spy.get("ret_1m" if bars == 21 else "ret_3m")
                    if spy_ret is not None:
                        alpha = ticker_ret - spy_ret
                        vs    = "underperforming SPY" if alpha < 0 else "outperforming SPY"
                        note  = " ← favorable for short thesis" if alpha < 0 else " ← headwind for short thesis"
                        print(f"  {ticker} vs SPY {label}: {pct(ticker_ret)} vs {pct(spy_ret)} "
                              f"({vs} by {pct(abs(alpha))}){note}")
    else:
        print("  ⚠ Could not fetch macro data")

    # ── ─────────────────────────────────────────────────────────────────────
    # SECTION 1: Current State (Short Perspective)
    # ── ─────────────────────────────────────────────────────────────────────
    print(_h2("1 · CURRENT STATE  (short perspective)"))

    bar_chg = (float(last["close"]) - float(last["open"])) / float(last["open"])
    print(f"  Last bar : {bar_str(last_ts)}   Close: {dol(last_close).strip()}  "
          f"({pct(bar_chg)} on bar)")

    for bars, label in [(5, "1wk"), (21, "1mo"), (63, "3mo")]:
        if len(df) > bars:
            past = float(df.iloc[-bars]["close"])
            chg  = (last_close - past) / past
            note = " ← downtrend" if chg < -0.05 else (" ← flat" if abs(chg) < 0.03 else " ← uptrend")
            print(f"  {label} price change: {pct(chg):>8}   "
                  f"({dol(past).strip()} → {dol(last_close).strip()}){note}")
    print()

    # Regime — for shorts, CONFIRMED bear regime (not just transitioning) is ideal
    sma200      = float(last["sma200"])
    sma200_dir  = "↑ rising" if float(last["sma200"]) > float(df.iloc[-2]["sma200"]) else "↓ falling"
    dist_sma200 = (last_close - sma200) / sma200
    regime_sym  = {"bull": "🐂 BULL (⚠ AVOID SHORT)", "bear": "🐻 BEAR (✅ short-friendly)",
                   "neutral": "🔀 NEUTRAL/TRANSITIONING (⚠ reduced conviction)"}
    print(f"  Regime    : {regime_sym[bear_bucket['regime']]}")
    print(f"  SMA200    : {dol(sma200).strip()}  {sma200_dir}   "
          f"Price is {pct(dist_sma200)} {'above' if dist_sma200 > 0 else 'below'}")

    sma50 = float(last["sma50"])
    dist_sma50 = (last_close - sma50) / sma50
    below_both = last_close < sma200 and last_close < sma50
    print(f"  SMA50     : {dol(sma50).strip()}   "
          f"Price is {pct(dist_sma50)} {'above' if dist_sma50 > 0 else 'below'}   "
          f"{'✅ below both SMAs' if below_both else '⚠ not below both SMAs'}")
    print()

    # Bear score
    bear_score = int(last.get("bear_score", 0))
    bull_score = int(last.get("bull_score", 0))
    bear_filled = "▓" * bear_score + "░" * (16 - bear_score)
    bear_lbl = (
        "🔥 HIGH — strong short setup"   if bear_score >= 9
        else "🟡 MODERATE"               if bear_score >= 6
        else "❌ LOW — weak short signal"
    )
    print(f"  Bear score: {bear_score}/16  [{bear_filled}]  {bear_lbl}")
    print(f"  Bull score: {bull_score}/16  (high bull score = avoid shorting)")

    # ADX
    adx     = float(last.get("adx", 0))
    adx_ok  = adx >= strat.adx_threshold
    adx_sym = "🟢 trending" if adx_ok else "🔴 choppy"
    adx_note = "downtrend has momentum" if adx_ok else "choppy/sideways — shorts more risky"
    print(f"  ADX       : {adx:.1f}  [{adx_sym}]  ({adx_note})")

    # RSI — most important for short entry timing
    rsi = float(last["rsi"])
    rsi_zone, rsi_note = _rsi_short_zone(rsi)
    print(f"  RSI       : {rsi:.1f}  [{rsi_zone}]")
    print(f"              → {rsi_note}")

    # MACD
    macd_l = float(last["macd_line"])
    sig_l  = float(last["signal_line"])
    hist   = float(last.get("macd_hist", 0))
    macd_bearish = macd_l < sig_l
    cross_str = ""
    if last.get("macd_crossunder"):
        cross_str = "  ⚡ CROSSED DOWN this bar — fresh entry signal"
    elif last.get("macd_cross"):
        cross_str = "  ⚡ CROSSED UP this bar — weakening short thesis"
    macd_sym = "🔴 MACD below signal (bearish)" if macd_bearish else "🟢 MACD above signal (bullish)"
    print(f"  MACD      : {macd_sym}  (line {macd_l:.2f} / sig {sig_l:.2f} / hist {hist:.2f}){cross_str}")
    print()

    # Active signals on last bar
    active = []
    for col, label in [
        ("sell_signal",  "🔻 SELL"),
        ("bear_div",     "✕ Bear Div"),
        ("buy_signal",   "🔺 BUY (⚠ counter-signal)"),
        ("bounce_signal","⬤ BOUNCE (⚠ counter-signal)"),
        ("bull_div",     "✕ Bull Div (⚠ cover risk)"),
    ]:
        if last.get(col):
            active.append(label)
    print(f"  Signal now: {', '.join(active) if active else '— no signal firing'}")

    # Recent bearish signal events
    recent_bear = []
    recent_bull = []
    for col, label, side in [
        ("sell_signal",  "🔻 SELL",          "bear"),
        ("bear_div",     "✕ Bear Div",        "bear"),
        ("buy_signal",   "🔺 BUY",            "bull"),
        ("bounce_signal","⬤ BOUNCE",          "bull"),
        ("bull_div",     "✕ Bull Div",        "bull"),
        ("vwap_bounce",  "◆ VWAP BOUNCE",     "bull"),
    ]:
        if col not in df.columns:
            continue
        fired = df[df[col] == True]
        for ts, row in fired.iterrows():
            entry = (ts, label, float(row["close"]))
            if side == "bear":
                recent_bear.append(entry)
            else:
                recent_bull.append(entry)

    recent_bear.sort(key=lambda x: x[0])
    recent_bull.sort(key=lambda x: x[0])

    if recent_bear:
        print(f"  Recent BEAR signals (last 5):")
        for ts, lbl, price in recent_bear[-5:]:
            marker = " ◀ latest" if ts == recent_bear[-1][0] else ""
            print(f"       {bar_str(ts)}  {lbl:<18}  {dol(price).strip()}{marker}")
    if recent_bull:
        print(f"  Recent BULL signals (last 3) — counter-signals to monitor:")
        for ts, lbl, price in recent_bull[-3:]:
            print(f"       {bar_str(ts)}  {lbl:<18}  {dol(price).strip()}  (⚠ watch for reversal)")
    print()

    # ── ─────────────────────────────────────────────────────────────────────
    # SECTION 2: Short Base Rates
    # What % of the time did price FALL in this same bear state?
    # ── ─────────────────────────────────────────────────────────────────────
    print(_h2(f"2 · SHORT BASE RATES  (state: {bear_bucket_label(bear_bucket)})"))
    print(f"  'Short WR' = fraction of past instances where price FELL (short won)\n")

    horizons = [5, 10, 21, 42, 63]
    h_labels = {5: "1wk", 10: "2wk", 21: "1mo", 42: "2mo", 63: "3mo"}

    stats = compute_short_base_rates(df, bear_bucket, horizons, strat.adx_threshold, exact_match=True)
    n_exact = stats["_n_instances"]

    if n_exact < 10:
        print(f"  ⚠ Only {n_exact} exact-match instances — using regime+bear_score (looser match)")
        stats = compute_short_base_rates(df, bear_bucket, horizons, strat.adx_threshold, exact_match=False)
        n_used = stats["_n_instances"]
        match_desc = f"regime={bear_bucket['regime']} / bear_score={bear_bucket['bear_score']}"
    else:
        n_used = n_exact
        match_desc = bear_bucket_label(bear_bucket)

    print(f"  Matched {n_used} historical instances for: {match_desc}")
    print(f"  ({bar_str(df_long.index[0])} → {bar_str(df_long.index[-1])})\n")

    # Table header
    print(f"  {'Horizon':<8}  {'N':>4}  {'ShortWR':>8}  {'Median':>8}  "
          f"{'Best↓':>8}  {'Worst↑':>8}  {'P25':>8}  {'P75':>8}")
    print(_sep())

    for h in horizons:
        s = stats.get(h, {})
        if not s or s.get("n", 0) == 0:
            print(f"  {h_labels[h]:<8}  {'—':>4}")
            continue
        short_wr_bar = _bar_chart(s["short_win_rate"], 10, invert=True)
        fav = "✅" if s["short_win_rate"] > 0.55 else ("🟡" if s["short_win_rate"] > 0.45 else "🔴")
        print(f"  {h_labels[h]:<8}  {s['n']:>4}  "
              f"{pct(s['short_win_rate']):>8}  {pct(s['median']):>8}  "
              f"{pct(s['best']):>8}  {pct(s['worst']):>8}  "
              f"{pct(s['p25']):>8}  {pct(s['p75']):>8}  {fav}")

    # Primary horizon callout
    s_h = stats.get(horizon_td, stats.get(21, {}))
    if s_h and s_h.get("n", 0) > 0:
        print()
        wr_bar  = _bar_chart(s_h["short_win_rate"], 20, invert=True)
        spread  = abs(s_h["p75"] - s_h["p25"])
        grade, grade_desc = _confidence_grade(
            n           = s_h["n"],
            spread_pct  = spread,
            exact_match = stats.get("_exact", True),
        )
        short_favorable = s_h["short_win_rate"] > 0.55
        bias_lbl = "✅ FAVORABLE for short" if short_favorable else \
                   ("🟡 MIXED" if s_h["short_win_rate"] > 0.45 else "🔴 UNFAVORABLE for short")
        print(f"  ★  {horizon_td}td base rate:  "
              f"Short WR {pct(s_h['short_win_rate'])}  {wr_bar}  [{bias_lbl}]")
        print(f"     Median price change: {pct(s_h['median'])}  "
              f"(P25: {pct(s_h['p25'])} / P75: {pct(s_h['p75'])})")
        print(f"     Best fall (short):   {pct(s_h['best'])}   "
              f"Worst rise (short loss): {pct(s_h['worst'])}")
        print(f"     Confidence:          Grade {grade} — {grade_desc}")

    # ── ─────────────────────────────────────────────────────────────────────
    # SECTION 3: Key Levels  (Resistance ABOVE = stops, Support BELOW = targets)
    # ── ─────────────────────────────────────────────────────────────────────
    print(_h2("3 · KEY LEVELS  (resistance above ↑ = stop zones / support below ↓ = targets)"))

    ema20  = float(last["ema20"])  if "ema20"  in df.columns else None
    vwap_w = float(last["vwap_w"]) if "vwap_w" in df.columns else None
    vwap_m = float(last["vwap_m"]) if "vwap_m" in df.columns else None
    poc    = float(last["poc"])    if "poc"    in df.columns else None
    recent_52w = df.iloc[-252:]["close"] if len(df) >= 252 else df["close"]
    high_52w   = float(recent_52w.max())
    low_52w    = float(recent_52w.min())

    # Sort all levels relative to current price
    all_levels = [
        ("52-week High",    high_52w,  "strong resistance"),
        ("SMA200",          sma200,    "regime line — now RESISTANCE if below"),
        ("Weekly VWAP",     vwap_w,    "VWAP resistance"),
        ("SMA50",           sma50,     "medium resistance"),
        ("Monthly VWAP",    vwap_m,    "secondary VWAP"),
        ("EMA20",           ema20,     "short-term resistance"),
        ("POC (vol. node)", poc,       "vol-based S/R"),
        ("Current Price",   last_close,""),
        ("52-week Low",     low_52w,   "major support — cover target"),
    ]
    all_levels = [(n, v, note) for n, v, note in all_levels if v is not None]
    all_levels.sort(key=lambda x: x[1], reverse=True)  # high to low

    print(f"  {'Level':<22}  {'Price':>10}  {'Dist':>8}  Notes")
    print(_sep())

    for name, val, note in all_levels:
        dist  = (val - last_close) / last_close   # positive = ABOVE price
        side  = "↑ ABOVE" if dist > 0.001 else ("↓ BELOW" if dist < -0.001 else "=  AT")
        role  = "stop-loss zone" if dist > 0.001 else ("cover target" if dist < -0.001 else "")
        marker = " ◀ CURRENT" if name == "Current Price" else ""
        tag   = f"{note}  [{role}]" if role else note
        print(f"  {name:<22}  {dol(val)}  {side} {abs(dist)*100:.1f}%   {tag}{marker}")

    # Stop-loss guidance
    print()
    nearest_resistance = None
    for name, val, _ in all_levels:
        if val > last_close * 1.005:   # at least 0.5% above
            nearest_resistance = (name, val)
            break
    if nearest_resistance:
        stop_dist = (nearest_resistance[1] - last_close) / last_close
        print(f"  📌 Nearest resistance (stop-loss guide): "
              f"{nearest_resistance[0]} at {dol(nearest_resistance[1]).strip()}  "
              f"(+{stop_dist * 100:.1f}% from current)")
        print(f"     Suggest stop above {nearest_resistance[0]} to avoid noise")

    # Fibonacci context
    if "near_fib" in df.columns:
        near = bool(last["near_fib"])
        print(f"  {'🎯 Price is near a Fibonacci level' if near else '   Price not near a Fib level'}")

    # ── ─────────────────────────────────────────────────────────────────────
    # SECTION 4: Short Decision Checklist
    # ── ─────────────────────────────────────────────────────────────────────
    print(_h2("4 · SHORT DECISION CHECKLIST"))

    def chk(c: bool) -> str: return "✅" if c else "❌"

    # ── ENTER SHORT conditions ───────────────────────────────────────────
    print(f"\n  ENTER SHORT — all should be ✅ for highest-conviction setup:")
    c_regime       = bear_bucket["regime"] == "bear"
    c_sma200_fall  = sma200_dir == "↓ falling"
    c_below_both   = below_both
    c_bear_score   = bear_score >= 9
    c_adx          = adx_ok
    c_rsi_not_os   = rsi > 30
    c_rsi_good     = rsi > 40
    c_macd_bear    = not macd_bearish is False and macd_l < sig_l
    c_no_earnings  = earnings is None or earnings["trading_days"] > 10
    c_no_bull_div  = not bool(last.get("bull_div", False))

    print(f"  {chk(c_regime      )}  Bear regime confirmed (price < falling SMA200)")
    print(f"  {chk(c_sma200_fall )}  SMA200 is falling (not just flat/rising)")
    print(f"  {chk(c_below_both  )}  Price below BOTH SMA200 and SMA50")
    print(f"  {chk(c_bear_score  )}  Bear score ≥ 9/16  (currently {bear_score}/16)")
    print(f"  {chk(c_adx         )}  ADX ≥ {strat.adx_threshold:.0f} — downtrend has momentum (currently {adx:.1f})")
    print(f"  {chk(c_rsi_not_os  )}  RSI > 30 — not oversold (currently {rsi:.1f})")
    print(f"  {chk(c_rsi_good    )}  RSI > 40 — ideal short zone (not late-stage exhaustion)")
    print(f"  {chk(c_macd_bear   )}  MACD below signal line (bearish momentum)")
    print(f"  {chk(c_no_earnings )}  No earnings within 10 trading days (gap-up risk)")
    print(f"  {chk(c_no_bull_div )}  No bull divergence firing (reversal signal)")

    enter_score = sum([c_regime, c_sma200_fall, c_below_both, c_bear_score,
                       c_adx, c_rsi_not_os, c_macd_bear, c_no_earnings, c_no_bull_div])
    enter_max   = 9
    enter_lbl   = (
        "✅ HIGH CONVICTION — conditions strongly support short entry"    if enter_score >= 7
        else "🟡 MODERATE — some conditions missing, size down"          if enter_score >= 5
        else "🔴 LOW — do not enter short, most gates failing"
    )
    print(f"  → Short entry readiness: {enter_score}/{enter_max}  [{enter_lbl}]")

    # ── HOLD SHORT conditions ─────────────────────────────────────────────
    print(f"\n  HOLD EXISTING SHORT — watch for these exit triggers:")
    h_rsi_warn  = rsi < 35
    h_bull_div  = bool(last.get("bull_div", False))
    h_macd_up   = bool(last.get("macd_cross", False))
    h_near_low  = last_close < low_52w * 1.05
    h_regime_wk = sma200_dir == "↑ rising" and bear_bucket["regime"] != "bull"

    cover_triggers = []
    if h_rsi_warn:  cover_triggers.append(f"RSI {rsi:.0f} < 35 — approaching oversold, consider partial cover")
    if h_bull_div:  cover_triggers.append("Bull divergence active — potential bounce, tighten stop")
    if h_macd_up:   cover_triggers.append("MACD crossed UP this bar — momentum shift, exit short")
    if h_near_low:  cover_triggers.append(f"Price near 52w low ({dol(low_52w).strip()}) — major support, book profits")
    if h_regime_wk: cover_triggers.append("SMA200 flattening/rising — bear regime may be weakening")

    if cover_triggers:
        print(f"  ⚠ ACTIVE COVER TRIGGERS:")
        for t in cover_triggers:
            print(f"     🔸 {t}")
    else:
        print(f"  ◆  No immediate cover triggers — short thesis intact")
        print(f"  ◆  Continue to monitor RSI (cover risk rises below 35)")
        print(f"  ◆  Watch for MACD crossover or bull divergence as exit signals")

    # ── AVOID / ABORT conditions ──────────────────────────────────────────
    print(f"\n  ⛔ ABORT / DO NOT SHORT if any of these:")
    abort_conditions = []
    if bear_bucket["regime"] == "bull":
        abort_conditions.append("BULL REGIME — SMA200 rising and price above it (NEVER short into bull trend)")
    if bull_score >= 11:
        abort_conditions.append(f"STRONG bull score ({bull_score}/16) — long momentum strongly favors longs")
    if rsi < 25:
        abort_conditions.append(f"RSI {rsi:.0f} — deep oversold, high bounce risk on any catalyst")
    if earnings and earnings["trading_days"] <= 5:
        abort_conditions.append(f"Earnings in {earnings['trading_days']}td — gap-up risk makes shorts highly dangerous")

    if abort_conditions:
        for a in abort_conditions:
            print(f"  🚫 {a}")
    else:
        print(f"  ◆  No hard abort conditions detected")

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────
    print(_h2("SUMMARY"))

    s_h = stats.get(horizon_td, stats.get(21, {}))
    regime_display = {
        "bull":    "🐂 BULL (⚠ unfavorable for shorts)",
        "bear":    "🐻 BEAR (✅ short-friendly)",
        "neutral": "🔀 NEUTRAL/TRANSITIONING"
    }
    print(f"\n  Symbol       : {ticker}   ${last_close:,.2f}  ({bar_str(last_ts)})")
    print(f"  Regime       : {regime_display[bear_bucket['regime']]}")
    print(f"  Bear score   : {bear_score}/16  (RSI {rsi:.0f}  ADX {adx:.0f})")
    print(f"  MACD         : {'🔴 below signal (bearish)' if macd_l < sig_l else '🟢 above signal (bullish)'}")
    print(f"  Short ready  : {enter_score}/{enter_max} conditions met  [{enter_lbl}]")

    if s_h and s_h.get("n", 0) > 0:
        spread = abs(s_h["p75"] - s_h["p25"])
        grade, _ = _confidence_grade(s_h["n"], spread, stats.get("_exact", True))
        bias = "price fell" if s_h["short_win_rate"] > 0.5 else "price rose"
        print(f"  {horizon_td}td history  : {pct(s_h['short_win_rate'])} of time {bias}  "
              f"(n={s_h['n']}, confidence: {grade})")

    if macro:
        print(f"  Market       : {_macro_short_note(macro)}")
    if earnings:
        td = earnings["trading_days"]
        warn = "  ⚠ AVOID new shorts" if td <= 10 else ""
        print(f"  Earnings in  : ~{td} trading days ({bar_str(earnings['date'])}){warn}")

    if cover_triggers:
        print(f"  Cover watch  : {len(cover_triggers)} trigger(s) active — review above")

    print()
    print(f"  ⚠  This is indicator data only — not investment advice.")
    print(f"  ⚠  Shorting carries unlimited theoretical loss risk.")
    print(f"  ⚠  Always set hard stop-losses. Past base rates ≠ future performance.")
    print(_sep("═"))
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = sys.argv[1:]
    if not args:
        tickers = ["SNAP"]
        horizon = 21
    else:
        tickers = [a.upper() for a in args if not a.startswith("--")]
        horizon = 21
        if "--horizon" in args:
            idx = args.index("--horizon")
            if idx + 1 < len(args):
                horizon = int(args[idx + 1])

    for ticker in tickers:
        run_short_decision_report(ticker, horizon_td=horizon)


if __name__ == "__main__":
    main()
