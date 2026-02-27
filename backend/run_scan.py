#!/usr/bin/env python3
"""
run_current_state.py  —  Current signal state scanner

Fetches recent data, runs all indicators, and reports the current state
of the Growth Signal Bot for any symbol. Useful for "what is the bot
reading right now?" — not investment advice, just signal data.

Usage:
    python run_current_state.py META
    python run_current_state.py GOOG --tf 1h
    python run_current_state.py META GOOG NVDA   (multi-symbol)
"""
from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from backtester.data   import fetch_ohlcv
from backtester.engine import BacktestEngine
from strategies.growth_signal_bot import GrowthSignalBot


# ── Helpers ───────────────────────────────────────────────────────────────────

def bar_str(ts) -> str:
    """Format a bar timestamp nicely."""
    try:
        return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(ts)

def pct(v: float) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v * 100:.2f}%"

def _signal_history(df: pd.DataFrame, n: int = 10) -> list[dict]:
    """Return last n signal events from a prepared dataframe."""
    events = []
    for col, label in [
        ("buy_signal",     "🔺 STRONG BUY" if False else "🔺 BUY"),
        ("bounce_signal",  "⬤  BOUNCE"),
        ("vwap_bounce",    "◆  VWAP BOUNCE"),
        ("sell_signal",    "🔻 SELL"),
        ("bull_div",       "✕  Bull Div"),
        ("bear_div",       "✕  Bear Div"),
    ]:
        if col not in df.columns:
            continue
        fired = df[df[col] == True]
        for ts, row in fired.iterrows():
            # Annotate BUY as Strong vs Moderate
            lbl = label
            if col == "buy_signal":
                score = row.get("bull_score", 0)
                strong_thresh = 11
                lbl = "🔥 STRONG BUY" if score >= strong_thresh else "🔺 BUY"
            events.append({
                "ts":     ts,
                "label":  lbl,
                "price":  row.get("close", float("nan")),
                "score":  row.get("bull_score" if "buy" in col or "bounce" in col else "bear_score", float("nan")),
            })
    events.sort(key=lambda x: x["ts"])
    return events[-n:]


def scan_symbol(ticker: str, tf: str = "1d") -> None:
    period = "1y" if tf == "1d" else "2y"
    print(f"\n{'═' * 64}")
    print(f"  {ticker}  ·  {tf.upper()}  ·  fetching {period} of data …")
    print(f"{'═' * 64}")

    df_raw = fetch_ohlcv(ticker, period=period, interval=tf)
    if df_raw is None or len(df_raw) < 50:
        print(f"  ⚠ Not enough data for {ticker}")
        return

    # ── Run strategy prepare() ────────────────────────────────────────────────
    if tf == "1h":
        strat = GrowthSignalBot.for_1h()
    else:
        strat = GrowthSignalBot()

    df = strat.prepare(df_raw.copy())

    last        = df.iloc[-1]
    prev        = df.iloc[-2]
    last_ts     = df.index[-1]
    last_close  = float(last["close"])
    last_open   = float(last["open"])
    bar_chg     = (last_close - last_open) / last_open

    # ── Current price & bar ───────────────────────────────────────────────────
    print(f"\n  📅  Last bar : {bar_str(last_ts)}")
    print(f"  💵  Close    : ${last_close:,.2f}  ({pct(bar_chg)} on bar)")
    print()

    # ── Regime ────────────────────────────────────────────────────────────────
    regime = (
        "🐂 BULL REGIME  (above rising SMA200)" if last["bull_regime"]
        else "🐻 BEAR REGIME  (below falling SMA200)" if last["bear_regime"]
        else "🔀 NEUTRAL / TRANSITIONING"
    )
    sma200      = float(last["sma200"])
    sma200_chg  = float(last["sma200"]) - float(prev["sma200"])
    rising      = "↑ rising" if sma200_chg > 0 else "↓ falling"
    print(f"  {regime}")
    print(f"     SMA200: ${sma200:,.2f}  ({rising})  "
          f"Price vs SMA200: {pct((last_close - sma200) / sma200)}")
    if "sma50" in df.columns:
        sma50 = float(last["sma50"])
        print(f"     SMA50:  ${sma50:,.2f}  "
              f"Price vs SMA50: {pct((last_close - sma50) / sma50)}")
    print()

    # ── Indicators ────────────────────────────────────────────────────────────
    rsi = float(last["rsi"])
    rsi_zone = (
        "🟢 bull zone" if strat.rsi_bull_min <= rsi <= strat.rsi_bull_max
        else "🔴 overbought" if rsi > strat.rsi_bull_max
        else "🟡 below bull zone"
    )
    print(f"  📊  RSI({strat.rsi_len}):  {rsi:.1f}  [{rsi_zone}]  "
          f"(bull zone: {strat.rsi_bull_min}–{strat.rsi_bull_max})")

    macd_l = float(last["macd_line"])
    sig_l  = float(last["signal_line"])
    hist   = float(last["macd_hist"])
    macd_state = "🟢 MACD above signal" if macd_l > sig_l else "🔴 MACD below signal"
    cross_str = ""
    if last.get("macd_cross"):
        cross_str = "  ⚡ CROSSED UP this bar"
    elif last.get("macd_crossunder"):
        cross_str = "  ⚡ CROSSED DOWN this bar"
    print(f"  📊  MACD:   line {macd_l:.3f}  sig {sig_l:.3f}  hist {hist:.3f}  "
          f"[{macd_state}]{cross_str}")

    adx = float(last["adx"]) if "adx" in df.columns else float("nan")
    adx_ok = last.get("adx_ok", False)
    adx_str = f"{adx:.1f}  ({'🟢 trending' if adx_ok else '🔴 choppy — buy_signal gated'})" \
              if not pd.isna(adx) else "n/a"
    print(f"  📶  ADX:    {adx_str}  (threshold: {strat.adx_threshold})")

    if "w_rsi" in df.columns:
        w_rsi = float(last["w_rsi"])
        w_bull = bool(last.get("weekly_bull", False))
        htf_label = "Daily" if tf == "1h" else "Weekly"
        print(f"  📊  {htf_label} MTF RSI: {w_rsi:.1f}  "
              f"[{'🟢 bullish' if w_bull else '🔴 bearish'}]")

    if "vwap_w" in df.columns:
        vwap_w = float(last["vwap_w"])
        above_vw = bool(last.get("above_wvwap", False))
        vwap_label = "Daily VWAP" if tf == "1h" else "Weekly VWAP"
        print(f"  💧  {vwap_label}: ${vwap_w:,.2f}  "
              f"[{'🟢 above' if above_vw else '🔴 below'}]")
    print()

    # ── Scores ────────────────────────────────────────────────────────────────
    bull_score = int(last.get("bull_score", 0))
    bear_score = int(last.get("bear_score", 0))
    max_score  = 16

    # Build score bar
    filled = "█" * bull_score
    empty  = "░" * (max_score - bull_score)
    score_bar = f"[{filled}{empty}]"
    thresh_mod = strat.score_moderate
    thresh_str = strat.score_strong

    score_label = (
        "🔥 STRONG" if bull_score >= thresh_str
        else "✅ MODERATE" if bull_score >= thresh_mod
        else "❌ below threshold"
    )
    print(f"  🏆  Bull score : {bull_score}/{max_score}  {score_bar}  [{score_label}]")
    print(f"                   (moderate ≥{thresh_mod}, strong ≥{thresh_str})")

    # Score breakdown
    score_parts = []
    for col, name in [
        ("s_regime",   "Regime"), ("s_sma50", "SMA50"), ("s_rsi", "RSI"),
        ("s_macd_pos", "MACD+"),  ("s_macd_cross", "MACDx"), ("s_vol", "Vol"),
        ("s_vwap_w",   "VWAP_p"), ("s_vwap_m", "VWAP_s"), ("s_consec", "Consec"),
        ("s_weekly_tf","MTF"),    ("s_poc",  "POC"),
        ("s_adx",      "ADX"),    ("s_fib",  "Fib"),
    ]:
        if col in df.columns:
            v = int(last[col])
            if v > 0:
                score_parts.append(f"{name}:{v}")
    print(f"                   Components: {' | '.join(score_parts) or 'none'}")
    print()

    # ── Active signals on last bar ────────────────────────────────────────────
    active = []
    if last.get("buy_signal"):
        active.append("🔥 STRONG BUY" if bull_score >= thresh_str else "🔺 BUY")
    if last.get("bounce_signal"):
        active.append("⬤  BOUNCE")
    if last.get("vwap_bounce"):
        active.append("◆  VWAP BOUNCE")
    if last.get("sell_signal"):
        active.append("🔻 SELL")
    if last.get("bull_div"):
        active.append("✕  Bull Divergence")
    if last.get("bear_div"):
        active.append("✕  Bear Divergence")

    print(f"  🚦  Signal on last bar: {', '.join(active) if active else '—  no signal'}")
    print()

    # ── Recent signal history (last 10 events) ───────────────────────────────
    events = _signal_history(df, n=12)
    print(f"  📜  Recent signal history (last {len(events)}):")
    for ev in events[-12:]:
        marker = "◀ latest" if ev["ts"] == events[-1]["ts"] else ""
        score_str = f"  score={int(ev['score'])}" if not pd.isna(ev['score']) else ""
        print(f"       {bar_str(ev['ts'])}  {ev['label']:<20}  "
              f"${float(ev['price']):,.2f}{score_str}  {marker}")
    print()

    # ── Key price levels ──────────────────────────────────────────────────────
    print(f"  📏  Key levels:")
    levels = {}
    levels["SMA200"] = float(last["sma200"])
    levels["SMA50"]  = float(last["sma50"]) if "sma50" in df.columns else None
    levels["EMA20"]  = float(last["ema20"]) if "ema20" in df.columns else None
    levels["VWAP_primary"]   = float(last["vwap_w"]) if "vwap_w" in df.columns else None
    levels["VWAP_secondary"] = float(last["vwap_m"]) if "vwap_m" in df.columns else None
    levels["POC"]    = float(last["poc"])   if "poc"   in df.columns else None

    for name, val in levels.items():
        if val is None:
            continue
        dist = (last_close - val) / val
        arrow = "↑" if dist > 0 else "↓"
        print(f"       {name:<16} ${val:>10,.2f}   {arrow} {abs(dist)*100:.1f}% from price")

    # Fibonacci levels
    if "near_fib" in df.columns:
        near = bool(last["near_fib"])
        fib_str = "🎯 price is near a Fib level" if near else "price not near a Fib level"
        print(f"       Fibonacci       {fib_str}")
    print()

    # ── 3-month context: what's the recent price trend ───────────────────────
    bars_3m = 63 if tf == "1d" else 63 * 7  # ~3 calendar months
    if len(df) >= bars_3m:
        price_3m_ago = float(df.iloc[-bars_3m]["close"])
        chg_3m = (last_close - price_3m_ago) / price_3m_ago
        print(f"  📈  3-month price change: {pct(chg_3m)}  "
              f"(${price_3m_ago:,.2f} → ${last_close:,.2f})")

    bars_1m = 21 if tf == "1d" else 21 * 7
    if len(df) >= bars_1m:
        price_1m_ago = float(df.iloc[-bars_1m]["close"])
        chg_1m = (last_close - price_1m_ago) / price_1m_ago
        print(f"  📈  1-month price change: {pct(chg_1m)}  "
              f"(${price_1m_ago:,.2f} → ${last_close:,.2f})")
    print()

    # ── Quick verdict ─────────────────────────────────────────────────────────
    print(f"  ── Strategy reading ─────────────────────────────────────────")
    if last["bull_regime"]:
        regime_msg = "Bull regime intact — SMA200 rising and price above it"
    elif last["bear_regime"]:
        regime_msg = "⚠ Bear regime — SMA200 falling and price below it"
    else:
        regime_msg = "⚠ Transitioning — mixed regime signals"

    print(f"     Regime  : {regime_msg}")
    print(f"     Score   : {bull_score}/16 ({score_label})")
    adx_msg = f"trending (ADX {adx:.0f} ≥ {strat.adx_threshold})" \
              if adx_ok else f"choppy (ADX {adx:.0f} < {strat.adx_threshold})"
    print(f"     Trend   : {adx_msg}")
    signal_msg = ", ".join(active) if active else "No signal firing — watching"
    print(f"     Signal  : {signal_msg}")
    print()
    print(f"  ⚠  This is indicator data only — not investment advice.")
    print(f"{'═' * 64}")


def main() -> None:
    args = sys.argv[1:]
    if not args:
        # Default: META daily + 1H
        tickers = ["META"]
        tf = "1d"
    else:
        tickers = [a.upper() for a in args if not a.startswith("--")]
        tf = "1h" if "--tf" in args and args[args.index("--tf") + 1] == "1h" else "1d"

    for ticker in tickers:
        scan_symbol(ticker, tf)
        scan_symbol(ticker, "1h" if tf == "1d" else "1d")


if __name__ == "__main__":
    main()
