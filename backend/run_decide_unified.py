#!/usr/bin/env python3
"""
run_decide_unified.py  —  Unified Long / Short Decision Engine  (CLI)

Runs BOTH bull and bear stacks, issues a single top-level verdict.
Analysis logic lives in analysis/decision_engine.py.

Usage:
    python run_decide_unified.py NFLX
    python run_decide_unified.py GOOG META GLD
    python run_decide_unified.py NFLX --horizon 42

⚠  NOT investment advice.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from analysis.decision_engine import analyze


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers (CLI only — no print in decision_engine.py)
# ─────────────────────────────────────────────────────────────────────────────

W = 68
def _sep(c="─"): return "  " + c * (W - 2)
def _hdr(t, c="═"): return f"\n  {c*(W-2)}\n  {t}\n  {c*(W-2)}"
def _h2(t): return f"\n  ── {t} {'─'*max(1,W-6-len(t))}"
def pct(v, d=1): return f"{'+'if v>=0 else''}{v*100:.{d}f}%"
def dol(v): return f"${v:,.2f}"
def chk(c): return "✅" if c else "❌"


def print_report(r: dict) -> None:
    """Render an analyze() result dict to terminal."""
    ticker     = r["ticker"]
    horizon_td = r["horizon"]

    profile_name = r.get("profile", "Large-Cap Growth")
    print(_hdr(f"  DECIDE  ·  {ticker}  [{profile_name}]  ·  {horizon_td}td horizon  (~{horizon_td//5}wk / {horizon_td//21}mo)"))

    if r.get("error"):
        print(f"\n  ⚠ {r['error']}")
        return

    verdict      = r["verdict"]
    action       = r["action"]
    long_score   = r["long_score"]
    short_score  = r["short_score"]
    regime       = r["regime"]
    rsi          = r["rsi"]
    adx          = r["adx"]
    adx_ok       = r["adx_ok"]
    macd_b       = r["macd_bull"]
    sma200       = r["sma200"]
    sma50        = r["sma50"]
    sma200_rising = r["sma200_rising"]
    bull_score   = r["bull_score"]
    bear_score   = r["bear_score"]
    earnings     = r["earnings"]
    macro        = r["macro"]
    base_rates   = r["base_rates"]
    direction    = r["base_rates_dir"]
    recent_sigs  = r["recent_signals"]
    active_sigs  = r["active_signals"]
    levels       = r["levels"]
    alpha_1m     = r["alpha_1m"]
    last_close   = r["price"]
    last_ts      = r["last_ts"]

    # ── Verdict box ───────────────────────────────────────────────────────
    print()
    print(_sep("═"))
    print(f"  VERDICT:  {verdict}")
    print()
    print(f"  {action}")
    print(_sep("═"))

    # ── Section 0: Earnings & Macro ───────────────────────────────────────
    print(_h2("0 · EARNINGS & MACRO"))
    if earnings:
        td, dt = earnings["trading_days"], earnings["date_str"]
        flag = "🚨 IMMINENT" if td <= 3 else ("⚠  UPCOMING" if td <= 10 else "📅")
        warn = "  ← ⚠ HIGH RISK" if td <= 10 else ""
        print(f"\n  {flag}  Next earnings: {dt}  (~{td} trading days away){warn}")
    else:
        print(f"\n  📅  Earnings: not found")

    if macro:
        print()
        bull_mkts = sum(1 for v in macro.values() if v["regime"] == "bull")
        bear_mkts = sum(1 for v in macro.values() if v["regime"] == "bear")
        for tkr, info in macro.items():
            reg = {"bull":"🐂","bear":"🐻","neutral":"🔀"}.get(info["regime"],"?")
            arrow = "↑" if info["sma200_rising"] else "↓"
            ret_1m = pct(info["ret_1m"]) if info.get("ret_1m") is not None else "n/a"
            print(f"  {tkr:<6} {reg} {info['regime'].upper():<8}  SMA200{arrow}  1mo {ret_1m:>7}")
        if bull_mkts == 2:
            note = "✅ Both indexes bull — tailwind for LONG, headwind for SHORT"
        elif bear_mkts == 2:
            note = "🔴 Both indexes bear — tailwind for SHORT, headwind for LONG"
        else:
            note = "🟡 Mixed macro — reduces conviction in either direction"
        print(f"\n  {note}")
        if alpha_1m is not None:
            vs = "outperforming" if alpha_1m > 0 else "underperforming"
            spy_ret = macro["SPY"].get("ret_1m", 0) or 0
            ticker_1m = alpha_1m + spy_ret
            print(f"  {ticker} 1mo vs SPY: {pct(ticker_1m)} vs {pct(spy_ret)} ({vs} by {pct(abs(alpha_1m))})")

    # ── Section 1: Dual score card ────────────────────────────────────────
    print(_h2("1 · DUAL SCORE CARD"))
    regime_icons = {"bull":"🐂 BULL", "bear":"🐻 BEAR", "neutral":"🔀 NEUTRAL/TRANSITIONING"}
    sma_dir = "↑ rising" if sma200_rising else "↓ falling"
    print(f"\n  {last_ts}  {dol(last_close)}   Regime: {regime_icons[regime]}   SMA200 {sma_dir}")
    print(f"  RSI {rsi:.0f}  |  ADX {adx:.0f} ({'✅trending' if adx_ok else '⚠choppy'})  |  "
          f"MACD {'🟢above sig' if macd_b else '🔴below sig'}")
    print()

    l_bar = "█" * long_score  + "░" * (10 - long_score)
    s_bar = "▓" * short_score + "░" * (10 - short_score)
    print(f"  LONG  score : {long_score:>2}/10  [{l_bar}]  {'HIGH' if long_score>=7 else 'MOD' if long_score>=5 else 'LOW'}")
    print(f"  SHORT score : {short_score:>2}/10  [{s_bar}]  {'HIGH' if short_score>=7 else 'MOD' if short_score>=5 else 'LOW'}")
    print()

    print(f"  Long conditions ({long_score}/10):")
    for ok, desc in r["long_conditions"]:
        print(f"    {chk(ok)}  {desc}")
    print(f"\n  Short conditions ({short_score}/10):")
    for ok, desc in r["short_conditions"]:
        print(f"    {chk(ok)}  {desc}")

    print()
    print(f"  Signals on last bar: {', '.join(active_sigs) if active_sigs else '— none firing'}")
    if recent_sigs:
        print(f"  Recent signals (last 6):")
        for ts, lbl, price in recent_sigs[-6:]:
            mrk = " ◀" if ts == recent_sigs[-1][0] else ""
            print(f"    {pd.Timestamp(ts).strftime('%Y-%m-%d')}  {lbl:<18}  {dol(price)}{mrk}")

    # ── Section 2: Base rates ─────────────────────────────────────────────
    print(_h2(f"2 · BASE RATES  ({direction.upper()} perspective)"))
    n_hist    = base_rates["_n"]
    h_lbl     = {5:"1wk", 10:"2wk", 21:"1mo", 42:"2mo"}
    horizons  = [5, 10, 21, 42]
    print(f"\n  {n_hist} historical instances in same state bucket\n")
    if n_hist > 0:
        print(f"  {'Horizon':<7}  {'N':>4}  {'WinRate':>8}  {'Median':>8}  {'P25':>8}  {'P75':>8}")
        print(_sep())
        for h in horizons:
            s = base_rates.get(h, {})
            if not s or s.get("n", 0) == 0:
                print(f"  {h_lbl[h]:<7}  {'—':>4}")
                continue
            fav = "✅" if s["win"] > 0.55 else ("🟡" if s["win"] > 0.45 else "🔴")
            print(f"  {h_lbl[h]:<7}  {s['n']:>4}  {pct(s['win']):>8}  "
                  f"{pct(s['median']):>8}  {pct(s['p25']):>8}  {pct(s['p75']):>8}  {fav}")
        s_prim = base_rates.get(horizon_td, base_rates.get(21, {}))
        if s_prim and s_prim.get("n", 0) > 0:
            lbl_dir = "price rose" if direction == "long" else "price fell"
            print(f"\n  ★  {horizon_td}td:  {pct(s_prim['win'])} of the time {lbl_dir}  "
                  f"(n={s_prim['n']}, median {pct(s_prim['median'])})")

    # ── Section 3: Key levels + action plan ───────────────────────────────
    print(_h2("3 · KEY LEVELS + ACTION PLAN"))
    print()
    for name, val in levels:
        dist  = (val - last_close) / last_close
        side  = "↑" if dist > 0.002 else ("↓" if dist < -0.002 else "=")
        role  = ""
        if direction == "long":
            role = "resistance" if dist > 0 else "support"
        else:
            role = "stop-loss" if dist > 0 else "cover target"
        marker = "  ◀ CURRENT" if abs(dist) < 0.002 else ""
        print(f"  {name:<12}  {dol(val):>12}   {side} {abs(dist)*100:>5.1f}%   {role}{marker}")

    print()
    print(f"  ACTION PLAN  ({verdict})")
    print(_sep())

    # Use precomputed values from analyze() (profile-aware)
    sl_pct   = r.get("sl_pct", 5.0)
    tp_pct   = r.get("tp_pct", 15.0)
    sl_price = r.get("sl_price")
    tp_price = r.get("tp_price")
    rr_ratio = r.get("rr_ratio")
    entry_lo = r.get("entry_lo")
    entry_hi = r.get("entry_hi")
    rr_str   = f"{rr_ratio:.1f}:1" if rr_ratio is not None else "—"

    if "STRONG LONG" in verdict or "LEAN LONG" in verdict:
        print(f"  Entry zone : {dol(entry_lo)} – {dol(entry_hi)}")
        print(f"  Stop-loss  : {dol(sl_price)}  (SL {sl_pct:.0f}% below entry)")
        print(f"  Target     : {dol(tp_price)}  (TP {tp_pct:.0f}%, R/R {rr_str})")
        print(f"  Invalidation: close below {dol(sma200)} SMA200 → exit long")
    elif "BOUNCE" in verdict:
        ema20 = r.get("ema20") or last_close * 1.04
        print(f"  Entry zone : {dol(entry_lo)} – {dol(entry_hi)}")
        print(f"  Stop-loss  : {dol(sl_price)}  (below 52w low / recent swing low)")
        print(f"  Target 1   : {dol(ema20)}  (EMA20)  R/R: {rr_str}")
        print(f"  Target 2   : {dol(r['sma50'])}  (SMA50)")
        print(f"  Size       : REDUCED — counter-trend bounce play")
    elif "SHORT" in verdict:
        print(f"  Entry zone : {dol(entry_lo)} – {dol(entry_hi)}")
        print(f"  Stop-loss  : {dol(sl_price)}  (above nearest resistance)")
        print(f"  Target     : {dol(tp_price)}  (~7% profit target)  R/R: {rr_str}")
        print(f"  RSI check  : must be > 30 before entering  (currently {rsi:.0f})")
        print(f"  Invalidation: close above {dol(sma200)} SMA200 → cover short")
    else:
        print(f"  No entry recommended — wait for scores to diverge.")
        print(f"  Watch: RSI {rsi:.0f} | MACD crossover | Price vs SMA200 {dol(sma200)}")

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print(_sep("═"))
    s_prim = base_rates.get(horizon_td, base_rates.get(21, {}))
    hist_note = (f"{pct(s_prim['win'])} historical win rate over {horizon_td}td  (n={s_prim['n']})"
                 if s_prim and s_prim.get("n", 0) > 0 else "insufficient historical data")
    print(f"\n  {ticker:<6}  {dol(last_close)}   {last_ts}   [{profile_name}]")
    print(f"  Verdict  : {verdict}")
    print(f"  Scores   : LONG {long_score}/10  vs  SHORT {short_score}/10   R/R: {rr_str}")
    print(f"  Regime   : {regime_icons[regime]}  |  RSI {rsi:.0f}  |  ADX {adx:.0f}")
    print(f"  History  : {hist_note}  ({direction.upper()} perspective)")
    if earnings:
        td = earnings["trading_days"]
        warn = "  ⚠ near earnings" if td <= 10 else ""
        print(f"  Earnings : ~{td}td  ({earnings['date_str']}){warn}")
    print()
    print(f"  ⚠  Not investment advice. Past base rates ≠ future returns.")
    print(_sep("═"))
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args    = sys.argv[1:]
    tickers = ["NFLX"] if not args else [a.upper() for a in args if not a.startswith("--")]
    horizon = 21
    if "--horizon" in args:
        idx = args.index("--horizon")
        if idx + 1 < len(args):
            horizon = int(args[idx + 1])
    for t in tickers:
        result = analyze(t, horizon_td=horizon)
        print_report(result)


if __name__ == "__main__":
    main()
