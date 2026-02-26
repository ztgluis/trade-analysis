#!/usr/bin/env python3
"""
run_1h_tune3.py  —  Round 3: ADX ceiling probe + Buy Only SL tuning

Round 2 findings:
  - Buy Only: ALL score/RSI changes have ZERO effect (all 16 trades already
    comfortably exceed any threshold we tested → market naturally self-selects)
  - ADX gate in Buy Only kills it (9 trades → Sharpe drops to 0.48-0.71)
  - AllSig + ADX≥35 + SL 1.5%/TP 6%: BEST so far (Sharpe 1.26, Sortino 1.29)
  - ADX sweep shows clear upward trend: 25→28→30→32→35 all progressively better
  - Combination rule: ADX≥30 + SL 1.5 HURT, but ADX≥32+ + SL 1.5 HELPED
    → ADX≥32 appears to be the threshold where quality overrides SL friction

Round 3 goals:
  A) Probe AllSig + ADX ceiling: 38, 40, 45 (with SL 1.5% / TP 6%)
  B) Test Buy Only + SL 1.5% / TP 6% (never tested this combo!)
  C) AllSig + ADX≥35 + TP 7% (between the two TP options we've tested)
  D) Confirm: does Buy Only already beat AllSig at its best?

Data: GOOG 2y 1H
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backtester.data   import fetch_ohlcv
from backtester.engine import BacktestEngine
from strategies.growth_signal_bot import GrowthSignalBot


def pct(v: float) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v * 100:.1f}%"

def f2(v: float) -> str:
    return f"{v:.2f}"

def row(label: str, s: dict) -> str:
    return (
        f"  {label:<48}  {s['n_trades']:>6}  "
        f"{pct(s['win_rate']):>6}  {f2(s['profit_factor']):>6}  "
        f"{pct(s['cagr']):>7}  {f2(s['sharpe']):>7}  "
        f"{pct(s['max_drawdown']):>7}  {f2(s['sortino']):>7}"
    )

def delta_row(label: str, s: dict, base: dict) -> str:
    ds  = s["sharpe"]        - base["sharpe"]
    dc  = s["cagr"]          - base["cagr"]
    dwr = s["win_rate"]      - base["win_rate"]
    dt  = s["n_trades"]      - base["n_trades"]
    dpf = s["profit_factor"] - base["profit_factor"]
    dd  = s["max_drawdown"]  - base["max_drawdown"]
    sign_s = "▲" if ds > 0 else "▼"
    sign_c = "▲" if dc > 0 else "▼"
    sign_w = "▲" if dwr > 0 else "▼"
    return (
        f"  {label:<48}  "
        f"Sharpe {sign_s}{ds:+.2f}  "
        f"CAGR {sign_c}{dc*100:+.1f}%  "
        f"WR {sign_w}{dwr*100:+.1f}%  "
        f"Trades {dt:+d}  "
        f"PF {dpf:+.2f}  "
        f"DD {dd*100:+.1f}%"
    )


ALL = dict(entry_mode="All Signals")
BUY = dict(entry_mode="Buy Only")

CONFIGS: list[tuple[str, dict]] = [
    # ── References from round 2 ───────────────────────────────────────────────
    ("REF: AllSig baseline",
        ALL),
    ("REF: BuyOnly baseline (SL 2% / TP 6%)",
        BUY),
    ("REF: AllSig | ADX≥35 + SL 1.5%/TP 6%  [R2 best]",
        {**ALL, "adx_threshold": 35, "sl_pct": 1.5, "tp_pct": 6.0}),

    # ── Section A: ADX ceiling (All Signals + SL 1.5% / TP 6%) ───────────────
    ("A1. AllSig | ADX≥38 + SL 1.5/6%",
        {**ALL, "adx_threshold": 38, "sl_pct": 1.5, "tp_pct": 6.0}),
    ("A2. AllSig | ADX≥40 + SL 1.5/6%",
        {**ALL, "adx_threshold": 40, "sl_pct": 1.5, "tp_pct": 6.0}),
    ("A3. AllSig | ADX≥45 + SL 1.5/6%",
        {**ALL, "adx_threshold": 45, "sl_pct": 1.5, "tp_pct": 6.0}),

    # ── Section B: Buy Only + SL 1.5% (never tested!) ────────────────────────
    ("B1. BuyOnly | SL 1.5% / TP 6%",
        {**BUY, "sl_pct": 1.5, "tp_pct": 6.0}),
    ("B2. BuyOnly | SL 1.5% / TP 7%",
        {**BUY, "sl_pct": 1.5, "tp_pct": 7.0}),
    ("B3. BuyOnly | SL 1.5% / TP 8%",
        {**BUY, "sl_pct": 1.5, "tp_pct": 8.0}),

    # ── Section C: AllSig + ADX≥35 + TP variants ─────────────────────────────
    ("C1. AllSig | ADX≥35 + SL 1.5% / TP 5%",
        {**ALL, "adx_threshold": 35, "sl_pct": 1.5, "tp_pct": 5.0}),
    ("C2. AllSig | ADX≥35 + SL 1.5% / TP 7%",
        {**ALL, "adx_threshold": 35, "sl_pct": 1.5, "tp_pct": 7.0}),
    ("C3. AllSig | ADX≥35 + SL 1.5% / TP 8%",
        {**ALL, "adx_threshold": 35, "sl_pct": 1.5, "tp_pct": 8.0}),
    ("C4. AllSig | ADX≥40 + SL 2.0% / TP 6%  (orig SL)",
        {**ALL, "adx_threshold": 40, "sl_pct": 2.0, "tp_pct": 6.0}),
]


def main() -> None:
    print("[data] Fetching GOOG 2y 1h …", flush=True)
    df = fetch_ohlcv("GOOG", period="2y", interval="1h")
    print(f"[data] {len(df)} bars  ({df.index[0].date()} → {df.index[-1].date()})\n")

    HEADER = (
        f"  {'Config':<48}  {'Trades':>6}  {'WR':>6}  {'PF':>6}  "
        f"{'CAGR':>7}  {'Sharpe':>7}  {'MaxDD':>7}  {'Sortino':>7}"
    )
    SEP = "  " + "─" * (len(HEADER) - 2)

    print(HEADER)
    print(SEP)

    results_list: list[tuple[str, dict]] = []
    for label, overrides in CONFIGS:
        strat  = GrowthSignalBot.for_1h(**overrides)
        strat.name = label
        engine = BacktestEngine(strategy=strat, data=df.copy(), initial_capital=10_000)
        s      = engine.run().summary()
        results_list.append((label, s))
        print(row(label, s))

    print(SEP)

    # Δ vs AllSig baseline
    base_all = results_list[0][1]
    base_buy = results_list[1][1]
    r2_best  = results_list[2][1]

    print(f"\n  Δ vs AllSig baseline\n" + SEP)
    for label, s in results_list[3:]:
        print(delta_row(label, s, base_all))
    print(SEP)

    print(f"\n  Δ vs BuyOnly baseline\n" + SEP)
    for label, s in results_list[6:9]:   # Section B only
        print(delta_row(label, s, base_buy))
    print(SEP)

    # Rankings
    ranked = sorted(results_list, key=lambda x: x[1]["sharpe"], reverse=True)
    print("\n  ── Top 6 overall by Sharpe ──")
    print(HEADER)
    for label, s in ranked[:6]:
        print(row(label, s))

    ranked_so = sorted(results_list, key=lambda x: x[1]["sortino"], reverse=True)
    print("\n  ── Top 6 by Sortino ──")
    print(HEADER)
    for label, s in ranked_so[:6]:
        print(row(label, s))

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
