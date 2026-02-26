#!/usr/bin/env python3
"""
run_1h_tune2.py  —  Round 2 tuning sweep for Growth Signal Bot 1H v1.3

Round 1 findings:
  - score_moderate / rsi_bull_max had ZERO effect in All Signals mode
    → bounce_signal (score≥8) and vwap_bounce (score≥7) dominate entries
    → those params only matter in "Buy Only" mode
  - ADX ≥ 30: best single lever (+0.11 Sharpe, WR 41.2%, MaxDD -14.0%)
  - SL 1.5% / TP 6%: tied Sharpe, best Sortino (1.33), lowest MaxDD (-13.4%)
  - Trailing stops: catastrophic on 1H (far too many exits)

Round 2 goals:
  A) Combine the two round-1 winners (ADX≥30 + SL 1.5/6%)
  B) Re-test score / RSI in "Buy Only" mode where they actually filter
  C) Test "Buy Only" with ADX≥30 (should work here — adx_gate applies to buy_signal)
  D) Best combo: ADX≥30 + SL 1.5/6% in both modes
  E) Explore finer ADX range (28, 32, 35) around the winning 30 threshold

Data: GOOG 2y 1H  (2 078 tradeable bars after 1 400-bar warmup)
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backtester.data   import fetch_ohlcv
from backtester.engine import BacktestEngine
from strategies.growth_signal_bot import GrowthSignalBot


# ── Helpers ───────────────────────────────────────────────────────────────────

def pct(v: float) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v * 100:.1f}%"

def f2(v: float) -> str:
    return f"{v:.2f}"

def row(label: str, s: dict) -> str:
    return (
        f"  {label:<46}  {s['n_trades']:>6}  "
        f"{pct(s['win_rate']):>6}  {f2(s['profit_factor']):>6}  "
        f"{pct(s['cagr']):>7}  {f2(s['sharpe']):>7}  "
        f"{pct(s['max_drawdown']):>7}  {f2(s['sortino']):>7}"
    )

def delta_row(label: str, s: dict, base: dict) -> str:
    ds  = s["sharpe"]        - base["sharpe"]
    dc  = s["cagr"]          - base["cagr"]
    dd  = s["max_drawdown"]  - base["max_drawdown"]
    dpf = s["profit_factor"] - base["profit_factor"]
    dt  = s["n_trades"]      - base["n_trades"]
    dwr = s["win_rate"]      - base["win_rate"]
    sign_s = "▲" if ds > 0 else "▼"
    sign_c = "▲" if dc > 0 else "▼"
    sign_w = "▲" if dwr > 0 else "▼"
    return (
        f"  {label:<46}  "
        f"Sharpe {sign_s}{ds:+.2f}  "
        f"CAGR {sign_c}{dc*100:+.1f}%  "
        f"WR {sign_w}{dwr*100:+.1f}%  "
        f"Trades {dt:+d}  "
        f"PF {dpf:+.2f}  "
        f"DD {dd*100:+.1f}%"
    )


# ── Config groups ─────────────────────────────────────────────────────────────
# All configs use for_1h() as base (entry_mode="All Signals" unless overridden)

ALL_SIGNALS = dict(entry_mode="All Signals")
BUY_ONLY    = dict(entry_mode="Buy Only")

CONFIGS: list[tuple[str, dict]] = [

    # ── Section A: Reference baselines ───────────────────────────────────────
    ("A1. AllSig Baseline (v1.3 defaults)",
        ALL_SIGNALS),
    ("A2. BuyOnly Baseline",
        BUY_ONLY),

    # ── Section B: Combine round-1 winners (All Signals) ─────────────────────
    ("B1. AllSig | ADX≥30",
        {**ALL_SIGNALS, "adx_threshold": 30}),
    ("B2. AllSig | SL 1.5% / TP 6%",
        {**ALL_SIGNALS, "sl_pct": 1.5, "tp_pct": 6.0}),
    ("B3. AllSig | ADX≥30 + SL 1.5% / TP 6%  ★ KEY",
        {**ALL_SIGNALS, "adx_threshold": 30, "sl_pct": 1.5, "tp_pct": 6.0}),
    ("B4. AllSig | ADX≥30 + SL 1.5% / TP 8%",
        {**ALL_SIGNALS, "adx_threshold": 30, "sl_pct": 1.5, "tp_pct": 8.0}),

    # ── Section C: Buy Only — score & RSI where they actually filter ──────────
    ("C1. BuyOnly | score_moderate=10",
        {**BUY_ONLY, "score_moderate": 10}),
    ("C2. BuyOnly | score_moderate=11",
        {**BUY_ONLY, "score_moderate": 11}),
    ("C3. BuyOnly | RSI≤70",
        {**BUY_ONLY, "rsi_bull_max": 70}),
    ("C4. BuyOnly | RSI≤65",
        {**BUY_ONLY, "rsi_bull_max": 65}),
    ("C5. BuyOnly | score≥10 + RSI≤70",
        {**BUY_ONLY, "score_moderate": 10, "rsi_bull_max": 70}),

    # ── Section D: Buy Only + ADX ─────────────────────────────────────────────
    ("D1. BuyOnly | ADX≥25",
        {**BUY_ONLY, "adx_threshold": 25}),
    ("D2. BuyOnly | ADX≥30",
        {**BUY_ONLY, "adx_threshold": 30}),
    ("D3. BuyOnly | ADX≥30 + SL 1.5% / TP 6%  ★ KEY",
        {**BUY_ONLY, "adx_threshold": 30, "sl_pct": 1.5, "tp_pct": 6.0}),
    ("D4. BuyOnly | ADX≥30 + score≥10 + RSI≤70",
        {**BUY_ONLY, "adx_threshold": 30, "score_moderate": 10, "rsi_bull_max": 70}),
    ("D5. BuyOnly | ADX≥30 + score≥10 + SL 1.5/6%",
        {**BUY_ONLY, "adx_threshold": 30, "score_moderate": 10, "sl_pct": 1.5, "tp_pct": 6.0}),

    # ── Section E: Fine-tune ADX threshold (All Signals) ─────────────────────
    ("E1. AllSig | ADX≥25 + SL 1.5/6%",
        {**ALL_SIGNALS, "adx_threshold": 25, "sl_pct": 1.5, "tp_pct": 6.0}),
    ("E2. AllSig | ADX≥28 + SL 1.5/6%",
        {**ALL_SIGNALS, "adx_threshold": 28, "sl_pct": 1.5, "tp_pct": 6.0}),
    ("E3. AllSig | ADX≥32 + SL 1.5/6%",
        {**ALL_SIGNALS, "adx_threshold": 32, "sl_pct": 1.5, "tp_pct": 6.0}),
    ("E4. AllSig | ADX≥35 + SL 1.5/6%",
        {**ALL_SIGNALS, "adx_threshold": 35, "sl_pct": 1.5, "tp_pct": 6.0}),
]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("[data] Fetching GOOG 2y 1h …", flush=True)
    df = fetch_ohlcv("GOOG", period="2y", interval="1h")
    print(f"[data] {len(df)} bars  ({df.index[0].date()} → {df.index[-1].date()})\n")

    HEADER = (
        f"  {'Config':<46}  {'Trades':>6}  {'WR':>6}  {'PF':>6}  "
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

    # ── Δ vs AllSig baseline ──────────────────────────────────────────────────
    base_allsig = results_list[0][1]
    base_buyonly = results_list[1][1]
    print(f"\n  Δ vs {results_list[0][0]}\n" + SEP)
    for label, s in results_list[2:]:
        print(delta_row(label, s, base_allsig))
    print(SEP)

    # ── Top 5 by Sharpe ───────────────────────────────────────────────────────
    ranked = sorted(results_list, key=lambda x: x[1]["sharpe"], reverse=True)
    print("\n  ── Top 5 by Sharpe ──")
    print(HEADER)
    for label, s in ranked[:5]:
        print(row(label, s))

    # ── Top 5 by Sortino ─────────────────────────────────────────────────────
    ranked_so = sorted(results_list, key=lambda x: x[1]["sortino"], reverse=True)
    print("\n  ── Top 5 by Sortino (downside-adjusted) ──")
    print(HEADER)
    for label, s in ranked_so[:5]:
        print(row(label, s))

    # ── Top 5 by Win Rate ─────────────────────────────────────────────────────
    ranked_wr = sorted(results_list, key=lambda x: x[1]["win_rate"], reverse=True)
    print("\n  ── Top 5 by Win Rate ──")
    print(HEADER)
    for label, s in ranked_wr[:5]:
        print(row(label, s))

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
