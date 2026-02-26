#!/usr/bin/env python3
"""
run_1h_tune.py  —  Systematic parameter sweep to tune Growth Signal Bot 1H v1.3

Sweeps the levers most likely to improve Sharpe / win rate:
  1. Entry quality (score_moderate, rsi_bull_max, adx_threshold)
  2. Risk / reward  (sl_pct / tp_pct ratio, trailing stop)
  3. Combined candidates (best single-lever picks)

Data: GOOG 2y 1H  (2 078 tradeable bars after 1 400-bar warmup)
Baseline: for_1h(entry_mode="Buy Only")  → 16 trades, 50% WR, Sharpe 1.22

Run from backend/:
    python run_1h_tune.py
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
        f"  {label:<40}  {s['n_trades']:>6}  "
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
        f"  {label:<40}  "
        f"Sharpe {sign_s}{ds:+.2f}  "
        f"CAGR {sign_c}{dc*100:+.1f}%  "
        f"WR {sign_w}{dwr*100:+.1f}%  "
        f"Trades {dt:+d}  "
        f"PF {dpf:+.2f}  "
        f"DD {dd*100:+.1f}%"
    )


# ── Config definitions ────────────────────────────────────────────────────────

CONFIGS: list[tuple[str, dict]] = [
    # ── Group 0: Baseline ────────────────────────────────────────────────────
    ("Baseline (score≥9, RSI≤75, ADX≥20)",     {}),

    # ── Group 1: Entry quality — score gate ──────────────────────────────────
    ("score_moderate = 10",                     dict(score_moderate=10)),
    ("score_moderate = 11",                     dict(score_moderate=11)),

    # ── Group 2: Entry quality — RSI max ─────────────────────────────────────
    ("rsi_bull_max = 70",                       dict(rsi_bull_max=70)),
    ("rsi_bull_max = 65",                       dict(rsi_bull_max=65)),
    ("rsi_bull_max = 60",                       dict(rsi_bull_max=60)),

    # ── Group 3: Entry quality — ADX threshold ───────────────────────────────
    ("adx_threshold = 25",                      dict(adx_threshold=25)),
    ("adx_threshold = 30",                      dict(adx_threshold=30)),

    # ── Group 4: Risk / reward — SL / TP ─────────────────────────────────────
    ("SL 1.5% / TP 6% (4:1 R:R)",              dict(sl_pct=1.5, tp_pct=6.0)),
    ("SL 2.0% / TP 8% (4:1 R:R)",              dict(sl_pct=2.0, tp_pct=8.0)),
    ("SL 2.0% / TP 10% (5:1 R:R)",             dict(sl_pct=2.0, tp_pct=10.0)),
    ("SL 1.5% / TP 8% (5.3:1 R:R)",            dict(sl_pct=1.5, tp_pct=8.0)),

    # ── Group 5: Trailing stop ────────────────────────────────────────────────
    ("Trailing stop 1.5%",                      dict(use_trail=True,  trail_pct=1.5)),
    ("Trailing stop 2.0%",                      dict(use_trail=True,  trail_pct=2.0)),
    ("Trailing stop 2.5%",                      dict(use_trail=True,  trail_pct=2.5)),

    # ── Group 6: Combined candidates ─────────────────────────────────────────
    ("score≥10 + RSI≤70",                       dict(score_moderate=10, rsi_bull_max=70)),
    ("score≥10 + ADX≥25",                       dict(score_moderate=10, adx_threshold=25)),
    ("RSI≤70 + ADX≥25",                         dict(rsi_bull_max=70, adx_threshold=25)),
    ("score≥10 + RSI≤70 + TP 8%",               dict(score_moderate=10, rsi_bull_max=70, tp_pct=8.0)),
    ("score≥10 + RSI≤70 + ADX≥25",              dict(score_moderate=10, rsi_bull_max=70, adx_threshold=25)),
    ("score≥10 + RSI≤70 + trail 1.5%",         dict(score_moderate=10, rsi_bull_max=70, use_trail=True, trail_pct=1.5)),
    ("RSI≤70 + ADX≥25 + trail 2%",             dict(rsi_bull_max=70, adx_threshold=25, use_trail=True, trail_pct=2.0)),
]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("[data] Fetching GOOG 2y 1h …", flush=True)
    df = fetch_ohlcv("GOOG", period="2y", interval="1h")
    print(f"[data] {len(df)} bars  ({df.index[0].date()} → {df.index[-1].date()})\n")

    HEADER = (
        f"  {'Config':<40}  {'Trades':>6}  {'WR':>6}  {'PF':>6}  "
        f"{'CAGR':>7}  {'Sharpe':>7}  {'MaxDD':>7}  {'Sortino':>7}"
    )
    SEP = "  " + "─" * (len(HEADER) - 2)

    print(HEADER)
    print(SEP)

    results_list: list[tuple[str, dict]] = []
    for label, overrides in CONFIGS:
        strat = GrowthSignalBot.for_1h(**overrides)
        strat.name = label
        engine  = BacktestEngine(strategy=strat, data=df.copy(), initial_capital=10_000)
        s       = engine.run().summary()
        results_list.append((label, s))
        print(row(label, s))

    print(SEP)

    # ── Δ vs baseline ─────────────────────────────────────────────────────────
    base_s = results_list[0][1]
    print(f"\n  Δ vs {results_list[0][0]}\n" + SEP)
    for label, s in results_list[1:]:
        print(delta_row(label, s, base_s))
    print(SEP)

    # ── Top 5 by Sharpe ───────────────────────────────────────────────────────
    ranked = sorted(results_list, key=lambda x: x[1]["sharpe"], reverse=True)
    print("\n  Top 5 by Sharpe\n" + SEP)
    print(HEADER)
    for label, s in ranked[:5]:
        print(row(label, s))
    print(SEP)

    # ── Top 5 by Win Rate ─────────────────────────────────────────────────────
    ranked_wr = sorted(results_list, key=lambda x: x[1]["win_rate"], reverse=True)
    print("\n  Top 5 by Win Rate\n" + SEP)
    print(HEADER)
    for label, s in ranked_wr[:5]:
        print(row(label, s))
    print(SEP)

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
