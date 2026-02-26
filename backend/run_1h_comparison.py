#!/usr/bin/env python3
"""
run_1h_comparison.py  —  Compare 1H v1.3 vs daily v4.3 on GOOG

Configs:
  1. Daily v4.3  — GOOG 5y daily  (existing benchmark)
  2. 1H v1.3 Buy Only  — GOOG 2y hourly  (yfinance 1H max: 730 days)
  3. 1H v1.3 All Signals  — same data, all three entry types

Note: Different lookback periods (5y vs 2y) so results aren't perfectly
apples-to-apples, but the CAGR/Sharpe figures normalise for time.  The key
questions are (a) does 1H produce enough trades, (b) is the per-trade quality
comparable, and (c) does execution on 1H signals actually improve outcomes vs
the fill-at-daily-close assumption of the daily backtester.

Run from backend/:
    python run_1h_comparison.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backtester.data   import fetch_ohlcv
from backtester.engine import BacktestEngine
from strategies.growth_signal_bot import GrowthSignalBot


# ── Helper formatters ─────────────────────────────────────────────────────────

def pct(v: float) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v * 100:.1f}%"


def f2(v: float) -> str:
    return f"{v:.2f}"


# ── Configs ───────────────────────────────────────────────────────────────────

CONFIGS: list[tuple[str, dict, GrowthSignalBot]] = [
    (
        "Daily v4.3  (GOOG 5y)",
        dict(ticker="GOOG", period="5y", interval="1d"),
        GrowthSignalBot(
            rsi_bull_max = 62,
            entry_mode   = "All Signals",
            req_macd_x   = True,
            sl_pct       = 5.0,
            tp_pct       = 15.0,
        ),
    ),
    (
        "1H v1.3 Buy Only   (GOOG 2y)",
        dict(ticker="GOOG", period="2y", interval="1h"),
        GrowthSignalBot.for_1h(
            entry_mode = "Buy Only",
        ),
    ),
    (
        "1H v1.3 All Signals (GOOG 2y)",
        dict(ticker="GOOG", period="2y", interval="1h"),
        GrowthSignalBot.for_1h(
            entry_mode = "All Signals",
        ),
    ),
]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    header = (
        f"  {'Config':<35}  {'Trades':>6}  {'WR':>6}  {'PF':>6}  "
        f"{'CAGR':>7}  {'Sharpe':>7}  {'MaxDD':>7}"
    )
    sep = "  " + "─" * (len(header) - 2)

    print()
    print(header)
    print(sep)

    results_list = []
    for label, fetch_kw, strat in CONFIGS:
        tf = fetch_kw.get("interval", "1d")
        print(f"  [data] Fetching {fetch_kw['ticker']} {fetch_kw['period']} {tf} …", flush=True)
        df = fetch_ohlcv(**fetch_kw)
        strat.name = label

        engine  = BacktestEngine(strategy=strat, data=df.copy(), initial_capital=10_000)
        results = engine.run()
        s       = results.summary()
        results_list.append((label, s, results))

        print(
            f"  {label:<35}  {s['n_trades']:>6}  "
            f"{pct(s['win_rate']):>6}  {f2(s['profit_factor']):>6}  "
            f"{pct(s['cagr']):>7}  {f2(s['sharpe']):>7}  "
            f"{pct(s['max_drawdown']):>7}"
        )

    print(sep)

    # ── Per-trade quality comparison ──────────────────────────────────────────
    print("\n  Per-trade quality\n" + sep)
    for label, s, results in results_list:
        print(
            f"  {label:<35}  "
            f"AvgWin {pct(s['avg_winner'])}  "
            f"AvgLoss {pct(s['avg_loser'])}  "
            f"Expect {pct(s['expectancy'])}"
        )

    print(sep)
    print("\n  Done.\n")


if __name__ == "__main__":
    main()
