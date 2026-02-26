#!/usr/bin/env python3
"""
run_1h_final.py  —  Final comparison: 1H v1.4 vs baselines

Shows the tuned v1.4 result alongside daily v4.3 and 1H v1.3 baselines.
GOOG 2y 1H for both 1H configs; GOOG 5y daily for the daily reference.
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
        f"  {label:<44}  {s['n_trades']:>6}  "
        f"{pct(s['win_rate']):>6}  {f2(s['profit_factor']):>6}  "
        f"{pct(s['cagr']):>7}  {f2(s['sharpe']):>7}  "
        f"{pct(s['max_drawdown']):>7}  {f2(s['sortino']):>7}"
    )


def main() -> None:
    print("[data] Fetching GOOG …", flush=True)
    df_1h = fetch_ohlcv("GOOG", period="2y", interval="1h")
    df_5y = fetch_ohlcv("GOOG", period="5y", interval="1d")
    print(f"[data] 1H: {len(df_1h)} bars  ({df_1h.index[0].date()} → {df_1h.index[-1].date()})")
    print(f"[data] 5y: {len(df_5y)} bars  ({df_5y.index[0].date()} → {df_5y.index[-1].date()})\n")

    CONFIGS = [
        ("Daily v4.3  (5y GOOG, All Signals)",
            GrowthSignalBot(),    df_5y),
        ("1H  v1.3  (2y GOOG, All Signals)",
            GrowthSignalBot.for_1h(adx_threshold=20.0), df_1h),
        ("1H  v1.4  (2y GOOG, All Signals) ← tuned",
            GrowthSignalBot.for_1h(), df_1h),
        ("1H  v1.4  Buy Only  (2y GOOG)",
            GrowthSignalBot.for_1h(entry_mode="Buy Only"), df_1h),
    ]

    HEADER = (
        f"  {'Config':<44}  {'Trades':>6}  {'WR':>6}  {'PF':>6}  "
        f"{'CAGR':>7}  {'Sharpe':>7}  {'MaxDD':>7}  {'Sortino':>7}"
    )
    SEP = "  " + "─" * (len(HEADER) - 2)

    print(HEADER)
    print(SEP)

    results = []
    for label, strat, df in CONFIGS:
        strat.name = label
        engine = BacktestEngine(strategy=strat, data=df.copy(), initial_capital=10_000)
        s = engine.run().summary()
        results.append((label, s))
        print(row(label, s))

    print(SEP)

    # Per-trade quality
    print(f"\n  Per-trade quality breakdown")
    print(SEP)
    for label, s in results:
        wr   = s["win_rate"]
        pf   = s["profit_factor"]
        # Expectancy = WR * AvgWin - (1-WR) * AvgLoss  (rough via PF)
        # PF = (WR * AvgWin) / ((1-WR) * AvgLoss)  → AvgWin/AvgLoss = PF*(1-WR)/WR
        avg_win_loss = pf * (1 - wr) / wr if wr > 0 else 0
        print(f"  {label:<44}  WR {wr*100:.1f}%  PF {pf:.2f}  AvgWin/AvgLoss {avg_win_loss:.2f}x")
    print(SEP)

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
