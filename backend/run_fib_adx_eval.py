#!/usr/bin/env python3
"""
run_fib_adx_eval.py  —  Evaluate Fibonacci + ADX score additions on GOOG 5y

Configs tested (all share v4.2 baseline params):
  0. v4.2 baseline          — no fib, no ADX
  1. + Fib score            — use_fib_score=True
  2. + ADX score            — use_adx_score=True
  3. + ADX hard gate        — use_adx_gate=True (no score change)
  4. + Fib + ADX score      — both score bits on
  5. + Fib + ADX score + gate — full package

Run from backend/:
    python run_fib_adx_eval.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backtester.data   import fetch_ohlcv
from backtester.engine import BacktestEngine
from strategies.growth_signal_bot import GrowthSignalBot

# ── Shared v4.2 base params ───────────────────────────────────────────────────

BASE = dict(
    rsi_bull_max = 62,
    entry_mode   = "All Signals",
    req_macd_x   = True,
    sl_pct       = 5.0,
    tp_pct       = 15.0,
)

CONFIGS: list[tuple[str, dict]] = [
    ("v4.2 baseline",              {}),
    ("+ Fib score",                dict(use_fib_score=True)),
    ("+ ADX score",                dict(use_adx_score=True)),
    ("+ ADX gate (hard)",          dict(use_adx_gate=True)),
    ("+ Fib + ADX score",          dict(use_fib_score=True, use_adx_score=True)),
    ("+ Fib + ADX score + gate",   dict(use_fib_score=True, use_adx_score=True, use_adx_gate=True)),
]

# ── Run ───────────────────────────────────────────────────────────────────────

def pct(v: float) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v * 100:.1f}%"

def f2(v: float) -> str:
    return f"{v:.2f}"


def main() -> None:
    print("[data] Fetching GOOG 5y …")
    df = fetch_ohlcv("GOOG", period="5y")
    print(f"[data] {len(df)} bars  ({df.index[0].date()} → {df.index[-1].date()})\n")

    header = (
        f"  {'Config':<30}  {'Trades':>6}  {'WR':>6}  {'PF':>6}  "
        f"{'CAGR':>7}  {'Sharpe':>7}  {'MaxDD':>7}"
    )
    sep = "  " + "─" * (len(header) - 2)
    print(header)
    print(sep)

    results_list = []
    for label, extra in CONFIGS:
        params = {**BASE, **extra}
        strat  = GrowthSignalBot(**params)
        strat.name = label

        engine  = BacktestEngine(strategy=strat, data=df.copy(), initial_capital=10_000)
        results = engine.run()
        s       = results.summary()
        results_list.append((label, s, results))

        print(
            f"  {label:<30}  {s['n_trades']:>6}  "
            f"{pct(s['win_rate']):>6}  {f2(s['profit_factor']):>6}  "
            f"{pct(s['cagr']):>7}  {f2(s['sharpe']):>7}  "
            f"{pct(s['max_drawdown']):>7}"
        )

    print(sep)

    # ── Delta table vs baseline ───────────────────────────────────────────────
    print("\n  Δ vs v4.2 baseline\n" + sep)
    base_s = results_list[0][1]
    for label, s, _ in results_list[1:]:
        d_sharpe = s["sharpe"]  - base_s["sharpe"]
        d_cagr   = s["cagr"]    - base_s["cagr"]
        d_dd     = s["max_drawdown"] - base_s["max_drawdown"]
        d_pf     = s["profit_factor"] - base_s["profit_factor"]
        d_trades = s["n_trades"] - base_s["n_trades"]

        sign_s = "▲" if d_sharpe > 0 else "▼"
        sign_c = "▲" if d_cagr   > 0 else "▼"

        print(
            f"  {label:<30}  "
            f"Sharpe {sign_s}{d_sharpe:+.2f}  "
            f"CAGR {sign_c}{d_cagr*100:+.1f}%  "
            f"Trades {d_trades:+d}  "
            f"PF {d_pf:+.2f}  "
            f"DD {d_dd*100:+.1f}%"
        )

    print(sep)
    print("\n  Done.\n")


if __name__ == "__main__":
    main()
