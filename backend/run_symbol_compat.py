#!/usr/bin/env python3
"""
run_symbol_compat.py  —  Multi-symbol compatibility sweep for Growth Signal Bot

Tests daily v4.3 (5y) and 1H v1.4 (2y) on a broad set of symbols to identify
which ones are "compatible" — i.e. the strategy's regime/momentum filters
naturally align with how that symbol trends.

Symbols tested:
  Large-cap tech/growth: GOOG, AAPL, MSFT, META, NVDA, AMD, AMZN, NFLX, TSLA
  Mid-cap growth:        SQ, PYPL, SNAP, UBER, SHOP, RIVN
  Macro / index / ETF:   SPY, GLD, SLV

Compatibility criteria (for a ✅ rating):
  Sharpe  ≥ 0.8   — decent risk-adjusted return
  CAGR    ≥ +8%   — meaningful absolute return
  MaxDD   ≤ -35%  — drawdown not catastrophic
  Trades  ≥ 5     — enough signals to be meaningful (low-frequency daily)

Run from backend/:
    python run_symbol_compat.py
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backtester.data   import fetch_ohlcv
from backtester.engine import BacktestEngine
from strategies.growth_signal_bot import GrowthSignalBot

# ── Symbol lists ──────────────────────────────────────────────────────────────

SYMBOLS: list[tuple[str, str]] = [
    # ticker,  display label
    ("GOOG",  "Alphabet"),
    ("AAPL",  "Apple"),
    ("MSFT",  "Microsoft"),
    ("META",  "Meta"),
    ("NVDA",  "NVIDIA"),
    ("AMD",   "AMD"),
    ("AMZN",  "Amazon"),
    ("NFLX",  "Netflix"),
    ("TSLA",  "Tesla"),
    ("SQ",    "Block (SQ)"),
    ("PYPL",  "PayPal"),
    ("SNAP",  "Snap"),
    ("UBER",  "Uber"),
    ("SHOP",  "Shopify"),
    ("RIVN",  "Rivian"),
    ("SPY",   "S&P 500 ETF"),
    ("GLD",   "Gold ETF"),
    ("SLV",   "Silver ETF"),
]

# ── Compatibility thresholds ──────────────────────────────────────────────────

MIN_SHARPE  = 0.8
MIN_CAGR    = 0.08    # 8%
MAX_DD      = -0.35   # -35%
MIN_TRADES  = 5

# ── Helpers ───────────────────────────────────────────────────────────────────

def pct(v: float) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v * 100:.1f}%"

def f2(v: float) -> str:
    return f"{v:.2f}"

def compat_badge(s: dict) -> str:
    if s["n_trades"] < MIN_TRADES:
        return "⚠ few trades"
    ok = (
        s["sharpe"]       >= MIN_SHARPE
        and s["cagr"]     >= MIN_CAGR
        and s["max_drawdown"] >= MAX_DD
    )
    if ok:
        return "✅ compatible"
    # partial credit
    pts = sum([
        s["sharpe"]       >= MIN_SHARPE,
        s["cagr"]         >= MIN_CAGR,
        s["max_drawdown"] >= MAX_DD,
    ])
    if pts == 2:
        return "🟡 partial"
    return "❌ poor fit"

def row(ticker: str, label: str, s: dict) -> str:
    badge = compat_badge(s)
    return (
        f"  {ticker:<5}  {label:<16}  {s['n_trades']:>6}  "
        f"{pct(s['win_rate']):>6}  {f2(s['profit_factor']):>5}  "
        f"{pct(s['cagr']):>7}  {f2(s['sharpe']):>6}  "
        f"{pct(s['max_drawdown']):>7}  {badge}"
    )

HEADER = (
    f"  {'Tick':<5}  {'Name':<16}  {'Trades':>6}  "
    f"{'WR':>6}  {'PF':>5}  {'CAGR':>7}  {'Sharpe':>6}  {'MaxDD':>7}  Compat"
)
SEP = "  " + "─" * (len(HEADER) - 2)

# ── Main ──────────────────────────────────────────────────────────────────────

def run_sweep(
    symbols: list[tuple[str, str]],
    make_strat,
    period: str,
    interval: str,
    label: str,
) -> list[tuple[str, str, dict | None]]:
    """Run one pass (daily or 1H) across all symbols. Returns list of results."""
    print(f"\n{'═' * 78}")
    print(f"  {label}   ({period} · {interval})")
    print(f"{'═' * 78}")
    print(HEADER)
    print(SEP)

    results = []
    for ticker, name in symbols:
        try:
            df = fetch_ohlcv(ticker, period=period, interval=interval)
            if df is None or len(df) < 100:
                print(f"  {ticker:<5}  {name:<16}  — insufficient data —")
                results.append((ticker, name, None))
                continue
            strat  = make_strat()
            strat.name = f"{ticker} {interval}"
            engine = BacktestEngine(strategy=strat, data=df.copy(), initial_capital=10_000)
            s      = engine.run().summary()
            results.append((ticker, name, s))
            print(row(ticker, name, s))
        except Exception as e:
            print(f"  {ticker:<5}  {name:<16}  ⚠ error: {e}")
            results.append((ticker, name, None))
        time.sleep(0.1)   # gentle rate limit

    print(SEP)
    return results


def print_rankings(results: list[tuple[str, str, dict | None]], top_n: int = 5) -> None:
    valid = [(t, n, s) for t, n, s in results if s is not None]

    by_sharpe = sorted(valid, key=lambda x: x[2]["sharpe"], reverse=True)
    by_wr     = sorted(valid, key=lambda x: x[2]["win_rate"], reverse=True)
    by_cagr   = sorted(valid, key=lambda x: x[2]["cagr"], reverse=True)

    print(f"\n  ── Top {top_n} by Sharpe ──")
    for t, n, s in by_sharpe[:top_n]:
        print(f"     {t:<5}  {n:<16}  Sharpe {f2(s['sharpe'])}  "
              f"CAGR {pct(s['cagr'])}  WR {pct(s['win_rate'])}  "
              f"MaxDD {pct(s['max_drawdown'])}  {compat_badge(s)}")

    print(f"\n  ── Top {top_n} by Win Rate ──")
    for t, n, s in by_wr[:top_n]:
        print(f"     {t:<5}  {n:<16}  WR {pct(s['win_rate'])}  "
              f"Sharpe {f2(s['sharpe'])}  CAGR {pct(s['cagr'])}  "
              f"MaxDD {pct(s['max_drawdown'])}  {compat_badge(s)}")

    print(f"\n  ── Top {top_n} by CAGR ──")
    for t, n, s in by_cagr[:top_n]:
        print(f"     {t:<5}  {n:<16}  CAGR {pct(s['cagr'])}  "
              f"Sharpe {f2(s['sharpe'])}  WR {pct(s['win_rate'])}  "
              f"MaxDD {pct(s['max_drawdown'])}  {compat_badge(s)}")

    compat   = [x for x in valid if compat_badge(x[2]) == "✅ compatible"]
    partial  = [x for x in valid if compat_badge(x[2]) == "🟡 partial"]
    poor     = [x for x in valid if compat_badge(x[2]) == "❌ poor fit"]
    few      = [x for x in valid if compat_badge(x[2]) == "⚠ few trades"]

    print(f"\n  ── Compatibility summary ──")
    print(f"     ✅ Compatible ({len(compat)}):   {', '.join(t for t,_,_ in compat)}")
    print(f"     🟡 Partial   ({len(partial)}): {', '.join(t for t,_,_ in partial)}")
    print(f"     ❌ Poor fit  ({len(poor)}):  {', '.join(t for t,_,_ in poor)}")
    if few:
        print(f"     ⚠ Too few   ({len(few)}):  {', '.join(t for t,_,_ in few)}")


def main() -> None:
    # ── Daily v4.3 — 5 year ──────────────────────────────────────────────────
    daily_results = run_sweep(
        SYMBOLS,
        make_strat = GrowthSignalBot,
        period     = "5y",
        interval   = "1d",
        label      = "DAILY v4.3  ·  5-year window",
    )
    print_rankings(daily_results)

    # ── 1H v1.4 — 2 year (yfinance max for 1H) ───────────────────────────────
    h1_results = run_sweep(
        SYMBOLS,
        make_strat = GrowthSignalBot.for_1h,
        period     = "2y",
        interval   = "1h",
        label      = "1H v1.4  ·  2-year window  (yfinance max for hourly data)",
    )
    print_rankings(h1_results)

    # ── Cross-timeframe winner summary ───────────────────────────────────────
    print(f"\n{'═' * 78}")
    print(f"  CROSS-TIMEFRAME: symbols that are ✅ compatible on BOTH timeframes")
    print(f"{'═' * 78}")
    daily_compat = {t for t, _, s in daily_results if s and compat_badge(s) == "✅ compatible"}
    h1_compat    = {t for t, _, s in h1_results   if s and compat_badge(s) == "✅ compatible"}
    both = daily_compat & h1_compat
    only_daily = daily_compat - h1_compat
    only_1h    = h1_compat - daily_compat

    print(f"  Both timeframes : {', '.join(sorted(both)) or 'none'}")
    print(f"  Daily only      : {', '.join(sorted(only_daily)) or 'none'}")
    print(f"  1H only         : {', '.join(sorted(only_1h)) or 'none'}")
    print()


if __name__ == "__main__":
    main()
