#!/usr/bin/env python3
"""
Backtesting CLI — run any strategy against any Yahoo Finance ticker.

Usage examples:
    python run_backtest.py                          # BTC-USD, 5y, default params
    python run_backtest.py --ticker ABNB --period 3y
    python run_backtest.py --ticker BTC-USD --period 10y --score-min 9 --trail 6
    python run_backtest.py --no-chart               # skip equity curve plot
    python run_backtest.py --no-macd-cross          # allow mid-trend entries

Run from the backend/ directory:
    cd /Users/ztgluis/dev/trader-bot/backend
    pip install -r requirements.txt
    python run_backtest.py
"""
import sys
import argparse
import textwrap
from pathlib import Path

# Make backtester importable regardless of cwd
sys.path.insert(0, str(Path(__file__).parent))

from backtester.data    import fetch_ohlcv
from backtester.engine  import BacktestEngine
from backtester         import metrics as m
from strategies         import BTCCycleRider


# ── CLI argument parser ───────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a backtest against Yahoo Finance data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Tickers:  BTC-USD  ETH-USD  ABNB  NVDA  TSLA  GOOG  NFLX  SPY  ^GSPC
        Periods:  1y  2y  3y  5y  10y  max
        """),
    )
    p.add_argument("--ticker",      default="BTC-USD", help="Yahoo Finance ticker (default: BTC-USD)")
    p.add_argument("--period",      default="5y",      help="History length (default: 5y)")
    p.add_argument("--capital",     default=10_000,    type=float, help="Initial capital USD (default: 10000)")
    p.add_argument("--commission",  default=0.1,       type=float, help="Commission %% per side (default: 0.1)")
    p.add_argument("--score-min",   default=7,         type=int,   help="Min entry score 1-12 (default: 7)")
    p.add_argument("--sl",          default=8.0,       type=float, help="Hard stop %% below entry (default: 8.0)")
    p.add_argument("--trail",       default=8.0,       type=float, help="Trailing stop %% from peak (default: 8.0)")
    p.add_argument("--no-macd-cross", action="store_true",         help="Allow entries without MACD crossover")
    p.add_argument("--no-chart",    action="store_true",            help="Skip equity curve chart")
    p.add_argument("--refresh",     action="store_true",            help="Force-refresh cached data")
    return p.parse_args()


# ── Pretty-print helpers ──────────────────────────────────────────────────────

W = 58   # table width

def _bar(label: str, value: str, pad: int = 28) -> str:
    return f"  {label:<{pad}} {value}"

def _pct(v: float, decimals: int = 1) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v * 100:.{decimals}f}%"

def _f(v: float, decimals: int = 2) -> str:
    return f"{v:.{decimals}f}"

def _trade_row(i: int, t) -> str:
    entry = t.entry_date.strftime("%Y-%m-%d") if t.entry_date else "—"
    exit_ = t.exit_date.strftime("%Y-%m-%d")  if t.exit_date  else "open"
    pnl   = _pct(t.pnl_pct) if t.pnl_pct is not None else "—"
    reason = (t.exit_reason or "")[:18]
    return f"  {i:>3}  {entry}  {exit_}  {t.entry_price:>10,.2f}  {t.exit_price or 0:>10,.2f}  {pnl:>8}  {reason}"


def print_results(results) -> None:
    s   = results.summary()
    eq  = results.equity
    sep = "═" * W

    print()
    print(sep)
    print(f"  {s['strategy']} — {results.data.index[0].year}–{results.data.index[-1].year}")
    print(sep)

    print(f"\n  RETURNS {'─' * (W - 10)}")
    print(_bar("Total Return",      _pct(s["total_return"])))
    print(_bar("CAGR",              _pct(s["cagr"])))
    print(_bar("Buy & Hold Return", _pct(s["buy_hold"])))

    print(f"\n  RISK {'─' * (W - 7)}")
    print(_bar("Max Drawdown",  _pct(s["max_drawdown"])))
    print(_bar("Sharpe Ratio",  _f(s["sharpe"])))
    print(_bar("Sortino Ratio", _f(s["sortino"])))
    print(_bar("Calmar Ratio",  _f(s["calmar"])))

    print(f"\n  TRADES {'─' * (W - 9)}")
    wins   = len(results.portfolio.winners())
    losses = len(results.portfolio.losers())
    print(_bar("Total Trades",   f"{s['n_trades']}"))
    print(_bar("Win Rate",       f"{_pct(s['win_rate'])}  ({wins}W / {losses}L)"))
    print(_bar("Profit Factor",  _f(s["profit_factor"])))
    print(_bar("Avg Winner",     _pct(s["avg_winner"])))
    print(_bar("Avg Loser",      _pct(s["avg_loser"])))
    print(_bar("Expectancy/trade", _pct(s["expectancy"])))

    best  = s["best_trade"]
    worst = s["worst_trade"]
    if best:
        print(_bar("Best Trade",
            f"{_pct(best.pnl_pct)}  ({best.entry_date.date()} → {best.exit_date.date() if best.exit_date else 'open'})"))
    if worst:
        print(_bar("Worst Trade",
            f"{_pct(worst.pnl_pct)}  ({worst.entry_date.date()} → {worst.exit_date.date() if worst.exit_date else 'open'})"))

    # Exit reason breakdown
    closed = results.trades
    if closed:
        reasons: dict[str, int] = {}
        for t in closed:
            r = t.exit_reason or "Unknown"
            reasons[r] = reasons.get(r, 0) + 1
        print(f"\n  EXIT REASONS {'─' * (W - 15)}")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            pct_r = count / len(closed) * 100
            print(_bar(reason, f"{count}  ({pct_r:.0f}%)"))

    print()
    print(sep)


def print_trade_list(results) -> None:
    trades = results.trades
    if not trades:
        print("\n  No completed trades.")
        return

    print(f"\n  {'#':>3}  {'Entry':^10}  {'Exit':^10}  {'Entry $':>10}  {'Exit $':>10}  {'P&L%':>8}  {'Reason'}")
    print("  " + "─" * 73)
    for i, t in enumerate(trades, 1):
        print(_trade_row(i, t))
    print()


# ── Chart ─────────────────────────────────────────────────────────────────────

def plot_equity_curve(results) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("[chart] matplotlib not installed — skipping chart.")
        return

    eq     = results.equity
    prices = results.data["close"]
    # Normalise buy-and-hold to same starting capital
    bh     = prices / prices.iloc[0] * eq.iloc[0]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1, 1]})
    fig.suptitle(f"{results.strategy_name} — Equity Curve", fontsize=13, y=0.98)

    # ── Panel 1: Equity curve vs Buy & Hold ──────────────────────────────────
    ax1 = axes[0]
    ax1.plot(eq.index,     eq.values,    color="#00c878", linewidth=2, label="Strategy")
    ax1.plot(bh.index,     bh.values,    color="#888888", linewidth=1, linestyle="--", label="Buy & Hold")

    # Shade drawdown
    peak  = eq.cummax()
    ax1.fill_between(eq.index, peak.values, eq.values,
                     where=eq.values < peak.values, alpha=0.25, color="red", label="Drawdown")

    # Mark trades
    for t in results.trades:
        if t.entry_date in eq.index:
            ax1.axvline(t.entry_date, color="lime",   alpha=0.3, linewidth=0.8)
        if t.exit_date and t.exit_date in eq.index:
            color = "red" if (t.pnl_pct or 0) < 0 else "orange"
            ax1.axvline(t.exit_date, color=color, alpha=0.3, linewidth=0.8)

    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.15)

    # ── Panel 2: Drawdown ─────────────────────────────────────────────────────
    ax2 = axes[1]
    dd  = (eq - peak) / peak * 100
    ax2.fill_between(dd.index, dd.values, 0, color="red", alpha=0.6)
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(alpha=0.15)

    # ── Panel 3: Price + 200 SMA (regime context) ─────────────────────────────
    ax3 = axes[2]
    ax3.plot(results.data.index, results.data["close"],  color="#cccccc", linewidth=0.8)
    ax3.plot(results.data.index, results.data["sma200"], color="#ff4444", linewidth=1, label="200 SMA")
    ax3.set_ylabel("Price")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(alpha=0.15)

    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Fetch data
    df = fetch_ohlcv(
        ticker=args.ticker,
        period=args.period,
        force_refresh=args.refresh,
    )

    # Build strategy with CLI overrides
    strategy = BTCCycleRider(
        score_min   = args.score_min,
        sl_pct      = args.sl,
        trail_pct   = args.trail,
        req_macd_x  = not args.no_macd_cross,
    )
    strategy.name = f"{strategy.name} [{args.ticker}]"

    # Run backtest
    engine = BacktestEngine(
        strategy        = strategy,
        data            = df,
        initial_capital = args.capital,
        commission_pct  = args.commission / 100,
    )
    results = engine.run()

    # Print results
    print_results(results)
    print_trade_list(results)

    # Chart
    if not args.no_chart:
        plot_equity_curve(results)


if __name__ == "__main__":
    main()
