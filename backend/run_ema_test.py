#!/usr/bin/env python3
"""
run_ema_test.py — Test the EMA Channel strategy across multiple tickers.

Usage:
    python run_ema_test.py --tickers GOOG,META,NVDA,AAPL,AMZN,TSLA,GLD,SLV --period 5y
    python run_ema_test.py --tickers GOOG,META,NVDA --exit-mode fast
    python run_ema_test.py --tickers GOOG,META --exit-mode all --atr-mult 1.5:3.0:0.5
    python run_ema_test.py --tickers GOOG --compare-exits   # compare all 3 exit modes
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backtester.data import fetch_ohlcv
from backtester.engine import BacktestEngine
from strategies.ema_channel_strategy import EmaChannelStrategy


DEFAULT_TICKERS = "GOOG,META,NVDA,AAPL,AMZN,TSLA,MSFT,GLD"


def run_backtest(strategy, ticker, period, interval, capital, commission):
    """Run a single backtest, return summary dict or None on failure."""
    try:
        df = fetch_ohlcv(ticker, period=period, interval=interval)
        if df is None or len(df) < 220:
            print(f"  [!] {ticker}: insufficient data, skipping")
            return None
        engine = BacktestEngine(strategy, df, initial_capital=capital,
                                commission_pct=commission)
        results = engine.run()
        summary = results.summary()
        summary["ticker"] = ticker
        return summary
    except Exception as exc:
        print(f"  [!] {ticker}: {exc}")
        return None


def print_table(rows, title=None):
    """Print a comparison table."""
    if title:
        print(f"\n{title}")
    print(f"{'Ticker':<8} {'Trades':>6} {'WinRate':>8} {'PF':>6} "
          f"{'TotRet':>8} {'CAGR':>7} {'B&H':>7} {'MaxDD':>7} {'Sharpe':>7}")
    print("-" * 72)
    for r in rows:
        print(
            f"{r['ticker']:<8} "
            f"{r['n_trades']:>6} "
            f"{r['win_rate']:>7.1%} "
            f"{r['profit_factor']:>6.2f} "
            f"{r['total_return']:>+7.1%} "
            f"{r['cagr']:>+6.1%} "
            f"{r['buy_hold']:>+6.1%} "
            f"{r['max_drawdown']:>+6.1%} "
            f"{r['sharpe']:>7.2f}"
        )

    if len(rows) > 1:
        keys = ['n_trades', 'win_rate', 'profit_factor',
                'total_return', 'cagr', 'buy_hold', 'max_drawdown', 'sharpe']
        avg = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
        print("-" * 72)
        print(
            f"{'AVG':<8} "
            f"{avg['n_trades']:>6.0f} "
            f"{avg['win_rate']:>7.1%} "
            f"{avg['profit_factor']:>6.2f} "
            f"{avg['total_return']:>+7.1%} "
            f"{avg['cagr']:>+6.1%} "
            f"{avg['buy_hold']:>+6.1%} "
            f"{avg['max_drawdown']:>+6.1%} "
            f"{avg['sharpe']:>7.2f}"
        )


def run_single_mode(tickers, args, exit_mode, atr_mult):
    """Run backtests for all tickers with one configuration."""
    strategy = EmaChannelStrategy(
        exit_mode=exit_mode,
        atr_mult=atr_mult,
        fast_len=args.fast_len,
        med_len=args.med_len,
        slow_len=args.slow_len,
    )
    rows = []
    for ticker in tickers:
        summary = run_backtest(strategy, ticker, args.period, args.interval,
                               args.capital, args.commission)
        if summary:
            rows.append(summary)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Test EMA Channel strategy across multiple tickers")
    parser.add_argument("--tickers", type=str, default=DEFAULT_TICKERS,
                        help=f"Comma-separated tickers (default: {DEFAULT_TICKERS})")
    parser.add_argument("--period", type=str, default="5y")
    parser.add_argument("--interval", type=str, default="1d")
    parser.add_argument("--capital", type=float, default=10_000)
    parser.add_argument("--commission", type=float, default=0.001)

    parser.add_argument("--exit-mode", type=str, default="all",
                        choices=["all", "fast", "any"],
                        help="Exit mode (default: all)")
    parser.add_argument("--atr-mult", type=str, default="2.0",
                        help="ATR multiplier or sweep spec min:max:step (default: 2.0)")
    parser.add_argument("--compare-exits", action="store_true",
                        help="Compare all 3 exit modes side by side")

    parser.add_argument("--fast-len", type=int, default=5, help="Fast EMA length")
    parser.add_argument("--med-len", type=int, default=100, help="Medium EMA length")
    parser.add_argument("--slow-len", type=int, default=200, help="Slow EMA length")

    args = parser.parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",")]

    print(f"EMA Channel Strategy  |  EMAs: {args.fast_len}/{args.med_len}/{args.slow_len}")
    print(f"Tickers: {', '.join(tickers)}  |  Period: {args.period}")

    # ── Compare exits mode ───────────────────────────────────────────
    if args.compare_exits:
        atr_mult = float(args.atr_mult.split(":")[0])
        print(f"ATR mult: {atr_mult}")

        for mode in ["all", "fast", "any"]:
            rows = run_single_mode(tickers, args, mode, atr_mult)
            if rows:
                print_table(rows, title=f"\n--- Exit mode: {mode.upper()} ---")
        return

    # ── ATR sweep mode ───────────────────────────────────────────────
    if ":" in args.atr_mult:
        parts = args.atr_mult.split(":")
        lo, hi, step = float(parts[0]), float(parts[1]), float(parts[2])
        atr_values = []
        v = lo
        while v <= hi + 1e-9:
            atr_values.append(round(v, 2))
            v += step

        print(f"Exit mode: {args.exit_mode}  |  ATR sweep: {atr_values}")

        for atr_mult in atr_values:
            rows = run_single_mode(tickers, args, args.exit_mode, atr_mult)
            if rows:
                print_table(rows, title=f"\n--- ATR mult: {atr_mult} ---")
        return

    # ── Standard single run ──────────────────────────────────────────
    atr_mult = float(args.atr_mult)
    print(f"Exit mode: {args.exit_mode}  |  ATR mult: {atr_mult}")

    rows = run_single_mode(tickers, args, args.exit_mode, atr_mult)
    if rows:
        print_table(rows)

        # Print detailed summary for single ticker
        if len(rows) == 1:
            r = rows[0]
            print(f"\n  Avg Winner:    {r['avg_winner']:>+7.1%}")
            print(f"  Avg Loser:     {r['avg_loser']:>+7.1%}")
            print(f"  Expectancy:    {r['expectancy']:>+7.1%}")
            print(f"  Sortino:       {r['sortino']:>7.2f}")
            print(f"  Calmar:        {r['calmar']:>7.2f}")


if __name__ == "__main__":
    main()
