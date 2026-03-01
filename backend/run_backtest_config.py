#!/usr/bin/env python3
"""
run_backtest_config.py — Backtest a signal configuration against historical data.

Usage:
    python run_backtest_config.py --config configs/signals.yaml --ticker GOOG --period 5y
    python run_backtest_config.py --config configs/signals.yaml --tickers GOOG,META,AMZN
    python run_backtest_config.py --ticker GOOG --period 5y  # uses legacy config
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backtester.data import fetch_ohlcv
from backtester.engine import BacktestEngine
from signals.config import SignalConfig
from strategies.configurable_strategy import ConfigurableStrategy


def print_summary(ticker: str, s: dict) -> None:
    """Print a formatted summary for one ticker."""
    print(f"\n{'='*55}")
    print(f"  {ticker}  —  {s['strategy']}")
    print(f"  {s['period']}  ({s['bars']} bars)")
    print(f"{'='*55}")
    print(f"\n  RETURNS:")
    print(f"    Total Return:       {s['total_return']:>+8.1%}")
    print(f"    CAGR:               {s['cagr']:>+8.1%}")
    print(f"    Buy & Hold Return:  {s['buy_hold']:>+8.1%}")
    print(f"\n  RISK:")
    print(f"    Max Drawdown:       {s['max_drawdown']:>+8.1%}")
    print(f"    Sharpe Ratio:       {s['sharpe']:>8.2f}")
    print(f"    Sortino Ratio:      {s['sortino']:>8.2f}")
    print(f"    Calmar Ratio:       {s['calmar']:>8.2f}")
    print(f"\n  TRADES:")
    print(f"    Total Trades:       {s['n_trades']:>8}")
    wr = s['win_rate']
    wins = round(s['n_trades'] * wr)
    losses = s['n_trades'] - wins
    print(f"    Win Rate:           {wr:>7.1%}  ({wins}W / {losses}L)")
    print(f"    Profit Factor:      {s['profit_factor']:>8.2f}")
    print(f"    Avg Winner:         {s['avg_winner']:>+8.1%}")
    print(f"    Avg Loser:          {s['avg_loser']:>+8.1%}")
    print(f"    Expectancy:         {s['expectancy']:>+8.1%}")


def print_comparison_table(rows: list[dict]) -> None:
    """Print a multi-ticker comparison table."""
    print(f"\n{'Ticker':<8} {'Trades':>6} {'WinRate':>8} {'PF':>6} "
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
        avg = {k: sum(r[k] for r in rows) / len(rows)
               for k in ['n_trades', 'win_rate', 'profit_factor',
                         'total_return', 'cagr', 'buy_hold', 'max_drawdown', 'sharpe']}
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


def main():
    parser = argparse.ArgumentParser(description="Backtest a signal config YAML")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to signal config YAML (default: legacy v4.3)")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Single ticker to backtest")
    parser.add_argument("--tickers", type=str, default=None,
                        help="Comma-separated tickers for comparison")
    parser.add_argument("--period", type=str, default="5y",
                        help="Data history period (default: 5y)")
    parser.add_argument("--interval", type=str, default="1d",
                        help="Bar interval (default: 1d)")
    parser.add_argument("--capital", type=float, default=10_000,
                        help="Initial capital (default: 10000)")
    parser.add_argument("--commission", type=float, default=0.001,
                        help="Commission per side (default: 0.1%%)")
    args = parser.parse_args()

    # Load config
    if args.config:
        config = SignalConfig.from_yaml(args.config)
    else:
        config = SignalConfig.from_legacy()

    print(f"Config: {config.name}")
    print(f"Max score: {config.max_score:.0f}  |  "
          f"Moderate: {config.score_moderate:.1f}  |  "
          f"Strong: {config.score_strong:.1f}")

    # Determine tickers
    if args.ticker:
        tickers = [args.ticker.upper()]
    elif args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    else:
        tickers = ["GOOG"]

    strategy = ConfigurableStrategy(config)
    rows = []

    for ticker in tickers:
        try:
            df = fetch_ohlcv(ticker, period=args.period, interval=args.interval)
            if df is None or len(df) < 220:
                print(f"\n[!] {ticker}: insufficient data, skipping")
                continue

            engine = BacktestEngine(
                strategy, df,
                initial_capital=args.capital,
                commission_pct=args.commission,
            )
            results = engine.run()
            summary = results.summary()
            summary["ticker"] = ticker
            rows.append(summary)

            if len(tickers) == 1:
                print_summary(ticker, summary)
        except Exception as exc:
            print(f"\n[!] {ticker}: {exc}")

    if len(tickers) > 1 and rows:
        print_comparison_table(rows)


if __name__ == "__main__":
    main()
