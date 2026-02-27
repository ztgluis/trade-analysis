#!/usr/bin/env python3
"""
Multi-Ticker Backtest Runner
Runs a strategy across a list of tickers and prints a ranked comparison table.

Usage:
    python run_multi.py                                      # default tickers, ABNB strategy
    python run_multi.py --strategy abnb                      # explicit
    python run_multi.py --strategy btc                       # BTC Cycle Rider on all tickers
    python run_multi.py --tickers TSLA NVDA NFLX GOOG       # custom ticker list
    python run_multi.py --tickers BTC-USD ETH-USD SOL-USD --strategy btc
    python run_multi.py --period 3y --sort cagr --no-chart
    python run_multi.py --entry-mode "Strong Buy Only"       # ABNB strategy option
    python run_multi.py --trail 18 --sl 10                   # override stops
    python run_multi.py --save-csv results.csv               # export table to CSV

Run from backend/:
    source venv/bin/activate
    python run_multi.py
"""
import sys
import csv
import argparse
import textwrap
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backtester.data    import fetch_ohlcv
from backtester.engine  import BacktestEngine, BacktestResults
from backtester         import metrics as m
from strategies         import BTCCycleRider, GrowthSignalBot


# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_STOCK_TICKERS = ["ABNB", "NVDA", "TSLA", "NFLX", "GOOG", "NET", "META", "AMZN"]
DEFAULT_CRYPTO_TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD"]

SORT_KEYS = {
    "sharpe":   lambda r: r.get("sharpe",  -999),
    "cagr":     lambda r: r.get("cagr",    -999),
    "pf":       lambda r: r.get("pf",      -999),
    "return":   lambda r: r.get("total_r", -999),
    "drawdown": lambda r: r.get("max_dd",   999),   # lower is better
    "winrate":  lambda r: r.get("win_rate", -999),
}

STATUS_EMOJI = {
    "great": "🟢",
    "ok":    "🟡",
    "bad":   "🔴",
    "error": "❌",
}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a strategy across multiple tickers and compare results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Strategies:
          abnb  — Growth Signal Bot v4 (stock-optimised: MACD+RSI+VWAP+score)
          btc   — BTC Cycle Rider v1 (crypto-optimised: regime+trail stop)

        Periods: 1y 2y 3y 5y 10y max
        Sort:    sharpe (default)  cagr  pf  return  drawdown  winrate
        """),
    )
    p.add_argument("--strategy",    default="abnb",    choices=["abnb", "btc"],
                   help="Strategy to run (default: abnb)")
    p.add_argument("--tickers",     nargs="+",         default=None,
                   help="Space-separated ticker list. Default: stocks for abnb, crypto for btc")
    p.add_argument("--period",      default="5y",
                   help="History length (default: 5y)")
    p.add_argument("--capital",     default=10_000,    type=float,
                   help="Initial capital per ticker USD (default: 10000)")
    p.add_argument("--commission",  default=0.1,       type=float,
                   help="Commission %% per side (default: 0.1)")
    p.add_argument("--sort",        default="sharpe",  choices=list(SORT_KEYS),
                   help="Sort column (default: sharpe)")
    p.add_argument("--sort-asc",    action="store_true",
                   help="Sort ascending (default is descending)")
    # ABNB-specific overrides
    p.add_argument("--entry-mode",  default=None,
                   choices=["Buy Only", "Strong Buy Only", "All Signals"],
                   help="ABNB entry mode override")
    p.add_argument("--sl",          default=None,      type=float, help="Stop loss %%")
    p.add_argument("--tp",          default=None,      type=float, help="Take profit %%")
    p.add_argument("--trail",       default=None,      type=float, help="Trailing stop %% (enables trailing)")
    p.add_argument("--score-min",   default=None,      type=int,   help="Min entry score")
    # BTC-specific overrides
    p.add_argument("--no-macd-cross", action="store_true",
                   help="BTC: allow entries without fresh MACD crossover")
    # Output
    p.add_argument("--no-chart",    action="store_true", help="Skip equity curve chart")
    p.add_argument("--save-csv",    default=None,      metavar="FILE",
                   help="Save results table to CSV file")
    p.add_argument("--refresh",     action="store_true", help="Force re-download data")
    return p.parse_args()


# ── Strategy builder ──────────────────────────────────────────────────────────

def build_strategy(args: argparse.Namespace):
    if args.strategy == "btc":
        kwargs = dict(req_macd_x=not args.no_macd_cross)
        if args.sl    is not None: kwargs["sl_pct"]    = args.sl
        if args.trail is not None: kwargs["trail_pct"] = args.trail
        if args.score_min is not None: kwargs["score_min"] = args.score_min
        return BTCCycleRider(**kwargs)
    else:
        kwargs = {}
        if args.entry_mode is not None: kwargs["entry_mode"] = args.entry_mode
        if args.sl         is not None: kwargs["sl_pct"]     = args.sl
        if args.trail is not None:
            kwargs["use_trail"]  = True
            kwargs["trail_pct"]  = args.trail
        elif args.tp is not None:
            kwargs["tp_pct"] = args.tp
        if args.score_min is not None: kwargs["score_moderate"] = args.score_min
        return GrowthSignalBot(**kwargs)


# ── Run one ticker ────────────────────────────────────────────────────────────

def run_one(ticker: str, strategy, args: argparse.Namespace) -> dict:
    """Returns a flat dict of metrics, or an error dict."""
    try:
        df = fetch_ohlcv(ticker, period=args.period, force_refresh=args.refresh)
        strategy.name = f"{strategy.__class__.__name__} [{ticker}]"

        engine  = BacktestEngine(
            strategy       = strategy,
            data           = df,
            initial_capital= args.capital,
            commission_pct = args.commission / 100,
        )
        results = engine.run()

        eq     = results.equity
        trades = results.trades
        prices = df["close"]

        return {
            "ticker":   ticker,
            "ok":       True,
            "n_trades": len(trades),
            "win_rate": m.win_rate(trades),
            "pf":       m.profit_factor(trades),
            "total_r":  m.total_return(eq),
            "cagr":     m.cagr(eq),
            "bah":      m.buy_hold_return(prices),
            "max_dd":   m.max_drawdown(eq),
            "sharpe":   m.sharpe_ratio(eq),
            "sortino":  m.sortino_ratio(eq),
            "avg_win":  m.avg_winner(trades),
            "avg_loss": m.avg_loser(trades),
            "expect":   m.expectancy(trades),
            "_results": results,   # keep for charting
        }
    except Exception as e:
        print(f"  [!] {ticker}: {e}")
        return {"ticker": ticker, "ok": False, "error": str(e)}


# ── Rating helper ─────────────────────────────────────────────────────────────

def _rate(r: dict) -> str:
    """Quick ✅/⚠️/❌ rating based on key metrics."""
    if not r.get("ok"):
        return STATUS_EMOJI["error"]
    pf = r.get("pf", 0)
    dd = r.get("max_dd", -1)
    sh = r.get("sharpe", 0)
    if pf >= 1.5 and sh >= 0.5 and dd >= -0.35:
        return STATUS_EMOJI["great"]
    if pf >= 1.0 and sh >= 0:
        return STATUS_EMOJI["ok"]
    return STATUS_EMOJI["bad"]


# ── Print summary table ───────────────────────────────────────────────────────

def print_table(rows: list[dict], sort_key: str, ascending: bool, strategy_name: str, period: str) -> None:
    W = 110
    hdr = (
        f"  {'Ticker':<10} {'Trades':>6}  {'WinRate':>7}  {'ProfFact':>8}  "
        f"{'TotRet':>8}  {'CAGR':>7}  {'B&H':>8}  {'MaxDD':>7}  {'Sharpe':>7}  {'Status':>6}"
    )

    print()
    print("═" * W)
    print(f"  MULTI-TICKER BACKTEST — {strategy_name} — {period} Daily   (sorted by {sort_key})")
    print("═" * W)
    print(hdr)
    print("  " + "─" * (W - 2))

    ok_rows    = [r for r in rows if r.get("ok")]
    error_rows = [r for r in rows if not r.get("ok")]

    reverse = not ascending
    ok_rows.sort(key=SORT_KEYS.get(sort_key, SORT_KEYS["sharpe"]), reverse=reverse)

    def _p(v, dec=1):
        return f"{v*100:>+.{dec}f}%"

    for r in ok_rows:
        status = _rate(r)
        print(
            f"  {r['ticker']:<10} "
            f"{r['n_trades']:>6}  "
            f"{r['win_rate']*100:>6.1f}%  "
            f"{r['pf']:>8.2f}  "
            f"{_p(r['total_r']):>8}  "
            f"{_p(r['cagr']):>7}  "
            f"{_p(r['bah']):>8}  "
            f"{_p(r['max_dd']):>7}  "
            f"{r['sharpe']:>7.2f}  "
            f"{status:>6}"
        )

    for r in error_rows:
        print(f"  {r['ticker']:<10}  {'ERROR — ' + r.get('error','')[:70]}")

    print("  " + "─" * (W - 2))

    # Averages row (ok tickers only)
    if ok_rows:
        avg = {k: sum(r[k] for r in ok_rows) / len(ok_rows)
               for k in ["win_rate", "pf", "total_r", "cagr", "bah", "max_dd", "sharpe"]}
        n   = sum(r["n_trades"] for r in ok_rows) // len(ok_rows)
        print(
            f"  {'AVG':<10} "
            f"{n:>6}  "
            f"{avg['win_rate']*100:>6.1f}%  "
            f"{avg['pf']:>8.2f}  "
            f"{_p(avg['total_r']):>8}  "
            f"{_p(avg['cagr']):>7}  "
            f"{_p(avg['bah']):>8}  "
            f"{_p(avg['max_dd']):>7}  "
            f"{avg['sharpe']:>7.2f}  "
            f"{'—':>6}"
        )
    print("═" * W)
    print()


# ── Save CSV ──────────────────────────────────────────────────────────────────

def save_csv(rows: list[dict], path: str) -> None:
    fields = ["ticker", "n_trades", "win_rate", "pf", "total_r",
              "cagr", "bah", "max_dd", "sharpe", "sortino",
              "avg_win", "avg_loss", "expect"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(r for r in rows if r.get("ok"))
    print(f"[csv] Saved {len([r for r in rows if r.get('ok')])} rows to {path}")


# ── Chart all equity curves ───────────────────────────────────────────────────

def plot_all(rows: list[dict], capital: float) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("[chart] matplotlib not installed — skipping.")
        return

    ok = [r for r in rows if r.get("ok") and "_results" in r]
    if not ok:
        return

    cols = min(3, len(ok))
    rows_n = (len(ok) + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(6 * cols, 4 * rows_n), sharex=False)
    axes = [axes] if rows_n == 1 and cols == 1 else (
        [ax for row in axes for ax in (row if hasattr(row, '__iter__') else [row])]
    )

    for i, r in enumerate(ok):
        ax     = axes[i]
        res    = r["_results"]
        eq     = res.equity
        prices = res.data["close"]
        bh     = prices / prices.iloc[0] * capital

        ax.plot(eq.index,  eq.values,  color="#00c878", linewidth=1.5, label="Strategy")
        ax.plot(bh.index,  bh.values,  color="#888",    linewidth=0.8, linestyle="--", label="B&H")
        peak = eq.cummax()
        ax.fill_between(eq.index, peak.values, eq.values,
                        where=eq.values < peak.values, alpha=0.25, color="red")
        ax.set_title(
            f"{r['ticker']}  CAGR {r['cagr']*100:+.1f}%  Sharpe {r['sharpe']:.2f}",
            fontsize=9
        )
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.15)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Hide unused axes
    for j in range(len(ok), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Multi-Ticker Equity Curves", fontsize=12)
    plt.tight_layout()
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Pick default ticker list based on strategy
    if args.tickers:
        tickers = args.tickers
    elif args.strategy == "btc":
        tickers = DEFAULT_CRYPTO_TICKERS
    else:
        tickers = DEFAULT_STOCK_TICKERS

    print(f"\n[multi] Strategy: {args.strategy.upper()}  |  Tickers: {', '.join(tickers)}  |  Period: {args.period}")
    print(f"[multi] Running {len(tickers)} backtests …\n")

    # Build a fresh strategy instance per ticker (avoids state leakage)
    rows: list[dict] = []
    for ticker in tickers:
        strategy = build_strategy(args)
        result   = run_one(ticker, strategy, args)
        rows.append(result)

    # Summary table
    strategy_label = build_strategy(args).name.split("[")[0].strip()
    print_table(rows, sort_key=args.sort, ascending=args.sort_asc,
                strategy_name=strategy_label, period=args.period)

    # Per-ticker detail (top 3 by sort key)
    ok = sorted(
        [r for r in rows if r.get("ok")],
        key=SORT_KEYS.get(args.sort, SORT_KEYS["sharpe"]),
        reverse=not args.sort_asc,
    )
    if ok:
        print(f"  Top performer: {ok[0]['ticker']}  "
              f"CAGR {ok[0]['cagr']*100:+.1f}%  "
              f"Sharpe {ok[0]['sharpe']:.2f}  "
              f"PF {ok[0]['pf']:.2f}")
        if len(ok) > 1:
            print(f"  Weakest:       {ok[-1]['ticker']}  "
                  f"CAGR {ok[-1]['cagr']*100:+.1f}%  "
                  f"Sharpe {ok[-1]['sharpe']:.2f}  "
                  f"PF {ok[-1]['pf']:.2f}")
    print()

    # Save CSV
    if args.save_csv:
        save_csv(rows, args.save_csv)

    # Chart
    if not args.no_chart:
        plot_all(rows, capital=args.capital)


if __name__ == "__main__":
    main()
