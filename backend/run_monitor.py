#!/usr/bin/env python3
"""
run_monitor.py — Start the signal monitoring agent.

Usage:
    python run_monitor.py                                     # default config + watchlist
    python run_monitor.py --config configs/signals.yaml       # custom config
    python run_monitor.py --watchlist GOOG,META --interval 60 # custom tickers, hourly
    python run_monitor.py --universe sp100                    # add S&P 100 scan
    python run_monitor.py --once                              # single scan, no loop
    python run_monitor.py --once --watchlist GOOG --json      # JSON output
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from signals.config import SignalConfig
from agent.monitor import MonitorAgent
from agent.alerts import ConsoleAlert, JsonLogAlert

DEFAULT_WATCHLIST = ["META", "AAPL", "AMZN", "NFLX", "GOOG", "NVDA", "TSLA", "SPY", "QQQ"]

# Top S&P 100 tickers (subset — full list from scanner/universe.py)
SP100 = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMD", "AMGN", "AMT", "AMZN",
    "AVGO", "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY", "BRK.B", "C",
    "CAT", "CHTR", "CI", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO",
    "CVS", "CVX", "DE", "DHR", "DIS", "DUK", "EMR", "EXC", "F", "FDX",
    "GD", "GE", "GILD", "GM", "GOOG", "GS", "HD", "HON", "IBM", "INTC",
    "INTU", "ISRG", "JNJ", "JPM", "KHC", "KO", "LIN", "LLY", "LMT", "LOW",
    "MA", "MCD", "MDLZ", "MDT", "MET", "META", "MMM", "MO", "MRK", "MS",
    "MSFT", "NEE", "NFLX", "NKE", "NOW", "NVDA", "ORCL", "PEP", "PFE", "PG",
    "PM", "PYPL", "QCOM", "RTX", "SBUX", "SCHW", "SO", "SPG", "T", "TGT",
    "TMO", "TMUS", "TSLA", "TXN", "UNH", "UNP", "UPS", "USB", "V", "VZ",
    "WFC", "WMT", "XOM",
]


def main():
    parser = argparse.ArgumentParser(description="Signal monitoring agent")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to signal config YAML (default: legacy v4.3)")
    parser.add_argument("--watchlist", type=str, default=None,
                        help="Comma-separated tickers (default: FAANG+)")
    parser.add_argument("--universe", type=str, default=None,
                        choices=["sp100", "none"],
                        help="Additional universe to scan")
    parser.add_argument("--interval", type=int, default=60,
                        help="Scan interval in minutes (default: 60)")
    parser.add_argument("--once", action="store_true",
                        help="Run single scan then exit")
    parser.add_argument("--json", action="store_true",
                        help="Output alerts as JSON lines")
    parser.add_argument("--period", type=str, default="1y",
                        help="Data history period (default: 1y)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel download workers (default: 4)")
    args = parser.parse_args()

    # Load config
    if args.config:
        config = SignalConfig.from_yaml(args.config)
    else:
        config = SignalConfig.from_legacy()

    # Watchlist
    if args.watchlist:
        watchlist = [t.strip().upper() for t in args.watchlist.split(",")]
    else:
        watchlist = DEFAULT_WATCHLIST

    # Universe
    universe = []
    if args.universe == "sp100":
        universe = SP100

    # Alert dispatcher
    dispatcher = JsonLogAlert() if args.json else ConsoleAlert()

    # Create and run agent
    agent = MonitorAgent(
        config=config,
        watchlist=watchlist,
        universe=universe,
        interval_minutes=args.interval,
        alert_dispatcher=dispatcher,
        data_period=args.period,
        max_workers=args.workers,
    )

    if args.once:
        agent.run_once()
    else:
        agent.run_loop()


if __name__ == "__main__":
    main()
