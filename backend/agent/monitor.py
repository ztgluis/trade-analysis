"""
agent/monitor.py — Standalone monitoring agent.

Periodically scans a watchlist + optional universe of tickers,
evaluates signals using SignalEvaluator, and dispatches alerts
when signal state changes.

Runs independently of Streamlit — launch via run_monitor.py CLI.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from backtester.data import fetch_ohlcv
from signals.config import SignalConfig
from signals.evaluator import SignalEvaluator, SignalResult
from .alerts import AlertDispatcher, ConsoleAlert


class MonitorAgent:
    """Signal monitoring agent with configurable scan interval."""

    def __init__(
        self,
        config: SignalConfig,
        watchlist: list[str],
        universe: list[str] | None = None,
        interval_minutes: int = 60,
        alert_dispatcher: AlertDispatcher | None = None,
        data_period: str = "1y",
        data_interval: str = "1d",
        max_workers: int = 4,
    ):
        self.config = config
        self.evaluator = SignalEvaluator(config)
        self.watchlist = list(watchlist)
        self.universe = list(universe) if universe else []
        self.interval = interval_minutes
        self.alerts = alert_dispatcher or ConsoleAlert()
        self.data_period = data_period
        self.data_interval = data_interval
        self.max_workers = max_workers

        # State tracking for debouncing
        self._last_signals: dict[str, str] = {}   # ticker → last signal_type
        self._last_verdicts: dict[str, str] = {}   # ticker → last verdict

    @property
    def all_tickers(self) -> list[str]:
        """Combined unique tickers from watchlist + universe."""
        seen = set()
        result = []
        for t in self.watchlist + self.universe:
            if t not in seen:
                seen.add(t)
                result.append(t)
        return result

    def scan_ticker(self, ticker: str) -> SignalResult | None:
        """Fetch latest data and evaluate signals for one ticker."""
        try:
            df = fetch_ohlcv(
                ticker,
                period=self.data_period,
                interval=self.data_interval,
                force_refresh=True,
            )
            if df is None or len(df) < 220:
                print(f"[monitor] {ticker}: insufficient data ({len(df) if df is not None else 0} bars)")
                return None
            return self.evaluator.evaluate_latest(df, ticker=ticker)
        except Exception as exc:
            print(f"[monitor] {ticker}: error — {exc}")
            return None

    def scan_all(self) -> dict[str, SignalResult]:
        """Scan all tickers in parallel. Returns {ticker: SignalResult}."""
        tickers = self.all_tickers
        results: dict[str, SignalResult] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self.scan_ticker, t): t for t in tickers}
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        results[ticker] = result
                except Exception as exc:
                    print(f"[monitor] {ticker}: exception — {exc}")

        return results

    def _is_new_signal(self, ticker: str, result: SignalResult) -> bool:
        """Debounce: only alert if verdict changed since last scan."""
        prev_verdict = self._last_verdicts.get(ticker)
        if prev_verdict is None:
            # First scan — alert on any actionable signal
            return result.verdict not in ("WAIT",)
        return result.verdict != prev_verdict

    def run_once(self) -> dict[str, SignalResult]:
        """Single scan cycle: evaluate all tickers, dispatch alerts for changes."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tickers = self.all_tickers
        print(f"\n[monitor] Scan started at {ts} — {len(tickers)} tickers")

        results = self.scan_all()

        # Summary table
        alerts_fired = 0
        print(f"\n{'Ticker':<8} {'Price':>10} {'Bull':>6} {'Bear':>6} {'Regime':<8} {'Verdict':<16} {'Alert'}")
        print("-" * 72)

        for ticker in self.all_tickers:
            result = results.get(ticker)
            if result is None:
                print(f"{ticker:<8} {'ERROR':>10}")
                continue

            is_new = self._is_new_signal(ticker, result)
            alert_str = " *** NEW ***" if is_new else ""

            print(
                f"{ticker:<8} "
                f"${result.price:>9.2f} "
                f"{result.bull_score:>5.1f} "
                f"{result.bear_score:>5.1f} "
                f"{result.regime:<8} "
                f"{result.verdict:<16} "
                f"{alert_str}"
            )

            if is_new:
                self.alerts.dispatch(ticker, result)
                alerts_fired += 1

            # Update state
            self._last_signals[ticker] = result.signal_type
            self._last_verdicts[ticker] = result.verdict

        print(f"\n[monitor] Scan complete — {len(results)} evaluated, {alerts_fired} alerts fired")
        return results

    def run_loop(self) -> None:
        """Blocking loop: runs scan_all at configured interval."""
        print(f"[monitor] Starting monitoring loop — interval {self.interval} min")
        print(f"[monitor] Watchlist: {self.watchlist}")
        if self.universe:
            print(f"[monitor] Universe: {len(self.universe)} tickers")
        print(f"[monitor] Config: {self.config.name}")
        print(f"[monitor] Press Ctrl+C to stop\n")

        try:
            while True:
                self.run_once()
                print(f"\n[monitor] Next scan in {self.interval} minutes...")
                time.sleep(self.interval * 60)
        except KeyboardInterrupt:
            print("\n[monitor] Stopped by user")
