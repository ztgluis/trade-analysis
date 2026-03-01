"""
agent/alerts.py — Pluggable alert dispatchers.

V1: ConsoleAlert — structured stdout logging.
Future: EmailAlert, SlackAlert, PushoverAlert, etc.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime

from signals.evaluator import SignalResult


class AlertDispatcher(ABC):
    """Base class for alert delivery mechanisms."""

    @abstractmethod
    def dispatch(self, ticker: str, result: SignalResult) -> None:
        """Send an alert for a fired signal."""
        ...


class ConsoleAlert(AlertDispatcher):
    """Print structured signal alerts to stdout."""

    def dispatch(self, ticker: str, result: SignalResult) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        bull_pct = result.bull_score_pct
        bear_pct = result.bear_score_pct

        # Header
        print(f"\n{'='*60}")
        print(f"  ALERT  {ts}")
        print(f"{'='*60}")
        print(f"  Ticker:   {ticker}")
        print(f"  Price:    ${result.price:.2f}")
        print(f"  Regime:   {result.regime.upper()}")
        print(f"  Verdict:  {result.verdict}")
        print(f"  Action:   {result.verdict_action}")

        # Scores
        print(f"\n  Bull Score: {result.bull_score:.1f} / {result.bull_max:.0f}  ({bull_pct:.0f}%)")
        print(f"  Bear Score: {result.bear_score:.1f} / {result.bear_max:.0f}  ({bear_pct:.0f}%)")

        # Component breakdown
        if result.bull_components:
            print(f"\n  Bull Components:")
            for name, val in sorted(result.bull_components.items(), key=lambda x: -x[1]):
                if val > 0:
                    print(f"    {name:<20} +{val:.1f}")

        # Signal
        if result.signal_type:
            print(f"\n  Signal: {result.signal_type.upper()}")
            print(f"  Entry:  {'YES' if result.should_enter else 'NO'}")

        if result.should_exit:
            print(f"\n  EXIT:   {result.exit_reason}")

        print(f"{'='*60}\n")


class JsonLogAlert(AlertDispatcher):
    """Output alerts as JSON lines (for log aggregation / piping)."""

    def dispatch(self, ticker: str, result: SignalResult) -> None:
        record = {
            "ts": datetime.utcnow().isoformat(),
            "ticker": ticker,
            "price": result.price,
            "regime": result.regime,
            "verdict": result.verdict,
            "bull_score": result.bull_score,
            "bear_score": result.bear_score,
            "bull_max": result.bull_max,
            "bear_max": result.bear_max,
            "signal_type": result.signal_type,
            "should_enter": result.should_enter,
            "should_exit": result.should_exit,
        }
        print(json.dumps(record))
