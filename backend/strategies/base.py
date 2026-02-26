"""
Abstract base class for all strategies.
Every strategy must implement: prepare(), entry_signal(), exit_signal(), get_stops().
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class BaseStrategy(ABC):
    """
    Contract every strategy must fulfil.

    The engine calls these methods:
      1. prepare(df)             — compute all indicators, add columns to df
      2. entry_signal(bar)       — should we enter on this bar?
      3. exit_signal(bar)        — should we exit the current trade on this bar?
      4. get_stops(bar)          — return (hard_stop_price, trail_pct) for a new entry

    Subclasses set `name` and `warmup_bars` as class attributes.
    """

    # Override in subclass
    name:        str = "Unnamed Strategy"
    warmup_bars: int = 210          # bars skipped before first entry (for SMA warmup)

    @abstractmethod
    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all indicators on the full OHLCV DataFrame.
        Mutate and return df — called ONCE before the bar loop (vectorized, fast).
        """
        ...

    @abstractmethod
    def entry_signal(self, bar: pd.Series) -> bool:
        """
        Return True if the strategy wants to enter a long position on this bar.
        Called at bar close. `bar` is a single row of the prepared DataFrame.
        """
        ...

    @abstractmethod
    def exit_signal(self, bar: pd.Series) -> tuple[bool, str]:
        """
        Return (True, reason) to exit the current trade at bar close.
        Return (False, "") to stay in.
        Called AFTER stop checks — this is the signal-based exit.
        """
        ...

    @abstractmethod
    def get_stops(self, bar: pd.Series) -> tuple[Optional[float], Optional[float]]:
        """
        Return (hard_stop_price, trail_pct) for a newly opened position.
        Called immediately after an entry.
          hard_stop_price : absolute price level (e.g. entry * 0.92 for 8% stop)
          trail_pct       : trailing stop distance as %, e.g. 8.0 for 8%
        Return (None, None) to disable stops.
        """
        ...
