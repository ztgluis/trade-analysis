"""
Portfolio and position management.
Simulates Pine Script strategy() with process_orders_on_close=True:
  - Signals fire at bar close → fills execute at that same close price
  - Trailing stop tracks the HIGH of each bar (intrabar peak)
  - Hard stop and trailing stop are both active simultaneously
  - No pyramiding (one open position at a time)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np


@dataclass
class Trade:
    """One completed (or open) trade."""
    entry_date:  pd.Timestamp
    entry_price: float
    direction:   str            # "long" (shorts not yet implemented)
    qty:         float          # units held (shares / BTC / etc.)

    exit_date:   Optional[pd.Timestamp] = None
    exit_price:  Optional[float]        = None
    exit_reason: Optional[str]          = None

    # ── Derived properties ──────────────────────────────────────────────────
    @property
    def is_open(self) -> bool:
        return self.exit_date is None

    @property
    def is_winner(self) -> Optional[bool]:
        p = self.pnl_pct
        return (p > 0) if p is not None else None

    @property
    def pnl_pct(self) -> Optional[float]:
        """Return as fraction of entry price, e.g. 0.12 = +12%."""
        if self.exit_price is None:
            return None
        if self.direction == "long":
            return (self.exit_price - self.entry_price) / self.entry_price
        return (self.entry_price - self.exit_price) / self.entry_price

    @property
    def pnl_dollar(self) -> Optional[float]:
        """Dollar P&L (after per-share commission was already deducted in Portfolio)."""
        if self.exit_price is None or self.pnl_pct is None:
            return None
        return self.qty * self.entry_price * self.pnl_pct

    @property
    def duration_bars(self) -> Optional[int]:
        return None  # set externally by engine if needed


class Portfolio:
    """
    Tracks capital, open position, stops, and equity curve.

    Args:
        initial_capital: Starting cash in USD
        commission_pct:  Round-trip commission per side as fraction (0.001 = 0.1%)
        position_pct:    Fraction of current equity to invest per trade (1.0 = 100%)
    """

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        commission_pct:  float = 0.001,
        position_pct:    float = 1.0,
    ) -> None:
        self.initial_capital = initial_capital
        self.commission_pct  = commission_pct
        self.position_pct    = position_pct

        self.cash:   float = initial_capital
        self.equity: float = initial_capital
        self.trades: list[Trade] = []

        # Stop state (reset after each exit)
        self._hard_stop:  Optional[float] = None
        self._trail_pct:  Optional[float] = None   # % below peak
        self._trail_peak: Optional[float] = None   # highest high since entry
        self._tp_price:   Optional[float] = None   # fixed take-profit price

        # (date, equity) pairs for equity curve
        self.equity_curve: list[tuple[pd.Timestamp, float]] = []

    # ── Read-only properties ─────────────────────────────────────────────────

    @property
    def open_trade(self) -> Optional[Trade]:
        return next((t for t in reversed(self.trades) if t.is_open), None)

    @property
    def in_position(self) -> bool:
        return self.open_trade is not None

    @property
    def trail_stop_level(self) -> Optional[float]:
        if self._trail_peak is not None and self._trail_pct is not None:
            return self._trail_peak * (1.0 - self._trail_pct / 100.0)
        return None

    # ── Entry ────────────────────────────────────────────────────────────────

    def enter_long(
        self,
        date:    pd.Timestamp,
        price:   float,
        comment: str = "",
    ) -> None:
        if self.in_position:
            return  # no pyramiding

        invest = self.cash * self.position_pct
        qty    = invest / price
        fee    = invest * self.commission_pct
        self.cash -= (invest + fee)
        self.trades.append(Trade(
            entry_date=date, entry_price=price, direction="long", qty=qty
        ))

    def set_stops(
        self,
        hard_stop: Optional[float],
        trail_pct: Optional[float],
        tp_price:  Optional[float] = None,
    ) -> None:
        """
        Call immediately after enter_long to register SL / trailing stop / TP.

        Modes (set the params that apply, leave others as None):
          Fixed SL + Fixed TP : hard_stop=price, trail_pct=None, tp_price=price
          Fixed SL + Trailing  : hard_stop=price, trail_pct=pct,  tp_price=None
          All three active     : hard_stop=price, trail_pct=pct,  tp_price=price
        """
        self._hard_stop  = hard_stop
        self._trail_pct  = trail_pct
        self._trail_peak = self.open_trade.entry_price if self.open_trade else None
        self._tp_price   = tp_price

    # ── Exit ─────────────────────────────────────────────────────────────────

    def exit_long(
        self,
        date:   pd.Timestamp,
        price:  float,
        reason: str = "",
    ) -> None:
        trade = self.open_trade
        if not trade:
            return

        fee        = trade.qty * price * self.commission_pct
        self.cash += trade.qty * price - fee

        trade.exit_date   = date
        trade.exit_price  = price
        trade.exit_reason = reason

        # Reset stop state
        self._hard_stop  = None
        self._trail_pct  = None
        self._trail_peak = None
        self._tp_price   = None
        self.equity = self.cash

    # ── Stop checking (called once per bar while in position) ─────────────────

    def update_and_check_stops(
        self,
        bar_high: float,
        bar_low:  float,
    ) -> tuple[Optional[float], Optional[str]]:
        """
        1. Update trailing peak with bar's high.
        2. Check if hard stop or trailing stop was breached by bar's low.

        Returns (exit_price, reason) or (None, None).

        Uses bar LOW for stop checks — conservative assumption (worst fill
        within the bar). Hard stop fires at the hard_stop level (not lower),
        which matches Pine Script's stop= behaviour.
        """
        if not self.in_position:
            return None, None

        # Update peak
        if self._trail_peak is not None:
            self._trail_peak = max(self._trail_peak, bar_high)

        trail_level = self.trail_stop_level

        # Take-profit: bar high reached TP level (checked first — optimistic fill)
        if self._tp_price and bar_high >= self._tp_price:
            return self._tp_price, "Take Profit"

        # Hard stop (protects against large gaps)
        if self._hard_stop and bar_low <= self._hard_stop:
            return self._hard_stop, "Hard Stop"

        # Trailing stop
        if trail_level and bar_low <= trail_level:
            return trail_level, "Trailing Stop"

        return None, None

    # ── Equity mark-to-market ─────────────────────────────────────────────────

    def update_equity(self, date: pd.Timestamp, close: float) -> None:
        if self.open_trade:
            self.equity = self.cash + self.open_trade.qty * close
        else:
            self.equity = self.cash
        self.equity_curve.append((date, self.equity))

    def equity_series(self) -> pd.Series:
        if not self.equity_curve:
            return pd.Series(dtype=float)
        dates, values = zip(*self.equity_curve)
        return pd.Series(list(values), index=pd.DatetimeIndex(list(dates)), name="equity")

    # ── Convenience ───────────────────────────────────────────────────────────

    def closed_trades(self) -> list[Trade]:
        return [t for t in self.trades if not t.is_open]

    def winners(self) -> list[Trade]:
        return [t for t in self.closed_trades() if t.is_winner]

    def losers(self) -> list[Trade]:
        return [t for t in self.closed_trades() if not t.is_winner]
