"""
Core backtesting engine — bar-by-bar simulation loop.

Design principles:
  1. Indicators are computed VECTORIZED on the full dataset first (fast).
  2. The bar loop handles ORDER LOGIC only (slow but unavoidable).
  3. No look-ahead bias:
       - process_orders_on_close=True (same as Pine Script default)
       - Signals computed from bar[i] → fill at close[i] (same bar)
       - Stop checks use bar[i].high/low (intrabar range)
  4. Stops are checked BEFORE signal exits on each bar (more conservative).
  5. One position at a time (no pyramiding).
"""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np

from .portfolio import Portfolio, Trade
from . import metrics as m


@dataclass
class BacktestResults:
    """Container for all backtest output — passed to reporting functions."""
    portfolio:   Portfolio
    data:        pd.DataFrame      # full OHLCV + indicators DataFrame
    strategy_name: str = ""

    # ── Convenience shortcuts ────────────────────────────────────────────────

    @property
    def trades(self) -> list[Trade]:
        return self.portfolio.closed_trades()

    @property
    def equity(self) -> pd.Series:
        return self.portfolio.equity_series()

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    def summary(self) -> dict:
        eq     = self.equity
        trades = self.trades
        prices = self.data["close"]

        # Auto-detect bar interval for correct Sharpe/Sortino annualisation.
        # Sharpe requires knowing bars-per-year (daily=252, 1H≈1638, weekly=52).
        # CAGR and MaxDD use actual calendar dates so they're always correct.
        if len(eq) > 10:
            median_gap_h = (
                pd.Series(eq.index.astype("int64")).diff().dropna().median()
                / 3_600_000_000_000   # nanoseconds → hours
            )
            if median_gap_h < 2:        # 1H (or sub-hour) bars
                periods_yr = int(round(252 * 6.5))   # ~1638
            elif median_gap_h < 25:     # daily bars
                periods_yr = 252
            elif median_gap_h < 200:    # weekly bars
                periods_yr = 52
            else:                       # monthly bars
                periods_yr = 12
        else:
            periods_yr = 252

        return {
            "strategy":       self.strategy_name,
            "period":         f"{eq.index[0].date()} → {eq.index[-1].date()}",
            "bars":           len(self.data),
            "total_return":   m.total_return(eq),
            "cagr":           m.cagr(eq),
            "buy_hold":       m.buy_hold_return(prices),
            "max_drawdown":   m.max_drawdown(eq),
            "sharpe":         m.sharpe_ratio(eq, periods_per_year=periods_yr),
            "sortino":        m.sortino_ratio(eq, periods_per_year=periods_yr),
            "calmar":         m.calmar_ratio(eq),
            "n_trades":       len(trades),
            "win_rate":       m.win_rate(trades),
            "profit_factor":  m.profit_factor(trades),
            "avg_winner":     m.avg_winner(trades),
            "avg_loser":      m.avg_loser(trades),
            "expectancy":     m.expectancy(trades),
            "best_trade":     m.best_trade(trades),
            "worst_trade":    m.worst_trade(trades),
        }


class BacktestEngine:
    """
    Runs a strategy against historical OHLCV data.

    Usage:
        engine  = BacktestEngine(strategy, df, initial_capital=10_000)
        results = engine.run()
        print_results(results)
    """

    def __init__(
        self,
        strategy,
        data:            pd.DataFrame,
        initial_capital: float = 10_000.0,
        commission_pct:  float = 0.001,      # 0.1% per side
        position_pct:    float = 1.0,        # 100% of equity per trade
    ) -> None:
        self.strategy        = strategy
        self.raw_data        = data.copy()
        self.initial_capital = initial_capital
        self.commission_pct  = commission_pct
        self.position_pct    = position_pct

    def run(self) -> BacktestResults:
        # ── Step 1: compute all indicators vectorized ────────────────────────
        print(f"[engine] Preparing indicators …")
        df = self.strategy.prepare(self.raw_data.copy())

        warmup = self.strategy.warmup_bars
        total  = len(df)
        print(f"[engine] {total} bars total, {warmup} warmup, "
              f"{total - warmup} tradeable bars")

        # ── Step 2: initialise portfolio ──────────────────────────────────────
        portfolio = Portfolio(
            initial_capital=self.initial_capital,
            commission_pct=self.commission_pct,
            position_pct=self.position_pct,
        )

        # ── Step 3: bar-by-bar simulation ─────────────────────────────────────
        for i in range(warmup, total):
            bar = df.iloc[i]

            # 3a. Check stops (uses bar HIGH/LOW — intrabar range)
            if portfolio.in_position:
                exit_px, reason = portfolio.update_and_check_stops(
                    bar_high=float(bar["high"]),
                    bar_low=float(bar["low"]),
                )
                if exit_px:
                    portfolio.exit_long(bar.name, exit_px, reason)

            # 3b. Check signal-based exits (at bar close)
            if portfolio.in_position:
                should_exit, exit_reason = self.strategy.exit_signal(bar)
                if should_exit:
                    portfolio.exit_long(bar.name, float(bar["close"]), exit_reason)

            # 3c. Check entries (at bar close, only when flat)
            if not portfolio.in_position:
                if self.strategy.entry_signal(bar):
                    entry_px  = float(bar["close"])
                    portfolio.enter_long(bar.name, entry_px)
                    hard_stop, trail_pct = self.strategy.get_stops(bar)
                    # Fixed TP: strategy may optionally expose get_tp()
                    tp_price = None
                    if hasattr(self.strategy, "get_tp"):
                        tp_price = self.strategy.get_tp(entry_px)
                    portfolio.set_stops(hard_stop, trail_pct, tp_price=tp_price)

            # 3d. Mark-to-market equity
            portfolio.update_equity(bar.name, float(bar["close"]))

        # ── Step 4: force-close any open position at end of data ──────────────
        if portfolio.in_position:
            last = df.iloc[-1]
            portfolio.exit_long(last.name, float(last["close"]), "End of Data")
            portfolio.update_equity(last.name, float(last["close"]))

        print(f"[engine] Done — {len(portfolio.closed_trades())} trades completed.")
        return BacktestResults(
            portfolio=portfolio,
            data=df,
            strategy_name=self.strategy.name,
        )
