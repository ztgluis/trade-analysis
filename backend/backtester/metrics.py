"""
Performance metrics for backtesting results.
All functions take a pandas Series or list of Trade objects.
"""
from __future__ import annotations
import math
import pandas as pd
import numpy as np
from .portfolio import Trade


# ── Return metrics ────────────────────────────────────────────────────────────

def total_return(equity: pd.Series) -> float:
    """Total return as a fraction, e.g. 1.45 = +145%."""
    if len(equity) < 2:
        return 0.0
    return (equity.iloc[-1] / equity.iloc[0]) - 1.0


def cagr(equity: pd.Series) -> float:
    """Compound Annual Growth Rate."""
    if len(equity) < 2:
        return 0.0
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    return (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0


def buy_hold_return(prices: pd.Series) -> float:
    """Simple buy-and-hold return over the same period."""
    if len(prices) < 2:
        return 0.0
    return (prices.iloc[-1] / prices.iloc[0]) - 1.0


# ── Risk metrics ──────────────────────────────────────────────────────────────

def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a fraction, e.g. -0.18 = -18%."""
    if len(equity) < 2:
        return 0.0
    peak = equity.cummax()
    dd   = (equity - peak) / peak
    return float(dd.min())


def sharpe_ratio(
    equity: pd.Series,
    risk_free_annual: float = 0.05,
    periods_per_year: int   = 252,
) -> float:
    """Annualised Sharpe ratio from daily equity series."""
    rets = equity.pct_change().dropna()
    if len(rets) < 2 or rets.std() == 0:
        return 0.0
    rf_daily = risk_free_annual / periods_per_year
    excess   = rets - rf_daily
    return float(math.sqrt(periods_per_year) * excess.mean() / rets.std())


def sortino_ratio(
    equity: pd.Series,
    risk_free_annual: float = 0.05,
    periods_per_year: int   = 252,
) -> float:
    """Annualised Sortino ratio (penalises only downside volatility)."""
    rets = equity.pct_change().dropna()
    if len(rets) < 2:
        return 0.0
    rf_daily  = risk_free_annual / periods_per_year
    excess    = rets - rf_daily
    downside  = rets[rets < 0]
    if len(downside) < 2 or downside.std() == 0:
        return 0.0
    return float(math.sqrt(periods_per_year) * excess.mean() / downside.std())


def calmar_ratio(equity: pd.Series) -> float:
    """CAGR / |Max Drawdown|. Higher = better risk-adjusted trend following."""
    dd = max_drawdown(equity)
    if dd == 0:
        return 0.0
    return cagr(equity) / abs(dd)


# ── Trade-level metrics ───────────────────────────────────────────────────────

def win_rate(trades: list[Trade]) -> float:
    """Fraction of closed trades that were profitable."""
    closed = [t for t in trades if t.pnl_pct is not None]
    if not closed:
        return 0.0
    winners = sum(1 for t in closed if t.pnl_pct > 0)
    return winners / len(closed)


def profit_factor(trades: list[Trade]) -> float:
    """Gross profit / gross loss. >1 = profitable, >2 = solid."""
    closed = [t for t in trades if t.pnl_pct is not None]
    gross_win  = sum(t.pnl_dollar for t in closed if t.pnl_dollar and t.pnl_dollar > 0)
    gross_loss = sum(abs(t.pnl_dollar) for t in closed if t.pnl_dollar and t.pnl_dollar < 0)
    if gross_loss == 0:
        return float("inf") if gross_win > 0 else 0.0
    return gross_win / gross_loss


def avg_winner(trades: list[Trade]) -> float:
    """Average P&L% of winning trades."""
    wins = [t.pnl_pct for t in trades if t.pnl_pct is not None and t.pnl_pct > 0]
    return float(np.mean(wins)) if wins else 0.0


def avg_loser(trades: list[Trade]) -> float:
    """Average P&L% of losing trades."""
    losses = [t.pnl_pct for t in trades if t.pnl_pct is not None and t.pnl_pct <= 0]
    return float(np.mean(losses)) if losses else 0.0


def expectancy(trades: list[Trade]) -> float:
    """
    Expected value per trade = (Win Rate × Avg Win) + (Loss Rate × Avg Loss).
    Positive = edge exists.
    """
    wr = win_rate(trades)
    return wr * avg_winner(trades) + (1 - wr) * avg_loser(trades)


def best_trade(trades: list[Trade]) -> Optional[Trade]:
    closed = [t for t in trades if t.pnl_pct is not None]
    return max(closed, key=lambda t: t.pnl_pct, default=None)


def worst_trade(trades: list[Trade]) -> Optional[Trade]:
    closed = [t for t in trades if t.pnl_pct is not None]
    return min(closed, key=lambda t: t.pnl_pct, default=None)


# Add Optional to avoid NameError
from typing import Optional
