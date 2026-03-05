"""
EMA Channel Strategy — 5-EMA trend alignment system.

Uses 5 EMAs with different sources to detect confirmed trend entries:
  - EMA(5, high)  + EMA(5, low)   → fast channel
  - EMA(100, high) + EMA(100, low) → medium channel
  - EMA(200, close)                → trend filter

Entry: ALL 4 channel EMAs flip above EMA200 (transition detection).
Exit:  Configurable — all/fast/any EMAs cross below EMA200.
Stop:  ATR-based hard stop.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from backtester import indicators as ind
from .base import BaseStrategy


class EmaChannelStrategy(BaseStrategy):
    """5-EMA channel breakout strategy with configurable exit modes."""

    name = "EMA Channel (5/100/200)"
    warmup_bars = 210

    def __init__(
        self,
        exit_mode: str = "all",
        atr_mult: float = 2.0,
        atr_len: int = 14,
        fast_len: int = 5,
        med_len: int = 100,
        slow_len: int = 200,
    ):
        """
        Args:
            exit_mode: "all" | "fast" | "any" — which EMAs must cross below 200 to exit
            atr_mult:  ATR multiplier for hard stop (entry - atr_mult * ATR)
            atr_len:   ATR period
            fast_len:  Fast EMA length (applied to high & low)
            med_len:   Medium EMA length (applied to high & low)
            slow_len:  Slow EMA length (applied to close)
        """
        if exit_mode not in ("all", "fast", "any"):
            raise ValueError(f"exit_mode must be 'all', 'fast', or 'any', got '{exit_mode}'")
        self.exit_mode = exit_mode
        self.atr_mult = atr_mult
        self.atr_len = atr_len
        self.fast_len = fast_len
        self.med_len = med_len
        self.slow_len = slow_len
        self.name = f"EMA Channel ({fast_len}/{med_len}/{slow_len}) exit={exit_mode}"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        # ── Compute the 5 EMAs ───────────────────────────────────────
        df["ema_fast_h"] = ind.ema(df["high"], self.fast_len)
        df["ema_fast_l"] = ind.ema(df["low"], self.fast_len)
        df["ema_med_h"] = ind.ema(df["high"], self.med_len)
        df["ema_med_l"] = ind.ema(df["low"], self.med_len)
        df["ema_slow"] = ind.ema(df["close"], self.slow_len)

        # ── ATR for stops ────────────────────────────────────────────
        df["atr"] = ind.atr(df["high"], df["low"], df["close"], self.atr_len)

        # ── Alignment states ─────────────────────────────────────────
        slow = df["ema_slow"]

        df["all_above"] = (
            (df["ema_fast_h"] > slow)
            & (df["ema_fast_l"] > slow)
            & (df["ema_med_h"] > slow)
            & (df["ema_med_l"] > slow)
        )

        # Entry: transition — just flipped to all-above
        df["entry_flip"] = df["all_above"] & ~df["all_above"].shift(1, fill_value=False)

        # Exit conditions per mode
        df["exit_all"] = (
            (df["ema_fast_h"] < slow)
            & (df["ema_fast_l"] < slow)
            & (df["ema_med_h"] < slow)
            & (df["ema_med_l"] < slow)
        )
        df["exit_fast"] = (
            (df["ema_fast_h"] < slow)
            & (df["ema_fast_l"] < slow)
        )
        df["exit_any"] = (
            (df["ema_fast_h"] < slow)
            | (df["ema_fast_l"] < slow)
            | (df["ema_med_h"] < slow)
            | (df["ema_med_l"] < slow)
        )

        return df

    def entry_signal(self, bar: pd.Series) -> bool:
        return bool(bar["entry_flip"])

    def exit_signal(self, bar: pd.Series) -> tuple[bool, str]:
        col = f"exit_{self.exit_mode}"
        if bar[col]:
            return True, f"EMAs crossed below EMA{self.slow_len} ({self.exit_mode})"
        return False, ""

    def get_stops(self, bar: pd.Series) -> tuple[Optional[float], Optional[float]]:
        atr_val = float(bar["atr"])
        hard_stop = float(bar["close"]) - self.atr_mult * atr_val
        return hard_stop, None
