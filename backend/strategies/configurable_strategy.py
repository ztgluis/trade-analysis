"""
ConfigurableStrategy — BaseStrategy subclass driven by a SignalConfig YAML.

Bridges the new configurable signal framework with the existing BacktestEngine.
The engine's contract (prepare, entry_signal, exit_signal, get_stops) is unchanged.

Usage:
    from signals.config import SignalConfig
    from strategies.configurable_strategy import ConfigurableStrategy

    config   = SignalConfig.from_yaml("configs/signals.yaml")
    strategy = ConfigurableStrategy(config)
    engine   = BacktestEngine(strategy, df, initial_capital=10_000)
    results  = engine.run()
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .base import BaseStrategy
from signals.config import SignalConfig
from signals.evaluator import SignalEvaluator


class ConfigurableStrategy(BaseStrategy):
    """Strategy driven by a SignalConfig. Plugs into BacktestEngine."""

    def __init__(self, config: SignalConfig):
        self.config = config
        self.evaluator = SignalEvaluator(config)
        self.name = f"Configurable: {config.name}"
        self.warmup_bars = config.indicator_params.get("warmup_bars", 210)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.evaluator.prepare(df)

    def entry_signal(self, bar: pd.Series) -> bool:
        return self.evaluator.entry_signal(bar)

    def exit_signal(self, bar: pd.Series) -> tuple[bool, str]:
        return self.evaluator.exit_signal(bar)

    def get_stops(self, bar: pd.Series) -> tuple[Optional[float], Optional[float]]:
        exits = self.config.exits
        hard_stop = float(bar["close"]) * (1.0 - exits.sl_pct / 100.0)
        trail = exits.trail_pct if exits.use_trail else None
        return hard_stop, trail

    def get_tp(self, entry_price: float) -> Optional[float]:
        exits = self.config.exits
        if exits.use_trail:
            return None
        return entry_price * (1.0 + exits.tp_pct / 100.0)
