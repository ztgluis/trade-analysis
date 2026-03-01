"""
signals/config.py — Signal configuration data structures and YAML I/O.

A SignalConfig defines:
  - Which scoring components are enabled and how they're weighted
  - Entry signal definitions (gates, required components)
  - Exit / risk management parameters
  - Global indicator parameters

Use from_legacy() to generate a config matching LongSignalStrategy v4.3 exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SignalComponent:
    """A single scoring component (e.g., regime, rsi_zone, volume)."""
    name: str
    enabled: bool = True
    weight: float = 1.0
    required: bool = False      # hard gate on entry signals that list it
    category: str = ""
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class EntrySignalDef:
    """Definition of an entry signal type (buy, bounce, vwap_bounce)."""
    name: str
    enabled: bool = True
    min_score_frac: float = 0.0   # minimum bull_score as fraction of bull_max
    gates: list[str] = field(default_factory=list)
    required_components: list[str] = field(default_factory=list)


@dataclass
class SellSignalDef:
    """Definition of the sell signal."""
    gates: list[str] = field(default_factory=list)
    use_macd_gate: bool = True


@dataclass
class ExitConfig:
    """Exit / risk management configuration."""
    sl_pct: float = 5.0
    tp_pct: float = 15.0
    use_trail: bool = False
    trail_pct: float = 5.0
    atr_len: int = 14
    atr_mult: float = 0.4
    score_min_sell_frac: float = 0.5714   # bear score threshold as fraction of bear_max


@dataclass
class SignalConfig:
    """Complete signal configuration."""
    name: str = "Unnamed Config"
    description: str = ""
    components: dict[str, SignalComponent] = field(default_factory=dict)
    entry_signals: dict[str, EntrySignalDef] = field(default_factory=dict)
    sell_signal: SellSignalDef = field(default_factory=SellSignalDef)
    entry_mode: str = "All Signals"
    thresholds: dict[str, float] = field(default_factory=lambda: {
        "score_moderate_frac": 0.5,
        "score_strong_frac": 0.6875,
    })
    exits: ExitConfig = field(default_factory=ExitConfig)
    indicator_params: dict[str, Any] = field(default_factory=dict)

    # ── Derived properties ──────────────────────────────────────────────────

    # Components that participate in bear scoring (ADX/fib excluded)
    _BEAR_EXCLUDED = {"adx", "fibonacci"}

    @property
    def max_score(self) -> float:
        """Sum of weights for all enabled bull components."""
        return sum(c.weight for c in self.components.values() if c.enabled)

    @property
    def bear_max_score(self) -> float:
        """Sum of weights for all enabled bear components."""
        return sum(
            c.weight for c in self.components.values()
            if c.enabled and c.name not in self._BEAR_EXCLUDED
        )

    @property
    def score_moderate(self) -> float:
        return self.thresholds["score_moderate_frac"] * self.max_score

    @property
    def score_strong(self) -> float:
        return self.thresholds["score_strong_frac"] * self.max_score

    @property
    def score_min_sell(self) -> float:
        return self.exits.score_min_sell_frac * self.bear_max_score

    # ── Serialisation ───────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str | Path) -> SignalConfig:
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d: dict) -> SignalConfig:
        components = {}
        for name, cd in d.get("components", {}).items():
            components[name] = SignalComponent(
                name=name,
                enabled=cd.get("enabled", True),
                weight=cd.get("weight", 1.0),
                required=cd.get("required", False),
                category=cd.get("category", ""),
                params=cd.get("params", {}),
            )

        entry_signals = {}
        for name, ed in d.get("entry_signals", {}).items():
            entry_signals[name] = EntrySignalDef(
                name=name,
                enabled=ed.get("enabled", True),
                min_score_frac=ed.get("min_score_frac", 0.0),
                gates=ed.get("gates", []),
                required_components=ed.get("required_components", []),
            )

        sell_d = d.get("sell_signal", {})
        sell_signal = SellSignalDef(
            gates=sell_d.get("gates", [
                "consec_below", "below_atr_buffer",
                "bear_regime", "no_bull_divergence",
            ]),
            use_macd_gate=sell_d.get("use_macd_gate", True),
        )

        exits_d = d.get("exits", {})
        exits = ExitConfig(
            sl_pct=exits_d.get("sl_pct", 5.0),
            tp_pct=exits_d.get("tp_pct", 15.0),
            use_trail=exits_d.get("use_trail", False),
            trail_pct=exits_d.get("trail_pct", 5.0),
            atr_len=exits_d.get("atr_len", 14),
            atr_mult=exits_d.get("atr_mult", 0.4),
            score_min_sell_frac=exits_d.get("score_min_sell_frac", 0.5714),
        )

        return cls(
            name=d.get("name", "Unnamed Config"),
            description=d.get("description", ""),
            components=components,
            entry_signals=entry_signals,
            sell_signal=sell_signal,
            entry_mode=d.get("entry_mode", "All Signals"),
            thresholds=d.get("thresholds", {
                "score_moderate_frac": 0.5,
                "score_strong_frac": 0.6875,
            }),
            exits=exits,
            indicator_params=d.get("indicator_params", {}),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "entry_mode": self.entry_mode,
            "thresholds": self.thresholds,
            "indicator_params": self.indicator_params,
            "components": {
                name: {
                    "enabled": c.enabled,
                    "weight": c.weight,
                    "required": c.required,
                    "category": c.category,
                    "params": c.params,
                } for name, c in self.components.items()
            },
            "entry_signals": {
                name: {
                    "enabled": e.enabled,
                    "min_score_frac": e.min_score_frac,
                    "gates": e.gates,
                    "required_components": e.required_components,
                } for name, e in self.entry_signals.items()
            },
            "sell_signal": {
                "gates": self.sell_signal.gates,
                "use_macd_gate": self.sell_signal.use_macd_gate,
            },
            "exits": {
                "sl_pct": self.exits.sl_pct,
                "tp_pct": self.exits.tp_pct,
                "use_trail": self.exits.use_trail,
                "trail_pct": self.exits.trail_pct,
                "atr_len": self.exits.atr_len,
                "atr_mult": self.exits.atr_mult,
                "score_min_sell_frac": self.exits.score_min_sell_frac,
            },
        }

    def to_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    # ── Legacy factory ──────────────────────────────────────────────────────

    @classmethod
    def from_legacy(cls) -> SignalConfig:
        """Generate a config matching LongSignalStrategy v4.3 defaults exactly.

        Every weight equals the component's max raw score, so the weighted
        bull/bear totals are identical to the legacy integer scores.
        """
        components = {
            "regime": SignalComponent(
                name="regime", weight=2, category="trend",
                params={"sma_slow": 200},
            ),
            "sma50": SignalComponent(
                name="sma50", weight=1, category="trend",
                params={"sma_mid": 50},
            ),
            "rsi_zone": SignalComponent(
                name="rsi_zone", weight=2, category="momentum",
                params={"rsi_len": 14, "bull_min": 42, "bull_max": 62},
            ),
            "macd_positive": SignalComponent(
                name="macd_positive", weight=1, category="momentum",
                params={"fast": 12, "slow": 26, "sig": 9},
            ),
            "macd_cross": SignalComponent(
                name="macd_cross", weight=1, category="momentum",
            ),
            "volume": SignalComponent(
                name="volume", weight=2, category="volume",
                params={"mult": 1.25, "strong": 1.8},
            ),
            "vwap_primary": SignalComponent(
                name="vwap_primary", weight=1, category="vwap",
                params={"freq": "W"},
            ),
            "vwap_secondary": SignalComponent(
                name="vwap_secondary", weight=1, category="vwap",
                params={"freq": "M"},
            ),
            "consecutive_bars": SignalComponent(
                name="consecutive_bars", weight=1, category="quality",
                params={"consec_len": 2},
            ),
            "weekly_mtf": SignalComponent(
                name="weekly_mtf", weight=1, category="quality",
                params={"freq": "W", "rsi_len": 14, "ema_len": 20},
            ),
            "rolling_poc": SignalComponent(
                name="rolling_poc", weight=1, category="structure",
                params={"lookback": 50},
            ),
            "adx": SignalComponent(
                name="adx", weight=1, category="trend_strength",
                params={"len": 14, "threshold": 20.0},
            ),
            "fibonacci": SignalComponent(
                name="fibonacci", weight=1, category="structure",
                params={"swing_len": 50, "threshold_pct": 1.5},
            ),
        }

        entry_signals = {
            "buy_signal": EntrySignalDef(
                name="buy_signal",
                min_score_frac=0.5,         # score_moderate = 8/16
                gates=[
                    "consec_above", "above_atr_buffer",
                    "bull_regime", "no_bear_divergence",
                ],
                required_components=["macd_cross", "adx"],
            ),
            "bounce_signal": EntrySignalDef(
                name="bounce_signal",
                min_score_frac=0.5,         # 8/16
                gates=[
                    "was_below_then_reclaimed_ema", "bull_candle",
                    "rsi_below_65", "bull_regime", "no_bear_divergence",
                ],
            ),
            "vwap_bounce": EntrySignalDef(
                name="vwap_bounce",
                min_score_frac=0.4375,      # 7/16
                gates=[
                    "near_vwap_lower_band", "bull_candle",
                    "rsi_35_60", "above_mvwap", "bull_regime",
                ],
            ),
        }

        sell_signal = SellSignalDef(
            gates=[
                "consec_below", "below_atr_buffer",
                "bear_regime", "no_bull_divergence",
            ],
            use_macd_gate=True,
        )

        return cls(
            name="Growth Signal v4.3 (Legacy)",
            description="Matches LongSignalStrategy v4.3 defaults exactly",
            components=components,
            entry_signals=entry_signals,
            sell_signal=sell_signal,
            entry_mode="All Signals",
            thresholds={
                "score_moderate_frac": 0.5,      # 8/16
                "score_strong_frac": 0.6875,     # 11/16
            },
            exits=ExitConfig(
                sl_pct=5.0, tp_pct=15.0,
                use_trail=False, trail_pct=5.0,
                atr_len=14, atr_mult=0.4,
                score_min_sell_frac=8 / 14,      # 8/14 ≈ 0.5714
            ),
            indicator_params={
                "ema_len": 20,
                "use_daily_regime": False,
                "warmup_bars": 210,
            },
        )
