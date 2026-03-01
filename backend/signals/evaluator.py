"""
signals/evaluator.py — Core signal evaluation engine.

SignalEvaluator replaces the hardcoded scoring in LongSignalStrategy.prepare().
It computes the same indicators via backtester.indicators, but applies
configurable weights from a SignalConfig.

Design:
  1. prepare(df) computes all indicators and raw s_* component columns,
     then aggregates into weighted bull_score / bear_score.
  2. Entry/exit signals are evaluated using config-defined gates and thresholds.
  3. evaluate_latest(df) is the main entry point for the monitoring agent.

For the legacy config (from_legacy()), the weighted scores are identical
to LongSignalStrategy v4.3 because each weight equals the component's max
raw score, making normalized * weight == raw.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from backtester import indicators as ind
from .config import SignalConfig


# ── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class SignalResult:
    """Output of evaluating a single bar."""
    ticker: str = ""
    timestamp: Any = None
    price: float = 0.0

    # Scores
    bull_score: float = 0.0
    bear_score: float = 0.0
    bull_max: float = 0.0
    bear_max: float = 0.0

    # Component breakdown: {component_name: weighted_contribution}
    bull_components: dict[str, float] = field(default_factory=dict)
    bear_components: dict[str, float] = field(default_factory=dict)

    # Signals
    signal_type: str = ""           # "buy", "bounce", "vwap_bounce", "" (none)
    should_enter: bool = False
    should_exit: bool = False
    exit_reason: str = ""

    # Regime
    regime: str = "neutral"         # "bull" | "bear" | "neutral"

    # Verdict (replaces decision_engine scoring)
    verdict: str = ""
    verdict_action: str = ""
    verdict_color: str = "grey"

    @property
    def has_actionable_signal(self) -> bool:
        return self.should_enter or self.should_exit

    @property
    def bull_score_pct(self) -> float:
        return (self.bull_score / self.bull_max * 100) if self.bull_max else 0.0

    @property
    def bear_score_pct(self) -> float:
        return (self.bear_score / self.bear_max * 100) if self.bear_max else 0.0


# ── Component max raw values (for normalization) ────────────────────────────
# Maps component name → max integer value the raw s_* column can take.
# Binary components have max 1; three-level components have max 2.

_BULL_MAX_RAW: dict[str, int] = {
    "regime": 2, "sma50": 1,
    "rsi_zone": 2, "macd_positive": 1, "macd_cross": 1,
    "volume": 2,
    "vwap_primary": 1, "vwap_secondary": 1,
    "consecutive_bars": 1, "weekly_mtf": 1,
    "rolling_poc": 1,
    "adx": 1, "fibonacci": 1,
}

_BEAR_MAX_RAW: dict[str, int] = {
    "regime": 2, "sma50": 1,
    "rsi_zone": 2, "macd_positive": 1, "macd_cross": 1,
    "volume": 2,
    "vwap_primary": 1, "vwap_secondary": 1,
    "consecutive_bars": 1, "weekly_mtf": 1,
    "rolling_poc": 1,
    # adx and fibonacci NOT in bear scoring
}

# Bull s_* column → bear s_* column mapping
_BULL_COL = {
    "regime": "s_regime", "sma50": "s_sma50",
    "rsi_zone": "s_rsi", "macd_positive": "s_macd_pos", "macd_cross": "s_macd_cross",
    "volume": "s_vol",
    "vwap_primary": "s_vwap_w", "vwap_secondary": "s_vwap_m",
    "consecutive_bars": "s_consec", "weekly_mtf": "s_weekly_tf",
    "rolling_poc": "s_poc",
    "adx": "s_adx", "fibonacci": "s_fib",
}

_BEAR_COL = {
    "regime": "s_bear_regime", "sma50": "s_bear_sma50",
    "rsi_zone": "s_bear_rsi", "macd_positive": "s_bear_macd_p",
    "macd_cross": "s_bear_macd_x",
    "volume": "s_vol",    # same column — volume is direction-agnostic
    "vwap_primary": "s_bear_vwap_w", "vwap_secondary": "s_bear_vwap_m",
    "consecutive_bars": "s_bear_consec", "weekly_mtf": "s_bear_weekly_tf",
    "rolling_poc": "s_bear_poc",
}


# ── Gate evaluators ──────────────────────────────────────────────────────────
# Each gate is a column name or a function that returns a boolean Series.

def _eval_gate(df: pd.DataFrame, gate_name: str) -> pd.Series:
    """Evaluate a named gate condition, returning a boolean Series."""
    if gate_name == "consec_above":
        return df["consec_above"]
    if gate_name == "consec_below":
        return df["consec_below"]
    if gate_name == "above_atr_buffer":
        return df["close"] > df["ema20"] + df["atr_buffer"]
    if gate_name == "below_atr_buffer":
        return df["close"] < df["ema20"] - df["atr_buffer"]
    if gate_name == "bull_regime":
        return df["bull_regime"]
    if gate_name == "bear_regime":
        return df["bear_regime"]
    if gate_name == "no_bear_divergence":
        return ~df["bear_div"]
    if gate_name == "no_bull_divergence":
        return ~df["bull_div"]
    if gate_name == "was_below_then_reclaimed_ema":
        return df["was_below_2bar"] & df["back_above_ema"]
    if gate_name == "bull_candle":
        return df["bull_candle"]
    if gate_name == "rsi_below_65":
        return df["rsi"] < 65
    if gate_name == "rsi_35_60":
        return (df["rsi"] > 35) & (df["rsi"] < 60)
    if gate_name == "near_vwap_lower_band":
        return df["near_wvwap_lower"]
    if gate_name == "above_mvwap":
        return df["above_mvwap"]
    raise ValueError(f"Unknown gate: {gate_name}")


def _eval_required_component(df: pd.DataFrame, comp_name: str) -> pd.Series:
    """Evaluate a required component as a boolean pass/fail."""
    if comp_name == "macd_cross":
        return df["macd_cross"]
    if comp_name == "adx":
        return df["adx_ok"]
    raise ValueError(f"Unknown required component: {comp_name}")


# ── Main evaluator ───────────────────────────────────────────────────────────

class SignalEvaluator:
    """Configurable signal evaluation engine driven by a SignalConfig."""

    def __init__(self, config: SignalConfig):
        self.config = config

    # ── Helper: merged params ────────────────────────────────────────────

    def _p(self, component_name: str, key: str, default: Any = None) -> Any:
        """Get a param from a component, falling back to indicator_params."""
        comp = self.config.components.get(component_name)
        if comp and key in comp.params:
            return comp.params[key]
        if key in self.config.indicator_params:
            return self.config.indicator_params[key]
        return default

    # ── Main prepare ─────────────────────────────────────────────────────

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all indicators, raw components, and weighted scores.

        This is the vectorized pass — called once before the bar loop.
        Mirrors LongSignalStrategy.prepare() but with config-driven weights.
        """
        cfg = self.config

        # ── Moving averages ──────────────────────────────────────────────
        ema_len  = self._p("consecutive_bars", "ema_len",
                           self.config.indicator_params.get("ema_len", 20))
        sma_mid  = self._p("sma50", "sma_mid", 50)
        sma_slow = self._p("regime", "sma_slow", 200)

        df["ema20"]  = ind.ema(df["close"], ema_len)
        df["sma50"]  = ind.sma(df["close"], sma_mid)

        use_daily = self.config.indicator_params.get("use_daily_regime", False)
        if use_daily:
            df["sma200"], _sma200_ago = ind.daily_sma200(df["close"], sma_len=sma_slow)
        else:
            df["sma200"]  = ind.sma(df["close"], sma_slow)
            _sma200_ago   = df["sma200"].shift(20)

        # ── RSI ──────────────────────────────────────────────────────────
        rsi_len = self._p("rsi_zone", "rsi_len", 14)
        df["rsi"] = ind.rsi(df["close"], rsi_len)

        # ── MACD ─────────────────────────────────────────────────────────
        macd_fast = self._p("macd_positive", "fast", 12)
        macd_slow = self._p("macd_positive", "slow", 26)
        macd_sig  = self._p("macd_positive", "sig", 9)
        df["macd_line"], df["signal_line"], df["macd_hist"] = ind.macd(
            df["close"], macd_fast, macd_slow, macd_sig,
        )
        df["macd_cross"]      = ind.crossover(df["macd_line"], df["signal_line"])
        df["macd_crossunder"] = ind.crossunder(df["macd_line"], df["signal_line"])

        # ── ATR + volume ─────────────────────────────────────────────────
        atr_len   = cfg.exits.atr_len
        atr_mult  = cfg.exits.atr_mult
        vol_mult  = self._p("volume", "mult", 1.25)
        vol_strong = self._p("volume", "strong", 1.8)

        df["atr"]        = ind.atr(df["high"], df["low"], df["close"], atr_len)
        df["avg_vol"]    = ind.avg_volume(df["volume"])
        df["high_vol"]   = df["volume"] >= df["avg_vol"] * vol_mult
        df["strong_vol"] = df["volume"] >= df["avg_vol"] * vol_strong
        df["s_vol"]      = np.where(df["strong_vol"], 2, np.where(df["high_vol"], 1, 0))

        # ── VWAP ─────────────────────────────────────────────────────────
        vwap_p_freq = self._p("vwap_primary", "freq", "W")
        vwap_s_freq = self._p("vwap_secondary", "freq", "M")

        df["vwap_w"], df["vwap_w_hi"], df["vwap_w_lo"] = ind.vwap_periodic(df, freq=vwap_p_freq)
        df["vwap_m"], _, _ = ind.vwap_periodic(df, freq=vwap_s_freq)

        df["above_wvwap"]      = df["close"] > df["vwap_w"]
        df["above_mvwap"]      = df["close"] > df["vwap_m"]
        df["near_wvwap_lower"] = (df["low"] <= df["vwap_w_lo"]) & (df["close"] > df["vwap_w_lo"])

        # ── Higher-Timeframe MTF ─────────────────────────────────────────
        mtf_freq = self._p("weekly_mtf", "freq", "W")
        mtf_rsi  = self._p("weekly_mtf", "rsi_len", 14)
        mtf_ema  = self._p("weekly_mtf", "ema_len", 20)

        mtf_enabled = cfg.components.get("weekly_mtf", None)
        if mtf_enabled and mtf_enabled.enabled:
            w_rsi, w_ema = ind.htf_mtf(df["close"], freq=mtf_freq, rsi_len=mtf_rsi, ema_len=mtf_ema)
            df["w_rsi"]       = w_rsi
            df["w_ema"]       = w_ema
            df["weekly_bull"] = (df["w_rsi"] > 45) & (df["close"] > df["w_ema"])
        else:
            df["weekly_bull"] = True

        # ── Regime ───────────────────────────────────────────────────────
        df["sma200_rising"]  = df["sma200"] > _sma200_ago
        df["sma200_falling"] = df["sma200"] < _sma200_ago
        df["bull_regime"]    = (df["close"] > df["sma200"]) & df["sma200_rising"]
        df["bear_regime"]    = (df["close"] < df["sma200"]) & df["sma200_falling"]

        # ── Fakeout filters ──────────────────────────────────────────────
        df["atr_buffer"] = df["atr"] * atr_mult
        _above_ema = df["close"] > df["ema20"]
        _below_ema = df["close"] < df["ema20"]

        consec_len = self._p("consecutive_bars", "consec_len", 2)
        df["consec_above"] = _above_ema.rolling(consec_len, min_periods=consec_len).sum() >= consec_len
        df["consec_below"] = _below_ema.rolling(consec_len, min_periods=consec_len).sum() >= consec_len
        df["bull_candle"]  = df["close"] > df["open"]
        df["was_below_2bar"] = _below_ema.shift(1).rolling(consec_len, min_periods=consec_len).sum() >= consec_len
        df["back_above_ema"] = df["close"] > df["ema20"]

        # ── RSI divergence ───────────────────────────────────────────────
        df["bull_div"] = ind.bull_divergence(df["close"], df["rsi"])
        df["bear_div"] = ind.bear_divergence(df["close"], df["rsi"])

        # ── Rolling POC ──────────────────────────────────────────────────
        poc_len = self._p("rolling_poc", "lookback", 50)
        df["poc"]       = ind.rolling_poc(df, lookback=poc_len)
        df["above_poc"] = df["close"] > df["poc"]
        df["below_poc"] = df["close"] < df["poc"]

        # ── ADX ──────────────────────────────────────────────────────────
        adx_ind_len = self._p("adx", "len", 14)
        adx_thr     = self._p("adx", "threshold", 20.0)
        df["adx"]    = ind.adx(df["high"], df["low"], df["close"], adx_ind_len)
        df["adx_ok"] = df["adx"] > adx_thr

        # ── Fibonacci ────────────────────────────────────────────────────
        fib_swing = self._p("fibonacci", "swing_len", 50)
        fib_thr   = self._p("fibonacci", "threshold_pct", 1.5)
        df["near_fib"] = ind.fib_near_level(
            df["high"], df["low"], df["close"],
            swing_len=fib_swing, threshold_pct=fib_thr,
        )

        # ── Raw component columns (same values as legacy) ───────────────
        self._compute_raw_bull_components(df)
        self._compute_raw_bear_components(df)

        # ── Weighted scores ──────────────────────────────────────────────
        self._compute_weighted_scores(df)

        # ── Entry/exit signals ───────────────────────────────────────────
        self._compute_entry_signals(df)
        self._compute_sell_signal(df)

        return df

    # ── Raw bull component columns ───────────────────────────────────────

    def _compute_raw_bull_components(self, df: pd.DataFrame) -> None:
        """Compute s_* columns matching legacy LongSignalStrategy exactly."""
        rsi_bull_min = self._p("rsi_zone", "bull_min", 42)
        rsi_bull_max = self._p("rsi_zone", "bull_max", 62)

        # TREND
        df["s_regime"] = np.where(df["bull_regime"], 2,
                         np.where(df["close"] > df["sma200"], 1, 0))
        df["s_sma50"]  = (df["close"] > df["sma50"]).astype(int)

        # MOMENTUM
        rsi_ideal = (df["rsi"] >= rsi_bull_min) & (df["rsi"] <= rsi_bull_max)
        rsi_ok    = (df["rsi"] >= 35) & (df["rsi"] < rsi_bull_min)
        df["s_rsi"]        = np.where(rsi_ideal, 2, np.where(rsi_ok, 1, 0))
        df["s_macd_pos"]   = (df["macd_line"] > df["signal_line"]).astype(int)
        df["s_macd_cross"] = df["macd_cross"].astype(int)

        # VOLUME — s_vol already computed in prepare()

        # VWAP
        df["s_vwap_w"] = df["above_wvwap"].astype(int)
        df["s_vwap_m"] = df["above_mvwap"].astype(int)

        # QUALITY
        df["s_consec"]    = df["consec_above"].astype(int)
        df["s_weekly_tf"] = df["weekly_bull"].astype(int)

        # PRICE STRUCTURE
        df["s_poc"] = df["above_poc"].astype(int)

        # ADX + FIBONACCI
        df["s_adx"] = df["adx_ok"].astype(int)
        df["s_fib"] = df["near_fib"].astype(int)

    # ── Raw bear component columns ───────────────────────────────────────

    def _compute_raw_bear_components(self, df: pd.DataFrame) -> None:
        """Compute bear s_* columns matching legacy LongSignalStrategy exactly."""
        rsi_bear_ideal = (df["rsi"] >= 32) & (df["rsi"] <= 62)

        df["s_bear_regime"]  = np.where(df["bear_regime"], 2,
                               np.where(df["close"] < df["sma200"], 1, 0))
        df["s_bear_sma50"]   = (df["close"] < df["sma50"]).astype(int)
        df["s_bear_rsi"]     = np.where(rsi_bear_ideal, 2, 1)
        df["s_bear_macd_p"]  = (df["macd_line"] < df["signal_line"]).astype(int)
        df["s_bear_macd_x"]  = df["macd_crossunder"].astype(int)
        # s_vol is shared (direction-agnostic)
        df["s_bear_vwap_w"]  = (~df["above_wvwap"]).astype(int)
        df["s_bear_vwap_m"]  = (~df["above_mvwap"]).astype(int)
        df["s_bear_consec"]  = df["consec_below"].astype(int)
        df["s_bear_weekly_tf"] = (~df["weekly_bull"]).astype(int)
        df["s_bear_poc"]     = df["below_poc"].astype(int)

    # ── Weighted score aggregation ───────────────────────────────────────

    def _compute_weighted_scores(self, df: pd.DataFrame) -> None:
        """Compute weighted bull_score and bear_score from raw components."""
        cfg = self.config

        bull_score = pd.Series(0.0, index=df.index)
        bear_score = pd.Series(0.0, index=df.index)

        for name, comp in cfg.components.items():
            if not comp.enabled:
                continue

            # Bull contribution
            if name in _BULL_COL and name in _BULL_MAX_RAW:
                raw_col = _BULL_COL[name]
                max_raw = _BULL_MAX_RAW[name]
                normalized = df[raw_col].astype(float) / max_raw
                bull_score = bull_score + normalized * comp.weight

            # Bear contribution (ADX and fibonacci excluded)
            if name in _BEAR_COL and name in _BEAR_MAX_RAW:
                raw_col = _BEAR_COL[name]
                max_raw = _BEAR_MAX_RAW[name]
                normalized = df[raw_col].astype(float) / max_raw
                bear_score = bear_score + normalized * comp.weight

        df["bull_score"] = bull_score
        df["bear_score"] = bear_score

    # ── Entry signal computation ─────────────────────────────────────────

    def _compute_entry_signals(self, df: pd.DataFrame) -> None:
        """Compute entry signals from config-defined gates and thresholds."""
        cfg = self.config

        for sig_name, sig_def in cfg.entry_signals.items():
            if not sig_def.enabled:
                df[sig_name] = False
                continue

            # Start with score threshold
            min_score = sig_def.min_score_frac * cfg.max_score
            signal = df["bull_score"] >= min_score

            # Apply all gates
            for gate in sig_def.gates:
                signal = signal & _eval_gate(df, gate)

            # Apply required components
            for comp_name in sig_def.required_components:
                signal = signal & _eval_required_component(df, comp_name)

            df[sig_name] = signal

    def _compute_sell_signal(self, df: pd.DataFrame) -> None:
        """Compute sell signal from config."""
        cfg = self.config
        sell_def = cfg.sell_signal

        min_sell = cfg.score_min_sell
        signal = df["bear_score"] >= min_sell

        for gate in sell_def.gates:
            signal = signal & _eval_gate(df, gate)

        if sell_def.use_macd_gate:
            signal = signal & df["macd_crossunder"]

        df["sell_signal"] = signal

    # ── Single-bar evaluation ────────────────────────────────────────────

    def entry_signal(self, bar: pd.Series) -> bool:
        """Check if any entry signal fires for a single bar."""
        cfg = self.config
        mode = cfg.entry_mode

        if mode == "Strong Buy Only":
            return bool(bar.get("buy_signal", False) and bar["bull_score"] >= cfg.score_strong)
        elif mode == "Buy Only":
            return bool(bar.get("buy_signal", False))
        else:  # "All Signals"
            return bool(
                bar.get("buy_signal", False)
                or bar.get("bounce_signal", False)
                or bar.get("vwap_bounce", False)
            )

    def exit_signal(self, bar: pd.Series) -> tuple[bool, str]:
        """Check if sell signal fires for a single bar."""
        if bar.get("sell_signal", False):
            return True, "Sell Signal"
        return False, ""

    # ── Evaluate latest bar (monitoring agent entry point) ───────────────

    def evaluate_latest(self, df: pd.DataFrame, ticker: str = "") -> SignalResult:
        """Prepare df and evaluate the most recent bar. Returns SignalResult."""
        df = self.prepare(df)
        bar = df.iloc[-1]
        cfg = self.config

        # Component breakdown
        bull_components: dict[str, float] = {}
        bear_components: dict[str, float] = {}
        for name, comp in cfg.components.items():
            if not comp.enabled:
                continue
            if name in _BULL_COL and name in _BULL_MAX_RAW:
                raw = float(bar[_BULL_COL[name]])
                bull_components[name] = raw / _BULL_MAX_RAW[name] * comp.weight
            if name in _BEAR_COL and name in _BEAR_MAX_RAW:
                raw = float(bar[_BEAR_COL[name]])
                bear_components[name] = raw / _BEAR_MAX_RAW[name] * comp.weight

        # Determine signal type
        signal_type = ""
        should_enter = self.entry_signal(bar)
        if should_enter:
            if bar.get("buy_signal"):
                signal_type = "buy"
            elif bar.get("bounce_signal"):
                signal_type = "bounce"
            elif bar.get("vwap_bounce"):
                signal_type = "vwap_bounce"

        should_exit, exit_reason = self.exit_signal(bar)

        # Regime
        if bool(bar.get("bull_regime")):
            regime = "bull"
        elif bool(bar.get("bear_regime")):
            regime = "bear"
        else:
            regime = "neutral"

        # Verdict
        bull_s = float(bar["bull_score"])
        bear_s = float(bar["bear_score"])
        verdict, action, color = self._determine_verdict(bar, bull_s, bear_s, regime)

        return SignalResult(
            ticker=ticker,
            timestamp=bar.name,
            price=float(bar["close"]),
            bull_score=bull_s,
            bear_score=bear_s,
            bull_max=cfg.max_score,
            bear_max=cfg.bear_max_score,
            bull_components=bull_components,
            bear_components=bear_components,
            signal_type=signal_type,
            should_enter=should_enter,
            should_exit=should_exit,
            exit_reason=exit_reason,
            regime=regime,
            verdict=verdict,
            verdict_action=action,
            verdict_color=color,
        )

    # ── Verdict logic (replaces decision_engine scoring) ─────────────────

    def _determine_verdict(
        self, bar: pd.Series, bull_score: float, bear_score: float, regime: str,
    ) -> tuple[str, str, str]:
        """Unified verdict using weighted scores.

        Thresholds are expressed as fractions of max_score, so they
        auto-scale when weights change.
        """
        cfg = self.config
        # Normalize to 0-10 scale for verdict thresholds (legacy compatibility)
        bull_max = cfg.max_score or 1
        bear_max = cfg.bear_max_score or 1
        long_10 = bull_score / bull_max * 10
        short_10 = bear_score / bear_max * 10

        rsi = float(bar.get("rsi", 50))
        macd_b = float(bar.get("macd_line", 0)) > float(bar.get("signal_line", 0))
        bounce_sig = bool(bar.get("bounce_signal", False))
        vwap_bounce = bool(bar.get("vwap_bounce", False))
        bull_div = bool(bar.get("bull_div", False))

        is_bounce = rsi < 38 and macd_b and (bounce_sig or vwap_bounce or bull_div or rsi < 32)

        if long_10 >= 8:
            return ("STRONG LONG",
                    "Bot conditions strongly favor a long entry. Size normally.",
                    "green")
        if long_10 >= 6 and long_10 > short_10 + 1:
            if is_bounce:
                return ("BOUNCE PLAY",
                        "Oversold + bullish reversal signals. Long on bounce — smaller size, tight stop.",
                        "lime")
            return ("LEAN LONG",
                    "More long conditions met than bear. Can lean long, size conservatively.",
                    "green")
        if is_bounce and long_10 >= 4:
            return ("BOUNCE PLAY",
                    "Oversold reversal signal. Long on bounce — reduced size, tight SL below recent low.",
                    "lime")
        if short_10 >= 8:
            return ("STRONG SHORT",
                    "Bear conditions dominant. Short entry favored — check RSI > 30.",
                    "red")
        if short_10 >= 6 and short_10 > long_10 + 1:
            return ("LEAN SHORT",
                    "More bear conditions met. Can lean short, size conservatively.",
                    "red")
        if abs(long_10 - short_10) <= 1:
            return ("WAIT",
                    "Long and short conditions roughly equal. No edge. Wait for a signal to tip the balance.",
                    "grey")
        return ("WAIT",
                "Neither direction has enough conditions met. Stand aside.",
                "grey")
