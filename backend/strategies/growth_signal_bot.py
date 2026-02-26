"""
Growth Signal Bot v4.3 — Python port of growth_signal_bot_v4.pine

16-point scoring system (v4.3 adds ADX + Fibonacci):
  Trend (3):        regime/200 SMA (2) + above 50 SMA (1)
  Momentum (4):     RSI zone (2) + MACD positive (1) + MACD cross (1)
  Volume (2):       strong vol (2) / high vol (1)
  VWAP (2):         above weekly VWAP (1) + above monthly VWAP (1)
  Quality (2):      consec above EMA (1) + weekly MTF (1)
  Price Struct (1): above rolling POC (1)          ← v4.1
  ADX (1):          ADX > 20, trending market       ← v4.3
  Fibonacci (1):    price near 38.2/50/61.8% level  ← v4.3

v4.2 tuning (gate analysis on GOOG 5y):
  rsi_bull_max=62        — tightened from 68; losers avg RSI 63.1 vs winners 61.1 at entry
  entry_mode="All Signals" — Sharpe 0.84 / WR 50% / PF 3.29 vs 0.26 on old Buy Only
  req_macd_x stays True  — MACD gate on buy_signal gives high-quality momentum entries;
                            bounce_signal + vwap_bounce naturally bypass it, adding pullback catches.
                            (req_macd_x=False + All Signals → only 0.65; MACD earns its keep here)

v4.3 tuning (backtest eval on GOOG 5y):
  use_fib_score=True     — +1 pt when price near 38.2/50/61.8% fib level
  use_adx_score=True     — +1 pt when ADX > 20 (trending market)
  use_adx_gate=True      — hard gate: entries only when ADX > threshold
  Combined result on GOOG 5y: Sharpe 0.82→1.06, CAGR 20→25.6%, PF 3.28→4.33, WR 50→55.6%

Tuned for large-cap growth stocks:
  RSI zone (42–62), ATR mult 0.4, SL 5% / TP 15%
  score_moderate >= 8/16,  score_strong >= 11/16

Exposes self._df after prepare() — used by run_gate_analysis.py for
signal funnel / false-positive / near-miss analysis.
"""
from __future__ import annotations
from typing import Optional

import pandas as pd
import numpy as np

from .base import BaseStrategy
from backtester import indicators as ind


class GrowthSignalBot(BaseStrategy):
    """
    Parameters mirror growth_signal_bot_v4.pine inputs.
    Override any param at instantiation:
        strategy = GrowthSignalBot(entry_mode="Strong Buy Only", sl_pct=6, tp_pct=18)

    For 1H charts use the factory (sets all 1H-appropriate defaults):
        strategy = GrowthSignalBot.for_1h(entry_mode="Buy Only")
    """

    name        = "Growth Signal Bot v4.3"
    warmup_bars = 210    # 200-bar SMA + RSI/MACD warmup buffer

    def __init__(
        self,
        # Moving averages
        ema_len:  int   = 20,
        sma_mid:  int   = 50,
        sma_slow: int   = 200,
        # RSI — zones for large-cap growth stock volatility (v4.2: max tightened 68→62)
        rsi_len:      int = 14,
        rsi_bull_min: int = 42,
        rsi_bull_max: int = 62,
        # MACD
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_sig:  int = 9,
        # Fakeout filters
        atr_len:    int   = 14,
        atr_mult:   float = 0.4,
        vol_mult:   float = 1.25,
        vol_strong: float = 1.8,
        req_macd_x: bool  = True,   # kept ON for buy_signal — bounce/vwap_bounce bypass it naturally
        # Scoring thresholds (max 14 in v4.1)
        score_strong:   int = 11,   # v4.1: raised from 10 (max now 14)
        score_moderate: int = 8,    # v4.1: raised from 7
        score_min_sell: int = 8,    # v4.1: raised from 7
        # Strategy / exits
        entry_mode: str   = "All Signals",  # v4.2: all three signal types enabled
        sl_pct:     float = 5.0,
        tp_pct:     float = 15.0,
        use_trail:  bool  = False,
        trail_pct:  float = 5.0,
        # Weekly MTF
        use_mtf: bool = True,
        # Rolling POC (v4.1)
        poc_len: int = 50,
        # ── v4.3 score components ─────────────────────────────────────────────
        # ADX — trend-strength filter
        use_adx_score: bool  = True,   # +1 point if ADX > adx_threshold
        use_adx_gate:  bool  = True,   # hard gate: entry only when ADX > threshold
        adx_len:       int   = 14,
        adx_threshold: float = 20.0,
        # Fibonacci retracement — price near key S/R level
        use_fib_score:     bool  = True,   # +1 point if price near fib level
        fib_swing_len:     int   = 50,
        fib_threshold_pct: float = 1.5,
        # ── Timeframe-specific VWAP anchoring ────────────────────────────────
        # Daily chart:  primary="W"  secondary="M"
        # Hourly chart: primary="D"  secondary="W"
        vwap_primary_freq:   str = "W",
        vwap_secondary_freq: str = "M",
        # ── 1H / generic timeframe tweaks ────────────────────────────────────
        # Consecutive bars above/below EMA required (daily: 2, 1H: 3)
        consec_len: int = 2,
        # MTF higher-timeframe frequency ("W" for daily chart, "D" for 1H chart)
        mtf_freq: str = "W",
        # Resample 1H close to daily for 200 SMA regime (set True on 1H)
        use_daily_regime: bool = False,
        # Warmup bars — override to ~1400 on 1H (200 daily bars × 6.5h + buffer)
        warmup_bars: int = 210,
    ) -> None:
        self.ema_len      = ema_len
        self.sma_mid      = sma_mid
        self.sma_slow     = sma_slow
        self.rsi_len      = rsi_len
        self.rsi_bull_min = rsi_bull_min
        self.rsi_bull_max = rsi_bull_max
        self.macd_fast    = macd_fast
        self.macd_slow    = macd_slow
        self.macd_sig     = macd_sig
        self.atr_len      = atr_len
        self.atr_mult     = atr_mult
        self.vol_mult     = vol_mult
        self.vol_strong   = vol_strong
        self.req_macd_x   = req_macd_x
        self.score_strong   = score_strong
        self.score_moderate = score_moderate
        self.score_min_sell = score_min_sell
        self.entry_mode   = entry_mode
        self.sl_pct       = sl_pct
        self.tp_pct       = tp_pct
        self.use_trail    = use_trail
        self.trail_pct    = trail_pct
        self.use_mtf      = use_mtf
        self.poc_len      = poc_len
        # v4.3 experimental
        self.use_adx_score    = use_adx_score
        self.use_adx_gate     = use_adx_gate
        self.adx_len          = adx_len
        self.adx_threshold    = adx_threshold
        self.use_fib_score    = use_fib_score
        self.fib_swing_len    = fib_swing_len
        self.fib_threshold_pct = fib_threshold_pct
        self.vwap_primary_freq   = vwap_primary_freq
        self.vwap_secondary_freq = vwap_secondary_freq
        self.consec_len          = consec_len
        self.mtf_freq            = mtf_freq
        self.use_daily_regime    = use_daily_regime
        self.warmup_bars         = warmup_bars   # shadows class attribute; allows 1H override
        self._df: pd.DataFrame | None = None   # exposed after prepare() for analysis

    # ── prepare() — vectorized indicator computation ─────────────────────────

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        # ── Moving averages ───────────────────────────────────────────────────
        df["ema20"] = ind.ema(df["close"], self.ema_len)
        df["sma50"] = ind.sma(df["close"], self.sma_mid)

        # SMA200 / regime anchor — on 1H, resample to daily for proper 10-month context
        # (200 × 1H bars = only ~31 days; useless as a trend filter on hourly charts)
        if self.use_daily_regime:
            df["sma200"], _sma200_ago = ind.daily_sma200(df["close"], sma_len=self.sma_slow)
        else:
            df["sma200"]  = ind.sma(df["close"], self.sma_slow)
            _sma200_ago   = df["sma200"].shift(20)

        # ── RSI ───────────────────────────────────────────────────────────────
        df["rsi"] = ind.rsi(df["close"], self.rsi_len)

        # ── MACD ──────────────────────────────────────────────────────────────
        df["macd_line"], df["signal_line"], df["macd_hist"] = ind.macd(
            df["close"], self.macd_fast, self.macd_slow, self.macd_sig
        )
        df["macd_cross"]      = ind.crossover(df["macd_line"],  df["signal_line"])
        df["macd_crossunder"] = ind.crossunder(df["macd_line"], df["signal_line"])

        # ── ATR + volume ──────────────────────────────────────────────────────
        df["atr"]     = ind.atr(df["high"], df["low"], df["close"], self.atr_len)
        df["avg_vol"] = ind.avg_volume(df["volume"])
        df["high_vol"]   = df["volume"] >= df["avg_vol"] * self.vol_mult
        df["strong_vol"] = df["volume"] >= df["avg_vol"] * self.vol_strong
        df["s_vol"]      = np.where(df["strong_vol"], 2, np.where(df["high_vol"], 1, 0))

        # ── VWAP (configurable periods — daily: W+M, hourly: D+W) ───────────
        df["vwap_w"], df["vwap_w_hi"], df["vwap_w_lo"] = ind.vwap_periodic(df, freq=self.vwap_primary_freq)
        df["vwap_m"], _,               _                = ind.vwap_periodic(df, freq=self.vwap_secondary_freq)

        df["above_wvwap"]      = df["close"] > df["vwap_w"]
        df["above_mvwap"]      = df["close"] > df["vwap_m"]
        df["near_wvwap_lower"] = (df["low"] <= df["vwap_w_lo"]) & (df["close"] > df["vwap_w_lo"])

        # ── Higher-Timeframe MTF (configurable: "W" for daily chart, "D" for 1H) ──
        if self.use_mtf:
            w_rsi, w_ema = ind.htf_mtf(df["close"], freq=self.mtf_freq, rsi_len=14, ema_len=20)
            df["w_rsi"]  = w_rsi
            df["w_ema"]  = w_ema
            df["weekly_bull"] = (df["w_rsi"] > 45) & (df["close"] > df["w_ema"])
        else:
            df["weekly_bull"] = True

        # ── Regime ───────────────────────────────────────────────────────────
        df["sma200_rising"]  = df["sma200"] > _sma200_ago
        df["sma200_falling"] = df["sma200"] < _sma200_ago
        df["bull_regime"]    = (df["close"] > df["sma200"]) & df["sma200_rising"]
        df["bear_regime"]    = (df["close"] < df["sma200"]) & df["sma200_falling"]

        # ── Fakeout filters ───────────────────────────────────────────────────
        df["atr_buffer"] = df["atr"] * self.atr_mult
        _above_ema = df["close"] > df["ema20"]
        _below_ema = df["close"] < df["ema20"]
        # consec_len=2 (daily default): rolling(2).sum()>=2 ≡ current + prev bar above EMA
        # consec_len=3 (1H default):    rolling(3).sum()>=3 ≡ 3 consecutive bars above EMA
        df["consec_above"] = _above_ema.rolling(self.consec_len, min_periods=self.consec_len).sum() >= self.consec_len
        df["consec_below"] = _below_ema.rolling(self.consec_len, min_periods=self.consec_len).sum() >= self.consec_len
        df["bull_candle"]  = df["close"] > df["open"]
        # was_below_nbars: previous consec_len bars were all below EMA (bounce precondition)
        df["was_below_2bar"] = _below_ema.shift(1).rolling(self.consec_len, min_periods=self.consec_len).sum() >= self.consec_len
        df["back_above_ema"] = df["close"] > df["ema20"]

        # ── RSI divergence ────────────────────────────────────────────────────
        df["bull_div"] = ind.bull_divergence(df["close"], df["rsi"])
        df["bear_div"] = ind.bear_divergence(df["close"], df["rsi"])

        # ── Rolling POC (v4.1 — price structure, max 1 pt) ───────────────────
        df["poc"] = ind.rolling_poc(df, lookback=self.poc_len)
        df["above_poc"] = df["close"] > df["poc"]
        df["below_poc"] = df["close"] < df["poc"]

        # ── Scoring system (0–14 pts) ─────────────────────────────────────────
        # TREND (max 3)
        df["s_regime"] = np.where(df["bull_regime"], 2,
                         np.where(df["close"] > df["sma200"], 1, 0))
        df["s_sma50"]  = (df["close"] > df["sma50"]).astype(int)

        # MOMENTUM (max 4)
        rsi_ideal = (df["rsi"] >= self.rsi_bull_min) & (df["rsi"] <= self.rsi_bull_max)
        rsi_ok    = (df["rsi"] >= 35) & (df["rsi"] < self.rsi_bull_min)
        df["s_rsi"]        = np.where(rsi_ideal, 2, np.where(rsi_ok, 1, 0))
        df["s_macd_pos"]   = (df["macd_line"] > df["signal_line"]).astype(int)
        df["s_macd_cross"] = df["macd_cross"].astype(int)

        # VOLUME (max 2)
        # s_vol already computed above

        # VWAP (max 2)
        df["s_vwap_w"] = df["above_wvwap"].astype(int)
        df["s_vwap_m"] = df["above_mvwap"].astype(int)

        # QUALITY (max 2)
        df["s_consec"]    = df["consec_above"].astype(int)
        df["s_weekly_tf"] = df["weekly_bull"].astype(int)

        # PRICE STRUCTURE (max 1) — v4.1 POC component
        df["s_poc"] = df["above_poc"].astype(int)

        # ── v4.3 experimental: ADX + Fibonacci ───────────────────────────────
        # Always compute so the data is available for inspection; flag controls
        # whether they contribute to the score / gate.
        df["adx"]      = ind.adx(df["high"], df["low"], df["close"], self.adx_len)
        df["adx_ok"]   = df["adx"] > self.adx_threshold
        df["near_fib"] = ind.fib_near_level(
            df["high"], df["low"], df["close"],
            swing_len=self.fib_swing_len,
            threshold_pct=self.fib_threshold_pct,
        )

        df["s_adx"] = df["adx_ok"].astype(int) if self.use_adx_score else 0
        df["s_fib"] = df["near_fib"].astype(int) if self.use_fib_score else 0

        # TOTAL BULL SCORE (max = 14 base + up to 2 from v4.3 experimental)
        df["bull_score"] = (
            df["s_regime"] + df["s_sma50"]
            + df["s_rsi"] + df["s_macd_pos"] + df["s_macd_cross"]
            + df["s_vol"]
            + df["s_vwap_w"] + df["s_vwap_m"]
            + df["s_consec"] + df["s_weekly_tf"]
            + df["s_poc"]
            + df["s_adx"]   # 0 unless use_adx_score=True
            + df["s_fib"]   # 0 unless use_fib_score=True
        )

        # BEAR SCORE (for sell signal)
        rsi_bear_ideal = (df["rsi"] >= 32) & (df["rsi"] <= 62)
        df["s_bear_regime"]  = np.where(df["bear_regime"], 2,
                               np.where(df["close"] < df["sma200"], 1, 0))
        df["s_bear_sma50"]   = (df["close"] < df["sma50"]).astype(int)
        df["s_bear_rsi"]     = np.where(rsi_bear_ideal, 2, 1)
        df["s_bear_macd_p"]  = (df["macd_line"] < df["signal_line"]).astype(int)
        df["s_bear_macd_x"]  = df["macd_crossunder"].astype(int)
        df["bear_score"]     = (
            df["s_bear_regime"] + df["s_bear_sma50"]
            + df["s_bear_rsi"] + df["s_bear_macd_p"] + df["s_bear_macd_x"]
            + df["s_vol"]
            + (~df["above_wvwap"]).astype(int)
            + (~df["above_mvwap"]).astype(int)
            + df["consec_below"].astype(int)
            + (~df["weekly_bull"]).astype(int)
            + df["below_poc"].astype(int)
        )

        # ── Signal pre-computation ────────────────────────────────────────────
        # buy_signal (scored, regime-gated, MACD optional, ADX gate optional)
        macd_gate = df["macd_cross"] if self.req_macd_x else pd.Series(True, index=df.index)
        adx_gate  = df["adx_ok"]     if self.use_adx_gate  else pd.Series(True, index=df.index)
        df["buy_signal"] = (
            (df["bull_score"] >= self.score_moderate)
            & df["consec_above"]
            & (df["close"] > df["ema20"] + df["atr_buffer"])
            & df["bull_regime"]
            & ~df["bear_div"]
            & macd_gate
            & adx_gate
        )

        # bounce_signal (v4.1: score floor raised 7→8 to match new max of 14)
        df["bounce_signal"] = (
            df["was_below_2bar"]
            & df["back_above_ema"]
            & df["bull_candle"]
            & (df["rsi"] < 65)
            & df["bull_regime"]
            & ~df["bear_div"]
            & (df["bull_score"] >= 8)
        )

        # vwap_bounce (v4.1: score floor raised 6→7)
        df["vwap_bounce"] = (
            df["near_wvwap_lower"]
            & df["bull_candle"]
            & (df["rsi"] > 35) & (df["rsi"] < 60)
            & df["above_mvwap"]
            & df["bull_regime"]
            & (df["bull_score"] >= 7)
        )

        # sell_signal
        sell_macd = df["macd_crossunder"] if self.req_macd_x else pd.Series(True, index=df.index)
        df["sell_signal"] = (
            (df["bear_score"] >= self.score_min_sell)
            & df["consec_below"]
            & (df["close"] < df["ema20"] - df["atr_buffer"])
            & df["bear_regime"]
            & ~df["bull_div"]
            & sell_macd
        )

        self._df = df   # expose for gate analysis (run_gate_analysis.py)
        return df

    # ── entry_signal() ────────────────────────────────────────────────────────

    def entry_signal(self, bar: pd.Series) -> bool:
        if self.entry_mode == "Strong Buy Only":
            return bool(bar["buy_signal"] and bar["bull_score"] >= self.score_strong)
        elif self.entry_mode == "Buy Only":
            return bool(bar["buy_signal"])
        else:  # "All Signals"
            return bool(bar["buy_signal"] or bar["bounce_signal"] or bar["vwap_bounce"])

    # ── exit_signal() ─────────────────────────────────────────────────────────

    def exit_signal(self, bar: pd.Series) -> tuple[bool, str]:
        """Sell signal fires at bar close (after stop checks)."""
        if bar["sell_signal"]:
            return True, "Sell Signal"
        return False, ""

    # ── get_stops() ───────────────────────────────────────────────────────────

    def get_stops(
        self, bar: pd.Series
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Returns (hard_stop_price, trail_pct_or_None).
        Also sets tp_price via the portfolio directly — handled in engine hook.
        """
        hard_stop = float(bar["close"]) * (1.0 - self.sl_pct / 100.0)
        trail     = self.trail_pct if self.use_trail else None
        return hard_stop, trail

    def get_tp(self, entry_price: float) -> Optional[float]:
        """Fixed take-profit price. Returns None if use_trail is True."""
        if self.use_trail:
            return None
        return entry_price * (1.0 + self.tp_pct / 100.0)

    # ── 1H factory ────────────────────────────────────────────────────────────

    @classmethod
    def for_1h(cls, **overrides) -> "GrowthSignalBot":
        """
        Factory: GrowthSignalBot pre-configured for 1H charts (v1.4 logic).

        Key differences from daily v4.3:
          - MACD 8/21/5           (faster 1H momentum response)
          - RSI bull zone 40–75   (hourly RSI swings harder than daily)
          - SL 2% / TP 6%         (scaled for 1H moves; R:R 3:1 maintained)
          - VWAP: Daily + Weekly  (was Weekly + Monthly on daily chart)
          - MTF:  Daily           (was Weekly — Pine: request.security(.."D"..))
          - Regime: daily 200 SMA via resampling (200 × 1H bars = only ~31 days)
          - POC lookback: 20 bars (≈ 2.5 trading days on 1H; was 50 on daily)
          - Fib swing:    20 bars (≈ 2.5 trading days on 1H; was 50 on daily)
          - Consec bars:  3       (tighter noise filter; daily uses 2)
          - Score thresholds: moderate 9/16, strong 11/16, sell 9
          - Warmup: 1400 bars     (covers ~200 daily bars × 6.5h + buffer)
          - ADX threshold: 40     (v1.4 calibration; filters choppy-market entries;
                                   buy_signal only; bounce/vwap_bounce unaffected)

        ⚠ ADX≥40 is calibrated for "All Signals" mode (the class default).
          In "Buy Only" mode it reduces entries to ~3/2y (too few).
          For "Buy Only" use, override: for_1h(entry_mode="Buy Only", adx_threshold=20)

        Override any param at call time:
            bot = GrowthSignalBot.for_1h(entry_mode="All Signals", sl_pct=1.5)
        """
        defaults: dict = dict(
            macd_fast           = 8,
            macd_slow           = 21,
            macd_sig            = 5,
            rsi_bull_min        = 40,
            rsi_bull_max        = 75,
            sl_pct              = 2.0,
            tp_pct              = 6.0,
            vwap_primary_freq   = "D",
            vwap_secondary_freq = "W",
            mtf_freq            = "D",
            use_daily_regime    = True,
            poc_len             = 20,
            fib_swing_len       = 20,
            consec_len          = 3,
            score_moderate      = 9,
            score_strong        = 11,
            score_min_sell      = 9,
            warmup_bars         = 1400,
            # v1.4 tuning: ADX≥40 gate filters choppy entries; backtest shows
            # Sharpe 1.35 / CAGR +33.8% / MaxDD -11.2% vs baseline 1.04/+27.9%/-15.9%
            adx_threshold       = 40.0,
        )
        defaults.update(overrides)
        bot      = cls(**defaults)
        bot.name = "Growth Signal Bot 1H v1.4"
        return bot
