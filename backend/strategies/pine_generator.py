"""
PineScriptGenerator — generates valid TradingView Pine Script v6 code
from configurable inputs, mirroring the structure of growth_signal_bot_v4.pine.

Supported indicators:
    rsi, macd, adx, ema20, sma50, sma200, vwap, atr, fib, volume

Entry modes (strategy only):
    "all_signals"      — buy_signal OR bounce_signal (EMA bounce, if EMA selected)
    "buy_only"         — scored buy_signal only
    "strong_buy_only"  — buy_signal AND bull_score >= strong threshold

Output types:
    "strategy"   — full strategy() script with entry/exit logic
    "indicator"  — indicator() script with calculations + plots only (no trade logic)
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


# ─── Default indicator parameters ─────────────────────────────────────────────

_INDICATOR_DEFAULTS: dict[str, dict[str, Any]] = {
    "rsi":    {"length": 14},
    "macd":   {"fast": 12, "slow": 26, "signal": 9},
    "adx":    {"length": 14, "smoothing": 14},
    "ema20":  {"length": 20},
    "sma50":  {"length": 50},
    "sma200": {"length": 200},
    "vwap":   {"anchor": "Week", "source": "HLC3"},
    "atr":    {"length": 14},
    "fib":    {"swing": 50},
    "volume": {"length": 20},
}

# Path to the linter (relative to this file's pine-scripts sibling directory)
_LINTER_PATH = Path(__file__).parent.parent.parent / "pine-scripts" / "lint_pine.py"


class PineScriptGenerator:
    """
    Generates Pine Script v6 strategy code from configurable inputs.

    Parameters
    ----------
    profile : dict
        Keys: rsi_bull_min, rsi_bull_max, adx_threshold, sl_pct, tp_pct, category
    indicators : list[str]
        Subset of: rsi, macd, adx, ema20, sma50, sma200, vwap, atr, fib, volume
    indicator_params : dict
        Per-indicator param overrides. e.g. {"rsi": {"length": 10}}
        Missing params fall back to _INDICATOR_DEFAULTS.
    strategy_name : str
    entry_mode : str
        "all_signals" | "buy_only" | "strong_buy_only"  (ignored for indicators)
    timeframe : str
        TradingView timeframe string: "D" (daily), "60" (1H), "240" (4H)
    output_type : str
        "strategy" (default) — full strategy() script with entry/exit logic
        "indicator"          — indicator() script with calculations + plots only
    """

    def __init__(
        self,
        profile: dict,
        indicators: list[str],
        indicator_params: dict,
        strategy_name: str = "Custom Strategy",
        entry_mode: str = "all_signals",
        timeframe: str = "D",
        output_type: str = "strategy",
    ) -> None:
        self.profile          = profile
        self.indicators       = [i.lower() for i in indicators]
        self.indicator_params = indicator_params or {}
        self.strategy_name    = strategy_name
        self.entry_mode       = entry_mode.lower()
        self.timeframe        = timeframe
        self.output_type      = output_type.lower()  # "strategy" or "indicator"

        # Resolved params: merge defaults with caller overrides
        self._params: dict[str, dict[str, Any]] = {}
        for key, defaults in _INDICATOR_DEFAULTS.items():
            overrides = self.indicator_params.get(key, {})
            self._params[key] = {**defaults, **overrides}

    # ── public API ─────────────────────────────────────────────────────────────

    def generate(self) -> str:
        """Return Pine Script v6 code as a string."""
        parts = [
            self._header(),
            self._inputs_section(),
            self._indicators_section(),
        ]
        if self.output_type != "indicator":
            parts.append(self._entry_conditions_section())
            parts.append(self._strategy_logic_section())
        parts.append(self._plots_section())
        return "\n".join(parts)

    def validate(self) -> tuple[bool, list[str]]:
        """
        Run the Pine Script linter on the generated code.

        Returns
        -------
        (is_valid, messages)
            is_valid  — True when there are zero errors (warnings are allowed)
            messages  — list of human-readable issue strings from the linter
        """
        code = self.generate()

        with tempfile.NamedTemporaryFile(
            suffix=".pine", mode="w", encoding="utf-8", delete=False
        ) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        try:
            result = subprocess.run(
                [sys.executable, str(_LINTER_PATH), tmp_path],
                capture_output=True,
                text=True,
            )
            output = result.stdout + result.stderr
            lines  = [ln for ln in output.splitlines() if ln.strip()]

            # The linter exits 0 when clean or warnings-only; 1 when errors exist.
            has_errors = result.returncode != 0
            return (not has_errors), lines
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # ── helpers ────────────────────────────────────────────────────────────────

    def _has(self, *keys: str) -> bool:
        """True if ALL of the given indicator keys are in self.indicators."""
        return all(k in self.indicators for k in keys)

    def _p(self, indicator: str, param: str) -> Any:
        """Return the resolved parameter value for an indicator."""
        return self._params[indicator][param]

    # ── code sections ──────────────────────────────────────────────────────────

    def _header(self) -> str:
        category = self.profile.get("category", "")
        desc = f"{self.strategy_name}"
        if category and category != "—":
            desc += f" — {category}"

        if self.output_type == "indicator":
            lines = [
                "//@version=6",
                f'// Generated by PineScriptGenerator  |  type=indicator  |  tf={self.timeframe}',
                f'indicator("{desc}", overlay=true, max_bars_back=500)',
            ]
        else:
            lines = [
                "//@version=6",
                f'// Generated by PineScriptGenerator  |  entry_mode={self.entry_mode}  |  tf={self.timeframe}',
                f'strategy("{desc}", overlay=true, max_bars_back=500,',
                "         initial_capital         = 10000,",
                "         default_qty_type        = strategy.percent_of_equity,",
                "         default_qty_value       = 10,",
                "         commission_type         = strategy.commission.percent,",
                "         commission_value        = 0.05,",
                "         slippage                = 2,",
                "         process_orders_on_close = true,",
                "         margin_long             = 100,",
                "         margin_short            = 100)",
            ]
        return "\n".join(lines)

    # ── INPUTS ─────────────────────────────────────────────────────────────────

    def _inputs_section(self) -> str:
        blocks: list[str] = [
            "",
            "// " + "\u2550" * 74,
            "// \u2500\u2500\u2500 Inputs " + "\u2500" * 60,
            "// " + "\u2550" * 74,
        ]

        sl_pct = self.profile.get("sl_pct", 5.0)
        tp_pct = self.profile.get("tp_pct", 15.0)

        # Strategy block (always present)
        blocks.append(
            "\n".join([
                "",
                "// Strategy / Exits",
                f'sl_pct    = input.float({sl_pct}, "Stop Loss %",   group="Strategy", step=0.5, minval=0.5)',
                f'tp_pct    = input.float({tp_pct}, "Take Profit %", group="Strategy", step=0.5, minval=1.0)',
                f'entry_mode = input.string("{self._pine_entry_mode()}", "Entry Mode",',
                '     options=["All Signals", "Buy Only", "Strong Buy Only"],',
                '     group="Strategy")',
            ])
        )

        # RSI inputs
        if self._has("rsi"):
            rsi_len     = self._p("rsi", "length")
            rsi_min     = self.profile.get("rsi_bull_min", 42)
            rsi_max     = self.profile.get("rsi_bull_max", 62)
            blocks.append(
                "\n".join([
                    "",
                    "// RSI",
                    f'rsi_len      = input.int({rsi_len}, "Length",        group="RSI")',
                    f'rsi_bull_min = input.int({rsi_min}, "Bull Zone Min", group="RSI")',
                    f'rsi_bull_max = input.int({rsi_max}, "Bull Zone Max", group="RSI")',
                ])
            )

        # MACD inputs
        if self._has("macd"):
            fast = self._p("macd", "fast")
            slow = self._p("macd", "slow")
            sig  = self._p("macd", "signal")
            blocks.append(
                "\n".join([
                    "",
                    "// MACD",
                    f'macd_fast = input.int({fast}, "Fast",   group="MACD")',
                    f'macd_slow = input.int({slow}, "Slow",   group="MACD")',
                    f'macd_sig  = input.int({sig},  "Signal", group="MACD")',
                ])
            )

        # ADX inputs
        if self._has("adx"):
            adx_len   = self._p("adx", "length")
            adx_smth  = self._p("adx", "smoothing")
            adx_thr   = self.profile.get("adx_threshold", 20.0)
            blocks.append(
                "\n".join([
                    "",
                    "// ADX",
                    f'adx_len       = input.int({adx_len},  "ADX Length",    group="ADX Filter", minval=1)',
                    f'adx_smoothing = input.int({adx_smth}, "ADX Smoothing", group="ADX Filter", minval=1)',
                    f'adx_threshold = input.float({adx_thr}, "ADX Threshold", group="ADX Filter", step=1.0, minval=5.0)',
                ])
            )

        # EMA20 inputs
        if self._has("ema20"):
            ema_len = self._p("ema20", "length")
            blocks.append(
                "\n".join([
                    "",
                    "// Moving Averages",
                    f'ema_len = input.int({ema_len}, "Fast EMA", group="Moving Averages", minval=1)',
                ])
            )

        # SMA50 inputs
        if self._has("sma50"):
            sma50_len = self._p("sma50", "length")
            prefix = "// Moving Averages\n" if not self._has("ema20") else ""
            blocks.append(
                f"{prefix}"
                f'sma_mid = input.int({sma50_len}, "Mid SMA", group="Moving Averages", minval=1)'
            )

        # SMA200 inputs
        if self._has("sma200"):
            sma200_len = self._p("sma200", "length")
            prefix = "// Moving Averages\n" if not self._has("ema20") and not self._has("sma50") else ""
            blocks.append(
                f"{prefix}"
                f'sma_slow = input.int({sma200_len}, "Slow SMA", group="Moving Averages", minval=1)'
            )

        # ATR inputs
        if self._has("atr"):
            atr_len = self._p("atr", "length")
            blocks.append(
                "\n".join([
                    "",
                    "// ATR",
                    f'atr_len  = input.int({atr_len}, "ATR Length", group="ATR", minval=1)',
                    'atr_mult = input.float(0.4, "ATR Buffer Multiplier", group="ATR", step=0.05)',
                ])
            )

        # Fibonacci inputs
        if self._has("fib"):
            fib_swing = self._p("fib", "swing")
            blocks.append(
                "\n".join([
                    "",
                    "// Fibonacci",
                    f'fib_len     = input.int({fib_swing}, "Swing Lookback (bars)", group="Fibonacci", minval=10, maxval=300)',
                    'fib_tol_pct = input.float(1.5, "Level Tolerance %",   group="Fibonacci", step=0.1, minval=0.1)',
                ])
            )

        # Volume inputs
        if self._has("volume"):
            vol_len = self._p("volume", "length")
            blocks.append(
                "\n".join([
                    "",
                    "// Volume",
                    f'vol_ma_len = input.int({vol_len}, "Volume MA Length", group="Volume", minval=1)',
                    'vol_mult   = input.float(1.25, "Volume Multiplier", group="Volume", step=0.05)',
                ])
            )

        # Signal scoring thresholds (always present)
        blocks.append(
            "\n".join([
                "",
                "// Signal Scoring",
                'score_strong   = input.int(11, "Strong Signal Min Score",   group="Signal Scoring", minval=1)',
                'score_moderate = input.int(8,  "Moderate Signal Min Score", group="Signal Scoring", minval=1)',
            ])
        )

        return "\n".join(blocks)

    # ── INDICATORS ─────────────────────────────────────────────────────────────

    def _indicators_section(self) -> str:
        blocks: list[str] = [
            "",
            "// \u2500\u2500\u2500 Indicators " + "\u2500" * 57,
            "",
        ]

        # Moving averages
        if self._has("ema20"):
            blocks.append("ema20 = ta.ema(close, ema_len)")
        if self._has("sma50"):
            blocks.append("sma50 = ta.sma(close, sma_mid)")
        if self._has("sma200"):
            blocks.append("sma200 = ta.sma(close, sma_slow)")

        # RSI
        if self._has("rsi"):
            blocks.append("rsi_val = ta.rsi(close, rsi_len)")

        # MACD
        if self._has("macd"):
            blocks.append("[macd_line, signal_line, macd_hist] = ta.macd(close, macd_fast, macd_slow, macd_sig)")

        # ADX — use ta.dmi() which returns [plus_di, minus_di, adx] in Pine v6
        if self._has("adx"):
            blocks.append("[_dmi_plus, _dmi_minus, adx_val] = ta.dmi(adx_len, adx_smoothing)")
            blocks.append("adx_ok = adx_val >= adx_threshold")

        # ATR
        if self._has("atr"):
            blocks.append("atr_val    = ta.atr(atr_len)")
            blocks.append("atr_buffer = atr_val * atr_mult")

        # VWAP — anchor-configurable implementation
        if self._has("vwap"):
            blocks.append(self._render_vwap())

        # Fibonacci
        if self._has("fib"):
            blocks.append(
                "\n".join([
                    "",
                    "// Fibonacci Retracement",
                    "fib_high  = ta.highest(high, fib_len)",
                    "fib_low   = ta.lowest(low,   fib_len)",
                    "fib_range = fib_high - fib_low",
                    "fib_382   = fib_low + 0.382 * fib_range",
                    "fib_500   = fib_low + 0.500 * fib_range",
                    "fib_618   = fib_low + 0.618 * fib_range",
                    "fib_tol   = close * fib_tol_pct / 100.0",
                    "near_fib  = math.abs(close - fib_382) <= fib_tol or",
                    "             math.abs(close - fib_500) <= fib_tol or",
                    "             math.abs(close - fib_618) <= fib_tol",
                ])
            )

        # Volume MA
        if self._has("volume"):
            blocks.append("")
            blocks.append("// Volume MA")
            blocks.append("vol_ma = ta.sma(volume, vol_ma_len)")

        # SL/TP in ticks (always computed for strategy.exit)
        blocks.append(
            "\n".join([
                "",
                "// Stop-loss and take-profit in ticks",
                "sl_points = strategy.position_avg_price * sl_pct / 100.0 / syminfo.mintick",
                "tp_points = strategy.position_avg_price * tp_pct / 100.0 / syminfo.mintick",
            ])
        )

        return "\n".join(blocks)

    # ── ENTRY CONDITIONS ───────────────────────────────────────────────────────

    def _entry_conditions_section(self) -> str:
        blocks: list[str] = [
            "",
            "// \u2500\u2500\u2500 Entry Conditions " + "\u2500" * 50,
            "",
        ]

        # ── gate booleans ──────────────────────────────────────────────────────
        gate_lines: list[str] = []

        if self._has("rsi"):
            gate_lines.append("rsi_gate    = rsi_val >= rsi_bull_min and rsi_val <= rsi_bull_max")
        if self._has("macd"):
            gate_lines.append("macd_gate   = macd_line > signal_line")
        if self._has("adx"):
            gate_lines.append("adx_gate    = adx_ok")
        if self._has("sma200"):
            gate_lines.append("trend_gate  = close > sma200")
        if self._has("ema20"):
            gate_lines.append("ema_gate    = close > ema20")
        if self._has("vwap"):
            gate_lines.append("vwap_gate   = close > vwap_val")
        if self._has("volume"):
            gate_lines.append("vol_gate    = volume > vol_ma")

        blocks.extend(gate_lines)
        blocks.append("")

        # ── bull_signal composite ──────────────────────────────────────────────
        # Assemble the list of active gates
        gate_names = []
        if self._has("rsi"):
            gate_names.append("rsi_gate")
        if self._has("macd"):
            gate_names.append("macd_gate")
        if self._has("adx"):
            gate_names.append("adx_gate")
        if self._has("sma200"):
            gate_names.append("trend_gate")
        if self._has("ema20"):
            gate_names.append("ema_gate")
        if self._has("vwap"):
            gate_names.append("vwap_gate")
        if self._has("volume"):
            gate_names.append("vol_gate")

        if gate_names:
            # Multi-line buy_signal
            first = gate_names[0]
            rest  = gate_names[1:]
            buy_lines = [f"buy_signal = {first}"]
            for g in rest:
                buy_lines.append(f"     and {g}")
            blocks.extend(buy_lines)
        else:
            blocks.append("buy_signal = true  // no gates selected")

        blocks.append("")

        # ── bull_strong (for strong_buy_only mode) ────────────────────────────
        # Count active gates as a proxy for score; reuse a simple sum
        if self.entry_mode == "strong_buy_only":
            # Provide a confirmations count — count of true gates beyond the first
            confirm_parts = []
            if self._has("macd"):
                confirm_parts.append("(macd_gate ? 1 : 0)")
            if self._has("adx"):
                confirm_parts.append("(adx_gate ? 1 : 0)")
            if self._has("sma200"):
                confirm_parts.append("(trend_gate ? 1 : 0)")
            if self._has("ema20"):
                confirm_parts.append("(ema_gate ? 1 : 0)")
            if self._has("vwap"):
                confirm_parts.append("(vwap_gate ? 1 : 0)")
            if self._has("volume"):
                confirm_parts.append("(vol_gate ? 1 : 0)")
            if self._has("rsi"):
                confirm_parts.append("(rsi_gate ? 1 : 0)")

            if confirm_parts:
                sum_expr = " + ".join(confirm_parts)
                blocks.append(f"confirm_count = {sum_expr}")
                blocks.append("bull_strong   = buy_signal and confirm_count >= 2")
            else:
                blocks.append("bull_strong = buy_signal")
        else:
            blocks.append("bull_strong = buy_signal and score_strong >= score_strong  // placeholder")

        blocks.append("")

        # ── bounce_signal (EMA bounce, only when ema20 is selected) ───────────
        if self._has("ema20"):
            bounce_conditions = [
                "was_below_2bar = close[1] < ema20[1] and close[2] < ema20[2]",
                "back_above_ema = close > ema20",
                "bull_candle    = close > open",
                "",
                "bounce_signal =",
                "     was_below_2bar",
                "     and back_above_ema",
                "     and bull_candle",
            ]
            if self._has("rsi"):
                bounce_conditions.append("     and rsi_val < 65")
            if self._has("sma200"):
                bounce_conditions.append("     and trend_gate")
            blocks.extend(bounce_conditions)
        else:
            blocks.append("bounce_signal = false  // ema20 not selected")

        blocks.append("")

        # ── qualify_buy — applies entry_mode logic ─────────────────────────────
        pine_mode = self._pine_entry_mode()
        if self.entry_mode == "strong_buy_only":
            blocks.append(
                f'qualify_buy = entry_mode == "Strong Buy Only" ? (buy_signal and bull_strong) : buy_signal'
            )
        elif self.entry_mode == "buy_only":
            blocks.append(
                'qualify_buy = entry_mode == "Buy Only" ? buy_signal : buy_signal'
            )
        else:  # all_signals
            if self._has("ema20"):
                blocks.append(
                    'qualify_buy = entry_mode == "All Signals" ? (buy_signal or bounce_signal) : buy_signal'
                )
            else:
                blocks.append(
                    'qualify_buy = entry_mode == "All Signals" ? buy_signal : buy_signal'
                )

        return "\n".join(blocks)

    # ── STRATEGY LOGIC ─────────────────────────────────────────────────────────

    def _strategy_logic_section(self) -> str:
        lines = [
            "",
            "// \u2500\u2500\u2500 Strategy Logic " + "\u2500" * 51,
            "",
            "long_entry = qualify_buy and strategy.position_size == 0",
            "",
            "if long_entry",
            '    strategy.entry("Long", strategy.long)',
            "",
            'strategy.exit("Exit Long", "Long", loss=sl_points, profit=tp_points)',
        ]
        return "\n".join(lines)

    # ── PLOTS ──────────────────────────────────────────────────────────────────

    def _plots_section(self) -> str:
        blocks: list[str] = [
            "",
            "// \u2500\u2500\u2500 Plots " + "\u2500" * 61,
            "",
        ]

        if self._has("ema20"):
            blocks.append('plot(ema20,  "20 EMA",  color=color.new(color.blue,   0), linewidth=1)')
        if self._has("sma50"):
            blocks.append('plot(sma50,  "50 SMA",  color=color.new(color.orange, 0), linewidth=2)')
        if self._has("sma200"):
            blocks.append('plot(sma200, "200 SMA", color=color.new(color.red,    0), linewidth=2)')

        if self._has("vwap"):
            anchor = self._params["vwap"].get("anchor", "Week")
            source = self._params["vwap"].get("source", "HLC3")
            vwap_label = f"{anchor} VWAP ({source})"
            blocks.append(
                f'plot(vwap_val, "{vwap_label}", color=color.new(color.purple, 0), linewidth=2)'
            )

        if self._has("fib"):
            blocks.extend([
                'plot(fib_382, "Fib 38.2%", color=color.new(color.yellow, 40), linewidth=1)',
                'plot(fib_500, "Fib 50.0%", color=color.new(color.yellow, 20), linewidth=1)',
                'plot(fib_618, "Fib 61.8%", color=color.new(color.yellow, 40), linewidth=1)',
            ])

        # Buy signal shapes
        blocks.append("")
        blocks.append(
            'plotshape(buy_signal, title="Buy", style=shape.triangleup, '
            'location=location.belowbar, color=color.new(color.green, 0), size=size.normal)'
        )
        if self._has("ema20"):
            blocks.append(
                'plotshape(bounce_signal, title="Bounce", style=shape.circle, '
                'location=location.belowbar, color=color.new(color.aqua, 0), size=size.small)'
            )

        # Regime background (when sma200 selected)
        if self._has("sma200"):
            blocks.extend([
                "",
                "bull_regime = close > sma200",
                "bear_regime = close < sma200",
                "regime_bg   = bull_regime ? color.new(color.green, 95) :",
                "              bear_regime ? color.new(color.red, 95) : na",
                'bgcolor(regime_bg, title="Regime BG")',
            ])

        return "\n".join(blocks)

    # ── VWAP renderer ──────────────────────────────────────────────────────────

    def _render_vwap(self) -> str:
        """
        Generate the Pine Script VWAP block based on the configured anchor and source.

        Anchor options:
            "Session"  — uses the built-in ta.vwap() which resets each trading session.
            "Week"     — manual cumulative reset every new calendar week.
            "Month"    — manual cumulative reset every new calendar month.
            "Quarter"  — manual cumulative reset at the start of each quarter (Jan/Apr/Jul/Oct).
            "Year"     — manual cumulative reset every new calendar year.

        Source options: HLC3 (typical price), HL2, Close, OHLC4.
        """
        anchor = self._params["vwap"].get("anchor", "Week")
        source = self._params["vwap"].get("source", "HLC3")

        src_exprs = {
            "HLC3":  "(high + low + close) / 3",
            "HL2":   "(high + low) / 2",
            "Close": "close",
            "OHLC4": "(open + high + low + close) / 4",
        }
        src_expr = src_exprs.get(source, "(high + low + close) / 3")

        anchor_labels = {
            "Session": "Session",
            "Week":    "Weekly",
            "Month":   "Monthly",
            "Quarter": "Quarterly",
            "Year":    "Yearly",
        }
        label = anchor_labels.get(anchor, anchor)

        lines = [
            "",
            f"// {label} VWAP  (source: {source})",
            f"vwap_src = {src_expr}",
        ]

        if anchor == "Session":
            # Pine Script built-in resets automatically each session
            lines.append("vwap_val = ta.vwap(vwap_src)")
        else:
            reset_conditions = {
                "Week":    "weekofyear != weekofyear[1] or year != year[1]",
                "Month":   "month != month[1] or year != year[1]",
                "Quarter": (
                    "year != year[1] or "
                    "(month != month[1] and (month == 1 or month == 4 or month == 7 or month == 10))"
                ),
                "Year":    "year != year[1]",
            }
            reset_cond = reset_conditions.get(anchor, reset_conditions["Week"])

            lines += [
                f"new_period = {reset_cond}",
                "",
                "var float _vwap_vol_px  = 0.0",
                "var float _vwap_vol     = 0.0",
                "var float _vwap_sum_sq  = 0.0",
                "var int   _vwap_bar_cnt = 0",
                "",
                "if new_period",
                "    _vwap_vol_px  := 0.0",
                "    _vwap_vol     := 0.0",
                "    _vwap_sum_sq  := 0.0",
                "    _vwap_bar_cnt := 0",
                "",
                "_vwap_vol_px  += vwap_src * volume",
                "_vwap_vol     += volume",
                "_vwap_bar_cnt += 1",
                "",
                "vwap_val      = _vwap_vol_px / _vwap_vol",
                "_vwap_sum_sq += math.pow(vwap_src - vwap_val, 2)",
                "vwap_stdev    = math.sqrt(_vwap_sum_sq / _vwap_bar_cnt)",
            ]

        return "\n".join(lines)

    # ── utilities ──────────────────────────────────────────────────────────────

    def _pine_entry_mode(self) -> str:
        """Convert internal entry_mode key to Pine Script display string."""
        mapping = {
            "all_signals":     "All Signals",
            "buy_only":        "Buy Only",
            "strong_buy_only": "Strong Buy Only",
        }
        return mapping.get(self.entry_mode, "All Signals")
