#!/usr/bin/env python3
"""
run_decide.py  —  Growth Signal Bot Decision Engine  (v2 — earnings + macro)

Answers the question: "Given the current market state for TICKER,
what does the strategy's data say about holding vs exiting over HORIZON trading days?"

Sections:
  0. Earnings & Market Alert — ⚠ earnings date warning + SPY/QQQ macro regime
  1. Current State           — what every indicator is reading right now
  2. Historical Base Rates   — in all past instances with this same state,
                               what happened over the next N trading days?
                               (with confidence grade A–D)
  3. Key Levels & Scenarios  — bull/base/bear targets and regime implications
  4. Decision Checklist      — specific conditions that would trigger each action

Usage:
    python run_decide.py META
    python run_decide.py META --horizon 63        # 63 td ≈ 3 months (default)
    python run_decide.py META GOOG NVDA           # multi-symbol
    python run_decide.py META --horizon 21        # 1-month horizon

NOT investment advice — objective indicator data only.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy  as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent))

from backtester.data   import fetch_ohlcv
from strategies.growth_signal_bot import GrowthSignalBot


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

W = 66   # report width

def _sep(char="─"): return "  " + char * (W - 2)
def _hdr(title, char="═"): return f"\n  {char * (W - 2)}\n  {title}\n  {char * (W - 2)}"
def _h2(title): return f"\n  ── {title} {'─' * max(1, W - 6 - len(title))}"

def pct(v: float, decimals: int = 1) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v * 100:.{decimals}f}%"

def dol(v: float) -> str:
    return f"${v:>10,.2f}"

def bar_str(ts) -> str:
    try:
        return pd.Timestamp(ts).strftime("%Y-%m-%d")
    except Exception:
        return str(ts)


# ─────────────────────────────────────────────────────────────────────────────
# State bucketing — defines "similar historical states"
# ─────────────────────────────────────────────────────────────────────────────

def _regime_bucket(row: pd.Series) -> str:
    if bool(row.get("bull_regime", False)):
        return "bull"
    if bool(row.get("bear_regime", False)):
        return "bear"
    return "neutral"

def _score_bucket(score: int) -> str:
    if score >= 11:
        return "strong"
    if score >= 8:
        return "moderate"
    return "weak"

def _adx_bucket(adx: float, threshold: float) -> str:
    return "trending" if adx >= threshold else "choppy"

def _macd_bucket(row: pd.Series) -> str:
    return "positive" if float(row.get("macd_line", 0)) > float(row.get("signal_line", 0)) else "negative"

def _div_bucket(row: pd.Series) -> str:
    if bool(row.get("bull_div", False)):
        return "bull_div"
    if bool(row.get("bear_div", False)):
        return "bear_div"
    return "none"

def get_state_bucket(row: pd.Series, adx_threshold: float) -> dict:
    return {
        "regime":  _regime_bucket(row),
        "score":   _score_bucket(int(row.get("bull_score", 0))),
        "adx":     _adx_bucket(float(row.get("adx", 0)), adx_threshold),
        "macd":    _macd_bucket(row),
    }

def bucket_label(b: dict) -> str:
    return f"regime={b['regime']} / score={b['score']} / ADX={b['adx']} / MACD={b['macd']}"


# ─────────────────────────────────────────────────────────────────────────────
# Historical base rate computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_base_rates(
    df:            pd.DataFrame,
    current_bucket: dict,
    horizons:      list[int],
    adx_threshold: float,
    exact_match:   bool = True,
) -> dict:
    """
    Find all historical bars matching current_bucket, compute forward returns
    at each horizon. Returns stats dict keyed by horizon.

    exact_match=True  → all 4 bucket dimensions must match
    exact_match=False → only regime + score must match (looser)
    """
    results = {}

    # Precompute close array for fast forward-return lookup
    closes  = df["close"].values
    indices = np.arange(len(closes))

    # Find matching bars (exclude last `max(horizons)` bars — no forward data)
    max_h   = max(horizons)
    valid   = df.iloc[: -max_h] if len(df) > max_h else df.iloc[:0]

    match_mask = pd.Series(True, index=valid.index)
    for i, (ts, row) in enumerate(valid.iterrows()):
        b = get_state_bucket(row, adx_threshold)
        if exact_match:
            match_mask.iloc[i] = (b == current_bucket)
        else:
            # loose: regime + score only
            match_mask.iloc[i] = (
                b["regime"] == current_bucket["regime"]
                and b["score"] == current_bucket["score"]
            )

    matching_idx = np.where(match_mask.values)[0]
    n_instances  = len(matching_idx)

    for h in horizons:
        fwd_returns = []
        for idx in matching_idx:
            future_idx = idx + h
            if future_idx < len(closes):
                fwd_ret = (closes[future_idx] - closes[idx]) / closes[idx]
                fwd_returns.append(fwd_ret)

        if not fwd_returns:
            results[h] = {"n": 0}
            continue

        arr = np.array(fwd_returns)
        results[h] = {
            "n":        len(arr),
            "win_rate": float(np.mean(arr > 0)),
            "median":   float(np.median(arr)),
            "mean":     float(np.mean(arr)),
            "p25":      float(np.percentile(arr, 25)),
            "p75":      float(np.percentile(arr, 75)),
            "worst":    float(np.min(arr)),
            "best":     float(np.max(arr)),
            "std":      float(np.std(arr)),
        }

    results["_n_instances"] = n_instances
    results["_exact"] = exact_match
    return results


def _bar_chart(win_rate: float, width: int = 20) -> str:
    filled = round(win_rate * width)
    empty  = width - filled
    return f"[{'█' * filled}{'░' * empty}]"


# ─────────────────────────────────────────────────────────────────────────────
# Earnings date detection
# ─────────────────────────────────────────────────────────────────────────────

def get_earnings_info(ticker: str) -> dict | None:
    """
    Fetch next scheduled earnings date via yfinance.
    Returns dict with date, calendar days, and approximate trading days until.
    Returns None if unavailable or no upcoming date found.
    """
    try:
        t   = yf.Ticker(ticker)
        cal = t.calendar          # dict or DataFrame depending on yf version

        dates = []
        if isinstance(cal, dict):
            raw = cal.get("Earnings Date", [])
            dates = list(raw) if raw is not None else []
        elif cal is not None and hasattr(cal, "loc"):
            # Older yfinance: DataFrame with dates as columns
            if "Earnings Date" in cal.index:
                dates = list(cal.loc["Earnings Date"].values)

        if not dates:
            return None

        now = pd.Timestamp.now(tz="UTC").normalize()
        upcoming = []
        for d in dates:
            try:
                d_ts = pd.Timestamp(d)
                if d_ts.tzinfo is None:
                    d_ts = d_ts.tz_localize("UTC")
                if d_ts >= now:
                    upcoming.append(d_ts)
            except Exception:
                continue

        if not upcoming:
            return None

        next_e         = min(upcoming)
        cal_days       = (next_e - now).days
        trading_days   = max(0, round(cal_days * 5 / 7))
        return {
            "date":          next_e,
            "cal_days":      cal_days,
            "trading_days":  trading_days,
        }
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Macro overlay — SPY + QQQ regime context
# ─────────────────────────────────────────────────────────────────────────────

def get_macro_context() -> dict[str, dict]:
    """
    Run the daily strategy on SPY and QQQ to get market-wide regime context.
    Returns dict keyed by ticker with regime, score, ADX, and recent returns.
    """
    result: dict[str, dict] = {}
    strat  = GrowthSignalBot()

    for mkt_ticker, label in [("SPY", "S&P 500 (SPY)"), ("QQQ", "Nasdaq 100 (QQQ)")]:
        try:
            df = fetch_ohlcv(mkt_ticker, period="1y", interval="1d")
            if df is None or len(df) < 220:
                continue
            df_p = strat.prepare(df.copy())
            last = df_p.iloc[-1]
            close = float(last["close"])
            sma200 = float(last["sma200"])
            sma200_rising = close_prev = float(df_p.iloc[-2]["sma200"])
            result[mkt_ticker] = {
                "label":         label,
                "close":         close,
                "regime":        _regime_bucket(last),
                "bull_score":    int(last.get("bull_score", 0)),
                "adx":           float(last.get("adx", 0)),
                "sma200":        sma200,
                "sma200_rising": float(last["sma200"]) > float(df_p.iloc[-2]["sma200"]),
                "ret_1m":        (close - float(df_p.iloc[-21]["close"])) / float(df_p.iloc[-21]["close"]) if len(df_p) > 21 else None,
                "ret_3m":        (close - float(df_p.iloc[-63]["close"])) / float(df_p.iloc[-63]["close"]) if len(df_p) > 63 else None,
                "macd_positive": float(last.get("macd_line", 0)) > float(last.get("signal_line", 0)),
                "rsi":           float(last.get("rsi", 0)),
            }
        except Exception:
            continue

    return result


def _macro_regime_line(info: dict) -> str:
    regime_icons = {"bull": "🐂 BULL", "bear": "🐻 BEAR", "neutral": "🔀 NEUTRAL"}
    regime = regime_icons.get(info["regime"], info["regime"])
    sma200_arrow = "↑" if info["sma200_rising"] else "↓"
    ret_1m = pct(info["ret_1m"]) if info["ret_1m"] is not None else "n/a"
    ret_3m = pct(info["ret_3m"]) if info["ret_3m"] is not None else "n/a"
    return (f"  {info['label']:<20}  {regime:<16}  "
            f"SMA200 {sma200_arrow}   1mo {ret_1m:>7}   3mo {ret_3m:>7}   "
            f"ADX {info['adx']:.0f}  RSI {info['rsi']:.0f}")


def _macro_summary_note(macro: dict[str, dict]) -> str:
    """Plain-language note on overall market regime."""
    bull_count = sum(1 for v in macro.values() if v["regime"] == "bull")
    bear_count = sum(1 for v in macro.values() if v["regime"] == "bear")
    if bull_count == len(macro):
        return "✅ Both SPY and QQQ in confirmed bull regime — market tailwind"
    if bear_count == len(macro):
        return "🔴 Both SPY and QQQ in bear regime — strong market headwind"
    if bull_count > 0:
        return "🟡 Mixed market regime — SPY/QQQ not aligned, reduced signal quality"
    return "🔀 Market in transition — reduced conviction on individual stock signals"


# ─────────────────────────────────────────────────────────────────────────────
# Base rate confidence grading
# ─────────────────────────────────────────────────────────────────────────────

def _confidence_grade(n: int, spread_pct: float, exact_match: bool) -> tuple[str, str]:
    """
    Grade the statistical reliability of the base rate.
    spread_pct = P75 - P25 as a decimal (e.g. 0.40 = 40pp spread)
    Returns (grade, explanation)
    """
    score = 0
    if n >= 60:   score += 3
    elif n >= 30: score += 2
    elif n >= 15: score += 1

    if spread_pct < 0.25:  score += 3   # very tight = predictive
    elif spread_pct < 0.40: score += 2
    elif spread_pct < 0.60: score += 1

    if exact_match: score += 1

    if score >= 6:
        return "A", f"High confidence  (n={n}, spread={pct(spread_pct)})"
    if score >= 4:
        return "B", f"Moderate confidence  (n={n}, spread={pct(spread_pct)})"
    if score >= 2:
        return "C", f"Low confidence — wide variance  (n={n}, spread={pct(spread_pct)})"
    return "D", f"Very low confidence — treat as noise  (n={n}, spread={pct(spread_pct)})"


# ─────────────────────────────────────────────────────────────────────────────
# Main report function
# ─────────────────────────────────────────────────────────────────────────────

def run_decision_report(ticker: str, horizon_td: int = 63) -> None:
    """Full decision engine report for a single ticker."""

    print(_hdr(f"  DECISION ENGINE  ·  {ticker}  ·  {horizon_td}td horizon (~{horizon_td//21}mo)"))

    # ── 1. Fetch data ─────────────────────────────────────────────────────────
    print(f"\n  Fetching data …", end="", flush=True)
    df_long = fetch_ohlcv(ticker, period="max", interval="1d")
    df_1h   = fetch_ohlcv(ticker, period="2y",  interval="1h")
    if df_long is None or len(df_long) < 300:
        df_long = fetch_ohlcv(ticker, period="10y", interval="1d")
    if df_long is None or len(df_long) < 150:
        print(f"\n  ⚠ Not enough data for {ticker}")
        return
    print(f" {len(df_long)} daily bars ({bar_str(df_long.index[0])} → {bar_str(df_long.index[-1])})")

    # Fetch macro + earnings in parallel with indicator prep
    print(f"  Fetching macro (SPY/QQQ) + earnings …", end="", flush=True)
    macro    = get_macro_context()
    earnings = get_earnings_info(ticker)
    print(" done")

    # ── 2. Run strategy indicators ────────────────────────────────────────────
    strat = GrowthSignalBot()     # daily v4.3 defaults
    df    = strat.prepare(df_long.copy())

    last       = df.iloc[-1]
    last_close = float(last["close"])
    last_ts    = df.index[-1]

    # 1H indicators for intraday context
    if df_1h is not None and len(df_1h) > 1400:
        strat_1h = GrowthSignalBot.for_1h()
        df_1h_p  = strat_1h.prepare(df_1h.copy())
        last_1h  = df_1h_p.iloc[-1]
        has_1h   = True
    else:
        has_1h   = False

    # Current state bucket
    bucket = get_state_bucket(last, strat.adx_threshold)

    # ── ─────────────────────────────────────────────────────────────────────
    # SECTION 0: Earnings Alert + Market Context
    # ── ─────────────────────────────────────────────────────────────────────
    print(_h2("0 · EARNINGS & MARKET CONTEXT"))

    # Earnings warning
    if earnings:
        td  = earnings["trading_days"]
        cal = earnings["cal_days"]
        dt  = bar_str(earnings["date"])
        if td <= 3:
            flag = "🚨 IMMINENT"
        elif td <= 10:
            flag = "⚠  UPCOMING"
        else:
            flag = "📅"
        warn = " — ⚠ SIGNALS UNRELIABLE NEAR EARNINGS" if td <= 10 else ""
        print(f"\n  {flag}  Earnings: {dt}  "
              f"({cal} calendar days / ~{td} trading days away){warn}")
    else:
        print(f"\n  📅  Earnings: not found / not scheduled in near term")

    # Market regime context
    print()
    if macro:
        for info in macro.values():
            print(_macro_regime_line(info))
        print(f"\n  {_macro_summary_note(macro)}")

        # Relative performance vs SPY
        spy = macro.get("SPY")
        if spy:
            for bars, label in [(21, "1mo"), (63, "3mo")]:
                if len(df) > bars:
                    ticker_ret  = (last_close - float(df.iloc[-bars]["close"])) / float(df.iloc[-bars]["close"])
                    spy_ret     = spy.get(f"ret_{label}", spy.get("ret_1m" if bars == 21 else "ret_3m"))
                    if spy_ret is not None:
                        alpha = ticker_ret - spy_ret
                        vs    = "outperforming" if alpha > 0 else "underperforming"
                        print(f"  {ticker} vs SPY {label}: {pct(ticker_ret)} vs {pct(spy_ret)} "
                              f"({vs} by {pct(abs(alpha))})")
    else:
        print("  ⚠ Could not fetch macro data")

    # ── ─────────────────────────────────────────────────────────────────────
    # SECTION 1: Current State
    # ── ─────────────────────────────────────────────────────────────────────
    print(_h2("1 · CURRENT STATE"))

    # Price
    bar_chg = (float(last["close"]) - float(last["open"])) / float(last["open"])
    print(f"  Last bar : {bar_str(last_ts)}   Close: {dol(last_close)}  "
          f"({pct(bar_chg)} on bar)")

    # 1m / 3m price change
    for bars, label in [(21, "1mo"), (63, "3mo"), (126, "6mo")]:
        if len(df) > bars:
            past = float(df.iloc[-bars]["close"])
            chg  = (last_close - past) / past
            print(f"  {label} price change: {pct(chg):>8}   "
                  f"({dol(past).strip()} → {dol(last_close).strip()})")
    print()

    # Regime
    sma200      = float(last["sma200"])
    sma200_dir  = "↑ rising"  if float(last["sma200"]) > float(df.iloc[-2]["sma200"]) else "↓ falling"
    dist_sma200 = (last_close - sma200) / sma200
    regime_sym  = {"bull": "🐂 BULL", "bear": "🐻 BEAR", "neutral": "🔀 NEUTRAL/TRANSITIONING"}
    print(f"  Regime    : {regime_sym[bucket['regime']]}")
    print(f"  SMA200    : {dol(sma200).strip()}  {sma200_dir}   "
          f"Price is {pct(dist_sma200)} {'above' if dist_sma200 > 0 else 'below'}")
    print(f"  SMA50     : {dol(float(last['sma50'])).strip()}   "
          f"Price is {pct((last_close - float(last['sma50'])) / float(last['sma50']))} "
          f"{'above' if last_close > float(last['sma50']) else 'below'}")
    print()

    # Scores
    bull_score = int(last.get("bull_score", 0))
    bear_score = int(last.get("bear_score", 0))
    filled     = "█" * bull_score + "░" * (16 - bull_score)
    score_lbl  = (
        "🔥 STRONG" if bull_score >= strat.score_strong
        else "✅ MODERATE" if bull_score >= strat.score_moderate
        else "❌ below threshold"
    )
    print(f"  Bull score: {bull_score}/16  [{filled}]  {score_lbl}")
    print(f"  Bear score: {bear_score}/16  (sell threshold: {strat.score_min_sell})")

    adx = float(last.get("adx", 0))
    adx_sym = "🟢 trending" if adx >= strat.adx_threshold else "🔴 choppy"
    print(f"  ADX       : {adx:.1f}  [{adx_sym}]  (gate: ≥{strat.adx_threshold})")

    rsi   = float(last["rsi"])
    macd_l = float(last["macd_line"])
    sig_l  = float(last["signal_line"])
    rsi_zone = "🟢 bull zone" if strat.rsi_bull_min <= rsi <= strat.rsi_bull_max else ("🔴 over" if rsi > strat.rsi_bull_max else "🟡 under zone")
    print(f"  RSI       : {rsi:.1f}  [{rsi_zone}]  "
          f"MACD: {'🟢 above' if macd_l > sig_l else '🔴 below'} signal ({macd_l:.2f} vs {sig_l:.2f})")
    print()

    # Active & recent signals
    active_signals = []
    for col, label in [("buy_signal","BUY"), ("bounce_signal","BOUNCE"),
                       ("vwap_bounce","VWAP BOUNCE"), ("sell_signal","SELL"),
                       ("bull_div","Bull Div"), ("bear_div","Bear Div")]:
        if last.get(col):
            active_signals.append(label)
    print(f"  Signal now: {', '.join(active_signals) if active_signals else '— no signal (watching)'}")

    # Most recent signal events (last 5)
    recent = []
    for col, label in [("buy_signal","🔺 BUY"), ("bounce_signal","⬤ BOUNCE"),
                       ("vwap_bounce","◆ VWAP BOUNCE"), ("sell_signal","🔻 SELL"),
                       ("bull_div","✕ Bull Div"), ("bear_div","✕ Bear Div")]:
        if col not in df.columns:
            continue
        fired = df[df[col] == True]
        for ts, row in fired.iterrows():
            recent.append((ts, label, float(row["close"])))
    recent.sort(key=lambda x: x[0])
    if recent:
        print(f"  Last signals:")
        for ts, lbl, price in recent[-5:]:
            marker = " ◀ latest" if ts == recent[-1][0] else ""
            print(f"       {bar_str(ts)}  {lbl:<16}  {dol(price).strip()}{marker}")
    print()

    # 1H context
    if has_1h:
        score_1h = int(last_1h.get("bull_score", 0))
        adx_1h   = float(last_1h.get("adx", 0))
        rsi_1h   = float(last_1h.get("rsi", 0))
        macd_1h  = float(last_1h.get("macd_line", 0))
        sig_1h   = float(last_1h.get("signal_line", 0))
        print(f"  1H context: score {score_1h}/16  "
              f"ADX {adx_1h:.0f}  RSI {rsi_1h:.0f}  "
              f"MACD {'↑' if macd_1h > sig_1h else '↓'}")

    # ── ─────────────────────────────────────────────────────────────────────
    # SECTION 2: Historical Base Rates
    # ── ─────────────────────────────────────────────────────────────────────
    print(_h2(f"2 · HISTORICAL BASE RATES  (state: {bucket_label(bucket)})"))

    horizons = [5, 10, 21, 42, 63, 126]
    h_labels = {5: "1wk", 10: "2wk", 21: "1mo", 42: "2mo", 63: "3mo", 126: "6mo"}

    # Exact match first, fall back to loose
    stats = compute_base_rates(df, bucket, horizons, strat.adx_threshold, exact_match=True)
    n_exact = stats["_n_instances"]

    if n_exact < 10:
        print(f"  ⚠ Only {n_exact} exact-match instances — using regime+score bucket (looser match)")
        stats = compute_base_rates(df, bucket, horizons, strat.adx_threshold, exact_match=False)
        n_used = stats["_n_instances"]
        match_desc = f"regime={bucket['regime']} / score={bucket['score']}"
    else:
        n_used = n_exact
        match_desc = bucket_label(bucket)

    print(f"  Matched {n_used} historical instances for: {match_desc}")
    print(f"  ({bar_str(df_long.index[0])} → {bar_str(df_long.index[-1])})")
    print()

    # Buy-and-hold benchmark at same horizons
    bh = {}
    for h in horizons:
        closes_arr = df["close"].values
        rets = []
        for i in range(len(closes_arr) - h):
            rets.append((closes_arr[i + h] - closes_arr[i]) / closes_arr[i])
        if rets:
            bh[h] = {"win_rate": np.mean(np.array(rets) > 0), "median": np.median(rets)}

    # Table header
    col_w = 10
    print(f"  {'Horizon':<8}  {'N':>4}  {'WinRate':>8}  {'Median':>8}  "
          f"{'P25':>8}  {'P75':>8}  {'Worst':>8}  {'Best':>8}  "
          f"{'B&H WR':>8}  {'B&H Med':>8}")
    print(_sep())

    for h in horizons:
        s = stats.get(h, {})
        if not s or s.get("n", 0) == 0:
            print(f"  {h_labels[h]:<8}  {'—':>4}")
            continue
        bh_s = bh.get(h, {})
        wr_bar = _bar_chart(s["win_rate"], 10)
        print(f"  {h_labels[h]:<8}  {s['n']:>4}  "
              f"{pct(s['win_rate']):>8}  {pct(s['median']):>8}  "
              f"{pct(s['p25']):>8}  {pct(s['p75']):>8}  "
              f"{pct(s['worst']):>8}  {pct(s['best']):>8}  "
              f"{pct(bh_s.get('win_rate',0)):>8}  {pct(bh_s.get('median',0)):>8}")

    # Key 3-month stat callout + confidence grade
    s3m = stats.get(horizon_td, stats.get(63, {}))
    if s3m and s3m.get("n", 0) > 0:
        print()
        wr_bar = _bar_chart(s3m["win_rate"])
        spread = s3m["p75"] - s3m["p25"]
        grade, grade_desc = _confidence_grade(
            n           = s3m["n"],
            spread_pct  = spread,
            exact_match = stats.get("_exact", True),
        )
        print(f"  ★  {horizon_td}td ({h_labels.get(horizon_td,'custom')}) base rate:  "
              f"Win rate {pct(s3m['win_rate'])}  {wr_bar}")
        print(f"     Median outcome : {pct(s3m['median'])}   "
              f"(P25: {pct(s3m['p25'])} / P75: {pct(s3m['p75'])})")
        print(f"     Range          : {pct(s3m['worst'])} to {pct(s3m['best'])}")
        print(f"     Confidence     : Grade {grade} — {grade_desc}")

    # ── ─────────────────────────────────────────────────────────────────────
    # SECTION 3: Key Levels & Scenarios
    # ── ─────────────────────────────────────────────────────────────────────
    print(_h2("3 · KEY LEVELS & SCENARIOS"))

    sma50  = float(last["sma50"])
    ema20  = float(last["ema20"])
    vwap_w = float(last["vwap_w"]) if "vwap_w" in df.columns else None
    vwap_m = float(last["vwap_m"]) if "vwap_m" in df.columns else None
    poc    = float(last["poc"])    if "poc"    in df.columns else None

    # 52-week high/low
    recent_52w  = df.iloc[-252:]["close"] if len(df) >= 252 else df["close"]
    high_52w    = float(recent_52w.max())
    low_52w     = float(recent_52w.min())

    print(f"  {'Level':<22}  {'Price':>10}  {'Dist':>8}  Notes")
    print(_sep())

    levels = [
        ("52-week High",    high_52w,  "resistance"),
        ("POC (vol. node)", poc,       "vol S/R"),
        ("Monthly VWAP",    vwap_m,    "secondary VWAP"),
        ("SMA200 ← KEY",    sma200,    "bull/bear regime line"),
        ("Weekly VWAP",     vwap_w,    "primary VWAP"),
        ("SMA50",           sma50,     "medium-term trend"),
        ("EMA20",           ema20,     "short-term trend"),
        ("Current Price",   last_close,""),
        ("52-week Low",     low_52w,   "support"),
    ]
    for name, val, note in levels:
        if val is None:
            continue
        dist = (last_close - val) / val
        arrow = "▲" if val > last_close else ("=" if abs(dist) < 0.001 else "▼")
        marker = " ◀" if name == "Current Price" else ""
        print(f"  {name:<22}  {dol(val)}  {pct(dist):>8}  {note}{marker}")

    # Scenarios at horizon
    print(f"\n  Scenario analysis at {horizon_td}td ({h_labels.get(horizon_td,'custom')} horizon):")
    print(f"  {'Scenario':<22}  {'Target':>10}  {'Chg':>8}  Regime implication")
    print(_sep())

    scenarios = [
        ("Bear  (–15%)",  last_close * 0.85),
        ("Bear  (–10%)",  last_close * 0.90),
        ("Base  (±0%)",   last_close * 1.00),
        ("Bull  (+10%)",  last_close * 1.10),
        ("Bull  (+20%)",  last_close * 1.20),
    ]

    for name, target in scenarios:
        chg = (target - last_close) / last_close
        above_sma200  = target > sma200
        regime_note = (
            "Reclaims SMA200 → potential regime flip to BULL" if above_sma200 and bucket["regime"] != "bull"
            else "Above SMA200 — bull regime continues" if above_sma200
            else f"Below SMA200 (${sma200:,.0f}) — bear/neutral regime"
        )
        print(f"  {name:<22}  {dol(target)}  {pct(chg):>8}  {regime_note}")

    # What price needed to flip regime
    to_sma200 = (sma200 - last_close) / last_close
    print()
    if bucket["regime"] != "bull":
        print(f"  ⚡ Regime flip to BULL needs: {dol(sma200).strip()} "
              f"({pct(to_sma200)} from current price)")
    else:
        print(f"  ⚡ Regime flip to BEAR would occur below SMA200: "
              f"{dol(sma200).strip()} ({pct(to_sma200)} from current price)")

    # ── ─────────────────────────────────────────────────────────────────────
    # SECTION 4: Decision Checklist
    # ── ─────────────────────────────────────────────────────────────────────
    print(_h2("4 · DECISION CHECKLIST"))

    def chk(condition: bool) -> str:
        return "✅" if condition else "❌"

    # What would make the bot BUY MORE
    print(f"\n  If you were considering buying MORE:")
    needs_bull_regime   = bucket["regime"] == "bull"
    needs_score_mod     = bull_score >= strat.score_moderate
    needs_adx           = adx >= strat.adx_threshold
    needs_macd_cross    = bool(last.get("macd_cross", False))
    needs_rsi_zone      = strat.rsi_bull_min <= rsi <= strat.rsi_bull_max
    print(f"  {chk(needs_bull_regime)}  Bull regime confirmed (price > rising SMA200)")
    print(f"  {chk(needs_score_mod )}  Score ≥ {strat.score_moderate}/16  (currently {bull_score}/16)")
    print(f"  {chk(needs_adx       )}  ADX ≥ {strat.adx_threshold} (currently {adx:.1f})")
    print(f"  {chk(needs_rsi_zone  )}  RSI in bull zone {strat.rsi_bull_min}–{strat.rsi_bull_max} (currently {rsi:.1f})")
    print(f"  {chk(needs_macd_cross)}  MACD crossed up (golden cross momentum)")
    buy_ready = sum([needs_bull_regime, needs_score_mod, needs_adx, needs_rsi_zone])
    print(f"  → Bot buy readiness: {buy_ready}/4 core conditions met")

    # What would make the bot SELL / EXIT
    print(f"\n  If you're considering EXITING / SELLING:")
    sell_bear_regime = bucket["regime"] == "bear"
    sell_score_ok    = bear_score >= strat.score_min_sell
    sell_macd_cross  = bool(last.get("macd_crossunder", False))
    sell_consec      = bool(last.get("consec_below", False))
    print(f"  {chk(sell_bear_regime)}  Bear regime confirmed (price < falling SMA200)")
    print(f"  {chk(sell_score_ok   )}  Bear score ≥ {strat.score_min_sell}/16  (currently {bear_score}/16)")
    print(f"  {chk(sell_macd_cross )}  MACD crossed down (momentum reversal)")
    print(f"  {chk(sell_consec     )}  N consecutive closes below EMA20")
    sell_proximity = sum([sell_bear_regime, sell_score_ok, sell_macd_cross, sell_consec])
    print(f"  → Bot sell signal proximity: {sell_proximity}/4 conditions met")

    # HOLD signals
    print(f"\n  Reasons the bot currently says HOLD / WAIT:")
    hold_reasons = []
    if not active_signals:
        hold_reasons.append("No buy OR sell signal is firing — no action triggered")
    if recent:
        last_sig_ts, last_sig_lbl, _ = recent[-1]
        hold_reasons.append(f"Last signal was '{last_sig_lbl}' on {bar_str(last_sig_ts)} — no new trigger")
    if bool(last.get("bull_div", False)):
        hold_reasons.append("Bull divergence active — possible exhaustion of downward move")
    if sma200_dir == "↑ rising" and bucket["regime"] != "bear":
        hold_reasons.append("SMA200 is rising — long-term uptrend structure intact")
    for r in hold_reasons:
        print(f"  ◆  {r}")

    # Watch levels — what to monitor
    print(f"\n  📌 Watch levels (monitor these over the next {horizon_td}td):")
    print(f"     {dol(sma200).strip()}  SMA200 — reclaiming this would restore bull regime")
    print(f"     {dol(sma50).strip()}   SMA50  — immediate resistance / support")
    if poc:
        print(f"     {dol(poc).strip()}    POC    — high-volume price node, strong S/R")
    if last_close > sma200 * 0.95:
        print(f"     SMA200 is within 5% — key test approaching")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(_h2("SUMMARY"))

    s3m = stats.get(horizon_td, stats.get(63, {}))
    print(f"\n  Symbol       : {ticker}   ${last_close:,.2f}  ({bar_str(last_ts)})")
    print(f"  Regime       : {regime_sym[bucket['regime']]}  —  {bucket_label(bucket)}")
    print(f"  Bot action   : {'🚫 No signal — watching' if not active_signals else ', '.join(active_signals)}")

    if s3m and s3m.get("n", 0) > 0:
        spread = s3m["p75"] - s3m["p25"]
        grade, _ = _confidence_grade(s3m["n"], spread, stats.get("_exact", True))
        print(f"  {horizon_td}td base rate: {pct(s3m['win_rate'])} positive  /  "
              f"median {pct(s3m['median'])}  (n={s3m['n']}, confidence: {grade})")
    if macro:
        print(f"  Market regime: {_macro_summary_note(macro)}")
    if earnings:
        td = earnings["trading_days"]
        warn = "  ⚠ EARNINGS IMMINENT — reduce signal weight" if td <= 10 else ""
        print(f"  Earnings in  : ~{td} trading days ({bar_str(earnings['date'])}){warn}")
    print()
    print(f"  ⚠  This is indicator data only — not investment advice.")
    print(f"  ⚠  Past base rates do not predict future performance.")
    print(_sep("═"))
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = sys.argv[1:]
    if not args:
        tickers  = ["META"]
        horizon  = 63
    else:
        tickers = [a.upper() for a in args if not a.startswith("--")]
        horizon = 63
        if "--horizon" in args:
            idx = args.index("--horizon")
            if idx + 1 < len(args):
                horizon = int(args[idx + 1])

    for ticker in tickers:
        run_decision_report(ticker, horizon_td=horizon)


if __name__ == "__main__":
    main()
