"""
Gate Analysis — Growth Signal Bot
══════════════════════════════════════════════════════════════════════════════
Answers two questions:

  1. FALSE NEGATIVES — which gates are blocking good trades?
     For each required gate (MACD, ATR, consec_above, bear_div), find bars where
     the core setup was valid (score + regime) but THAT gate blocked entry.
     Then check forward returns: if price rose after the block, it was a missed win.

  2. FALSE POSITIVES — what conditions correlate with losing trades?
     For trades that fired and hit the stop loss, extract what score / conditions
     were present at entry. This reveals borderline setups that slipped through.

  3. GATE REMOVAL IMPACT — full backtest with each gate individually removed
     Shows the trade-off: more trades vs. lower win rate.

  4. GRID SEARCH — sweeps req_macd_x × atr_mult × score_moderate
     Finds the Pareto-optimal settings (best Sharpe).

Usage:
    python run_gate_analysis.py
    python run_gate_analysis.py --ticker GOOG --period 5y
    python run_gate_analysis.py --ticker NVDA --period 3y --fwd-bars 10
    python run_gate_analysis.py --no-grid          # skip grid search (faster)
"""
from __future__ import annotations
import argparse
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from backtester.data    import fetch_ohlcv
from backtester.engine  import BacktestEngine
from backtester.metrics import win_rate, profit_factor
from strategies         import GrowthSignalBot


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Gate sensitivity analysis for Growth Signal Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples:
          python run_gate_analysis.py                        # GOOG 5y
          python run_gate_analysis.py --ticker NVDA --period 3y
          python run_gate_analysis.py --ticker META --no-grid
          python run_gate_analysis.py --fwd-bars 5           # 1-week forward window
        """),
    )
    p.add_argument("--ticker",    default="GOOG",  help="Ticker (default: GOOG)")
    p.add_argument("--period",    default="5y",    help="Lookback: 1y 2y 3y 5y 10y (default: 5y)")
    p.add_argument("--fwd-bars",  type=int, default=10,
                   help="Forward-return window in bars for near-miss analysis (default: 10 ≈ 2 weeks)")
    p.add_argument("--capital",   type=float, default=10000)
    p.add_argument("--no-grid",   action="store_true", help="Skip grid search")
    p.add_argument("--refresh",   action="store_true", help="Force re-download data")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(df_raw: pd.DataFrame, capital: float = 10000, **strategy_kwargs) -> dict:
    """Run a full backtest and return a flat result dict."""
    strat   = GrowthSignalBot(**strategy_kwargs)
    engine  = BacktestEngine(
        strategy        = strat,
        data            = df_raw.copy(),
        initial_capital = capital,
        commission_pct  = 0.0005,   # 0.05% per side
    )
    result  = engine.run()
    trades  = result.trades
    summary = result.summary()
    wr = win_rate(trades)      if trades else 0.0
    pf = profit_factor(trades) if trades else 0.0
    return {
        "_strategy": strat,   # expose prepared df
        "_trades":   trades,
        "trades":    len(trades),
        "win_rate":  round(wr * 100, 1),
        "pf":        round(pf, 2),
        "total_ret": round(summary["total_return"], 1),
        "cagr":      round(summary["cagr"], 1),
        "max_dd":    round(summary["max_drawdown"], 1),
        "sharpe":    round(summary["sharpe"], 2),
    }


def _pnl_pct(trade) -> float:
    """Long trade P&L as a percentage."""
    if trade.exit_price is None or trade.entry_price == 0:
        return 0.0
    return (trade.exit_price - trade.entry_price) / trade.entry_price * 100


def sep(char="═", n=68):
    print(char * n)

def section(title: str):
    print()
    sep()
    print(f"  {title}")
    sep()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Signal Funnel
# ─────────────────────────────────────────────────────────────────────────────

def print_funnel(df: pd.DataFrame, score_moderate: int) -> None:
    section(f"SIGNAL FUNNEL  (score_moderate = {score_moderate})")

    valid = df.dropna(subset=["bull_score"])
    n = len(valid)

    def pct(a, b):
        return f"({a/b*100:.0f}%)" if b else ""

    score_pass   = int((valid["bull_score"] >= score_moderate).sum())
    regime_pass  = int(((valid["bull_score"] >= score_moderate) & valid["bull_regime"]).sum())
    consec_pass  = int(((valid["bull_score"] >= score_moderate) & valid["bull_regime"] & valid["consec_above"]).sum())
    atr_pass     = int(((valid["bull_score"] >= score_moderate) & valid["bull_regime"] & valid["consec_above"]
                        & (valid["close"] > valid["ema20"] + valid["atr_buffer"])).sum())
    div_pass     = int(((valid["bull_score"] >= score_moderate) & valid["bull_regime"] & valid["consec_above"]
                        & (valid["close"] > valid["ema20"] + valid["atr_buffer"])
                        & ~valid["bear_div"]).sum())
    macd_pass    = int(valid["buy_signal"].sum())

    print(f"  Total bars (after warmup):       {n:>5}")
    print(f"  Score >= {score_moderate}:                    {score_pass:>5}  {pct(score_pass, n)}")
    print(f"  + bull_regime:                   {regime_pass:>5}  {pct(regime_pass, score_pass)}  filtered {score_pass - regime_pass}")
    print(f"  + consec_above (2 bars):         {consec_pass:>5}  {pct(consec_pass, regime_pass)}  filtered {regime_pass - consec_pass}")
    atr_filtered = consec_pass - atr_pass
    print(f"  + ATR buffer:                    {atr_pass:>5}  {pct(atr_pass, consec_pass)}  filtered {atr_filtered}")
    div_filtered = atr_pass - div_pass
    print(f"  + ~bear_div:                     {div_pass:>5}  {pct(div_pass, atr_pass)}  filtered {div_filtered}")
    macd_filtered = div_pass - macd_pass
    print(f"  + MACD crossover (required):     {macd_pass:>5}  {pct(macd_pass, div_pass)}  filtered {macd_filtered}")
    print(f"\n  ➤  Final buy_signal fires:       {macd_pass:>5}")
    if macd_filtered > 0:
        print(f"     MACD is the biggest filter — {macd_filtered} setups blocked")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Near-Miss Quality (False Negative analysis)
# ─────────────────────────────────────────────────────────────────────────────

def print_near_misses(df: pd.DataFrame, score_moderate: int, fwd_bars: int) -> None:
    section(f"NEAR-MISS QUALITY  (forward {fwd_bars} bars ≈ {fwd_bars // 5 or 1} weeks)")
    print(f"  Shows bars where the CORE setup passed (score + regime + ~bear_div)")
    print(f"  but one specific gate blocked entry.  Hit% = % where price later rose > +0.5%\n")

    # Forward return at fwd_bars from each bar
    fwd_ret = df["close"].pct_change(fwd_bars).shift(-fwd_bars) * 100

    # Core = score + regime + not bear divergence
    core = (
        (df["bull_score"] >= score_moderate)
        & df["bull_regime"]
        & ~df["bear_div"]
    )

    gates = {
        "consec_above":  df["consec_above"],
        "ATR buffer":    df["close"] > df["ema20"] + df["atr_buffer"],
        "MACD cross":    df["macd_cross"],
    }

    print(f"  {'Gate':<22} {'Blocked':>7}  {'Hit%':>5}  {'Avg Fwd Ret':>12}  Interpretation")
    print(f"  {'─' * 70}")

    for gate_name, gate_series in gates.items():
        # Other gates that must also pass (so we isolate just this one)
        other = {k: v for k, v in gates.items() if k != gate_name}
        other_mask = pd.Series(True, index=df.index)
        for v in other.values():
            other_mask = other_mask & v

        # Near-miss: core passes, other gates pass, but THIS gate fails
        nm_mask = core & other_mask & ~gate_series
        nm_df   = df[nm_mask].copy()
        nm_ret  = fwd_ret[nm_mask]
        valid   = nm_ret.dropna()

        if len(valid) == 0:
            print(f"  {gate_name:<22} {'0':>7}  {'—':>5}  {'—':>12}  no near-misses")
            continue

        hit_pct = (valid > 0.5).mean() * 100
        avg_ret = valid.mean()

        if hit_pct >= 55:
            note = "⚠  HIGH FN — many profitable setups blocked"
        elif hit_pct >= 45:
            note = "△  moderate FN — worth reviewing threshold"
        else:
            note = "✓  gate is effective (most blocks were correct)"

        print(f"  {gate_name:<22} {len(nm_df):>7}  {hit_pct:>4.0f}%  {avg_ret:>+11.1f}%  {note}")

    # Random baseline (how often does price just go up anyway?)
    rand_fwd   = fwd_ret.dropna()
    rand_hit   = (rand_fwd > 0.5).mean() * 100
    rand_avg   = rand_fwd.mean()
    print(f"\n  {'Baseline (any bar)':22} {'—':>7}  {rand_hit:>4.0f}%  {rand_avg:>+11.1f}%  random baseline")
    print(f"\n  If a gate's Hit% > baseline ({rand_hit:.0f}%), it's blocking genuinely good setups.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Gate Removal Impact
# ─────────────────────────────────────────────────────────────────────────────

def print_gate_removal(df_raw: pd.DataFrame, baseline_kwargs: dict, capital: float) -> None:
    section("GATE REMOVAL IMPACT  (full backtest per config)")
    print(f"  {'Configuration':<38} {'Trades':>6}  {'WR%':>5}  {'PF':>5}  {'CAGR%':>6}  {'Sharpe':>6}  {'MaxDD%':>7}")
    print(f"  {'─' * 75}")

    configs = [
        ("Baseline (all gates on)",           {}),
        ("No MACD required",                  {"req_macd_x": False}),
        ("No MACD + score >= 9",              {"req_macd_x": False, "score_moderate": 9}),
        ("ATR mult → 0.2  (looser)",          {"atr_mult": 0.2}),
        ("ATR mult → 0.3",                    {"atr_mult": 0.3}),
        ("Score >= 7  (looser threshold)",    {"score_moderate": 7}),
        ("Score >= 9  (stricter threshold)",  {"score_moderate": 9}),
        ("Score >= 10 (strictest)",           {"score_moderate": 10}),
        ("All Signals mode",                  {"entry_mode": "All Signals"}),
    ]

    for label, overrides in configs:
        kwargs = {**baseline_kwargs, **overrides}
        # strip internal keys
        clean  = {k: v for k, v in kwargs.items() if not k.startswith("_")}
        r = run_backtest(df_raw, capital=capital, **clean)
        marker = "  ◄ baseline" if label.startswith("Baseline") else ""
        print(f"  {label:<38} {r['trades']:>6}  {r['win_rate']:>5}  {r['pf']:>5}  {r['cagr']:>6}  {r['sharpe']:>6}  {r['max_dd']:>7}{marker}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. False Positive Analysis
# ─────────────────────────────────────────────────────────────────────────────

def print_fp_analysis(trades: list, df: pd.DataFrame) -> None:
    section("FALSE POSITIVE (FP) ANALYSIS")

    if not trades:
        print("  No trades to analyze.")
        return

    completed = [t for t in trades if t.exit_price is not None]
    winners   = [t for t in completed if _pnl_pct(t) > 0]
    losers    = [t for t in completed if _pnl_pct(t) <= 0]

    print(f"  Trades completed:    {len(completed)}")
    print(f"  Winners (TP hit):    {len(winners)}  ({len(winners)/len(completed)*100:.1f}%)")
    print(f"  Losers  (SL hit):    {len(losers)}  ({len(losers)/len(completed)*100:.1f}%)")

    if not losers:
        print("\n  No losing trades — nothing to investigate for FP.")
        return

    # Extract score and key conditions at each trade entry date
    def get_entry_row(trade):
        try:
            return df.loc[trade.entry_date]
        except KeyError:
            # Try nearest date
            idx = df.index.get_indexer([trade.entry_date], method="nearest")
            return df.iloc[idx[0]] if len(idx) else None

    winner_scores, loser_scores   = [], []
    winner_rsi,    loser_rsi      = [], []
    loser_conditions              = []

    for t in winners:
        row = get_entry_row(t)
        if row is not None:
            winner_scores.append(row["bull_score"])
            winner_rsi.append(row["rsi"])

    for t in losers:
        row = get_entry_row(t)
        if row is not None:
            loser_scores.append(row["bull_score"])
            loser_rsi.append(row["rsi"])
            loser_conditions.append({
                "score":        row["bull_score"],
                "rsi":          row["rsi"],
                "s_poc":        row.get("s_poc", np.nan),
                "s_weekly_tf":  row.get("s_weekly_tf", np.nan),
                "s_vwap_w":     row.get("s_vwap_w", np.nan),
                "macd_cross":   row.get("macd_cross", np.nan),
            })

    print(f"\n  Score at entry:")
    if winner_scores:
        print(f"    Winners: avg {np.mean(winner_scores):.1f}  "
              f"(range {min(winner_scores):.0f}–{max(winner_scores):.0f})")
    if loser_scores:
        print(f"    Losers:  avg {np.mean(loser_scores):.1f}  "
              f"(range {min(loser_scores):.0f}–{max(loser_scores):.0f})")

    print(f"\n  RSI at entry:")
    if winner_rsi:
        print(f"    Winners: avg {np.mean(winner_rsi):.1f}")
    if loser_rsi:
        print(f"    Losers:  avg {np.mean(loser_rsi):.1f}")

    if loser_conditions:
        lc = pd.DataFrame(loser_conditions)
        print(f"\n  Conditions on losing trades (borderline flags):")
        if "s_poc" in lc:
            below_poc_losers = (lc["s_poc"] == 0).sum()
            print(f"    Below POC at entry:         {below_poc_losers}/{len(losers)}  "
                  f"({'⚠' if below_poc_losers > len(losers)//3 else '✓'})")
        if "s_weekly_tf" in lc:
            weak_weekly = (lc["s_weekly_tf"] == 0).sum()
            print(f"    Weekly TF NOT confirming:   {weak_weekly}/{len(losers)}  "
                  f"({'⚠' if weak_weekly > len(losers)//3 else '✓'})")
        if "s_vwap_w" in lc:
            below_vwap = (lc["s_vwap_w"] == 0).sum()
            print(f"    Below weekly VWAP:          {below_vwap}/{len(losers)}  "
                  f"({'⚠' if below_vwap > len(losers)//3 else '✓'})")

        # Score distribution of losers
        score_counts = lc["score"].value_counts().sort_index()
        print(f"\n  Score distribution of losing trades:")
        for score, cnt in score_counts.items():
            bar = "█" * int(cnt)
            print(f"    score {int(score):>2}: {bar} {cnt}")

        min_score = int(lc["score"].min())
        borderline = (lc["score"] <= min_score + 1).sum()
        if borderline >= len(losers) // 2:
            print(f"\n  ⚠  {borderline}/{len(losers)} losses had score <= {min_score+1} (borderline entries)")
            print(f"     → Consider raising score_moderate by 1–2 pts to filter these out")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Grid Search
# ─────────────────────────────────────────────────────────────────────────────

def print_grid_search(df_raw: pd.DataFrame, baseline_kwargs: dict, capital: float) -> None:
    section("GRID SEARCH  (sorted by Sharpe)")

    base = {k: v for k, v in baseline_kwargs.items() if not k.startswith("_")}

    req_macd_opts   = [True, False]
    atr_mult_opts   = [0.2, 0.3, 0.4]
    score_mod_opts  = [7, 8, 9, 10]

    total = len(req_macd_opts) * len(atr_mult_opts) * len(score_mod_opts)
    print(f"  Running {total} combinations …  (this takes ~{total//4}–{total//2}s)\n")

    results = []
    done    = 0
    for req_x in req_macd_opts:
        for atr in atr_mult_opts:
            for score in score_mod_opts:
                kwargs = {**base, "req_macd_x": req_x, "atr_mult": atr, "score_moderate": score}
                r      = run_backtest(df_raw, capital=capital, **kwargs)
                r.update({"req_macd_x": req_x, "atr_mult": atr, "score": score})
                results.append(r)
                done += 1
                print(f"\r  {done}/{total} …", end="", flush=True)

    print()  # newline after progress
    results.sort(key=lambda x: x["sharpe"], reverse=True)

    print(f"  {'#':>3}  {'macd':>5}  {'atr':>5}  {'sc':>3}  "
          f"{'Trades':>6}  {'WR%':>5}  {'PF':>5}  {'CAGR%':>6}  {'Sharpe':>6}  {'MaxDD%':>7}")
    print(f"  {'─' * 68}")

    for i, r in enumerate(results[:12], 1):
        marker = "  ◄ best" if i == 1 else ("  ◄ baseline" if (
            r["req_macd_x"] == base.get("req_macd_x")
            and r["atr_mult"] == base.get("atr_mult")
            and r["score"] == base.get("score_moderate")
        ) else "")
        print(f"  {i:>3}  {str(r['req_macd_x']):>5}  {r['atr_mult']:>5.1f}  {r['score']:>3}  "
              f"{r['trades']:>6}  {r['win_rate']:>5}  {r['pf']:>5}  {r['cagr']:>6}  "
              f"{r['sharpe']:>6}  {r['max_dd']:>7}{marker}")

    best = results[0]
    print(f"\n  ➤  Best config: req_macd_x={best['req_macd_x']}, "
          f"atr_mult={best['atr_mult']}, score_moderate={best['score']}")
    print(f"     Sharpe {best['sharpe']}  |  CAGR {best['cagr']}%  |  "
          f"Win rate {best['win_rate']}%  |  {best['trades']} trades")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    print(f"\n{'─' * 68}")
    print(f"  GATE ANALYSIS — {args.ticker}  ({args.period})")
    print(f"{'─' * 68}")

    # ── Fetch data ─────────────────────────────────────────────────────────
    df_raw = fetch_ohlcv(args.ticker, args.period, force_refresh=args.refresh)
    print(f"  Data: {len(df_raw)} bars  "
          f"({df_raw.index[0].date()} → {df_raw.index[-1].date()})")

    # ── Baseline config ────────────────────────────────────────────────────
    baseline_kwargs = dict(
        req_macd_x    = True,
        atr_mult      = 0.4,
        score_moderate= 8,
        score_strong  = 11,
        poc_len       = 50,
        entry_mode    = "Buy Only",
    )

    # ── Baseline backtest + grab prepared df ──────────────────────────────
    print(f"  Running baseline backtest …")
    baseline = run_backtest(df_raw, capital=args.capital, **baseline_kwargs)
    strat    = baseline["_strategy"]
    df       = strat._df
    trades   = baseline["_trades"]

    print(f"  Baseline: {baseline['trades']} trades  |  "
          f"WR {baseline['win_rate']}%  |  PF {baseline['pf']}  |  "
          f"CAGR {baseline['cagr']}%  |  Sharpe {baseline['sharpe']}")

    # ── Analyses ───────────────────────────────────────────────────────────
    print_funnel(df, baseline_kwargs["score_moderate"])
    print_near_misses(df, baseline_kwargs["score_moderate"], args.fwd_bars)
    print_gate_removal(df_raw, baseline_kwargs, args.capital)
    print_fp_analysis(trades, df)

    if not args.no_grid:
        print_grid_search(df_raw, baseline_kwargs, args.capital)

    print()
    sep()
    print(f"  Analysis complete — {args.ticker} / {args.period}")
    sep()
    print()


if __name__ == "__main__":
    main()
