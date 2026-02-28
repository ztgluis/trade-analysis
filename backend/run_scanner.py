#!/usr/bin/env python3
"""
run_scanner.py  —  Ticker Scanner: S&P 500 + NASDAQ 100

Scans ~550 tickers via Finviz, ranks by attention score,
and returns the top 100 most interesting tickers for the day.

Usage:
    python run_scanner.py
    python run_scanner.py --top 50
    python run_scanner.py --category gainer
    python run_scanner.py --refresh
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from scanner.fetcher import fetch_scanner_data
from scanner.scorer  import score_and_rank


# ── Helpers ──────────────────────────────────────────────────────────────────

def _fmt_vol(v: float) -> str:
    try:
        v = float(v)
        if v >= 1_000_000:
            return f"{v / 1_000_000:.1f}M"
        if v >= 1_000:
            return f"{v / 1_000:.0f}K"
        return str(int(v))
    except (ValueError, TypeError):
        return "--"


def _fmt_pct(v: float, plus: bool = True) -> str:
    try:
        v = float(v)
        sign = "+" if plus and v >= 0 else ""
        return f"{sign}{v * 100:.1f}%"
    except (ValueError, TypeError):
        return "  --"


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = sys.argv[1:]

    top_n = 100
    category_filter = None
    force_refresh = "--refresh" in args

    if "--top" in args:
        idx = args.index("--top")
        if idx + 1 < len(args):
            top_n = int(args[idx + 1])

    if "--category" in args:
        idx = args.index("--category")
        if idx + 1 < len(args):
            category_filter = args[idx + 1]

    print(f"\n{'═' * 90}")
    print(f"  TICKER SCANNER  ·  S&P 500 + NASDAQ 100")
    print(f"{'═' * 90}")

    # 1. Fetch data from Finviz
    print()
    raw = fetch_scanner_data(force_refresh=force_refresh)
    print(f"  Received data for {len(raw)} tickers")

    # 2. Score and rank
    ranked = score_and_rank(raw, top_n=top_n)

    if category_filter:
        ranked = ranked[ranked["category"] == category_filter].reset_index(drop=True)

    # 3. Print results
    print(f"\n  Top {len(ranked)} tickers by attention score:")
    print(f"  {'─' * 86}")
    print(f"  {'#':>3}  {'Ticker':<7} {'Company':<26} {'Sector':<16} "
          f"{'Price':>8} {'Chg':>7} {'Volume':>9} {'RelVol':>7} "
          f"{'Gap':>7} {'Score':>5}  {'Cat':<8}")
    print(f"  {'─' * 86}")

    for i, row in ranked.iterrows():
        company = str(row.get("Company", ""))[:25]
        sector  = str(row.get("Sector", ""))[:15]
        price   = row.get("Price", 0) or 0
        change  = row.get("Change", 0) or 0
        volume  = row.get("Volume", 0) or 0
        rel_vol = row.get("Rel Volume", 0) or 0
        gap     = row.get("Gap", 0) or 0
        score   = row.get("attention_score", 0) or 0
        cat     = row.get("category", "")

        print(f"  {i + 1:>3}  {row['Ticker']:<7} {company:<26} {sector:<16} "
              f"${price:>7.2f} {_fmt_pct(change):>7} {_fmt_vol(volume):>9} "
              f"{rel_vol:>6.1f}x {_fmt_pct(gap):>7} {score:>5.1f}  {cat:<8}")

    print(f"  {'─' * 86}")

    # Category summary
    if "category" in ranked.columns:
        cats = ranked["category"].value_counts()
        parts = [f"{k}: {v}" for k, v in cats.items()]
        print(f"\n  Categories:  {' · '.join(parts)}")

    print(f"\n  ⚠  This is screening data only — not investment advice.")
    print(f"{'═' * 90}\n")


if __name__ == "__main__":
    main()
