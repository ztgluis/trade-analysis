"""
Composite attention scoring and ranking.

Takes raw Finviz data and computes a 0–100 "attention score" for each
ticker using percentile-ranked sub-scores for price action, volume,
and gaps.  Returns the top N tickers sorted by score.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

# Weights for each criterion (sum to 100)
WEIGHTS = {
    "abs_change":  25,
    "rel_volume":  25,
    "abs_volume":  10,
    "abs_gap":     20,
    "directional": 20,
}


def _percentile_rank(series: pd.Series) -> pd.Series:
    """Convert a series to 0–100 percentile ranks."""
    return series.rank(pct=True, method="average") * 100


def score_and_rank(
    df: pd.DataFrame,
    top_n: int = 100,
) -> pd.DataFrame:
    """
    Compute attention scores and return top N tickers.

    Expected input columns (from fetcher):
        Ticker, Company, Sector, Price, Change, Volume,
        Avg Volume, Rel Volume, Gap, Market Cap

    Returns DataFrame sorted by attention_score desc, with added columns:
        attention_score  (0–100 composite)
        category         (gainer / loser / volume / gap_up / gap_down)
    """
    if df.empty:
        return df

    out = df.copy()

    # Ensure numeric types
    for col in ["Change", "Volume", "Avg Volume", "Rel Volume", "Gap", "Price"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out["Change"]     = out.get("Change", pd.Series(0, index=out.index)).fillna(0)
    out["Volume"]     = out.get("Volume", pd.Series(0, index=out.index)).fillna(0)
    out["Gap"]        = out.get("Gap", pd.Series(0, index=out.index)).fillna(0)

    # Compute Rel Volume if missing
    if "Rel Volume" not in out.columns or out["Rel Volume"].isna().all():
        avg = out.get("Avg Volume", pd.Series(1, index=out.index)).replace(0, 1)
        out["Rel Volume"] = out["Volume"] / avg
    out["Rel Volume"] = out["Rel Volume"].fillna(1)

    # ── Sub-scores (each 0–100) ──────────────────────────────────────────────
    out["_s_abs_change"] = _percentile_rank(out["Change"].abs())
    out["_s_rel_volume"] = _percentile_rank(out["Rel Volume"])
    out["_s_abs_volume"] = _percentile_rank(out["Volume"])
    out["_s_abs_gap"]    = _percentile_rank(out["Gap"].abs())

    # Directional bonus: both extreme gainers AND extreme losers score high
    change_rank_asc  = out["Change"].rank(pct=True)   # 1.0 = biggest gainer
    change_rank_desc = 1 - change_rank_asc              # 1.0 = biggest loser
    out["_s_directional"] = np.maximum(change_rank_asc, change_rank_desc) * 100

    # ── Composite score (weighted sum, max ~100) ─────────────────────────────
    out["attention_score"] = (
        out["_s_abs_change"]  * WEIGHTS["abs_change"]  / 100 +
        out["_s_rel_volume"]  * WEIGHTS["rel_volume"]  / 100 +
        out["_s_abs_volume"]  * WEIGHTS["abs_volume"]  / 100 +
        out["_s_abs_gap"]     * WEIGHTS["abs_gap"]     / 100 +
        out["_s_directional"] * WEIGHTS["directional"] / 100
    )

    # ── Categorise by primary signal ─────────────────────────────────────────
    out["category"] = out.apply(_categorize, axis=1)

    # ── Clean up and return ──────────────────────────────────────────────────
    result_cols = [
        "Ticker", "Company", "Sector", "Industry", "Price", "Change",
        "Volume", "Avg Volume", "Rel Volume", "Gap",
        "Market Cap", "attention_score", "category",
    ]
    existing_cols = [c for c in result_cols if c in out.columns]

    result = out[existing_cols].sort_values("attention_score", ascending=False)
    return result.head(top_n).reset_index(drop=True)


def _categorize(row: pd.Series) -> str:
    """Assign each ticker a primary category based on its strongest signal."""
    change = row.get("Change", 0) or 0
    gap = row.get("Gap", 0) or 0
    rel_vol = row.get("Rel Volume", 0) or 0

    # Normalise to comparable scales:
    # Change/Gap as percentage points (e.g. 0.168 → 16.8)
    # Rel Volume as-is (e.g. 5.5)
    signals = {
        "gainer":   abs(change) * 100 if change > 0 else 0,
        "loser":    abs(change) * 100 if change < 0 else 0,
        "volume":   rel_vol if rel_vol > 2 else 0,
        "gap_up":   abs(gap) * 100 if gap > 0 else 0,
        "gap_down": abs(gap) * 100 if gap < 0 else 0,
    }
    return max(signals, key=signals.get)  # type: ignore[arg-type]
